"""One-day-at-a-time simulator shared by the backtest engine and the paper runner.

Timeline for trading day d:
  on_open(d):  fill orders decided at the previous close at d's OPEN (+slippage, +costs).
  (stops)      at d's close, a position whose stop was breached during d (raw low <= stop) gets a SELL
               for the NEXT open: the bot runs once after close and has no resting stop order.
  on_close(d): mark to market at d's close, apply circuit breakers, then ask the strategy for
               signals using history up to and including d. Orders wait for the next open.
Nothing here ever sees a bar later than d.

Price inputs: on_open gets the RAW bar (fills and stops use real exchange prints immediately, even a bar the
spike filter is still holding out); on_close gets the CLEANED history (signals and mark-to-market), so a data
glitch cannot by itself create signals or trip the kill-switch. Trade-off: a raw glitch can still cause a
fake stop-out; `nsebot crosscheck` is the guard for that.
"""
from __future__ import annotations

import math
from typing import Protocol

import pandas as pd

from nsebot.config import Config
from nsebot.costs import apply_slippage, trade_costs
from nsebot.portfolio import Book, Order, Position
from nsebot.risk.manager import RiskManager
from nsebot.strategies.base import Signal

HISTORY_WINDOW = 400  # bars passed to strategies; identical in backtest and paper mode


class Strategy(Protocol):
    source: str

    def generate(self, symbol: str, history: pd.DataFrame, in_position: bool) -> Signal: ...


def _d(date) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


class Simulator:
    def __init__(self, book: Book, cfg: Config, strategy: Strategy | None, symbols: list[str],
                 buy_and_hold: bool = False, kill_switch: bool = True):
        self.book, self.cfg, self.strategy, self.symbols = book, cfg, strategy, symbols
        self.buy_and_hold = buy_and_hold
        self.kill_switch = kill_switch  # False only for DIAGNOSTIC backtests
        self.risk = RiskManager(cfg.risk)
        self.signals: list[Signal] = []
        self._today_raw: dict[str, pd.Series] = {}

    # ---------- fills ----------
    def _record(self, date, symbol, side, qty, price, costs, reason, source, contaminated=False):
        self.book.trades.append({
            "date": _d(date), "symbol": symbol, "side": side, "qty": int(qty), "price": round(price, 4),
            "costs": round(costs, 4), "reason": reason, "signal_source": source,
            "contaminated": bool(contaminated), "book": self.book.name,
        })

    def _sell(self, date, symbol, raw_price, reason, source, contaminated=False):
        pos = self.book.positions.pop(symbol)
        px = apply_slippage("SELL", raw_price, self.cfg.costs)
        c = trade_costs("SELL", pos.qty, px, self.cfg.costs).total
        self.book.cash += pos.qty * px - c
        self.book.last_prices[symbol] = px
        self._record(date, symbol, "SELL", pos.qty, px, c, reason, source, contaminated)

    def _buy(self, date, order: Order, raw_price):
        if order.stop is not None and raw_price <= order.stop:
            return  # opened at/below the planned stop: entry no longer valid, skip
        px = apply_slippage("BUY", raw_price, self.cfg.costs)
        qty = order.qty
        if order.stop is not None:
            # Re-apply the 1%-risk and 20%-cap limits at the real fill price (a gap up changes both).
            qty = min(qty, self.risk.size(self.book.equity(), self.book.cash, px, order.stop))
        # A gap up can make the order unaffordable: shrink it to fit cash including costs.
        while qty > 0 and qty * px + trade_costs("BUY", qty, px, self.cfg.costs).total > self.book.cash:
            qty = min(qty - 1, math.floor(self.book.cash / (px * 1.002)))
        if qty <= 0:
            return
        c = trade_costs("BUY", qty, px, self.cfg.costs).total
        self.book.cash -= qty * px + c
        self.book.positions[order.symbol] = Position(order.symbol, qty, px, order.stop, _d(date), order.source)
        self.book.last_prices[order.symbol] = px
        self._record(date, order.symbol, "BUY", qty, px, c, order.reason, order.source, order.contaminated)

    def on_open(self, date, bars_today: dict[str, pd.Series]) -> None:
        still_pending: list[Order] = []
        orders = sorted(self.book.pending, key=lambda o: o.side != "SELL")  # sells free cash first
        for o in orders:
            bar = bars_today.get(o.symbol)
            if bar is None:
                if o.side == "SELL":
                    still_pending.append(o)  # exits wait for the next session the symbol trades
                # BUYs expire: an entry must fill at the very next session's open or not at all
                continue
            if o.side == "SELL":
                if o.symbol in self.book.positions:
                    self._sell(date, o.symbol, float(bar["open"]), o.reason, o.source, o.contaminated)
            elif o.symbol not in self.book.positions:
                self._buy(date, o, float(bar["open"]))
        self.book.pending = still_pending
        self._today_raw = bars_today  # the close step checks stops against today's raw low

    # ---------- close ----------
    def on_close(self, date, histories: dict[str, pd.DataFrame]) -> list[Signal]:
        eq = self._mark_and_guard(date, histories)
        if eq is None:
            return []
        return self._decide(date, histories, eq)

    def _mark_and_guard(self, date, histories: dict[str, pd.DataFrame]) -> float | None:
        """Mark to market, apply the kill-switch / daily-loss gate and queue stop exits.
        Returns equity, or None when the book is halted (no new decisions)."""
        b, ds = self.book, _d(date)
        for sym, h in histories.items():
            if len(h) and h.index[-1] == pd.Timestamp(date):
                b.last_prices[sym] = float(h["close"].iloc[-1])
        eq = b.equity()
        prev = b.last_equity if b.last_equity > 0 else eq
        b.peak_equity = max(b.peak_equity, eq)
        b.equity_curve.append((ds, round(eq, 2)))
        b.invested_curve.append((ds, round(eq - b.cash, 2)))
        b.last_equity, b.last_date = eq, ds

        if b.halted:
            return None
        # The passive benchmark is never subject to the bot's circuit breakers.
        if self.kill_switch and not self.buy_and_hold and self.risk.drawdown_breached(b.peak_equity, eq):
            dd = 1 - eq / b.peak_equity
            b.halted = True
            b.halt_reason = f"KILL-SWITCH {ds}: drawdown {dd:.1%} >= {self.cfg.risk.max_drawdown_pct:.0%}"
            b.pending = [Order(s, "SELL", p.qty, b.halt_reason, "risk", ds) for s, p in b.positions.items()]
            return None
        b.block_new_entries = self.risk.daily_loss_breached(prev, eq)

        # Stops: the bot runs once after the close and has no resting order at a broker, so a stop
        # breached during day d (raw low <= stop, including a gap below it) is SOLD AT THE NEXT OPEN.
        for sym, pos in b.positions.items():
            bar = self._today_raw.get(sym)
            if pos.stop is None or bar is None or not self.risk.stop_breached(pos.stop, bar):
                continue
            if any(o.symbol == sym and o.side == "SELL" for o in b.pending):
                continue
            b.pending.append(Order(sym, "SELL", pos.qty,
                                   f"STOP breached {ds}: low {float(bar['low']):.2f} <= stop {pos.stop:.2f} "
                                   f"(open {float(bar['open']):.2f}); sell at next open", pos.source, ds))
        return eq

    def _decide(self, date, histories: dict[str, pd.DataFrame], eq: float) -> list[Signal]:
        """v1 per-symbol signal logic (and the buy-and-hold benchmark)."""
        b, ds = self.book, _d(date)
        todays: list[Signal] = []
        if self.buy_and_hold:
            if not b.positions and not b.pending and not b.trades:
                sym = self.symbols[0]
                px = b.last_prices.get(sym)
                if px:
                    qty = int(b.cash / (px * 1.01))
                    b.pending.append(Order(sym, "BUY", qty, "Buy-and-hold benchmark: invest all cash once",
                                           "buyhold", ds))
            return todays

        assert self.strategy is not None
        pending_syms = {o.symbol for o in b.pending}
        for sym in self.symbols:
            h = histories.get(sym)
            if h is None or not len(h) or h.index[-1] != pd.Timestamp(date):
                continue  # no bar for this symbol today
            sig = self.strategy.generate(sym, h.tail(HISTORY_WINDOW), sym in b.positions)
            todays.append(sig)
            if sym in pending_syms:
                continue
            if sig.action == "SELL" and sym in b.positions:
                b.pending.append(Order(sym, "SELL", b.positions[sym].qty, sig.reason, sig.source, ds,
                                       contaminated=sig.contaminated))
            elif sig.action == "BUY" and sym not in b.positions:
                if b.block_new_entries:
                    continue
                close = float(h["close"].iloc[-1])
                committed = sum(o.qty * b.last_prices.get(o.symbol, 0) for o in b.pending if o.side == "BUY")
                qty = self.risk.size(eq, b.cash - committed, close, sig.stop)
                if qty > 0:
                    b.pending.append(Order(sym, "BUY", qty, sig.reason, sig.source, ds, stop=sig.stop,
                                           contaminated=sig.contaminated))
        self.signals.extend(todays)
        return todays
