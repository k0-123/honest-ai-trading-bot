"""Simulator for strategy-v2a: "v1 rules + NIFTYBEES parking" on the point-in-time Nifty 50.

Rules (fixed before the first v2a backtest):
- Signals: the frozen strategy-v1 BaselineStrategy, unchanged. Entries only for stocks that are Nifty 50 members on
  the decision date; a held stock that leaves the index keeps its v1 exit rules and stop. A held stock whose data ends
  (merger/delisting) is sold at its last close.
- Idle cash is parked in NIFTYBEES. At each open: stock sells first; then just enough NIFTYBEES is sold to fund the
  stock buys at that open's prices (+slippage, costs, cash buffer); then the stocks are bought; then any idle cash above
  a Rs 1,000 buffer is put back into NIFTYBEES (sale and stop-out proceeds included). The first open parks all capital.
- Sizing: equity includes NIFTYBEES; each stock is limited by the 1% risk rule and the 20% cap; NIFTYBEES counts as
  available funds. NIFTYBEES itself has no stop and no 20% cap (it is the parking asset).
- Risk: v1's daily-loss gate (blocks new stock entries only), 2xATR stops sold at the next open, and the 10%
  kill-switch, which sells everything including NIFTYBEES and then stays in cash (no re-parking).
"""
from __future__ import annotations

import math

import pandas as pd

from nsebot.config import Config
from nsebot.costs import apply_slippage, trade_costs
from nsebot.portfolio import Book, Order, Position
from nsebot.risk.manager import _CASH_BUFFER
from nsebot.sim import HISTORY_WINDOW, Simulator, Strategy, _d

PARK_BUFFER = 1_000.0      # cash kept outside NIFTYBEES
DATA_ENDED_DAYS = 7


class ParkingSimulator(Simulator):
    def __init__(self, book: Book, cfg: Config, universe, strategy: Strategy, calendar_days=None,
                 kill_switch: bool = True):
        super().__init__(book, cfg, strategy, [], kill_switch=kill_switch)
        self.universe, self.bench = universe, cfg.benchmark

    # ---------- fills ----------
    def _sell_part(self, date, qty: int, raw_price: float, reason: str) -> None:
        pos = self.book.positions[self.bench]
        qty = min(qty, pos.qty)
        if qty <= 0:
            return
        px = apply_slippage("SELL", raw_price, self.cfg.costs)
        c = trade_costs("SELL", qty, px, self.cfg.costs).total
        self.book.cash += qty * px - c
        pos.qty -= qty
        if pos.qty == 0:
            self.book.positions.pop(self.bench)
        self.book.last_prices[self.bench] = px
        self._record(date, self.bench, "SELL", qty, px, c, reason, "parking")

    def _park(self, date, raw_price: float) -> None:
        b = self.book
        spare = b.cash - PARK_BUFFER
        px = apply_slippage("BUY", raw_price, self.cfg.costs)
        if spare < px * 1.01:
            return
        qty = math.floor(spare / (px * 1.002))
        while qty > 0 and qty * px + trade_costs("BUY", qty, px, self.cfg.costs).total > spare:
            qty -= 1
        if qty <= 0:
            return
        c = trade_costs("BUY", qty, px, self.cfg.costs).total
        b.cash -= qty * px + c
        if self.bench in b.positions:
            p = b.positions[self.bench]
            p.entry_price = (p.qty * p.entry_price + qty * px) / (p.qty + qty)
            p.qty += qty
        else:
            b.positions[self.bench] = Position(self.bench, qty, px, None, _d(date), "parking")
        b.last_prices[self.bench] = px
        self._record(date, self.bench, "BUY", qty, px, c, "PARKING: idle cash into NIFTYBEES", "parking")

    def on_open(self, date, bars_today: dict[str, pd.Series]) -> None:
        b, still = self.book, []
        for o in [o for o in b.pending if o.side == "SELL"]:
            bar = bars_today.get(o.symbol)
            if bar is None:
                still.append(o)
                continue
            if o.symbol in b.positions:
                self._sell(date, o.symbol, float(bar["open"]), o.reason, o.source, o.contaminated)
        buys = [o for o in b.pending if o.side == "BUY" and o.symbol in bars_today and o.symbol not in b.positions
                and not (o.stop is not None and float(bars_today[o.symbol]["open"]) <= o.stop)]
        nb_bar = bars_today.get(self.bench)
        need = 0.0
        for o in buys:
            px = apply_slippage("BUY", float(bars_today[o.symbol]["open"]), self.cfg.costs)
            need += o.qty * px * (1 + _CASH_BUFFER) + trade_costs("BUY", o.qty, px, self.cfg.costs).total
        if need > b.cash and self.bench in b.positions and nb_bar is not None:
            nb_px = apply_slippage("SELL", float(nb_bar["open"]), self.cfg.costs)
            per_unit = nb_px * (1 - self.cfg.costs.stt_pct - 0.0001)
            qty = math.ceil((need - b.cash + self.cfg.costs.dp_charge_per_sell) / per_unit)
            self._sell_part(date, qty, float(nb_bar["open"]),
                            "PARKING: sell NIFTYBEES to fund " + ", ".join(o.symbol for o in buys))
        for o in buys:
            self._buy(date, o, float(bars_today[o.symbol]["open"]))
        b.pending = still  # unfilled BUYs expire, unfilled SELLs wait (as in v1)
        self._today_raw = bars_today
        if not b.halted and nb_bar is not None:
            self._park(date, float(nb_bar["open"]))

    # ---------- decisions ----------
    def _decide(self, date, histories: dict[str, pd.DataFrame], eq: float) -> list:
        b, d, ds = self.book, pd.Timestamp(date), _d(date)
        self._exit_ended(d, histories)
        members = self.universe.members_on(d)
        held = {k for k in b.positions if k != self.bench}
        nb = b.positions.get(self.bench)
        available = b.cash + (nb.qty * b.last_prices.get(self.bench, nb.entry_price) * 0.997 if nb else 0.0)
        available -= sum(o.qty * b.last_prices.get(o.symbol, 0) for o in b.pending if o.side == "BUY")
        pending_syms = {o.symbol for o in b.pending}
        todays = []
        for sym in sorted((members | held) - {self.bench}):
            h = histories.get(sym)
            if h is None or not len(h) or h.index[-1] != d:
                continue
            sig = self.strategy.generate(sym, h.tail(HISTORY_WINDOW), sym in b.positions)
            todays.append(sig)
            if sym in pending_syms:
                continue
            if sig.action == "SELL" and sym in b.positions:
                b.pending.append(Order(sym, "SELL", b.positions[sym].qty, sig.reason, sig.source, ds))
            elif sig.action == "BUY" and sym not in b.positions and sym in members and not b.block_new_entries:
                close = float(h["close"].iloc[-1])
                qty = self.risk.size(eq, available, close, sig.stop)
                if qty > 0:
                    available -= qty * close * (1 + _CASH_BUFFER)
                    b.pending.append(Order(sym, "BUY", qty, sig.reason, sig.source, ds, stop=sig.stop))
        self.signals.extend(todays)
        return todays

    def _exit_ended(self, d: pd.Timestamp, histories: dict[str, pd.DataFrame]) -> None:
        for k in [k for k in self.book.positions if k != self.bench]:
            h = histories.get(k)
            if h is None or not len(h):
                continue
            last = h.index[-1]
            if (d - last).days >= DATA_ENDED_DAYS:
                self.book.pending = [o for o in self.book.pending if o.symbol != k]
                self._sell(d, k, float(h["close"].iloc[-1]),
                           f"data ended {_d(last)} (merger/delisting): exit at last close", "baseline")
