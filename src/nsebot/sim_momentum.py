"""Simulator for strategy-v2b (monthly momentum). Reuses v1's fills, costs, stops, daily-loss gate and
kill-switch from Simulator; only the decision step differs. See strategies/momentum.py for the rules."""
from __future__ import annotations

import logging
import math

import pandas as pd

from nsebot.config import Config
from nsebot.indicators import atr
from nsebot.portfolio import Book, Order
from nsebot.sim import HISTORY_WINDOW, Simulator, _d
from nsebot.strategies.momentum import MomentumCfg, plan_rebalance, rank_members

log = logging.getLogger(__name__)
SOURCE = "v2b"
DATA_ENDED_DAYS = 7  # a held stock with no bar for this many calendar days is treated as delisted/merged


class MomentumSimulator(Simulator):
    def __init__(self, book: Book, cfg: Config, universe, mcfg: MomentumCfg, calendar_days=None,
                 kill_switch: bool = True, holidays=()):
        super().__init__(book, cfg, None, [], kill_switch=kill_switch)
        self.universe, self.mcfg, self.bench = universe, mcfg, cfg.benchmark
        self.days = pd.DatetimeIndex(sorted(calendar_days)) if calendar_days is not None else pd.DatetimeIndex([])
        self.holidays = {pd.Timestamp(h) for h in holidays}
        self.last_ranks = pd.Series(dtype=float)
        self.decisions: list[tuple[pd.Timestamp, pd.Timestamp]] = []  # (decision close, next session)

    # ---------- calendar ----------
    def next_session(self, d: pd.Timestamp) -> pd.Timestamp:
        """True next trading day when the calendar is known (backtest); else next weekday that is not a listed
        NSE holiday (paper mode, where tomorrow's bar does not exist yet)."""
        i = self.days.searchsorted(d, side="right")
        if i < len(self.days):
            return self.days[i]
        n = d + pd.Timedelta(days=1)
        while n.dayofweek >= 5 or n in self.holidays:
            n += pd.Timedelta(days=1)
        return n

    def _is_rebalance_close(self, d: pd.Timestamp) -> tuple[bool, str]:
        nxt = self.next_session(d)
        target = nxt.strftime("%Y-%m")
        if nxt.month != d.month:
            return True, target
        last = self.book.meta.get("last_rebalance_month")
        cur = d.strftime("%Y-%m")
        if last is not None and last < cur:  # month end was missed (unlisted holiday): rebalance one session late
            log.warning("v2b: month-end rebalance for %s missed; rebalancing late at close of %s", cur, _d(d))
            return True, cur
        return False, target

    # ---------- decisions ----------
    def _decide(self, date, histories: dict[str, pd.DataFrame], eq: float) -> list:
        b, d, ds = self.book, pd.Timestamp(date), _d(date)
        self._exit_ended(d, histories)
        ok, target = self._is_rebalance_close(d)
        if not ok:
            return []
        b.meta["last_rebalance_month"] = target
        self.decisions.append((d, self.next_session(d)))
        ranks = rank_members(histories, self.universe, d, self.mcfg)
        self.last_ranks = ranks
        held = set(b.positions)
        sells, buys = plan_rebalance(held, ranks, self.mcfg)
        pending_sells = {o.symbol for o in b.pending if o.side == "SELL"}
        est_cash = b.cash
        for k in sells:
            pos = b.positions[k]
            est_cash += pos.qty * b.last_prices.get(k, pos.entry_price) * 0.997
            if k in pending_sells:
                continue
            why = f"rank {int(ranks[k])} > {self.mcfg.sell_rank}" if k in ranks.index else "no rank (left Nifty 50 or no data)"
            b.pending.append(Order(k, "SELL", pos.qty, f"REBALANCE SELL {ds}: {why}", SOURCE, ds))

        nb = histories.get(self.bench)
        nbc = nb["close"].loc[:d] if nb is not None else pd.Series(dtype=float)
        sma = float(nbc.tail(self.mcfg.regime_sma).mean()) if len(nbc) >= self.mcfg.regime_sma else float("nan")
        nb_close = float(nbc.iloc[-1]) if len(nbc) else float("nan")
        risk_on = math.isnan(sma) or nb_close >= sma  # unknown SMA (too little history) -> treated as risk-on

        if not risk_on:
            # risk-off: hold cash. Sells above still happen; no new stocks this month.
            log.info("v2b %s: risk-off (NIFTYBEES %.2f < SMA200 %.2f): no new buys, holding cash", ds, nb_close, sma)
            return []
        if b.block_new_entries:
            return []
        for k in buys:
            h = histories[k].loc[:d].tail(HISTORY_WINDOW)
            close = float(h["close"].iloc[-1])
            a = float(atr(h, 14).iloc[-1])
            stop = close - self.cfg.risk.atr_stop_mult * a
            qty = min(math.floor(eq * self.mcfg.weight / close), self.risk.size(eq, est_cash, close, stop))
            if qty <= 0:
                continue
            est_cash -= qty * close * 1.003
            b.pending.append(Order(k, "BUY", qty,
                                   f"REBALANCE BUY {ds}: rank {int(ranks[k])}, 12-1 momentum "
                                   f"{self._mom(histories[k], d):+.1%}; stop {stop:.2f} "
                                   f"({self.cfg.risk.atr_stop_mult:.1f}xATR {a:.2f})", SOURCE, ds, stop=stop))
        return []

    def _mom(self, h: pd.DataFrame, d: pd.Timestamp) -> float:
        c = h["close"].loc[:d]
        return float(c.iloc[-1 - self.mcfg.skip] / c.iloc[-1 - self.mcfg.lookback] - 1)

    def _exit_ended(self, d: pd.Timestamp, histories: dict[str, pd.DataFrame]) -> None:
        """A held stock whose data stopped (merger/delisting) is sold at its last available close."""
        for k in list(self.book.positions):
            h = histories.get(k)
            if h is None or not len(h):
                continue
            last = h.index[-1]
            if (d - last).days >= DATA_ENDED_DAYS:
                self.book.pending = [o for o in self.book.pending if o.symbol != k]
                self._sell(d, k, float(h["close"].iloc[-1]),
                           f"data ended {_d(last)} (merger/delisting): exit at last close", SOURCE)
