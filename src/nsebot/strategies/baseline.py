"""Rule-based baseline: SMA trend filter + RSI entry cross, with SMA/RSI exits.

Uses only the `history` frame passed in (bars up to and including day t).
"""
from __future__ import annotations

import math

import pandas as pd

from nsebot.config import BaselineCfg
from nsebot.indicators import atr, rsi, sma
from nsebot.strategies.base import Signal

CONFIDENCE = 0.6


class BaselineStrategy:
    source = "baseline"

    def __init__(self, cfg: BaselineCfg, atr_stop_mult: float = 2.0):
        self.cfg = cfg
        self.atr_stop_mult = atr_stop_mult

    def generate(self, symbol: str, history: pd.DataFrame, in_position: bool) -> Signal:
        cfg = self.cfg
        date = history.index[-1] if len(history) else pd.NaT

        def hold(reason: str) -> Signal:
            return Signal(date=date, symbol=symbol, action="HOLD", confidence=0.0,
                          reason=reason, source=self.source)

        if len(history) < cfg.sma_slow + 2:
            return hold("insufficient history")

        close = history["close"].astype(float)
        s_fast = sma(close, cfg.sma_fast)
        s_slow = sma(close, cfg.sma_slow)
        r = rsi(close, cfg.rsi_period)
        a = atr(history, cfg.atr_period)
        t = len(history) - 1
        k = cfg.exit_below_fast_days

        needed = [close.iloc[t], s_fast.iloc[t], s_slow.iloc[t], r.iloc[t], r.iloc[t - 1], a.iloc[t]]
        needed += list(s_fast.iloc[t - k: t + 1]) if k > 0 else []
        if any(math.isnan(float(x)) for x in needed):
            return hold("insufficient history")

        c_t = float(close.iloc[t])
        f_t = float(s_fast.iloc[t])
        s_t = float(s_slow.iloc[t])
        r_prev, r_t = float(r.iloc[t - 1]), float(r.iloc[t])
        atr_t = float(a.iloc[t])
        fast_lbl, slow_lbl = f"SMA{cfg.sma_fast}", f"SMA{cfg.sma_slow}"

        if not in_position:
            trend = c_t > s_t and f_t > s_t
            crossed_up = r_prev < cfg.rsi_entry <= r_t
            exit_now = self._exit_reason(close, s_fast, s_slow, r, t) is not None
            # Consistency rule: never BUY on a bar where the SELL rule would already fire (found via the
            # first backtest: overlapping rules caused buy-today/sell-tomorrow churn).
            if trend and crossed_up and not exit_now:
                stop = c_t - self.atr_stop_mult * atr_t
                reason = (
                    f"BUY: uptrend (close {c_t:.1f} > {slow_lbl} {s_t:.1f}, "
                    f"{fast_lbl} {f_t:.1f} > {slow_lbl}); "
                    f"RSI({cfg.rsi_period}) recovered {r_prev:.1f}->{r_t:.1f} through {cfg.rsi_entry:g}; "
                    f"stop {stop:.1f} ({self.atr_stop_mult:.1f}xATR {atr_t:.1f})"
                )
                return Signal(date=date, symbol=symbol, action="BUY", confidence=CONFIDENCE,
                              reason=reason, source=self.source, stop=stop)
            return hold("HOLD: no entry condition met")

        why = self._exit_reason(close, s_fast, s_slow, r, t)
        if why is not None:
            return self._sell(symbol, date, why)
        return hold("HOLD: holding, no exit condition met")

    def _exit_reason(self, close, s_fast, s_slow, r, t) -> str | None:
        cfg, k = self.cfg, self.cfg.exit_below_fast_days
        c_t, f_t, s_t, r_t = float(close.iloc[t]), float(s_fast.iloc[t]), float(s_slow.iloc[t]), float(r.iloc[t])
        fast_lbl, slow_lbl = f"SMA{cfg.sma_fast}", f"SMA{cfg.sma_slow}"
        if r_t > cfg.rsi_exit:
            return f"SELL: RSI({cfg.rsi_period}) {r_t:.1f} > {cfg.rsi_exit:g}"
        # Optional (k=0 disables, the default): exit after k closes below a falling fast SMA. Disabled by
        # default because RSI pullbacks nearly always sit below the fast SMA, so it fights the entry.
        below_fast = k > 0 and bool((close.iloc[t - k + 1: t + 1] < s_fast.iloc[t - k + 1: t + 1]).all())
        f_prev = float(s_fast.iloc[t - k]) if k > 0 else f_t
        if below_fast and f_t < f_prev:
            return (f"SELL: close {c_t:.1f} below {fast_lbl} {f_t:.1f} for {k} closes in a row "
                    f"and {fast_lbl} falling ({f_prev:.1f}->{f_t:.1f})")
        if c_t < s_t:
            return f"SELL: close {c_t:.1f} below {slow_lbl} {s_t:.1f} (trend broken)"
        return None

    def _sell(self, symbol: str, date: pd.Timestamp, reason: str) -> Signal:
        return Signal(date=date, symbol=symbol, action="SELL", confidence=CONFIDENCE,
                      reason=reason, source=self.source)
