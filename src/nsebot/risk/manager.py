"""Position sizing, stop fills and portfolio-level circuit breakers."""
from __future__ import annotations

import math

import pandas as pd

from nsebot.config import RiskCfg

# Rough round-trip cost buffer used only to keep a cash-limited order fillable.
_CASH_BUFFER = 0.003
_EPS = 1e-9  # so a loss of exactly the limit counts as a breach despite float rounding


class RiskManager:
    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg

    def size(self, equity: float, cash: float, entry_price: float, stop: float | None) -> int:
        """Shares to buy: min(1% risk / per-share risk, 20% equity cap, cash). 0 if no valid stop."""
        if stop is None or entry_price <= 0 or stop <= 0 or stop >= entry_price or equity <= 0:
            return 0
        by_risk = (equity * self.cfg.risk_per_trade_pct) / (entry_price - stop)
        by_cap = (equity * self.cfg.max_position_pct) / entry_price
        by_cash = max(cash, 0.0) / (entry_price * (1 + _CASH_BUFFER))
        return max(int(math.floor(min(by_risk, by_cap, by_cash))), 0)

    @staticmethod
    def stop_fill(stop: float, bar: pd.Series) -> float | None:
        """Raw (pre-slippage) exit price if the day's bar hits the stop. Gaps fill at the open."""
        if bar["open"] <= stop:
            return float(bar["open"])
        if bar["low"] <= stop:
            return float(stop)
        return None

    def daily_loss_breached(self, prev_equity: float, equity: float) -> bool:
        return prev_equity > 0 and (equity / prev_equity - 1) <= -self.cfg.daily_loss_limit_pct + _EPS

    def drawdown_breached(self, peak_equity: float, equity: float) -> bool:
        return peak_equity > 0 and (1 - equity / peak_equity) >= self.cfg.max_drawdown_pct - _EPS
