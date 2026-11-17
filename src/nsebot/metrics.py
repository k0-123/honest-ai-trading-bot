"""Performance and risk metrics for equity curves and round-trip trades."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple periodic pct changes; non-finite and leading NaN values are dropped."""
    r = equity.astype(float).pct_change()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods: int = 252) -> float:
    """Compound annual growth. Years = (number of return observations) / periods."""
    if len(equity) < 2:
        return 0.0
    first, last = float(equity.iloc[0]), float(equity.iloc[-1])
    years = (len(equity) - 1) / periods
    if first <= 0 or last <= 0 or years <= 0:
        return 0.0
    return float((last / first) ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods: int = 252) -> float:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if not math.isfinite(sd) or sd == 0.0:
        return 0.0
    excess = r - rf_annual / periods
    return float(excess.mean() / sd * math.sqrt(periods))


def sortino(returns: pd.Series, rf_annual: float = 0.0, periods: int = 252) -> float:
    """Mean excess return over downside deviation (sqrt of mean squared shortfall over all obs)."""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf_annual / periods
    if not (r < 0).any():
        return 0.0
    downside = excess.clip(upper=0.0)
    dd = math.sqrt(float((downside**2).mean()))
    if dd == 0.0:
        return 0.0
    return float(excess.mean() / dd * math.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall as a positive fraction (0.12 = 12%)."""
    eq = equity.astype(float)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (1.0 - eq / peak).fillna(0.0)
    return float(max(dd.max(), 0.0))


def var_cvar(returns: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical VaR and CVaR as positive loss fractions.

    VaR is the `level` quantile of losses (-returns, linear interpolation);
    CVaR is the mean of losses at or beyond VaR.
    """
    losses = -pd.Series(returns, dtype=float).dropna()
    if losses.empty:
        return 0.0, 0.0
    var = float(np.quantile(losses.to_numpy(), level))
    tail = losses[losses >= var]
    return var, float(tail.mean())


def win_rate(pnls: Sequence[float]) -> float:
    """Share of round-trip pnls strictly above zero; 0.0 when empty."""
    if len(pnls) == 0:
        return 0.0
    arr = np.asarray(pnls, dtype=float)
    return float((arr > 0).mean())


def round_trip_pnls(trades: pd.DataFrame) -> list[float]:
    """FIFO round-trip pnl per SELL fill, net of costs.

    Each SELL fill closes shares against the oldest open BUY lots of the same symbol.
    Its pnl is the sum over the lots it consumes of (sell - buy) * qty, minus the
    proportional buy costs and the proportional sell costs. One entry per SELL fill
    that matched at least one share. Unmatched sells and open lots produce no entry.
    """
    if trades is None or len(trades) == 0:
        return []
    df = trades.sort_values("date", kind="stable") if "date" in trades.columns else trades
    lots: dict[str, deque[list[float]]] = {}
    pnls: list[float] = []
    for row in df.itertuples(index=False):
        qty = int(row.qty)
        if qty <= 0:
            continue
        price = float(row.price)
        cost_ps = float(row.costs) / qty
        queue = lots.setdefault(row.symbol, deque())
        if row.side == "BUY":
            queue.append([float(qty), price, cost_ps])
            continue
        remaining = float(qty)
        matched = 0.0
        pnl = 0.0
        while remaining > 0 and queue:
            lot = queue[0]
            take = min(lot[0], remaining)
            pnl += (price - lot[1]) * take - lot[2] * take
            matched += take
            remaining -= take
            lot[0] -= take
            if lot[0] <= 0:
                queue.popleft()
        if matched > 0:
            pnls.append(float(pnl - cost_ps * matched))
    return pnls


def summarize(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    r = daily_returns(equity)
    var, cvar = var_cvar(r, 0.95)
    if trades is None:
        n_trades, total_costs, wr = 0, 0.0, 0.0
    else:
        n_trades = len(trades)
        total_costs = float(trades["costs"].sum()) if n_trades else 0.0
        wr = win_rate(round_trip_pnls(trades))
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(equity),
        "var95": var,
        "cvar95": cvar,
        "win_rate": wr,
        "n_trades": n_trades,
        "total_costs": total_costs,
    }
