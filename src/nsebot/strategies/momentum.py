"""strategy-v2b: monthly 12-1 momentum on the point-in-time Nifty 50 (rules fixed before any backtest).

Rules:
- Rank Nifty 50 members (as of the decision date) by 12-month return excluding the most recent month:
  close 21 sessions ago / close 252 sessions ago - 1. A stock needs >= 253 bars up to the decision date.
- Hold the top 10, about 10% each. A held stock is sold only when its rank is worse than 20 (or it has no rank
  because it left the index); freed slots go to the best-ranked stocks not held.
- If NIFTYBEES closes below its 200-day average at the decision ("risk-off"), no new stocks are bought that month
  and idle cash stays as CASH (v2b never holds NIFTYBEES; it only reads its price). Sell rules work as normal.
  (Rule fixed by the user BEFORE any v2b backtest was run: the first draft parked cash in NIFTYBEES.)
Only data up to and including the decision date is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MomentumCfg:
    top_n: int = 10
    sell_rank: int = 20            # keep a holding while its rank <= this
    lookback: int = 252            # sessions
    skip: int = 21                 # most recent sessions excluded
    weight: float = 0.10           # target weight of a new position
    regime_sma: int = 200          # NIFTYBEES trend filter (risk-off -> no new buys, hold cash)

    @property
    def min_bars(self) -> int:
        return self.lookback + 1


def momentum_scores(histories: dict[str, pd.DataFrame], members, date, mcfg: MomentumCfg) -> pd.Series:
    """12-1 momentum for each member with enough history and a bar ON `date`. Uses only rows <= date."""
    d = pd.Timestamp(date)
    out = {}
    for key in sorted(members):
        h = histories.get(key)
        if h is None or not len(h):
            continue
        c = h["close"].loc[:d]
        if len(c) < mcfg.min_bars or c.index[-1] != d:
            continue
        out[key] = float(c.iloc[-1 - mcfg.skip] / c.iloc[-1 - mcfg.lookback] - 1)
    return pd.Series(out, dtype=float, name="momentum")


def rank_members(histories: dict[str, pd.DataFrame], universe, date, mcfg: MomentumCfg) -> pd.Series:
    """Rank 1 = strongest. Index = price keys; only point-in-time members of `universe` on `date`."""
    s = momentum_scores(histories, universe.members_on(date), date, mcfg)
    if s.empty:
        return pd.Series(dtype=float, name="rank")
    order = sorted(s.index, key=lambda k: (-s[k], k))  # ties broken by symbol, deterministic
    return pd.Series(range(1, len(order) + 1), index=order, dtype=float, name="rank")


def plan_rebalance(held: set[str], ranks: pd.Series, mcfg: MomentumCfg) -> tuple[list[str], list[str]]:
    """Return (sells, buys). Sell holdings ranked worse than sell_rank or unranked; buy best-ranked not held
    until top_n stocks are held."""
    sells = sorted(k for k in held if k not in ranks.index or ranks[k] > mcfg.sell_rank)
    keep = held - set(sells)
    buys = [k for k in ranks.index if k not in keep][: max(mcfg.top_n - len(keep), 0)]
    return sells, buys
