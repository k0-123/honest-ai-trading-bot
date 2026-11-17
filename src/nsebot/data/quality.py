"""Corporate-action (split/bonus) and data-quality audit for every symbol."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nsebot.data.loader import drop_spikes

BIG_MOVE = 0.15


@dataclass
class SymbolAudit:
    symbol: str
    n_bars: int
    actions: list[tuple[str, float]] = field(default_factory=list)          # (date, split ratio) from Yahoo
    action_max_move: dict[str, float] = field(default_factory=dict)        # max |ret| within +-3 bars
    big_moves: list[tuple[str, float]] = field(default_factory=list)       # |ret| > 15%
    held_out: list[str] = field(default_factory=list)                      # bars the spike filter holds out
    nse_checks: dict[str, str] = field(default_factory=dict)               # action date -> result text


def yahoo_splits(symbol: str) -> list[tuple[str, float]]:
    import yfinance as yf
    s = yf.Ticker(symbol).splits
    return [(pd.Timestamp(d).tz_localize(None).strftime("%Y-%m-%d"), float(v)) for d, v in s.items()]


def audit_symbol(symbol: str, raw: pd.DataFrame, actions: list[tuple[str, float]],
                 nse_fetcher=None) -> SymbolAudit:
    a = SymbolAudit(symbol, len(raw), actions)
    ret = raw["close"].pct_change()
    a.big_moves = [(d.strftime("%Y-%m-%d"), round(float(v), 4)) for d, v in ret[ret.abs() > BIG_MOVE].items()]
    clean = drop_spikes(raw)
    a.held_out = [d.strftime("%Y-%m-%d") for d in raw.index.difference(clean.index)]
    for date, _ratio in actions:
        d = pd.Timestamp(date)
        pos = raw.index.searchsorted(d)
        win = ret.iloc[max(pos - 3, 1): pos + 4]
        a.action_max_move[date] = round(float(win.abs().max()), 4) if len(win) else float("nan")
        if nse_fetcher is not None:
            from nsebot.data.crosscheck import cross_check
            s, e = (d - pd.Timedelta(days=15)).strftime("%Y-%m-%d"), (d + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
            try:
                nse = nse_fetcher(symbol, s, e)
                r = cross_check(raw.loc[s:e], nse, symbol, tol_pct=0.01)
                a.nse_checks[date] = (f"{r.n_compared} days compared, {r.n_mismatch} return mismatches >1%, "
                                      f"max diff {r.max_abs_pct_diff:.2%}")
            except Exception as ex:  # noqa: BLE001  NSE frequently blocks automated access
                a.nse_checks[date] = f"NSE fetch failed ({type(ex).__name__})"
    return a


def render(audits: list[SymbolAudit]) -> str:
    out = ["# Data check: splits, bonuses and bad bars", "",
           ("Yahoo prices are split/bonus adjusted. For each corporate action Yahoo records, we check that the "
           "adjusted series has no artificial jump (max |daily return| within ±3 bars) and, where NSE allows, "
           "that daily returns match NSE's own (unadjusted) prices around the event."), "",
           ("| Symbol | Bars | Yahoo splits/bonuses | Max move near action | NSE return check | Moves >15% | "
           "Held-out bars |"), "|---|---|---|---|---|---|---|"]
    for a in audits:
        acts = "<br>".join(f"{d} (x{r:g})" for d, r in a.actions) or "none"
        near = "<br>".join(f"{a.action_max_move[d]:.1%}" for d, _ in a.actions) or "-"
        nse = "<br>".join(a.nse_checks.get(d, "not run") for d, _ in a.actions) or "-"
        big = "<br>".join(f"{d} ({v:+.1%})" for d, v in a.big_moves) or "none"
        out.append(f"| {a.symbol} | {a.n_bars} | {acts} | {near} | {nse} | {big} | "
                   f"{', '.join(a.held_out) or 'none'} |")
    return "\n".join(out) + "\n"
