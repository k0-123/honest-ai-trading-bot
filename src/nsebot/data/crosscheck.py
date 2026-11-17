"""Cross-check Yahoo (primary) daily bars against NSE bhavcopy-style bars from jugaad-data (secondary).

Why returns, not price levels: Yahoo bars are split/dividend ADJUSTED (auto_adjust=True) while NSE bars
from jugaad-data are RAW. Adjustment rescales historical closes, so the levels legitimately differ by a
factor that changes at every corporate action. Daily close-to-close returns are invariant to a constant
scale factor, so they are the right thing to compare. Dividends shift levels slightly, which is a known
and small source of return difference; the tolerance absorbs it.

Method:
- Restrict both series to dates present in both sources (so a multi-day gap in one source never shows up as
  a return that the other source does not have).
- Compare simple returns on that common-date series. A mismatch is |ret_yahoo - ret_nse| > tol_pct.
- ``pct_diff`` and ``max_abs_pct_diff`` are fractions (0.005 = 0.5%), the same unit as ``tol_pct``.
- Dates present in only one source are reported (``missing_in_primary`` lists dates in the secondary
  source that the primary lacks, and vice versa), as ISO ``YYYY-MM-DD`` strings.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from nsebot.data.loader import normalize

MISMATCH_COLS = ["date", "yahoo_close", "nse_close", "pct_diff"]


@dataclass
class CrossCheckResult:
    symbol: str
    n_compared: int
    n_mismatch: int
    max_abs_pct_diff: float
    mismatches: pd.DataFrame  # cols: date, yahoo_close, nse_close, pct_diff (fraction)
    missing_in_primary: list[str] = field(default_factory=list)
    missing_in_secondary: list[str] = field(default_factory=list)


def jugaad_fetcher(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch NSE EQ-series daily bars via jugaad-data and return them in the bars convention.

    ``symbol`` may carry the Yahoo ".NS" suffix; it is stripped. ``end=None`` means today.
    jugaad_data is imported lazily so the package works without it installed.
    """
    from jugaad_data.nse import stock_df

    bare = symbol.removesuffix(".NS")
    from_date = dt.date.fromisoformat(start)
    to_date = dt.date.today() if end is None else dt.date.fromisoformat(end)
    raw = stock_df(symbol=bare, from_date=from_date, to_date=to_date, series="EQ")
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "date" in raw.columns:
        # jugaad-data returns IST midnight as a naive UTC timestamp (e.g. 2024-10-28 18:30 == 29 Oct IST).
        # Interpret as UTC and convert to IST before taking the date, else every bar lands one day early.
        d = pd.to_datetime(raw["date"])
        if d.dt.tz is None:
            d = d.dt.tz_localize("UTC")
        raw.index = d.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        raw = raw.drop(columns="date")
    return normalize(raw)


def _iso(idx: pd.Index) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in idx]


def cross_check(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    symbol: str,
    tol_pct: float = 0.005,
) -> CrossCheckResult:
    """Compare daily close-to-close returns of ``primary`` (Yahoo) and ``secondary`` (NSE) on common dates."""
    p_idx = pd.DatetimeIndex(primary.index)
    s_idx = pd.DatetimeIndex(secondary.index)
    common = p_idx.intersection(s_idx).sort_values()

    p_close = primary["close"].astype(float).reindex(common)
    s_close = secondary["close"].astype(float).reindex(common)

    ret_p = p_close.pct_change().iloc[1:]
    ret_s = s_close.pct_change().iloc[1:]
    diff = ret_p - ret_s
    abs_diff = diff.abs()

    flagged = abs_diff > tol_pct
    flagged_idx = ret_p.index[flagged.to_numpy()]
    mismatches = pd.DataFrame(
        {
            "date": flagged_idx,
            "yahoo_close": p_close.loc[flagged_idx].to_numpy(),
            "nse_close": s_close.loc[flagged_idx].to_numpy(),
            "pct_diff": diff.loc[flagged_idx].to_numpy(),
        },
        columns=MISMATCH_COLS,
    )

    return CrossCheckResult(
        symbol=symbol,
        n_compared=len(diff),
        n_mismatch=int(flagged.sum()),
        max_abs_pct_diff=float(abs_diff.max()) if len(abs_diff) else 0.0,
        mismatches=mismatches,
        missing_in_primary=_iso(s_idx.difference(p_idx)),
        missing_in_secondary=_iso(p_idx.difference(s_idx)),
    )
