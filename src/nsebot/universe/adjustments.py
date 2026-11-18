"""Documented price corrections for the v2b universe (corporate actions neither source adjusts, and Yahoo
adjustment errors). Each row: multiply OHLC of bars strictly BEFORE `date` by `factor` (volume / factor).
The file lives at data/nifty50/adjustments.csv and every row carries its evidence. v1 does not use this."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT = Path(__file__).resolve().parents[3] / "data" / "nifty50" / "adjustments.csv"
_OVERRIDES = Path(__file__).resolve().parents[3] / "data" / "nifty50" / "bar_overrides.csv"


def load_adjustments(path: str | Path = _DEFAULT) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], comment="#")
    return df.sort_values(["key", "date"]).reset_index(drop=True)


def apply_adjustments(bars: pd.DataFrame, key: str, adj: pd.DataFrame) -> pd.DataFrame:
    rows = adj[adj["key"] == key]
    if rows.empty or bars.empty:
        return bars
    out = bars.copy()
    for r in rows.itertuples():
        before = out.index < pd.Timestamp(r.date)
        out.loc[before, ["open", "high", "low", "close"]] *= float(r.factor)
        if "volume" in out:
            out.loc[before, "volume"] /= float(r.factor)
    return out


def load_bar_overrides(path: str | Path = _OVERRIDES) -> pd.DataFrame:
    """Stale/bad single Yahoo bars, replaced by NSE's real OHLC. Stored in NSE (raw) prices together with NSE's
    previous close, and rescaled to Yahoo's adjustment basis at load time (robust to later re-adjustments)."""
    df = pd.read_csv(path, parse_dates=["date"], comment="#")
    return df.sort_values(["key", "date"]).reset_index(drop=True)


def apply_bar_overrides(bars: pd.DataFrame, key: str, ov: pd.DataFrame) -> pd.DataFrame:
    rows = ov[ov["key"] == key]
    if rows.empty or bars.empty:
        return bars
    out = bars.copy()
    for r in rows.itertuples():
        d = pd.Timestamp(r.date)
        if d not in out.index:
            continue
        prev = out.loc[:d].iloc[:-1]
        if prev.empty:
            continue
        scale = float(prev["close"].iloc[-1]) / float(r.nse_prev_close)
        for col in ("open", "high", "low", "close"):
            out.loc[d, col] = float(getattr(r, col)) * scale
    return out
