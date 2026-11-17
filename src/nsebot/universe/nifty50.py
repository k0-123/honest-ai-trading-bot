"""NIFTY 50 point-in-time membership and the price-data check for it.

Membership is rebuilt from the current official list by undoing every change whose effective date is after
the requested date (see data/nifty50/README.md). Only the data files are used, so a date never sees a
constituent that was not known on that date.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

SNAPSHOT_DATE = "2026-11-17"
_SNAPSHOT_TS = pd.Timestamp(SNAPSHOT_DATE)

# official symbol in releases -> symbol used for prices today (same company, renamed)
CANONICAL = {
    "MCDOWELL-N": "UNITDSPR",
    "INFRATEL": "INDUSTOWER",
    "IBULHSGFIN": "SAMMAANCAP",
    "ZOMATO": "ETERNAL",
    "TATAMOTORS": "TMPV",
    "LTIM": "LTM",
}
# no Yahoo history: prices come from NSE via jugaad-data (EQ series only)
NSE_SOURCED = {"HDFC.NS", "CAIRN.NS", "TATAMTRDVR.NS", "IDFC.NS"}

_DEFAULT_DATA_DIR = "data/nifty50"
_CHECK_LOOKBACK_DAYS = 400


def price_key(symbol: str) -> str:
    """Price-data key (Yahoo-style ticker) for an official NSE symbol."""
    return CANONICAL.get(symbol, symbol) + ".NS"


class MissingPriceData(Exception):
    """Price data does not cover a NIFTY 50 membership span."""


def _ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


class Nifty50Universe:
    def __init__(self, data_dir: str | Path = _DEFAULT_DATA_DIR):
        data_dir = Path(data_dir)
        snap = pd.read_csv(data_dir / f"current_{SNAPSHOT_DATE}.csv", dtype=str)
        self._snapshot: frozenset[str] = frozenset(price_key(s) for s in snap["symbol"])

        changes = pd.read_csv(data_dir / "changes.csv", dtype=str)
        bad = set(changes["action"]) - {"IN", "OUT"}
        if bad:
            raise ValueError(f"unknown action(s) in changes.csv: {sorted(bad)}")
        changes["effective_date"] = pd.to_datetime(changes["effective_date"])
        changes = changes.sort_values("effective_date", kind="stable")
        # ascending by date; stored as (date, action, price key)
        self._changes: list[tuple[pd.Timestamp, str, str]] = [
            (r.effective_date, r.action, price_key(r.symbol)) for r in changes.itertuples(index=False)
        ]
        self._change_dates: list[pd.Timestamp] = sorted({c[0] for c in self._changes})

    def members_on(self, date) -> frozenset[str]:
        """Price keys of the index on ``date``: the snapshot with every change effective after ``date`` undone."""
        d = _ts(date)
        if d > _SNAPSHOT_TS:
            return self._snapshot
        members = set(self._snapshot)
        for eff, action, key in reversed(self._changes):  # newest first
            if eff <= d:
                break
            if action == "IN":
                members.discard(key)
            else:
                members.add(key)
        return frozenset(members)

    def ever_members(self, start, end) -> frozenset[str]:
        """Union of ``members_on`` over [start, end]. Membership only moves on effective dates."""
        return frozenset(k for _, _, members in self._segments(start, end) for k in members)

    def membership_spans(self, start, end) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        """Inclusive membership spans per key, clipped to [start, end]."""
        spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for seg_start, seg_end, members in self._segments(start, end):
            for key in members:
                intervals = spans.setdefault(key, [])
                if intervals and intervals[-1][1] == seg_start - pd.Timedelta(days=1):
                    intervals[-1] = (intervals[-1][0], seg_end)  # contiguous: extend
                else:
                    intervals.append((seg_start, seg_end))
        return spans

    def _segments(self, start, end) -> list[tuple[pd.Timestamp, pd.Timestamp, frozenset[str]]]:
        """Constant-membership segments covering [start, end]: (seg_start, seg_end, members), inclusive."""
        s, e = _ts(start), _ts(end)
        if e < s:
            raise ValueError(f"end {e.date()} is before start {s.date()}")
        boundaries = [s] + [d for d in self._change_dates if s < d <= e]
        out = []
        for i, b in enumerate(boundaries):
            seg_end = (boundaries[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(boundaries) else e
            out.append((b, seg_end, self.members_on(b)))
        return out


class UniversePriceLoader:
    """Same ``.get()`` signature as PriceLoader; routes NSE_SOURCED keys to the NSE loader, others to Yahoo."""

    def __init__(self, yahoo_loader, nse_loader):
        self.yahoo_loader = yahoo_loader
        self.nse_loader = nse_loader

    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True) -> pd.DataFrame:
        loader = self.nse_loader if symbol in NSE_SOURCED else self.yahoo_loader
        return loader.get(symbol, start, end, as_of=as_of, refresh=refresh, clean=clean)


def check_prices(universe: Nifty50Universe, loader, start: str, end: str, tolerance_days: int = 7) -> None:
    """Verify price data covers every membership span in [start, end]; raise MissingPriceData otherwise.

    For each key: the raw history from ``start`` minus 400 days to ``end`` must be non-empty, start no later
    than each span's start + tolerance, and end no earlier than each span's end - tolerance.
    """
    spans = universe.membership_spans(start, end)
    tol = pd.Timedelta(days=tolerance_days)
    lookback_start = (pd.Timestamp(start) - pd.Timedelta(days=_CHECK_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    problems: list[str] = []
    for key in sorted(spans):
        intervals = spans[key]
        try:
            bars = loader.get(key, lookback_start, end, clean=False)
        except Exception as exc:  # noqa: BLE001 - any fetcher error is recorded and raised below, never skipped
            problems.append(f"{key} {_fmt(intervals)}: fetch failed ({exc!r})")
            continue
        if bars is None or len(bars) == 0:
            problems.append(f"{key} {_fmt(intervals)}: no price data")
            continue
        first, last = pd.Timestamp(bars.index.min()), pd.Timestamp(bars.index.max())
        for s, e in intervals:
            if first > s + tol:
                problems.append(f"{key} span {_d(s)}..{_d(e)}: data starts {_d(first)}, too late")
            if last < e - tol:
                problems.append(f"{key} span {_d(s)}..{_d(e)}: data ends {_d(last)}, too early")
    if problems:
        raise MissingPriceData(
            f"price data does not cover NIFTY 50 membership for {start}..{end}:\n  " + "\n  ".join(problems)
        )


def _d(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def _fmt(intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    return ", ".join(f"{_d(s)}..{_d(e)}" for s, e in intervals)
