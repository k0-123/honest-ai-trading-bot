"""Train / validation / test periods and walk-forward windows (no overlap, no leakage)."""
from __future__ import annotations

import pandas as pd


def split_periods(start: str, end: str, train_end: str, val_end: str) -> dict[str, tuple[str, str]]:
    s, e, t, v = map(pd.Timestamp, (start, end, train_end, val_end))
    if not (s < t < v < e):
        raise ValueError("need start < train_end < val_end < end")
    one = pd.Timedelta(days=1)
    f = lambda x: x.strftime("%Y-%m-%d")
    return {"train": (f(s), f(t)), "val": (f(t + one), f(v)), "test": (f(v + one), f(e))}


def walk_forward_windows(start: str, end: str, train_years: int = 3, test_months: int = 6
                         ) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Rolling (train, test) windows; each test window starts the day after its train window ends."""
    out, cur = [], pd.Timestamp(start)
    end_ts, f = pd.Timestamp(end), (lambda x: x.strftime("%Y-%m-%d"))
    while True:
        tr_end = cur + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        te_start = tr_end + pd.Timedelta(days=1)
        te_end = te_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if te_start > end_ts:
            break
        out.append(((f(cur), f(tr_end)), (f(te_start), f(min(te_end, end_ts)))))
        cur = cur + pd.DateOffset(months=test_months)
    return out
