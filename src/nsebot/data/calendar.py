"""Session calendar helpers. Holidays are handled data-first: a day is a session iff a bar exists for it."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def is_weekday(d: date | datetime | str | pd.Timestamp) -> bool:
    return pd.Timestamp(d).weekday() < 5


def has_session(bars: pd.DataFrame, d: date | datetime | str | pd.Timestamp) -> bool:
    return pd.Timestamp(d).normalize() in bars.index
