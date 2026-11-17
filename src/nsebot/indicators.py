"""Causal technical indicators. The value at index t depends only on data at or before t."""
from __future__ import annotations

import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    """Simple moving average over n values; NaN until n values are available."""
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI: ewm(alpha=1/n, adjust=False, min_periods=n) of gains and losses.

    All gains (no losses) gives 100. A perfectly flat window (no gains, no losses) gives 50.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR: ewm(alpha=1/n, adjust=False, min_periods=n) of the true range.

    True range = max(high - low, |high - prev_close|, |low - prev_close|). The first bar has no
    previous close, so its true range is high - low.
    """
    high, low = bars["high"], bars["low"]
    prev_close = bars["close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=True)
    return true_range.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
