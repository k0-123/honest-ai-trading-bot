"""Daily price loader with a parquet cache. Output follows the "bars" convention (see docs/contracts.md)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

Fetcher = Callable[[str, str, str | None], pd.DataFrame]  # (symbol, start, end) -> raw yfinance-style frame

BAR_COLS = ["open", "high", "low", "close", "volume"]
OHLC_COLS = ["open", "high", "low", "close"]
REFRESH_OVERLAP_DAYS = 5


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (Price, Ticker) even for one ticker; keep the price level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def yfinance_fetcher(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch daily bars for [start, end] inclusive (yfinance's end is exclusive, so one day is added)."""
    yf_end = None if end is None else (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(symbol, start=start, end=yf_end, auto_adjust=True, progress=False)
    return _flatten_columns(raw)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort ascending and keep the last row per date (later rows win)."""
    df = df.sort_index(kind="stable")  # stable: keeps input order among equal dates so "last" really is last
    return df[~df.index.duplicated(keep="last")]


def _empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="date")
    return pd.DataFrame({c: pd.Series(dtype=float) for c in BAR_COLS}, index=idx)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw yfinance-style frame -> bars: DatetimeIndex 'date' (tz-naive, midnight, ascending, unique),
    float columns open/high/low/close/volume. Rows with NaN OHLC or non-positive prices are dropped."""
    if raw is None or len(raw) == 0:
        return _empty_bars()
    df = _flatten_columns(raw)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in OHLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"raw frame lacks OHLC columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df[BAR_COLS].astype(float)
    df.index = pd.DatetimeIndex(idx.normalize().to_numpy(), name="date")  # no freq attribute

    df = df.dropna(subset=OHLC_COLS)
    df = df[(df[OHLC_COLS] > 0).all(axis=1)]
    df["volume"] = df["volume"].fillna(0.0)
    return _finalize(df)  # raw; cleaning happens causally on read (see drop_spikes)


class PriceLoader:
    """Cached daily bars per symbol. Cache file: {cache_dir}/{symbol}.parquet."""

    def __init__(self, cache_dir: str | Path, fetcher: Fetcher | None = None):
        self.cache_dir = Path(cache_dir)
        self.fetcher: Fetcher = fetcher or yfinance_fetcher

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.parquet"

    def _read_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index).to_numpy(), name="date")
        return df

    def _write_cache(self, symbol: str, bars: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(self._path(symbol))

    def _fetch(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        return normalize(self.fetcher(symbol, start, end))

    def get(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        as_of: str | pd.Timestamp | None = None,
        refresh: bool = False,
        clean: bool = True,
    ) -> pd.DataFrame:
        """clean=True: causal spike filter applied (signals). clean=False: raw bars (fills and stops)."""
        cached = self._read_cache(symbol)
        if cached is None:
            bars = self._fetch(symbol, start, end)
            if len(bars):
                self._write_cache(symbol, bars)
        else:
            bars = cached
            changed = False
            if refresh and len(bars):
                last = bars.index.max()
                fetch_start = (last - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)).strftime("%Y-%m-%d")
                new = self._fetch(symbol, fetch_start, None)
                if len(new):
                    bars = _finalize(pd.concat([bars, new]))
                    changed = True
            elif refresh:
                bars = self._fetch(symbol, start, None)
                changed = True
            # requested history older than what is cached: backfill the gap
            if len(bars) and pd.Timestamp(start) < bars.index.min():
                gap_end = (bars.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                older = self._fetch(symbol, start, gap_end)
                if len(older):
                    bars = _finalize(pd.concat([older, bars]))
                    changed = True
            if changed:
                self._write_cache(symbol, bars)

        if clean:
            bars = drop_spikes(bars)  # causal cleaning on the full raw history, before any slicing
        out = bars.loc[pd.Timestamp(start):]
        if end is not None:
            out = out.loc[: pd.Timestamp(end)]
        if as_of is not None:
            out = out.loc[: pd.Timestamp(as_of)]
        out = out.copy()
        out.index.name = "date"
        return out


def drop_spikes(bars: pd.DataFrame, threshold: float = 0.5, max_run: int = 3) -> pd.DataFrame:
    """CAUSAL bad-bar filter: whether bar i is kept depends only on bars <= i, so cleaning a prefix gives
    exactly the prefix of the cleaned full series (backtest and paper mode see identical data).

    A close more than `threshold` away from the last good close is held out. If the new level persists for
    more than `max_run` consecutive bars it is accepted as genuine from then on (earlier held-out bars stay
    out: we never revise the past). Cost: a genuine >50% one-day move is seen up to `max_run` days late."""
    if len(bars) < 2:
        return bars
    close = bars["close"].to_numpy()
    keep = [True] * len(close)
    last_good, run = close[0], 0
    for i in range(1, len(close)):
        if abs(close[i] / last_good - 1) > threshold and run < max_run:
            keep[i] = False
            run += 1
        else:
            last_good, run = close[i], 0
    if not all(keep):
        dates = [bars.index[k].strftime("%Y-%m-%d") for k, ok in enumerate(keep) if not ok]
        log.warning("holding out %d suspicious price bar(s): %s", len(dates), ", ".join(dates))
    return bars[keep]
