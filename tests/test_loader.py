from __future__ import annotations

import pandas as pd
import pytest

from nsebot.data.calendar import has_session, is_weekday
from nsebot.data.loader import PriceLoader, normalize

BAR_COLS = ["open", "high", "low", "close", "volume"]


def _raw_frame(dates, *, multiindex: bool = False, symbol: str = "RELIANCE.NS") -> pd.DataFrame:
    """yfinance-style raw frame (capitalized columns) for the given dates."""
    idx = pd.DatetimeIndex(dates, name="Date")
    base = pd.Series(range(len(idx)), index=idx, dtype=float)
    close = 100.0 + base * 0.5
    df = pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1000.0 + base,
        },
        index=idx,
    )
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, [symbol]], names=["Price", "Ticker"])
    return df


class FakeFetcher:
    """Fetcher stand-in: (symbol, start, end) -> raw frame, inclusive of start and end, counts calls."""

    def __init__(self, dates, *, multiindex: bool = False, overrides: dict | None = None):
        self.dates = pd.DatetimeIndex(dates)
        self.multiindex = multiindex
        self.overrides = overrides or {}
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        mask = self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        raw = _raw_frame(self.dates[mask], multiindex=self.multiindex)
        for day, values in self.overrides.items():
            if pd.Timestamp(day) in raw.index:
                for col, val in values.items():
                    raw.loc[pd.Timestamp(day), col] = val
        return raw

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def _weekdays(start: str, end: str) -> list[pd.Timestamp]:
    return [d for d in pd.bdate_range(start, end)]


# ---------- normalize ----------

def test_normalize_lowercases_columns_and_sets_date_index():
    raw = _raw_frame(["2024-01-02", "2024-01-03", "2024-01-04"])
    out = normalize(raw)
    assert list(out.columns) == BAR_COLS
    assert out.index.name == "date"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique
    assert all(out[c].dtype.kind == "f" for c in BAR_COLS)


def test_normalize_flattens_multiindex_columns():
    raw = _raw_frame(["2024-01-02", "2024-01-03"], multiindex=True)
    assert isinstance(raw.columns, pd.MultiIndex)
    out = normalize(raw)
    assert list(out.columns) == BAR_COLS
    assert out["close"].iloc[0] == pytest.approx(100.0)


def test_normalize_strips_timezone_and_time_of_day():
    raw = _raw_frame(["2024-01-02", "2024-01-03"])
    raw.index = raw.index.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
    out = normalize(raw)
    assert out.index.tz is None
    assert list(out.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_normalize_drops_nan_ohlc_rows():
    raw = _raw_frame(["2024-01-02", "2024-01-03", "2024-01-04"])
    raw.loc[pd.Timestamp("2024-01-03"), "Close"] = float("nan")
    out = normalize(raw)
    assert pd.Timestamp("2024-01-03") not in out.index
    assert len(out) == 2


def test_normalize_drops_non_positive_prices():
    raw = _raw_frame(["2024-01-02", "2024-01-03", "2024-01-04"])
    raw.loc[pd.Timestamp("2024-01-03"), "Low"] = 0.0
    out = normalize(raw)
    assert pd.Timestamp("2024-01-03") not in out.index
    assert (out[["open", "high", "low", "close"]] > 0).all().all()


def test_normalize_drops_duplicate_dates():
    raw = _raw_frame(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"])
    out = normalize(raw)
    assert out.index.is_unique
    assert len(out) == 3


# ---------- PriceLoader.get ----------

def test_get_returns_bars_in_requested_range(tmp_path):
    days = _weekdays("2024-01-01", "2024-02-29")
    fake = FakeFetcher(days)
    df = PriceLoader(tmp_path, fetcher=fake).get("RELIANCE.NS", "2024-01-10", "2024-01-20")
    assert list(df.columns) == BAR_COLS
    assert df.index.name == "date"
    assert df.index.min() >= pd.Timestamp("2024-01-10")
    assert df.index.max() <= pd.Timestamp("2024-01-20")
    assert len(df) == len(_weekdays("2024-01-10", "2024-01-20"))


def test_cache_written_to_parquet_and_reused(tmp_path):
    days = _weekdays("2024-01-01", "2024-02-29")
    fake = FakeFetcher(days)
    first = PriceLoader(tmp_path, fetcher=fake).get("RELIANCE.NS", "2024-01-01", "2024-02-01")
    assert (tmp_path / "RELIANCE.NS.parquet").exists()
    assert fake.n_calls == 1

    # new loader instance, same cache dir, same range: must not fetch again
    second = PriceLoader(tmp_path, fetcher=FakeFetcher(days)).get("RELIANCE.NS", "2024-01-01", "2024-02-01")
    pd.testing.assert_frame_equal(first, second)

    same_loader = PriceLoader(tmp_path, fetcher=fake)
    same_loader.get("RELIANCE.NS", "2024-01-01", "2024-02-01")
    assert fake.n_calls == 1


def test_refresh_fetches_from_last_cached_minus_5_days_and_merges(tmp_path):
    days = _weekdays("2024-01-01", "2024-03-29")
    early = FakeFetcher([d for d in days if d <= pd.Timestamp("2024-02-01")])
    loader = PriceLoader(tmp_path, fetcher=early)
    before = loader.get("RELIANCE.NS", "2024-01-01")
    assert before.index.max() == pd.Timestamp("2024-02-01")

    # the provider now also has later data and a revised value for 2024-01-30
    later = FakeFetcher(days, overrides={"2024-01-30": {"Close": 123.45}})
    refreshed = PriceLoader(tmp_path, fetcher=later).get("RELIANCE.NS", "2024-01-01", refresh=True)

    assert later.n_calls == 1
    _, fetch_start, _ = later.calls[0]
    assert pd.Timestamp(fetch_start) == pd.Timestamp("2024-02-01") - pd.Timedelta(days=5)

    # new rows appended, old rows kept, overlapping date takes the new value
    assert refreshed.index.max() == pd.Timestamp("2024-03-29")
    assert refreshed.index.is_unique and refreshed.index.is_monotonic_increasing
    assert refreshed.loc[pd.Timestamp("2024-01-30"), "close"] == pytest.approx(123.45)
    assert refreshed.loc[pd.Timestamp("2024-01-02"), "close"] == pytest.approx(
        before.loc[pd.Timestamp("2024-01-02"), "close"]
    )
    # the cache on disk reflects the merge
    on_disk = pd.read_parquet(tmp_path / "RELIANCE.NS.parquet")
    assert on_disk.index.max() == pd.Timestamp("2024-03-29")


def test_as_of_excludes_all_later_rows(tmp_path):
    days = _weekdays("2024-01-01", "2024-03-29")
    loader = PriceLoader(tmp_path, fetcher=FakeFetcher(days))
    df = loader.get("RELIANCE.NS", "2024-01-01", "2024-03-29", as_of="2024-02-09")
    assert df.index.max() == pd.Timestamp("2024-02-09")
    assert (df.index <= pd.Timestamp("2024-02-09")).all()

    # cache already holds future rows; as_of must still hide them
    df_all = loader.get("RELIANCE.NS", "2024-01-01", "2024-03-29")
    assert df_all.index.max() == pd.Timestamp("2024-03-29")
    df_ts = loader.get("RELIANCE.NS", "2024-01-01", "2024-03-29", as_of=pd.Timestamp("2024-02-09"))
    assert df_ts.index.max() == pd.Timestamp("2024-02-09")


def test_missing_weekday_stays_missing_no_forward_fill(tmp_path):
    days = [d for d in _weekdays("2024-01-01", "2024-01-31") if d != pd.Timestamp("2024-01-17")]
    loader = PriceLoader(tmp_path, fetcher=FakeFetcher(days))
    df = loader.get("RELIANCE.NS", "2024-01-15", "2024-01-19")
    assert pd.Timestamp("2024-01-17") not in df.index
    assert list(df.index) == [
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-16"),
        pd.Timestamp("2024-01-18"),
        pd.Timestamp("2024-01-19"),
    ]
    assert not df.isna().any().any()


def test_backfill_when_requested_start_precedes_cache(tmp_path):
    days = _weekdays("2024-01-01", "2024-03-29")
    fake = FakeFetcher(days)
    loader = PriceLoader(tmp_path, fetcher=fake)
    loader.get("RELIANCE.NS", "2024-02-01", "2024-03-01")
    assert fake.n_calls == 1
    older = loader.get("RELIANCE.NS", "2024-01-01", "2024-03-01")
    assert older.index.min() == pd.Timestamp("2024-01-01")
    assert older.index.max() == pd.Timestamp("2024-03-01")


def test_empty_fetch_returns_empty_bars(tmp_path):
    fake = FakeFetcher([])
    df = PriceLoader(tmp_path, fetcher=fake).get("NOSUCH.NS", "2024-01-01", "2024-02-01")
    assert len(df) == 0
    assert list(df.columns) == BAR_COLS
    assert df.index.name == "date"


# ---------- calendar ----------

def test_is_weekday():
    assert is_weekday("2024-01-01") is True   # Monday
    assert is_weekday(pd.Timestamp("2024-01-05")) is True   # Friday
    assert is_weekday("2024-01-06") is False  # Saturday
    assert is_weekday("2024-01-07") is False  # Sunday


def test_has_session_true_and_false():
    bars = normalize(_raw_frame(["2024-01-02", "2024-01-03", "2024-01-05"]))
    assert has_session(bars, "2024-01-02") is True
    assert has_session(bars, pd.Timestamp("2024-01-03")) is True
    # Jan 4 is a weekday with no bar (e.g. exchange holiday): no session
    assert has_session(bars, "2024-01-04") is False
    # weekend
    assert has_session(bars, "2024-01-06") is False


def test_drop_spikes_removes_split_glitch_but_keeps_real_crash():
    import pandas as _pd

    from nsebot.data.loader import drop_spikes
    idx = _pd.bdate_range("2019-12-16", periods=8, name="date")
    c = [128.0, 129.0, 12.9, 13.0, 130.0, 129.5, 129.0, 60.0]  # 2-bar glitch; last bar a real -53% crash
    df = _pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    out = drop_spikes(df)
    assert list(out["close"]) == [128.0, 129.0, 130.0, 129.5, 129.0, 60.0]


def test_drop_spikes_keeps_permanent_level_shift():
    import pandas as _pd

    from nsebot.data.loader import drop_spikes
    idx = _pd.bdate_range("2020-01-01", periods=6, name="date")
    c = [100.0, 100.0, 40.0, 41.0, 40.0, 42.0]  # never reverts -> genuine, keep all
    df = _pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    assert len(drop_spikes(df)) == 6
