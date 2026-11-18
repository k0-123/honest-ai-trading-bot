from __future__ import annotations

import datetime as dt

import jugaad_data.nse as jnse
import pandas as pd
import pytest

from nsebot.data.crosscheck import CrossCheckResult, cross_check, jugaad_fetcher


def _bars(closes, start="2024-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=idx,
    )


CLOSES = [100.0, 101.0, 99.5, 102.0, 103.5, 101.0, 104.0, 105.0]


def test_identical_returns_with_constant_adjustment_factor_no_mismatch():
    nse = _bars(CLOSES)
    yahoo = _bars([c * 0.8 for c in CLOSES])  # split/dividend-adjusted levels differ, returns do not
    res = cross_check(yahoo, nse, "RELIANCE")
    assert isinstance(res, CrossCheckResult)
    assert res.symbol == "RELIANCE"
    assert res.n_compared == len(CLOSES) - 1
    assert res.n_mismatch == 0
    assert res.mismatches.empty
    assert res.max_abs_pct_diff == pytest.approx(0.0, abs=1e-12)
    assert res.missing_in_primary == []
    assert res.missing_in_secondary == []


def test_single_bad_close_produces_mismatches_including_that_date():
    nse = _bars(CLOSES)
    bad = list(CLOSES)
    bad[3] = 150.0  # one corrupted close on the NSE side
    yahoo = _bars(CLOSES)
    res = cross_check(yahoo, _bars(bad), "RELIANCE", tol_pct=0.005)
    assert res.n_mismatch >= 1
    bad_date = nse.index[3]
    assert bad_date in set(pd.DatetimeIndex(res.mismatches["date"]))
    assert res.max_abs_pct_diff > 0.005
    cols = list(res.mismatches.columns)
    assert cols == ["date", "yahoo_close", "nse_close", "pct_diff"]
    row = res.mismatches.set_index("date").loc[bad_date]
    assert row["nse_close"] == pytest.approx(150.0)
    assert row["yahoo_close"] == pytest.approx(CLOSES[3])


def test_small_difference_within_tolerance_is_not_a_mismatch():
    yahoo = _bars(CLOSES)
    nse = _bars([c * (1.0 + (0.001 if i == 2 else 0.0)) for i, c in enumerate(CLOSES)])
    res = cross_check(yahoo, nse, "RELIANCE", tol_pct=0.005)
    assert res.n_mismatch == 0


def test_missing_dates_are_reported_on_each_side():
    full = _bars(CLOSES)
    primary = full.drop(full.index[4])
    secondary = full.drop(full.index[6])
    res = cross_check(primary, secondary, "RELIANCE")
    assert res.missing_in_primary == [full.index[4].strftime("%Y-%m-%d")]
    assert res.missing_in_secondary == [full.index[6].strftime("%Y-%m-%d")]


def test_no_common_dates_compares_nothing():
    a = _bars(CLOSES, start="2024-01-01")
    b = _bars(CLOSES, start="2025-01-01")
    res = cross_check(a, b, "RELIANCE")
    assert res.n_compared == 0
    assert res.n_mismatch == 0
    assert res.max_abs_pct_diff == 0.0
    assert res.mismatches.empty


def test_jugaad_fetcher_normalizes_columns_with_monkeypatched_stock_df(monkeypatch):
    calls = {}

    def fake_stock_df(symbol, from_date, to_date, series="EQ"):
        calls.update(symbol=symbol, from_date=from_date, to_date=to_date, series=series)
        return pd.DataFrame(
            {
                "DATE": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "SERIES": ["EQ", "EQ", "EQ"],
                "OPEN": [10.0, 11.0, 12.0],
                "HIGH": [11.0, 12.0, 13.0],
                "LOW": [9.0, 10.0, 11.0],
                "CLOSE": [10.5, 11.5, 12.5],
                "VOLUME": [100, 200, 300],
            }
        )

    monkeypatch.setattr(jnse, "stock_df", fake_stock_df)
    bars = jugaad_fetcher("RELIANCE.NS", "2024-01-01", "2024-01-31")

    assert calls == {
        "symbol": "RELIANCE",
        "from_date": dt.date(2024, 1, 1),
        "to_date": dt.date(2024, 1, 31),
        "series": "EQ",
    }
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.name == "date"
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert bars.index.is_monotonic_increasing
    assert bars["close"].tolist() == pytest.approx([10.5, 11.5, 12.5])
    assert bars["volume"].dtype.kind == "f"


def test_jugaad_fetcher_end_none_defaults_to_today(monkeypatch):
    seen = {}

    def fake_stock_df(symbol, from_date, to_date, series="EQ"):
        seen["to_date"] = to_date
        return pd.DataFrame(
            {"DATE": pd.to_datetime(["2024-01-02"]), "OPEN": [1.0], "HIGH": [1.0], "LOW": [1.0], "CLOSE": [1.0]}
        )

    monkeypatch.setattr(jnse, "stock_df", fake_stock_df)
    jugaad_fetcher("TCS", "2024-01-01", None)
    assert isinstance(seen["to_date"], dt.date)


def test_jugaad_utc_1830_timestamps_map_to_correct_ist_date(monkeypatch):
    import sys
    import types

    import pandas as _pd

    from nsebot.data.crosscheck import jugaad_fetcher

    def fake_stock_df(symbol, from_date, to_date, series):
        return _pd.DataFrame({
            "DATE": _pd.to_datetime(["2024-10-28 18:30:00", "2024-10-27 18:30:00", "2024-10-24 18:30:00"]),
            "OPEN": [1, 1, 1], "HIGH": [1, 1, 1], "LOW": [1, 1, 1], "CLOSE": [1340.0, 1334.35, 2655.7],
            "VOLUME": [1, 1, 1]})

    nse_mod = types.ModuleType("jugaad_data.nse")
    nse_mod.stock_df = fake_stock_df
    monkeypatch.setitem(sys.modules, "jugaad_data", types.ModuleType("jugaad_data"))
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", nse_mod)
    df = jugaad_fetcher("RELIANCE.NS", "2024-10-24", "2024-10-29")
    assert [d.strftime("%Y-%m-%d") for d in df.index] == ["2024-10-25", "2024-10-28", "2024-10-29"]
    assert all(d.dayofweek < 5 for d in df.index)  # no Sunday bars


def test_jugaad_fetcher_drops_non_eq_series_rows(monkeypatch):
    """NSE mixes other series (e.g. HDFC warrants 'W2') into the EQ request; they must not reach the bars.

    The W2 rows are placed AFTER the EQ rows on the same dates, so without the filter normalize()'s
    keep-last-per-date would pick them and the closes would be wrong."""

    def fake_stock_df(symbol, from_date, to_date, series="EQ"):
        return pd.DataFrame(
            {
                "DATE": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
                "SERIES": ["EQ", "EQ", "W2", "W2"],
                "OPEN": [10.0, 11.0, 500.0, 600.0],
                "HIGH": [11.0, 12.0, 500.0, 600.0],
                "LOW": [9.0, 10.0, 500.0, 600.0],
                "CLOSE": [10.5, 11.5, 500.0, 600.0],
                "VOLUME": [100, 200, 5, 5],
            }
        )

    monkeypatch.setattr(jnse, "stock_df", fake_stock_df)
    bars = jugaad_fetcher("HDFC.NS", "2024-01-01", "2024-01-31")

    assert bars["close"].tolist() == pytest.approx([10.5, 11.5])
    assert bars["volume"].tolist() == pytest.approx([100.0, 200.0])
    assert bars.index.is_unique
    assert len(bars) == 2
