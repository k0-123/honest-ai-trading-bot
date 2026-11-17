from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from nsebot.universe import (
    CANONICAL,
    NSE_SOURCED,
    MissingPriceData,
    Nifty50Universe,
    UniversePriceLoader,
    check_prices,
    price_key,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nifty50"
CHANGES = pd.read_csv(DATA_DIR / "changes.csv", dtype=str)
SNAPSHOT = pd.read_csv(DATA_DIR / "current_2026-11-17.csv", dtype=str)
DVR_START = pd.Timestamp("2016-04-01")
DVR_END = pd.Timestamp("2017-09-28")


@pytest.fixture(scope="module")
def uni() -> Nifty50Universe:
    return Nifty50Universe(DATA_DIR)


def _snapshot_keys() -> frozenset[str]:
    return frozenset(price_key(s) for s in SNAPSHOT["symbol"])


# ---------------------------------------------------------------- constants and keys


def test_price_key_applies_canonical_rename_and_suffix():
    assert price_key("MCDOWELL-N") == "UNITDSPR.NS"
    assert price_key("ZOMATO") == "ETERNAL.NS"
    assert price_key("RELIANCE") == "RELIANCE.NS"
    assert CANONICAL["INFRATEL"] == "INDUSTOWER"
    assert "HDFC.NS" in NSE_SOURCED and "TATAMTRDVR.NS" in NSE_SOURCED


def test_snapshot_has_exactly_50_members():
    assert len(SNAPSHOT) == 50
    assert len(_snapshot_keys()) == 50


# ---------------------------------------------------------------- members_on


def test_members_on_has_50_keys_on_sample_dates_outside_dvr_period(uni):
    sample = ["2014-01-01", "2015-10-01", "2016-03-31", "2017-09-29", "2018-06-15", "2021-01-04",
              "2023-08-01", "2024-02-15", "2025-06-30", "2026-01-02", "2026-09-29"]
    for d in sample:
        assert len(uni.members_on(d)) == 50, d


def test_members_on_has_51_keys_during_dvr_period_and_50_outside_daily(uni):
    day = pd.Timestamp("2014-01-01")
    last = pd.Timestamp("2026-11-17")
    while day <= last:
        n = len(uni.members_on(day))
        expected = 51 if DVR_START <= day <= DVR_END else 50
        assert n == expected, day.date()
        day += pd.Timedelta(days=1)


def test_members_on_during_dvr_period_contains_tatamtrdvr(uni):
    assert len(uni.members_on("2016-04-01")) == 51
    assert len(uni.members_on("2017-09-28")) == 51
    assert "TATAMTRDVR.NS" in uni.members_on("2016-06-01")
    assert "TATAMTRDVR.NS" in uni.members_on("2017-09-28")
    assert "TATAMTRDVR.NS" not in uni.members_on("2017-10-03")


def test_members_on_after_snapshot_equals_snapshot(uni):
    snap = _snapshot_keys()
    assert uni.members_on("2026-11-17") == snap
    assert uni.members_on("2026-12-01") == snap
    assert uni.members_on(pd.Timestamp("2027-06-30")) == snap


def test_members_on_accepts_strings_and_timestamps(uni):
    assert uni.members_on("2023-08-01") == uni.members_on(pd.Timestamp("2023-08-01"))


def test_every_change_row_flips_membership_on_its_effective_date(uni):
    for _, row in CHANGES.iterrows():
        d = pd.Timestamp(row["effective_date"])
        key = price_key(row["symbol"])
        before = uni.members_on(d - pd.Timedelta(days=1))
        on = uni.members_on(d)
        if row["action"] == "IN":
            assert key not in before, f"{key} should not be a member on {d.date() - dt.timedelta(days=1)}"
            assert key in on, f"{key} should be a member on {d.date()}"
        elif row["action"] == "OUT":
            assert key in before, f"{key} should be a member on {d.date() - dt.timedelta(days=1)}"
            assert key not in on, f"{key} should not be a member on {d.date()}"
        else:
            pytest.fail(f"unexpected action {row['action']}")


@pytest.mark.parametrize(
    "date, key, present",
    [
        ("2023-07-12", "HDFC.NS", True),
        ("2023-07-13", "HDFC.NS", False),
        ("2023-07-12", "LTM.NS", False),
        ("2023-07-13", "LTM.NS", True),
        ("2024-09-29", "LTM.NS", True),
        ("2024-09-30", "LTM.NS", False),
        ("2016-06-01", "TATAMTRDVR.NS", True),
        ("2017-10-03", "TATAMTRDVR.NS", False),
        ("2026-09-29", "BSE.NS", False),
        ("2026-09-30", "BSE.NS", True),
        ("2023-08-01", "JIOFIN.NS", False),
        ("2025-03-28", "ETERNAL.NS", True),
        ("2025-03-27", "ETERNAL.NS", False),
    ],
)
def test_specific_membership_facts(uni, date, key, present):
    assert (key in uni.members_on(date)) is present


# ---------------------------------------------------------------- ever_members / membership_spans


def test_ever_members_is_union_of_members_on_over_range(uni):
    start, end = "2023-07-10", "2023-07-20"
    union = set()
    for d in pd.date_range(start, end):
        union |= uni.members_on(d)
    assert uni.ever_members(start, end) == frozenset(union)
    assert "HDFC.NS" in uni.ever_members(start, end)
    assert "LTM.NS" in uni.ever_members(start, end)


def test_membership_spans_are_inclusive_and_clipped(uni):
    spans = uni.membership_spans("2015-10-01", "2026-11-16")
    assert spans["HDFC.NS"] == [(pd.Timestamp("2015-10-01"), pd.Timestamp("2023-07-12"))]
    assert spans["LTM.NS"] == [(pd.Timestamp("2023-07-13"), pd.Timestamp("2024-09-29"))]
    assert spans["BSE.NS"] == [(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-11-16"))]
    for key, intervals in spans.items():
        for s, e in intervals:
            assert pd.Timestamp("2015-10-01") <= s <= e <= pd.Timestamp("2026-11-16"), key


def test_membership_spans_cover_exactly_the_members_each_day(uni):
    start, end = "2016-03-25", "2016-04-05"
    spans = uni.membership_spans(start, end)
    for d in pd.date_range(start, end):
        from_spans = {k for k, ivs in spans.items() if any(s <= d <= e for s, e in ivs)}
        assert from_spans == set(uni.members_on(d)), d.date()


# ---------------------------------------------------------------- UniversePriceLoader routing


class _RecordingLoader:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple] = []

    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
        self.calls.append((symbol, start, end, as_of, refresh, clean))
        return pd.DataFrame()


def test_universe_price_loader_routes_nse_sourced_keys_to_nse_loader():
    yahoo, nse = _RecordingLoader("yahoo"), _RecordingLoader("nse")
    loader = UniversePriceLoader(yahoo, nse)
    loader.get("HDFC.NS", "2015-01-01", "2016-01-01", clean=False)
    loader.get("RELIANCE.NS", "2015-01-01", "2016-01-01")
    assert nse.calls == [("HDFC.NS", "2015-01-01", "2016-01-01", None, False, False)]
    # the inner loader is always asked for RAW bars: adjustments are applied, then the spike filter
    assert yahoo.calls == [("RELIANCE.NS", "2015-01-01", "2016-01-01", None, False, False)]


def test_universe_price_loader_routes_every_nse_sourced_key_to_nse(uni):
    yahoo, nse = _RecordingLoader("yahoo"), _RecordingLoader("nse")
    loader = UniversePriceLoader(yahoo, nse)
    for key in sorted(NSE_SOURCED):
        loader.get(key, "2015-01-01")
    assert [c[0] for c in nse.calls] == sorted(NSE_SOURCED)
    assert yahoo.calls == []


# ---------------------------------------------------------------- check_prices


def _bars(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end, name="date")
    close = pd.Series(100.0, index=idx)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


class _FakeLoader:
    """Returns bars from a dict {key: (first, last)}; missing keys return an empty frame."""

    def __init__(self, coverage: dict[str, tuple[str, str]]):
        self.coverage = coverage
        self.calls: list[tuple] = []

    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
        self.calls.append((symbol, start, end, clean))
        if symbol not in self.coverage:
            return pd.DataFrame()
        first, last = self.coverage[symbol]
        return _bars(first, last)


def _full_coverage(uni: Nifty50Universe, start="2013-01-01", end="2026-12-31") -> dict[str, tuple[str, str]]:
    return {k: (start, end) for k in uni.ever_members("2015-10-01", "2026-11-16")}


def test_check_prices_passes_on_complete_data(uni):
    loader = _FakeLoader(_full_coverage(uni))
    check_prices(uni, loader, "2015-10-01", "2026-11-16")  # must not raise


def test_check_prices_requests_start_minus_400_days_and_raw_bars(uni):
    loader = _FakeLoader(_full_coverage(uni))
    check_prices(uni, loader, "2015-10-01", "2026-11-16")
    expected_start = (pd.Timestamp("2015-10-01") - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    assert loader.calls, "loader should be called"
    for symbol, start, end, clean in loader.calls:
        assert start == expected_start
        assert end == "2026-11-16"
        assert clean is False


def test_check_prices_raises_when_member_data_is_missing(uni):
    cov = _full_coverage(uni)
    del cov["HDFC.NS"]
    with pytest.raises(MissingPriceData) as exc:
        check_prices(uni, _FakeLoader(cov), "2015-10-01", "2026-11-16")
    assert "HDFC.NS" in str(exc.value)


def test_check_prices_raises_when_data_starts_too_late(uni):
    cov = _full_coverage(uni)
    cov["BSE.NS"] = ("2026-10-20", "2026-12-31")  # span starts 2026-09-30, tolerance 7 days
    with pytest.raises(MissingPriceData) as exc:
        check_prices(uni, _FakeLoader(cov), "2015-10-01", "2026-11-16")
    assert "BSE.NS" in str(exc.value)


def test_check_prices_raises_when_data_ends_too_early(uni):
    cov = _full_coverage(uni)
    cov["LTM.NS"] = ("2013-01-01", "2024-08-01")  # span ends 2024-09-29
    with pytest.raises(MissingPriceData) as exc:
        check_prices(uni, _FakeLoader(cov), "2015-10-01", "2026-11-16")
    assert "LTM.NS" in str(exc.value)


def test_check_prices_tolerates_a_few_days_of_slack(uni):
    cov = _full_coverage(uni)
    cov["BSE.NS"] = ("2026-10-03", "2026-12-31")  # 3 days after span start: within 7-day tolerance
    check_prices(uni, _FakeLoader(cov), "2015-10-01", "2026-11-16")


def test_check_prices_reports_every_offending_key(uni):
    cov = _full_coverage(uni)
    del cov["HDFC.NS"]
    cov["LTM.NS"] = ("2013-01-01", "2024-08-01")
    with pytest.raises(MissingPriceData) as exc:
        check_prices(uni, _FakeLoader(cov), "2015-10-01", "2026-11-16")
    assert "HDFC.NS" in str(exc.value) and "LTM.NS" in str(exc.value)


def test_universe_price_loader_applies_documented_adjustments_then_cleans():
    idx = pd.bdate_range("2025-10-08", periods=4, name="date")
    c = pd.Series([655.0, 655.0, 392.0, 393.0], index=idx)
    raw = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})

    class L:
        def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
            return raw if as_of is None else raw.loc[: pd.Timestamp(as_of)]

    adj = pd.DataFrame([{"key": "TMPV.NS", "date": idx[2], "factor": 392.0 / 655.0, "kind": "demerger",
                         "evidence": "test fixture"}])
    up = UniversePriceLoader(L(), L(), adjustments=adj)
    out = up.get("TMPV.NS", "2025-10-01")
    assert out.close.pct_change().abs().max() < 0.01                    # demerger gap removed
    early = up.get("TMPV.NS", "2025-10-01", as_of=idx[1])               # paper mode before the ex-date
    assert list(early.close) == [655.0, 655.0]                          # correction not applied yet
