import pandas as pd
import pytest

from nsebot.universe.adjustments import apply_adjustments, load_adjustments


def bars(closes, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 100.0})


def test_factor_applies_only_to_bars_before_the_date():
    df = bars([100.0, 100.0, 60.0, 61.0])
    adj = pd.DataFrame([{"key": "X.NS", "date": df.index[2], "factor": 0.6, "kind": "demerger", "evidence": "t"}])
    out = apply_adjustments(df, "X.NS", adj)
    assert list(out.close) == pytest.approx([60.0, 60.0, 60.0, 61.0])
    assert list(out.volume) == pytest.approx([100 / 0.6, 100 / 0.6, 100, 100])
    assert out.close.pct_change().abs().max() < 0.02   # the artificial gap is gone


def test_other_symbols_untouched_and_input_not_mutated():
    df = bars([100.0, 50.0])
    adj = pd.DataFrame([{"key": "X.NS", "date": df.index[1], "factor": 0.5, "kind": "bonus", "evidence": "t"}])
    out = apply_adjustments(df, "Y.NS", adj)
    pd.testing.assert_frame_equal(out, df)
    apply_adjustments(df, "X.NS", adj)
    assert df.close.iloc[0] == 100.0


def test_adjustments_file_is_valid():
    adj = load_adjustments()
    assert set(adj.columns) >= {"key", "date", "factor", "kind", "evidence"}
    assert (adj.factor > 0).all() and (adj.factor < 1.5).all()
    assert adj.evidence.str.len().gt(10).all()        # every fix documents why
    assert not adj.duplicated(["key", "date"]).any()


def test_bar_override_replaces_stale_bar_in_yahoo_basis():
    from nsebot.universe.adjustments import apply_bar_overrides
    idx = pd.bdate_range("2025-03-14", periods=4, name="date")
    # Yahoo basis is 0.98 x NSE (a past dividend); 2025-03-18 is a stale copy of the previous close
    y = pd.DataFrame({"open": [196.0, 199.6, 199.6, 219.0], "high": [197.0, 200.0, 199.6, 220.0],
                      "low": [195.0, 198.0, 199.6, 218.0], "close": [196.0, 199.6, 199.6, 219.1],
                      "volume": [1.0, 1.0, 0.0, 1.0]}, index=idx)
    ov = pd.DataFrame([{"key": "X.NS", "date": idx[2], "open": 205.0, "high": 223.0, "low": 204.0,
                        "close": 222.6, "nse_prev_close": 203.7, "evidence": "stale"}])
    out = apply_bar_overrides(y, "X.NS", ov)
    scale = 199.6 / 203.7
    assert out.loc[idx[2], "close"] == pytest.approx(222.6 * scale)
    assert out.loc[idx[2], "low"] == pytest.approx(204.0 * scale)
    assert out.drop(idx[2]).equals(y.drop(idx[2]))


def test_bar_overrides_file_is_valid():
    from nsebot.universe.adjustments import load_bar_overrides
    ov = load_bar_overrides()
    assert set(ov.columns) >= {"key", "date", "open", "high", "low", "close", "nse_prev_close", "evidence"}
    if len(ov):
        assert (ov[["open", "high", "low", "close", "nse_prev_close"]] > 0).all().all()
        assert (ov.low <= ov[["open", "close"]].min(axis=1) + 1e-9).all()
        assert not ov.duplicated(["key", "date"]).any()
