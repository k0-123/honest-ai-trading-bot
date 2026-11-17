from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nsebot.indicators import atr, rsi, sma


def make_bars(n: int, seed: int = 0) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    close = 100.0 * np.cumprod(1 + 0.01 * rs.standard_normal(n))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + 0.002 * rs.standard_normal(n))
    high = np.maximum(open_, close) * (1 + 0.005 * np.abs(rs.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - 0.005 * np.abs(rs.standard_normal(n)))
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6},
        index=idx,
    )


def loop_rsi(close: list[float], n: int) -> list[float]:
    """Reference: Wilder-style recursion as ewm(alpha=1/n, adjust=False) seeded on the first gain/loss."""
    alpha = 1.0 / n
    gains = [None] + [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [None] + [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    out = [math.nan] * len(close)
    ag = al = None
    for i in range(1, len(close)):
        g, l = gains[i], losses[i]
        if ag is None:
            ag, al = g, l
        else:
            ag = ag + alpha * (g - ag)
            al = al + alpha * (l - al)
        if i >= n:
            out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def loop_atr(bars: pd.DataFrame, n: int) -> list[float]:
    """Reference: true range (gap-aware) smoothed by the same ewm recursion."""
    h, l, c = bars["high"].tolist(), bars["low"].tolist(), bars["close"].tolist()
    tr = [h[0] - l[0]]
    for i in range(1, len(c)):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    alpha = 1.0 / n
    out = [math.nan] * len(tr)
    cur = None
    for i, x in enumerate(tr):
        cur = x if cur is None else cur + alpha * (x - cur)
        if i >= n - 1:
            out[i] = cur
    return out


# ---------- sma ----------

def test_sma_reference_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = sma(s, 3)
    assert math.isnan(out.iloc[0]) and math.isnan(out.iloc[1])
    assert out.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_sma_nan_until_window_full():
    s = pd.Series(np.arange(1.0, 31.0))
    out = sma(s, 10)
    assert out.iloc[:9].isna().all()
    assert not out.iloc[9:].isna().any()


# ---------- rsi ----------

def test_rsi_all_up_is_100():
    close = pd.Series(np.arange(100.0, 160.0))
    out = rsi(close, 14)
    assert out.iloc[:14].isna().all()
    assert (out.iloc[14:] == 100.0).all()


def test_rsi_all_down_is_0():
    close = pd.Series(np.arange(160.0, 100.0, -1.0))
    out = rsi(close, 14)
    assert out.iloc[14:].tolist() == pytest.approx([0.0] * (len(close) - 14), abs=1e-9)


def test_rsi_hand_example():
    # gains: +1 (i=1), 0 (i=2), +3 (i=3); losses: 0, 2, 0. Seeded on first observation.
    # i=2: ag=1-(1/3)=0.666667, al=0+(2/3)=0.666667
    # i=3: ag=0.666667+(3-0.666667)/3=1.444444, al=0.666667-0.222222=0.444444
    # RSI = 100 - 100/(1+1.444444/0.444444) = 76.470588
    close = pd.Series([100.0, 101.0, 99.0, 102.0])
    out = rsi(close, 3)
    assert math.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(76.470588, abs=1e-6)


def test_rsi_matches_loop_reference():
    bars = make_bars(300, seed=3)
    out = rsi(bars["close"], 14)
    ref = loop_rsi(bars["close"].tolist(), 14)
    for i, r in enumerate(ref):
        if math.isnan(r):
            assert math.isnan(out.iloc[i]), i
        else:
            assert out.iloc[i] == pytest.approx(r, abs=1e-6), i


# ---------- atr ----------

def test_atr_n1_is_true_range_including_gap_up():
    # previous close 100; next bar gaps up: high-low=2 but |high-prev_close|=6
    bars = pd.DataFrame(
        {"open": [100.0, 105.0], "high": [101.0, 106.0], "low": [99.0, 104.0],
         "close": [100.0, 105.0], "volume": [1.0, 1.0]},
        index=pd.bdate_range("2021-01-01", periods=2, name="date"),
    )
    out = atr(bars, 1)
    assert out.iloc[1] == pytest.approx(6.0, abs=1e-9)


def test_atr_true_range_including_gap_down():
    # previous close 100; next bar gaps down: high-low=2 but |low-prev_close|=8
    bars = pd.DataFrame(
        {"open": [100.0, 92.0], "high": [101.0, 94.0], "low": [99.0, 92.0],
         "close": [100.0, 93.0], "volume": [1.0, 1.0]},
        index=pd.bdate_range("2021-01-01", periods=2, name="date"),
    )
    out = atr(bars, 1)
    assert out.iloc[1] == pytest.approx(8.0, abs=1e-9)


def test_atr_matches_loop_reference():
    bars = make_bars(250, seed=7)
    out = atr(bars, 5)
    ref = loop_atr(bars, 5)
    for i, r in enumerate(ref):
        if math.isnan(r):
            assert math.isnan(out.iloc[i]), i
        else:
            assert out.iloc[i] == pytest.approx(r, abs=1e-6), i


def test_atr_nan_warmup():
    bars = make_bars(60, seed=1)
    out = atr(bars, 14)
    assert out.iloc[:13].isna().all()
    assert not out.iloc[13:].isna().any()


def test_atr_positive():
    bars = make_bars(120, seed=2)
    out = atr(bars, 14).dropna()
    assert (out > 0).all()


# ---------- causality ----------

@pytest.mark.parametrize("k", [5, 20, 60, 150, 299])
def test_indicators_are_causal(k):
    bars = make_bars(300, seed=11)
    full_sma = sma(bars["close"], 20)
    full_rsi = rsi(bars["close"], 14)
    full_atr = atr(bars, 14)

    part = bars.iloc[:k]
    pd.testing.assert_series_equal(sma(part["close"], 20), full_sma.iloc[:k], check_names=False)
    pd.testing.assert_series_equal(rsi(part["close"], 14), full_rsi.iloc[:k], check_names=False)
    pd.testing.assert_series_equal(atr(part, 14), full_atr.iloc[:k], check_names=False)


def test_future_wild_rows_do_not_change_past_values():
    bars = make_bars(200, seed=5)
    wild = bars.copy()
    wild.loc[wild.index[120]:, ["open", "high", "low", "close"]] *= 37.0
    for fn in (lambda b: sma(b["close"], 20), lambda b: rsi(b["close"], 14), lambda b: atr(b, 14)):
        a, b = fn(bars), fn(wild)
        pd.testing.assert_series_equal(a.iloc[:120], b.iloc[:120], check_names=False)
