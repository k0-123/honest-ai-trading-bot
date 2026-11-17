from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsebot.config import BaselineCfg
from nsebot.indicators import atr, rsi, sma
from nsebot.strategies.base import Signal
from nsebot.strategies.baseline import BaselineStrategy

SYM = "TEST.NS"


def _bars_from_close(close: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + 0.003 * np.abs(rs.standard_normal(len(close))))
    low = np.minimum(open_, close) * (1 - 0.003 * np.abs(rs.standard_normal(len(close))))
    idx = pd.bdate_range("2019-01-01", periods=len(close), name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx
    )


def _path(rets) -> np.ndarray:
    return 100.0 * np.cumprod(1 + np.asarray(rets, dtype=float))


def _uptrend_rets(n: int, seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    return 0.002 + 0.004 * rs.standard_normal(n)


def dip_recovery_bars() -> pd.DataFrame:
    """Long uptrend, a 12-day dip of about -17%, then a 10-day rebound."""
    rets = np.concatenate([_uptrend_rets(260, 1), np.full(12, -0.015), np.full(10, 0.012)])
    return _bars_from_close(_path(rets), seed=1)


def rally_bars() -> pd.DataFrame:
    """Uptrend followed by a strong rally that pushes RSI above 70."""
    rets = np.concatenate([_uptrend_rets(260, 1), np.full(15, 0.015)])
    return _bars_from_close(_path(rets), seed=2)


def two_day_drop_bars() -> pd.DataFrame:
    rets = np.concatenate([_uptrend_rets(260, 1), [-0.06, -0.06]])
    return _bars_from_close(_path(rets), seed=3)


def one_day_drop_bars() -> pd.DataFrame:
    rets = np.concatenate([_uptrend_rets(260, 1), [-0.06]])
    return _bars_from_close(_path(rets), seed=3)


def crash_bars() -> pd.DataFrame:
    rets = np.concatenate([_uptrend_rets(260, 1), [-0.30]])
    return _bars_from_close(_path(rets), seed=4)


def buy_index(bars: pd.DataFrame, cfg: BaselineCfg) -> int:
    """Index of the bar where RSI crosses up through rsi_entry with the trend intact."""
    c = bars["close"]
    r = rsi(c, cfg.rsi_period)
    s_fast, s_slow = sma(c, cfg.sma_fast), sma(c, cfg.sma_slow)
    hits = [
        t for t in range(1, len(c))
        if r.iloc[t - 1] < cfg.rsi_entry <= r.iloc[t] and c.iloc[t] > s_slow.iloc[t] and s_fast.iloc[t] > s_slow.iloc[t]
    ]
    assert hits, "fixture has no BUY crossing"
    return hits[-1]


@pytest.fixture
def cfg() -> BaselineCfg:
    return BaselineCfg()


@pytest.fixture
def strat(cfg) -> BaselineStrategy:
    return BaselineStrategy(cfg)


# ---------- BUY ----------

def test_buy_on_uptrend_rsi_cross(strat, cfg):
    bars = dip_recovery_bars()
    t = buy_index(bars, cfg)
    sig = strat.generate(SYM, bars.iloc[: t + 1], in_position=False)
    assert isinstance(sig, Signal)
    assert sig.action == "BUY"
    assert sig.source == "baseline"
    assert sig.symbol == SYM
    assert pd.Timestamp(sig.date) == bars.index[t]
    assert sig.confidence == pytest.approx(0.6)


def test_buy_stop_is_below_close_and_two_atr(strat, cfg):
    bars = dip_recovery_bars()
    t = buy_index(bars, cfg)
    hist = bars.iloc[: t + 1]
    sig = strat.generate(SYM, hist, in_position=False)
    close = float(hist["close"].iloc[-1])
    expected_atr = float(atr(hist, cfg.atr_period).iloc[-1])
    assert sig.stop is not None
    assert sig.stop < close
    assert sig.stop == pytest.approx(close - 2.0 * expected_atr, abs=1e-9)


def test_buy_reason_contains_numbers_and_stop(strat, cfg):
    bars = dip_recovery_bars()
    t = buy_index(bars, cfg)
    hist = bars.iloc[: t + 1]
    sig = strat.generate(SYM, hist, in_position=False)
    close = float(hist["close"].iloc[-1])
    assert "stop" in sig.reason.lower()
    assert f"{close:.1f}" in sig.reason
    assert f"{sig.stop:.1f}" in sig.reason
    assert "RSI(14)" in sig.reason
    assert "SMA200" in sig.reason


def test_in_position_prevents_buy(strat, cfg):
    bars = dip_recovery_bars()
    t = buy_index(bars, cfg)
    hist = bars.iloc[: t + 1]
    assert strat.generate(SYM, hist, in_position=False).action == "BUY"
    # same bar while holding: never BUY again, and (consistency rule) never SELL on a BUY bar
    sig = strat.generate(SYM, hist, in_position=True)
    assert sig.action == "HOLD"
    assert sig.stop is None


def test_no_buy_without_rsi_cross(strat, cfg):
    bars = dip_recovery_bars()
    t = buy_index(bars, cfg)
    # one bar before the crossing RSI is still below the threshold, so no cross occurs there
    sig = strat.generate(SYM, bars.iloc[: t], in_position=False)
    assert sig.action != "BUY"


# ---------- SELL ----------

def test_sell_on_rsi_above_exit(strat):
    bars = rally_bars()
    sig = strat.generate(SYM, bars, in_position=True)
    assert sig.action == "SELL"
    assert "RSI" in sig.reason
    assert sig.confidence == pytest.approx(0.6)
    assert sig.stop is None


def _bars_from_closes(closes):
    idx = pd.bdate_range("2020-01-01", periods=len(closes), name="date")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e5})


def test_sell_on_k_closes_below_falling_fast_sma():
    strat = BaselineStrategy(BaselineCfg(exit_below_fast_days=5))
    # uptrend, 60-day plateau, then a 6-day fade: close < SMA50, SMA50 now falling, still > SMA200
    closes = list(np.linspace(100, 300, 200)) + [300.0] * 60 + [300 - 3 * i for i in range(1, 7)]
    sig = strat.generate(SYM, _bars_from_closes(closes), in_position=True)
    assert sig.action == "SELL", sig.reason
    assert "SMA50" in sig.reason and "falling" in sig.reason


def test_two_closes_below_rising_fast_sma_is_a_hold(strat):
    # the old whipsaw case: a short dip under a still-rising SMA50 must NOT exit
    bars = two_day_drop_bars()
    sig = strat.generate(SYM, bars, in_position=True)
    assert sig.action != "SELL" or "SMA50" not in sig.reason


def test_one_close_below_fast_sma_is_not_enough(strat):
    bars = one_day_drop_bars()
    sig = strat.generate(SYM, bars, in_position=True)
    assert sig.action == "HOLD"


def test_sell_on_close_below_slow_sma(strat):
    bars = crash_bars()
    sig = strat.generate(SYM, bars, in_position=True)
    assert sig.action == "SELL"
    assert "SMA200" in sig.reason


def test_sell_requires_position(strat):
    bars = rally_bars()
    sig = strat.generate(SYM, bars, in_position=False)
    assert sig.action != "SELL"


# ---------- HOLD / insufficient history ----------

def test_insufficient_history_holds(strat, cfg):
    bars = dip_recovery_bars()
    short = bars.iloc[: cfg.sma_slow + 1]
    for in_pos in (False, True):
        sig = strat.generate(SYM, short, in_position=in_pos)
        assert sig.action == "HOLD"
        assert "insufficient history" in sig.reason
        assert sig.confidence == 0.0
        assert sig.stop is None


def test_minimum_history_is_evaluated(strat, cfg):
    bars = dip_recovery_bars()
    enough = bars.iloc[: cfg.sma_slow + 2]
    sig = strat.generate(SYM, enough, in_position=False)
    assert "insufficient history" not in sig.reason


# ---------- look-ahead invariance (mandatory) ----------

def _oracle_action(full: pd.DataFrame, t: int, in_position: bool, cfg: BaselineCfg) -> str:
    """Independent decision rule computed on the whole frame, read only at index t."""
    if t + 1 < cfg.sma_slow + 2:
        return "HOLD"
    c = full["close"]
    f, s = sma(c, cfg.sma_fast), sma(c, cfg.sma_slow)
    r = rsi(c, cfg.rsi_period)
    trend = c.iloc[t] > s.iloc[t] and f.iloc[t] > s.iloc[t]
    if not in_position:
        ok = trend and r.iloc[t - 1] < cfg.rsi_entry <= r.iloc[t] and not _oracle_exit(c, f, s, r, t, cfg)
        return "BUY" if ok else "HOLD"
    return "SELL" if _oracle_exit(c, f, s, r, t, cfg) else "HOLD"


def _oracle_exit(c, f, s, r, t, cfg) -> bool:
    k = cfg.exit_below_fast_days
    below_fast = k > 0 and all(c.iloc[j] < f.iloc[j] for j in range(t - k + 1, t + 1)) and f.iloc[t] < f.iloc[t - k]
    return bool(r.iloc[t] > cfg.rsi_exit or below_fast or c.iloc[t] < s.iloc[t])


def _wild_future(bars: pd.DataFrame, n_future: int, seed: int) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    last = float(bars["close"].iloc[-1])
    closes = last * np.exp(rs.standard_normal(n_future) * 0.8)  # wild jumps, both directions
    closes = np.maximum(closes, 0.5)
    idx = pd.bdate_range(bars.index[-1] + pd.offsets.BDay(1), periods=n_future, name="date")
    fut = pd.DataFrame(
        {"open": closes, "high": closes * 1.3, "low": closes * 0.7, "close": closes, "volume": 1e9},
        index=idx,
    )
    return pd.concat([bars, fut])


def _regime_bars(seed: int = 9) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    rets = np.concatenate([
        _uptrend_rets(260, 1),
        0.015 * rs.standard_normal(120) - 0.002,
        _uptrend_rets(120, 2),
        -0.02 + 0.03 * rs.standard_normal(40),
    ])
    return _bars_from_close(_path(rets), seed=seed)


@pytest.mark.parametrize("seed", [9, 21])
def test_future_rows_never_change_past_signals(cfg, seed):
    """generate(bars[:t+1]) must not change when the frame is extended with wild future rows."""
    strat = BaselineStrategy(cfg)
    clean = _regime_bars(seed)
    wild = _wild_future(clean, 60, seed=seed + 100)
    assert len(wild) == len(clean) + 60
    assert wild.index.is_monotonic_increasing and wild.index.is_unique

    entries: set[int] = set()
    exits: set[int] = set()
    for t in range(len(clean)):
        for in_pos in (False, True):
            a = strat.generate(SYM, clean.iloc[: t + 1], in_pos)
            b = strat.generate(SYM, wild.iloc[: t + 1], in_pos)
            assert a == b, (t, in_pos)
            assert a.action == _oracle_action(wild, t, in_pos, cfg), (t, in_pos)
            if a.action == "HOLD":
                assert a.confidence == 0.0 and a.stop is None, (t, in_pos)
            if a.action == "BUY":
                entries.add(t)
            if a.action == "SELL":
                exits.add(t)
    # the fixture must exercise both entries and exits, otherwise the check would be vacuous
    assert entries and exits


def test_wild_future_moves_full_frame_indicators(cfg):
    """Sanity: the wild rows really change full-frame indicators, so the oracle comparison has teeth."""
    clean = _regime_bars(9)
    wild = _wild_future(clean, 60, seed=109)
    assert not np.isclose(sma(wild["close"], cfg.sma_slow).iloc[-1], sma(clean["close"], cfg.sma_slow).iloc[-1])


def test_never_buys_on_a_bar_where_it_would_sell(strat):
    """Over random walks, any BUY bar must not also satisfy the SELL rule."""
    import numpy as _np
    for seed in range(3):
        rng = _np.random.default_rng(seed)
        c = 100 * _np.exp(_np.cumsum(rng.normal(0.0005, 0.02, 700)))
        idx = pd.bdate_range("2015-01-01", periods=700, name="date")
        bars = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=idx)
        for t in range(260, 700):
            h = bars.iloc[: t + 1]
            if strat.generate(SYM, h, in_position=False).action == "BUY":
                assert strat.generate(SYM, h, in_position=True).action != "SELL"
