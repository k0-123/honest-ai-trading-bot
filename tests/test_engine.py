import numpy as np
import pandas as pd
import pytest

from nsebot.backtest.engine import run_backtest
from nsebot.backtest.splits import split_periods, walk_forward_windows
from nsebot.config import Config
from nsebot.strategies.base import Signal
from nsebot.strategies.baseline import BaselineStrategy

CFG = Config(symbols=["A.NS", "B.NS"])


def synth(seed, n=900):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                         "close": c, "volume": 1e5}, index=idx)


PRICES = {"A.NS": synth(1), "B.NS": synth(2)}


def test_backtest_runs_and_accounts():
    res = run_backtest(PRICES, BaselineStrategy(CFG.baseline), CFG, "2019-01-01")
    assert res.equity.index[0] >= pd.Timestamp("2019-01-01")
    assert len(res.trades) > 0
    assert (res.trades["costs"] > 0).all()
    s = res.summary()
    assert set(s) >= {"total_return", "sharpe", "max_drawdown", "win_rate", "total_costs"}


def test_fills_happen_strictly_after_signal_day():
    res = run_backtest(PRICES, BaselineStrategy(CFG.baseline), CFG, "2019-01-01")
    sig_days = {(sg.symbol, sg.date.strftime("%Y-%m-%d")) for sg in res.signals if sg.action != "HOLD"}
    for _, t in res.trades.iterrows():
        if "STOP" in t["reason"] or "KILL" in t["reason"]:
            continue
        assert (t["symbol"], t["date"]) not in sig_days or any(
            d < t["date"] for (sym, d) in sig_days if sym == t["symbol"])


def test_future_data_does_not_change_past_results():
    cut = "2020-06-30"
    full = run_backtest(PRICES, BaselineStrategy(CFG.baseline), CFG, "2019-01-01", cut)
    trimmed = {k: v.loc[:cut] for k, v in PRICES.items()}
    part = run_backtest(trimmed, BaselineStrategy(CFG.baseline), CFG, "2019-01-01", cut)
    pd.testing.assert_series_equal(full.equity, part.equity)
    pd.testing.assert_frame_equal(full.trades, part.trades)


def test_contaminated_flag_propagates():
    class Fake:
        source = "llm"

        def generate(self, symbol, history, in_position):
            return Signal(history.index[-1], symbol, "HOLD", 0.0, "[CONTAMINATED] x", "llm", contaminated=True)

    res = run_backtest(PRICES, Fake(), CFG, "2019-01-01", "2019-02-01", name="llm")
    assert res.contaminated


def test_buy_and_hold_benchmark():
    res = run_backtest({"N.NS": synth(3)}, None, Config(symbols=["N.NS"]), "2019-01-01", buy_and_hold=True)
    assert len(res.trades) == 1


def test_splits_and_walk_forward_do_not_overlap():
    p = split_periods("2015-01-01", "2025-12-31", "2020-12-31", "2022-12-31")
    assert p["val"][0] == "2021-01-01" and p["test"][0] == "2023-01-01"
    with pytest.raises(ValueError):
        split_periods("2015-01-01", "2020-01-01", "2021-01-01", "2022-01-01")
    w = walk_forward_windows("2015-01-01", "2020-12-31", 3, 6)
    assert w and all(tr[1] < te[0] for tr, te in w)


def test_stop_uses_raw_open_even_when_spike_filter_holds_the_bar_out():
    """Fix #3: a genuine -60% gap is held out of the CLEANED series for up to 3 days, but the stop must fire
    on that same day at the RAW open, not 3 days later."""
    from nsebot.data.loader import drop_spikes
    cfg = Config(symbols=["X.NS"])
    idx = pd.bdate_range("2020-01-01", periods=300, name="date")
    c = np.linspace(100, 200, 300)
    raw = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)
    crash = idx[290]
    raw.loc[idx[290]:, ["open", "high", "low", "close"]] = 70.0  # permanent crash from ~197 to 70
    clean = drop_spikes(raw)
    assert crash not in clean.index  # the filter is holding the crash bar out

    class BuyOnce:
        source = "test"

        def generate(self, symbol, history, in_position):
            d = history.index[-1]
            act = "BUY" if (d == idx[285] and not in_position) else "HOLD"
            return Signal(d, symbol, act, 0.6, "t", "test", stop=float(history["close"].iloc[-1]) * 0.95)

    res = run_backtest({"X.NS": clean}, BuyOnce(), cfg, str(idx[280].date()), name="b", raw_prices={"X.NS": raw})
    sells = res.trades[res.trades.side == "SELL"]
    assert len(sells) == 1
    assert sells.iloc[0]["date"] == crash.strftime("%Y-%m-%d")      # same day, no 3-day delay
    assert sells.iloc[0]["price"] == pytest.approx(70.0 * 0.9995)   # raw gap open, minus slippage
    assert "STOP" in sells.iloc[0]["reason"]


def test_invested_series_tracks_exposure():
    res = run_backtest(PRICES, BaselineStrategy(CFG.baseline), CFG, "2019-01-01")
    assert len(res.invested) == len(res.equity)
    assert (res.invested >= 0).all() and (res.invested <= res.equity + 1e-6).all()
