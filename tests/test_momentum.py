"""strategy-v2b (Momentum) tests, written before the implementation."""
import numpy as np
import pandas as pd
import pytest

from nsebot.backtest.engine import run_backtest
from nsebot.config import Config
from nsebot.sim_momentum import MomentumSimulator
from nsebot.strategies.momentum import MomentumCfg, momentum_scores, plan_rebalance, rank_members

BENCH = "NB.NS"


class FakeUniverse:
    def __init__(self, spans):  # {key: (first_member_date, last_member_date or None)}
        self.spans = {k: (pd.Timestamp(a), pd.Timestamp(b) if b else pd.Timestamp("2100-01-01"))
                      for k, (a, b) in spans.items()}

    def members_on(self, date):
        d = pd.Timestamp(date)
        return frozenset(k for k, (a, b) in self.spans.items() if a <= d <= b)


def bars(closes, start="2019-01-01", days=None):
    idx = days if days is not None else pd.bdate_range(start, periods=len(closes), name="date")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=idx)


def trend(n, daily, start=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(np.full(n, daily) + rng.normal(0, 0.002, n)))


# ---------- ranking ----------
def test_score_is_12_month_return_excluding_last_month():
    c = np.arange(1, 301, dtype=float)
    h = bars(c)
    s = momentum_scores({"A.NS": h}, {"A.NS"}, h.index[-1], MomentumCfg())
    assert s["A.NS"] == pytest.approx(c[-22] / c[-253] - 1)


def test_ranking_has_no_look_ahead():
    """Scores for date d must not change when wild future bars are appended."""
    days = pd.bdate_range("2019-01-01", periods=420, name="date")
    hist = {f"S{i}.NS": bars(trend(420, 0.0003 * i, seed=i), days=days) for i in range(12)}
    members = set(hist)
    rng = np.random.default_rng(7)
    for t in (260, 300, 350, 400):
        d = days[t]
        past = {k: v.loc[:d] for k, v in hist.items()}
        wild = {k: pd.concat([v.loc[:d], bars(rng.uniform(1, 1000, 15),
                days=pd.bdate_range(d + pd.Timedelta(days=1), periods=15, name="date"))])
                for k, v in hist.items()}
        a = momentum_scores(past, members, d, MomentumCfg())
        b = momentum_scores(wild, members, d, MomentumCfg())
        pd.testing.assert_series_equal(a, b)


def test_only_members_known_on_that_date_are_ranked():
    days = pd.bdate_range("2019-01-01", periods=400, name="date")
    hist = {"OLD.NS": bars(trend(400, 0.0005), days=days), "NEW.NS": bars(trend(400, 0.003), days=days)}
    uni = FakeUniverse({"OLD.NS": ("2018-01-01", None), "NEW.NS": (str(days[350].date()), None)})
    before = rank_members(hist, uni, days[300], MomentumCfg())
    after = rank_members(hist, uni, days[360], MomentumCfg())
    assert "NEW.NS" not in before.index and list(after.index) == ["NEW.NS", "OLD.NS"]


def test_needs_253_bars_to_be_ranked():
    h = bars(np.arange(1, 253, dtype=float))  # 252 bars
    assert momentum_scores({"A.NS": h}, {"A.NS"}, h.index[-1], MomentumCfg()).empty


# ---------- selection rules ----------
def test_plan_keeps_holdings_ranked_20_or_better_and_fills_to_10():
    ranks = pd.Series(range(1, 31), index=[f"S{i:02d}.NS" for i in range(1, 31)], dtype=float)
    held = {"S05.NS", "S20.NS", "S21.NS", "S30.NS", "GONE.NS"}  # GONE left the index: no rank
    sells, buys = plan_rebalance(held, ranks, MomentumCfg())
    assert sells == ["GONE.NS", "S21.NS", "S30.NS"]           # rank > 20 or unranked
    assert buys == ["S01.NS", "S02.NS", "S03.NS", "S04.NS", "S06.NS", "S07.NS", "S08.NS", "S09.NS"]  # 2 kept + 8 new = 10


# ---------- simulator ----------
def make_market(n=520, seed=0, nb_daily=0.0006):
    days = pd.bdate_range("2019-01-01", periods=n, name="date")
    prices = {f"S{i:02d}.NS": bars(trend(n, 0.0001 * (i - 5), seed=i), days=days) for i in range(25)}
    prices[BENCH] = bars(trend(n, nb_daily, seed=99), days=days)
    return days, prices


SIMS = []


def run_v2b(prices, start, cfg=None, uni=None, kill_switch=True, end=None):
    cfg = cfg or Config(symbols=["S00.NS"], benchmark=BENCH)
    uni = uni or FakeUniverse({k: ("2018-01-01", None) for k in prices if k != BENCH})

    def factory(book, all_days):
        SIMS.append(MomentumSimulator(book, cfg, uni, MomentumCfg(), all_days, kill_switch=kill_switch))
        return SIMS[-1]
    return run_backtest(prices, None, cfg, start, end, "v2b", sim_factory=factory)


def first_sessions(days):
    s = pd.Series(days, index=days)
    return set(s.groupby([s.index.year, s.index.month]).min())


def test_rebalance_trades_only_on_first_trading_day_of_month():
    days, prices = make_market()
    # a holiday on the first weekday of a month: the first trading day is the next session
    hol = pd.Timestamp("2020-06-01")
    prices = {k: v.drop(hol, errors="ignore") for k, v in prices.items()}
    days = days.drop(hol, errors="ignore")
    res = run_v2b(prices, "2020-01-01")
    reb = res.trades[~res.trades.reason.str.contains("STOP|KILL")]
    assert len(reb) > 0
    assert set(pd.to_datetime(reb.date)) <= first_sessions(days)
    decisions = SIMS[-1].decisions
    # one decision per month, always at the last session of a month, aimed at the first session of the next
    assert all(nxt in first_sessions(days) and nxt.month != dec.month for dec, nxt in decisions)
    assert len({(dec.year, dec.month) for dec, _ in decisions}) == len(decisions)
    assert (pd.Timestamp("2020-05-29"), pd.Timestamp("2020-06-02")) in decisions  # June 1 holiday -> June 2


def test_top10_equal_weight_and_never_more_than_10_stocks():
    _, prices = make_market()
    res = run_v2b(prices, "2020-01-01")
    buys = res.trades[(res.trades.side == "BUY") & (res.trades.symbol != BENCH)]
    first = buys[buys.date == buys.date.min()]
    assert len(first) == 10
    val = first.qty * first.price
    # sized at the decision close; the fill is at the next open + slippage, so "~10%" not exactly 10%
    assert (val <= 0.105 * 100_000).all() and (val >= 0.08 * 100_000).all()


def test_regime_off_holds_cash_buys_nothing_and_never_trades_niftybees():
    """Rule (fixed before any v2b result was seen): if NIFTYBEES closes below its 200-day average at a
    rebalance, hold CASH: no new stocks, no NIFTYBEES. Sell rules keep working as normal."""
    _, prices = make_market(nb_daily=-0.002)  # NIFTYBEES in a steady downtrend -> below its 200-day SMA
    res = run_v2b(prices, "2020-03-01", end="2020-06-30")
    assert res.trades.empty                      # nothing bought: all cash
    assert (res.equity == 100_000).all()


def test_regime_off_still_sells_holdings_that_drop_out_of_the_top_20():
    days, prices = make_market(n=560)
    # risk-on first (NIFTYBEES rising) so stocks are bought, then NIFTYBEES collapses below its SMA200
    nb = prices[BENCH]
    cut = days[430]
    nb.loc[cut:, ["open", "high", "low", "close"]] *= 0.5
    # and the strongest stock collapses so its rank falls below 20
    strong = "S24.NS"
    prices[strong].loc[cut:, ["open", "high", "low", "close"]] *= 0.4
    res = run_v2b(prices, "2020-01-01", kill_switch=False)
    t = res.trades
    after = t[pd.to_datetime(t.date) > cut]
    assert BENCH not in set(t.symbol)                                   # never trades NIFTYBEES
    assert after[after.side == "BUY"].empty                             # risk-off: no new stocks
    assert ((after.symbol == strong) & (after.side == "SELL")).any()    # sell rules still work


def test_diagnostic_run_has_no_kill_switch():
    days, prices = make_market(n=520)
    crash = days[400]
    for k in prices:
        if k != BENCH:
            prices[k].loc[crash:, ["open", "high", "low", "close"]] *= 0.6
    with_ks = run_v2b(prices, "2020-01-01")
    diag = run_v2b(prices, "2020-01-01", kill_switch=False)
    assert with_ks.halted.startswith("KILL-SWITCH") and diag.halted == ""


def test_held_stock_whose_data_ends_is_sold_at_last_close():
    days, prices = make_market()
    gone = "S24.NS"  # strongest trend -> held; its data then ends (merger) and it leaves the index
    end_i = 330
    prices[gone] = prices[gone].iloc[:end_i]
    uni = FakeUniverse({**{k: ("2018-01-01", None) for k in prices if k != BENCH},
                        gone: ("2018-01-01", str(days[end_i].date()))})
    res = run_v2b(prices, "2020-01-01", uni=uni)
    s = res.trades[(res.trades.symbol == gone) & (res.trades.side == "SELL")]
    assert len(s) >= 1 and "data ended" in s.iloc[-1]["reason"]
    assert s.iloc[-1]["price"] == pytest.approx(prices[gone]["close"].iloc[-1] * 0.9995)
