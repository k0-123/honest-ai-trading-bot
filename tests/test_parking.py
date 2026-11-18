"""strategy-v2a ("v1 rules + NIFTYBEES parking") tests, written before the implementation."""
import numpy as np
import pandas as pd
import pytest

from nsebot.backtest.engine import run_backtest
from nsebot.config import Config
from nsebot.sim_parking import ParkingSimulator
from nsebot.strategies.base import Signal

BENCH = "NB.NS"
CFG = Config(symbols=["A.NS"], benchmark=BENCH)


class Uni:
    def __init__(self, spans):
        self.spans = {k: (pd.Timestamp(a), pd.Timestamp(b) if b else pd.Timestamp("2100-01-01")) for k, (a, b) in spans.items()}

    def members_on(self, d):
        d = pd.Timestamp(d)
        return frozenset(k for k, (a, b) in self.spans.items() if a <= d <= b)


class Scripted:
    source = "baseline"

    def __init__(self, script):
        self.script = script  # {(symbol, 'YYYY-MM-DD'): ('BUY'|'SELL', stop)}

    def generate(self, symbol, history, in_position):
        d = history.index[-1]
        act, stop = self.script.get((symbol, d.strftime("%Y-%m-%d")), ("HOLD", None))
        if act == "BUY" and in_position or act == "SELL" and not in_position:
            act = "HOLD"
        return Signal(d, symbol, act, 0.6, f"scripted {act}", "baseline", stop=stop)


def bars(closes, days, lows=None):
    c = np.asarray(closes, float)
    lo = np.asarray(lows, float) if lows is not None else c * 0.995
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": lo, "close": c, "volume": 1.0}, index=days)


DAYS = pd.bdate_range("2024-01-01", periods=12, name="date")


def run(prices, script, uni=None, kill_switch=True, start="2024-01-01"):
    uni = uni or Uni({k: ("2020-01-01", None) for k in prices if k != BENCH})
    holder = {}

    def f(book, days):
        holder["s"] = ParkingSimulator(book, CFG, uni, Scripted(script), days, kill_switch=kill_switch)
        return holder["s"]
    res = run_backtest(prices, None, CFG, start, None, "v2a", sim_factory=f)
    res.sim = holder["s"]
    return res


def flat(v, n=12):
    return [v] * n


def test_idle_cash_is_parked_in_niftybees_from_the_first_open():
    r = run({"A.NS": bars(flat(100), DAYS), BENCH: bars(flat(250), DAYS)}, {})
    t = r.trades
    assert t.iloc[0].symbol == BENCH and t.iloc[0].side == "BUY" and t.iloc[0].date == "2024-01-01"
    assert r.sim.book.cash < 1_000 + 250 * 1.01          # only the buffer + < 1 unit stays in cash
    assert r.sim.book.positions[BENCH].qty * 250 > 98_000


def test_buy_is_funded_by_selling_just_enough_niftybees_at_the_same_open():
    script = {("A.NS", "2024-01-03"): ("BUY", 95.0)}
    r = run({"A.NS": bars(flat(100), DAYS), BENCH: bars(flat(250), DAYS)}, script)
    t = r.trades[r.trades.date == "2024-01-04"]
    assert list(zip(t.symbol, t.side)) == [(BENCH, "SELL"), ("A.NS", "BUY")]
    stock_cost = t[t.symbol == "A.NS"].iloc[0]
    nb_sold = t[t.symbol == BENCH].iloc[0]
    # 1% risk re-checked at the real fill price 100.05 (as in v1): ~998 / (100.05 - 95) = 197 shares
    assert stock_cost.qty * (stock_cost.price - 95.0) <= 0.01 * 100_000 and stock_cost.qty >= 190
    proceeds = nb_sold.qty * nb_sold.price
    cost = stock_cost.qty * stock_cost.price + stock_cost.costs
    # cash on hand before the open is the ~Rs 1,000 buffer (+ < 1 NIFTYBEES unit): proceeds cover the rest
    assert proceeds >= cost - (1_000 + 250 * 1.01)
    assert proceeds <= cost + 0.004 * cost + 250            # no more than needed (+0.3% cash buffer, +1 unit)
    assert r.sim.book.cash >= 0
    assert BENCH in r.sim.book.positions                 # rest stays parked


def test_proceeds_of_a_stock_sale_are_reparked_at_the_same_open():
    script = {("A.NS", "2024-01-03"): ("BUY", 95.0), ("A.NS", "2024-01-08"): ("SELL", None)}
    r = run({"A.NS": bars(flat(100), DAYS), BENCH: bars(flat(250), DAYS)}, script)
    t = r.trades[r.trades.date == "2024-01-09"]
    assert list(zip(t.symbol, t.side)) == [("A.NS", "SELL"), (BENCH, "BUY")]
    assert r.sim.book.cash < 1_000 + 250 * 1.01


def test_stop_out_proceeds_are_reparked():
    lows = flat(99.5)
    lows[5] = 94.0                                      # stop 95 breached on day 6 -> sold at day 7 open
    script = {("A.NS", "2024-01-03"): ("BUY", 95.0)}
    r = run({"A.NS": bars(flat(100), DAYS, lows), BENCH: bars(flat(250), DAYS)}, script)
    sell_day = r.trades[(r.trades.symbol == "A.NS") & (r.trades.side == "SELL")].iloc[0].date
    assert sell_day == DAYS[6].strftime("%Y-%m-%d")
    t = r.trades[r.trades.date == sell_day]
    assert (t.symbol == BENCH).any() and (t[t.symbol == BENCH].side == "BUY").all()


def test_accounting_identity_and_no_negative_cash():
    script = {("A.NS", "2024-01-03"): ("BUY", 95.0), ("A.NS", "2024-01-08"): ("SELL", None)}
    days = DAYS
    a = bars(np.linspace(100, 110, 12), days)
    nb = bars(np.linspace(250, 240, 12), days)
    r = run({"A.NS": a, BENCH: nb}, script)
    b = r.sim.book
    assert b.cash >= 0
    marked = b.cash + sum(p.qty * (a.close.iloc[-1] if s == "A.NS" else nb.close.iloc[-1]) for s, p in b.positions.items())
    assert r.equity.iloc[-1] == pytest.approx(marked, abs=0.05)


def test_sizing_counts_niftybees_as_equity_and_caps_at_20pct():
    script = {("A.NS", "2024-01-03"): ("BUY", 99.9)}  # tiny risk per share -> 20% cap binds
    r = run({"A.NS": bars(flat(100), DAYS), BENCH: bars(flat(250), DAYS)}, script)
    buy = r.trades[(r.trades.symbol == "A.NS")].iloc[0]
    assert buy.qty * buy.price <= 0.20 * 100_000 * 1.001
    assert buy.qty * buy.price >= 0.19 * 100_000         # equity incl. NIFTYBEES (~100k), not just ~1k cash


def test_only_point_in_time_members_can_be_bought():
    script = {("A.NS", "2024-01-03"): ("BUY", 95.0)}
    uni = Uni({"A.NS": ("2024-01-05", None)})            # joins the index only on 01-05
    r = run({"A.NS": bars(flat(100), DAYS), BENCH: bars(flat(250), DAYS)}, script, uni=uni)
    assert not (r.trades.symbol == "A.NS").any()


def test_held_stock_that_leaves_the_index_keeps_its_exit_rules():
    script = {("A.NS", "2024-01-03"): ("BUY", 95.0), ("A.NS", "2024-01-10"): ("SELL", None)}
    uni = Uni({"A.NS": ("2020-01-01", "2024-01-05")})    # leaves the index after 01-05
    r = run({"A.NS": bars(flat(100), DAYS), BENCH: bars(flat(250), DAYS)}, script, uni=uni)
    sells = r.trades[(r.trades.symbol == "A.NS") & (r.trades.side == "SELL")]
    assert list(sells.date) == ["2024-01-11"]


def test_kill_switch_sells_everything_including_niftybees_and_stays_in_cash():
    days = pd.bdate_range("2024-01-01", periods=20, name="date")
    nb = np.full(20, 250.0)
    nb[8:] = 200.0                                      # NIFTYBEES -20% -> book drawdown > 10%
    r = run({"A.NS": bars(np.full(20, 100.0), days), BENCH: bars(nb, days)}, {})
    assert r.halted.startswith("KILL-SWITCH")
    last = r.trades.iloc[-1]
    assert last.symbol == BENCH and last.side == "SELL" and "KILL-SWITCH" in last.reason
    assert r.sim.book.positions == {}
    assert (r.trades.date > last.date).sum() == 0       # never re-parks after the halt
    d = run({"A.NS": bars(np.full(20, 100.0), days), BENCH: bars(nb, days)}, {}, kill_switch=False)
    assert d.halted == "" and BENCH in d.sim.book.positions
