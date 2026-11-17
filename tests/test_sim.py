import json

import pandas as pd
import pytest

from nsebot.config import Config
from nsebot.portfolio import Book
from nsebot.sim import Simulator
from nsebot.strategies.base import Signal

CFG = Config(symbols=["AAA.NS"])


class Scripted:
    """Returns pre-scripted actions by date string."""
    source = "test"

    def __init__(self, script):
        self.script = script

    def generate(self, symbol, history, in_position):
        d = history.index[-1]
        action, stop = self.script.get(d.strftime("%Y-%m-%d"), ("HOLD", None))
        return Signal(d, symbol, action, 0.6, f"scripted {action}", "test", stop=stop)


def make_bars(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    df = pd.DataFrame([r[1:] for r in rows], index=idx, columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    df.index.name = "date"
    return df


def run(sim, bars, sym="AAA.NS"):
    for d in bars.index:
        sim.on_open(d, {sym: bars.loc[d]})
        sim.on_close(d, {sym: bars.loc[:d]})


def test_buy_fills_next_open_then_stop_out():
    bars = make_bars([
        ("2024-01-01", 100, 101, 99, 100),
        ("2024-01-02", 102, 103, 101, 102),   # fill BUY at this open
        ("2024-01-03", 101, 101, 94, 95),     # stop 96 touched -> exit at 96
    ])
    book = Book("t", cash=100_000)
    sim = Simulator(book, CFG, Scripted({"2024-01-01": ("BUY", 96.0)}), ["AAA.NS"])
    run(sim, bars)
    buy, sell = book.trades
    assert buy["date"] == "2024-01-02" and buy["side"] == "BUY"
    assert buy["price"] == pytest.approx(102 * 1.0005)
    assert sell["date"] == "2024-01-03" and sell["price"] == pytest.approx(96 * 0.9995)
    assert "STOP" in sell["reason"]
    assert book.positions == {}
    # 1% risk: 1000 / (100-96) = 250 shares, 20% cap = 200 shares -> 200
    assert buy["qty"] == 200
    total_costs = buy["costs"] + sell["costs"]
    expected_cash = 100_000 - 200 * buy["price"] + 200 * sell["price"] - total_costs
    assert book.cash == pytest.approx(expected_cash, abs=0.01)


def test_accounting_identity_holds_every_day():
    bars = make_bars([("2024-01-0%d" % i, 100 + i, 101 + i, 99 + i, 100 + i) for i in range(1, 8)])
    book = Book("t", cash=100_000)
    sim = Simulator(book, CFG, Scripted({"2024-01-01": ("BUY", 90.0), "2024-01-04": ("SELL", None)}), ["AAA.NS"])
    for d in bars.index:
        sim.on_open(d, {"AAA.NS": bars.loc[d]})
        sim.on_close(d, {"AAA.NS": bars.loc[:d]})
        pos_val = sum(p.qty * bars.loc[d, "close"] for p in book.positions.values())
        assert book.equity() == pytest.approx(book.cash + pos_val)
    assert [t["side"] for t in book.trades] == ["BUY", "SELL"]
    assert book.trades[1]["date"] == "2024-01-05"


def test_gap_below_stop_skips_entry():
    bars = make_bars([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 95, 96, 94, 95)])
    book = Book("t", cash=100_000)
    run(Simulator(book, CFG, Scripted({"2024-01-01": ("BUY", 96.0)}), ["AAA.NS"]), bars)
    assert book.trades == [] and book.cash == 100_000


def test_gap_through_stop_triggers_kill_switch_and_halts():
    # stop 99 -> 1% risk sizing allows 1000 sh, 20% cap -> 200 sh. Gap open at 10 fills the stop at 10.
    bars = make_bars([
        ("2024-01-01", 100, 100, 100, 100),
        ("2024-01-02", 100, 100, 100, 100),      # buy 200 @ ~100
        ("2024-01-03", 10, 12, 9, 11),           # gap far below stop -> exit at open 10; equity -18%
        ("2024-01-04", 12, 13, 11, 12),
        ("2024-01-05", 12, 13, 11, 12),
    ])
    book = Book("t", cash=100_000)
    script = {"2024-01-01": ("BUY", 99.0), "2024-01-03": ("BUY", 5.0), "2024-01-04": ("BUY", 5.0)}
    run(Simulator(book, CFG, Scripted(script), ["AAA.NS"]), bars)
    buy, stop = book.trades
    assert buy["qty"] == 200
    assert stop["price"] == pytest.approx(10 * 0.9995) and "STOP" in stop["reason"]
    assert book.halted and "KILL-SWITCH" in book.halt_reason
    assert len(book.trades) == 2  # no new buys after the halt


def test_kill_switch_flattens_open_positions_next_open():
    bars = make_bars([("2024-01-01", 100, 100, 100, 100), ("2024-01-02", 100, 100, 100, 100),
                      ("2024-01-03", 100, 100, 100, 100), ("2024-01-04", 98, 99, 97, 98)])
    book = Book("t", cash=100_000)
    sim = Simulator(book, CFG, Scripted({"2024-01-01": ("BUY", 90.0)}), ["AAA.NS"])
    for d in bars.index[:2]:
        sim.on_open(d, {"AAA.NS": bars.loc[d]})
        sim.on_close(d, {"AAA.NS": bars.loc[:d]})
    assert "AAA.NS" in book.positions
    book.peak_equity = 200_000  # force a >10% drawdown versus peak
    d = bars.index[2]
    sim.on_open(d, {"AAA.NS": bars.loc[d]})
    sim.on_close(d, {"AAA.NS": bars.loc[:d]})
    assert book.halted and [o.side for o in book.pending] == ["SELL"]
    d = bars.index[3]
    sim.on_open(d, {"AAA.NS": bars.loc[d]})
    assert book.positions == {} and book.trades[-1]["date"] == "2024-01-04"
    assert "KILL-SWITCH" in book.trades[-1]["reason"]


def test_daily_loss_blocks_new_entries():
    # two symbols; A drops so equity falls >2% -> B's BUY signal that day is blocked
    cfg = Config(symbols=["A.NS", "B.NS"])
    # A: 200 sh with stop 99; gaps to 85 on 01-03 -> loss ~3% of equity -> B's BUY that day is blocked
    a = make_bars([("2024-01-01", 100, 100, 100, 100), ("2024-01-02", 100, 100, 100, 100),
                   ("2024-01-03", 85, 86, 84, 85), ("2024-01-04", 85, 86, 84, 85)])
    b = make_bars([("2024-01-01", 50, 50, 50, 50), ("2024-01-02", 50, 50, 50, 50),
                   ("2024-01-03", 50, 50, 50, 50), ("2024-01-04", 50, 50, 50, 50)])

    class S:
        source = "test"

        def generate(self, symbol, history, in_position):
            d = history.index[-1].strftime("%Y-%m-%d")
            if symbol == "A.NS" and d == "2024-01-01":
                return Signal(history.index[-1], symbol, "BUY", 0.6, "a", "test", stop=99.0)
            if symbol == "B.NS" and d == "2024-01-03":
                return Signal(history.index[-1], symbol, "BUY", 0.6, "b", "test", stop=45.0)
            return Signal(history.index[-1], symbol, "HOLD", 0.0, "h", "test")

    book = Book("t", cash=100_000)
    sim = Simulator(book, cfg, S(), cfg.symbols)
    for d in a.index:
        sim.on_open(d, {"A.NS": a.loc[d], "B.NS": b.loc[d]})
        sim.on_close(d, {"A.NS": a.loc[:d], "B.NS": b.loc[:d]})
    assert [(t["symbol"], t["side"]) for t in book.trades] == [("A.NS", "BUY"), ("A.NS", "SELL")]


def test_book_json_roundtrip():
    bars = make_bars([("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 102, 103, 101, 102)])
    book = Book("t", cash=100_000)
    run(Simulator(book, CFG, Scripted({"2024-01-01": ("BUY", 96.0)}), ["AAA.NS"]), bars)
    again = Book.from_dict(json.loads(json.dumps(book.to_dict())))
    assert again == book


def test_buy_and_hold_invests_once():
    bars = make_bars([("2024-01-0%d" % i, 100, 101, 99, 100) for i in range(1, 5)])
    book = Book("bh", cash=100_000)
    run(Simulator(book, CFG, None, ["AAA.NS"], buy_and_hold=True), bars)
    assert len(book.trades) == 1 and book.trades[0]["side"] == "BUY"
    assert book.cash >= 0 and book.cash < 2_000
