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
    # sized at signal close: min(1000/(100-96)=250, 20% cap=200) = 200; re-checked at the real fill
    # 102.05: 1% risk -> 1000/(102.05-96) = 165 shares
    assert buy["qty"] == 165
    total_costs = buy["costs"] + sell["costs"]
    expected_cash = 100_000 - 165 * buy["price"] + 165 * sell["price"] - total_costs
    assert book.cash == pytest.approx(expected_cash, abs=0.01)


def test_accounting_identity_holds_every_day():
    bars = make_bars([(f"2024-01-0{i}", 100 + i, 101 + i, 99 + i, 100 + i) for i in range(1, 8)])
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
    assert buy["qty"] == 199  # 20% cap checked at the slipped fill price: 20,000 / 100.05
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
    bars = make_bars([(f"2024-01-0{i}", 100, 101, 99, 100) for i in range(1, 5)])
    book = Book("bh", cash=100_000)
    run(Simulator(book, CFG, None, ["AAA.NS"], buy_and_hold=True), bars)
    assert len(book.trades) == 1 and book.trades[0]["side"] == "BUY"
    assert book.cash >= 0 and book.cash < 2_000


def test_gap_up_fill_respects_position_cap_and_risk():
    # sized at close 100 / stop 99 -> 200 sh (20% cap); opens at 110 -> must shrink to <=20% and <=1% risk
    bars = make_bars([("2024-01-01", 100, 100, 100, 100), ("2024-01-02", 110, 111, 109, 110)])
    book = Book("t", cash=100_000)
    run(Simulator(book, CFG, Scripted({"2024-01-01": ("BUY", 99.0)}), ["AAA.NS"]), bars)
    t = book.trades[0]
    assert t["qty"] * t["price"] <= 0.20 * 100_000 + 1e-6
    assert t["qty"] * (t["price"] - 99.0) <= 0.01 * 100_000 + 1e-6


def test_unfilled_buy_expires_after_one_session():
    cfg = Config(symbols=["A.NS", "B.NS"])
    a = make_bars([("2024-01-01", 100, 100, 100, 100), ("2024-01-03", 100, 100, 100, 100)])  # no bar 01-02
    b = make_bars([("2024-01-01", 50, 50, 50, 50), ("2024-01-02", 50, 50, 50, 50), ("2024-01-03", 50, 50, 50, 50)])
    book = Book("t", cash=100_000)
    sim = Simulator(book, cfg, Scripted({"2024-01-01": ("BUY", 95.0)}), cfg.symbols)
    for d in b.index:
        today = {s: df.loc[d] for s, df in {"A.NS": a, "B.NS": b}.items() if d in df.index}
        sim.on_open(d, today)
        sim.on_close(d, {"A.NS": a.loc[:d], "B.NS": b.loc[:d]})
    assert not any(t["symbol"] == "A.NS" for t in book.trades)


def test_halted_book_places_zero_buys_sells_until_flat_then_does_nothing():
    cfg = Config(symbols=["A.NS", "B.NS"])
    rows = lambda px: [(f"2024-01-0{i}", px[i - 1], px[i - 1], px[i - 1], px[i - 1]) for i in range(1, 9)]
    a = make_bars(rows([100, 100, 100, 100, 100, 100, 100, 100]))
    b = make_bars(rows([50, 50, 50, 50, 50, 50, 50, 50]))

    class AlwaysBuy:  # would buy every symbol every day if allowed
        source = "test"

        def generate(self, symbol, history, in_position):
            d = history.index[-1]
            stop = float(history["close"].iloc[-1]) * 0.9
            return Signal(d, symbol, "HOLD" if in_position else "BUY", 0.6, "always buy", "test", stop=stop)

    book = Book("t", cash=100_000)
    sim = Simulator(book, cfg, AlwaysBuy(), cfg.symbols)
    days = list(a.index)
    for d in days[:3]:  # day1 signals, day2 fills both, day3 holds
        sim.on_open(d, {"A.NS": a.loc[d], "B.NS": b.loc[d]})
        sim.on_close(d, {"A.NS": a.loc[:d], "B.NS": b.loc[:d]})
    assert set(book.positions) == {"A.NS", "B.NS"}
    book.peak_equity = 200_000  # force the kill-switch at the next close
    n_before = len(book.trades)
    snapshots = []
    for d in days[3:]:
        sim.on_open(d, {"A.NS": a.loc[d], "B.NS": b.loc[d]})
        sim.on_close(d, {"A.NS": a.loc[:d], "B.NS": b.loc[:d]})
        snapshots.append((len(book.trades), dict(book.positions), list(book.pending)))
    assert book.halted
    after = book.trades[n_before:]
    assert all(t["side"] == "SELL" for t in after)          # zero new BUYs once halted
    assert sorted(t["symbol"] for t in after) == ["A.NS", "B.NS"]
    assert book.positions == {} and book.pending == []       # flat
    flat_from = next(i for i, s in enumerate(snapshots) if not s[1])
    assert all(s == snapshots[flat_from] for s in snapshots[flat_from:])  # then does nothing at all
