import csv
from datetime import datetime

import pandas as pd

from nsebot.config import Config
from nsebot.paper.runner import IST, PaperRunner, effective_date
from nsebot.strategies.base import Signal

CFG = Config(symbols=["A.NS"], benchmark="NB.NS")
DAYS = pd.bdate_range("2024-01-01", "2024-01-10", name="date")


def frame(prices):
    df = pd.DataFrame(prices, index=DAYS[: len(prices)], columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    return df


A = frame([(100, 100, 100, 100), (101, 102, 100, 101), (102, 103, 94, 95), (96, 97, 95, 96),
           (97, 98, 96, 97), (98, 99, 97, 98), (99, 100, 98, 99), (100, 101, 99, 100)])
NB = frame([(250, 251, 249, 250)] * 8)


class FakeLoader:
    def get(self, symbol, start, end=None, as_of=None, refresh=False):
        df = {"A.NS": A, "NB.NS": NB}[symbol]
        return df.loc[: pd.Timestamp(as_of)] if as_of is not None else df


class BuyOnFirstDay:
    source = "test"

    def generate(self, symbol, history, in_position):
        d = history.index[-1]
        if d == DAYS[0]:
            return Signal(d, symbol, "BUY", 0.6, "test buy", "test", stop=96.0)
        return Signal(d, symbol, "HOLD", 0.0, "hold", "test")


def at(day, hh=16, mm=30):
    ts = pd.Timestamp(day)
    return datetime(ts.year, ts.month, ts.day, hh, mm, tzinfo=IST)


def test_effective_date_respects_market_close():
    assert effective_date(at("2024-01-03", 16, 30)) == pd.Timestamp("2024-01-03")
    assert effective_date(at("2024-01-03", 11, 0)) == pd.Timestamp("2024-01-02")


def test_three_day_cycle_signal_fill_stop(tmp_path):
    r = PaperRunner(CFG, FakeLoader(), {"baseline": BuyOnFirstDay()}, tmp_path)
    r.run(at(DAYS[0]))                      # day 1 close: BUY signal -> pending
    b = r.load_books()["baseline"]
    assert b.trades == [] and len(b.pending) == 1
    r.run(at(DAYS[1]))                      # day 2 open: filled
    b = r.load_books()["baseline"]
    assert b.trades[0]["date"] == "2024-01-02" and b.trades[0]["side"] == "BUY"
    r.run(at(DAYS[2]))                      # day 3: low 94 <= stop 96 -> stopped out
    b = r.load_books()["baseline"]
    assert b.trades[-1]["side"] == "SELL" and "STOP" in b.trades[-1]["reason"]
    rows = list(csv.DictReader(open(tmp_path / "trades.csv")))
    assert {"date", "symbol", "side", "qty", "price", "costs", "reason", "signal_source"} <= set(rows[0])
    assert any(row["book"] == "buyhold_niftybees" for row in rows)
    assert (tmp_path / "equity.csv").exists() and (tmp_path / "signals.csv").exists()


def test_same_day_rerun_is_a_noop(tmp_path):
    r = PaperRunner(CFG, FakeLoader(), {"baseline": BuyOnFirstDay()}, tmp_path)
    r.run(at(DAYS[0]))
    r.run(at(DAYS[1]))
    before = (tmp_path / "portfolio.json").read_text()
    res = r.run(at(DAYS[1], 18, 0))
    assert res.processed_days == [] and "Nothing to do" in res.message
    assert (tmp_path / "portfolio.json").read_text() == before


def test_catches_up_missed_days(tmp_path):
    r = PaperRunner(CFG, FakeLoader(), {"baseline": BuyOnFirstDay()}, tmp_path)
    r.run(at(DAYS[0]))
    res = r.run(at(DAYS[4]))                # VPS was down for 3 sessions
    assert res.processed_days == [d.strftime("%Y-%m-%d") for d in DAYS[1:5]]


def test_weekend_does_nothing(tmp_path):
    r = PaperRunner(CFG, FakeLoader(), {"baseline": BuyOnFirstDay()}, tmp_path)
    r.run(at("2024-01-05"))
    res = r.run(at("2024-01-06"))           # Saturday
    assert res.processed_days == []
