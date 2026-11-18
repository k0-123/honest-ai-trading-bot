"""Forward paper-trading kill-switch limits (decided before forward trading started):
v1 halts at 10% drawdown; v2a and v2b halt at 15%; the 2% daily-loss pause stays for all books."""
import pandas as pd

from nsebot.config import Config, load_config
from nsebot.paper.books import v2_universe_books
from nsebot.portfolio import Book
from nsebot.sim import Simulator


class Uni:
    def members_on(self, d):
        return frozenset()


def test_config_sets_forward_limits():
    cfg = load_config("config.yaml")
    assert cfg.risk.max_drawdown_pct == 0.10                       # v1 + all backtests: unchanged
    assert cfg.paper_max_drawdown == {"v2a_parking": 0.15, "v2b_momentum": 0.15}
    assert cfg.risk.daily_loss_limit_pct == 0.02                   # daily-loss pause unchanged


def test_v2_paper_books_use_15pct_and_v1_keeps_10pct():
    cfg = load_config("config.yaml")
    books = v2_universe_books(cfg, Uni())
    for name, factory in books.items():
        assert factory(Book(name, cash=100_000), []).max_drawdown_pct == 0.15
    assert Simulator(Book("baseline", cash=100_000), cfg, None, ["A.NS"]).max_drawdown_pct == 0.10


def _equity_path(sim, values):
    """Drive a simulator's close step with a held position whose value follows `values`."""
    from nsebot.portfolio import Position
    sim.book.cash = 0.0
    sim.book.positions["X.NS"] = Position("X.NS", 1000, 100.0, None, "2024-01-01", "t")
    days = pd.bdate_range("2024-01-01", periods=len(values), name="date")
    for d, v in zip(days, values):
        h = pd.DataFrame({"open": [v], "high": [v], "low": [v], "close": [v], "volume": [1.0]}, index=[d])
        sim._today_raw = {"X.NS": h.iloc[0]}
        sim._mark_and_guard(d, {"X.NS": h})
    return sim.book


def test_15pct_book_survives_a_12pct_drawdown_but_halts_at_15pct():
    cfg = Config(symbols=["X.NS"])
    s = Simulator(Book("v2b_momentum", cash=100_000), cfg, None, ["X.NS"])
    s.max_drawdown_pct = 0.15
    # step down by < 2% a day so the daily-loss pause is irrelevant here
    b = _equity_path(s, [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88])     # -12%
    assert not b.halted
    b = _equity_path(s, [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85])  # -15%
    assert b.halted and "15%" in b.halt_reason


def test_10pct_book_halts_at_10pct():
    cfg = Config(symbols=["X.NS"])
    s = Simulator(Book("baseline", cash=100_000), cfg, None, ["X.NS"])
    b = _equity_path(s, [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90])
    assert b.halted and "10%" in b.halt_reason


def test_daily_loss_pause_still_applies_to_v2_books():
    cfg = Config(symbols=["X.NS"])
    s = Simulator(Book("v2a_parking", cash=100_000), cfg, None, ["X.NS"])
    s.max_drawdown_pct = 0.15
    b = _equity_path(s, [100, 97])                                  # -3% in one day
    assert b.block_new_entries and not b.halted
