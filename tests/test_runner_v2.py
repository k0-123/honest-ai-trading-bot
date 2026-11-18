"""Daily paper run + replay with the v2a / v2b universe books (tests written before the integration)."""
from datetime import datetime

import numpy as np
import pandas as pd

from nsebot.config import Config
from nsebot.paper.books import v2_universe_books
from nsebot.paper.replay import replay
from nsebot.paper.runner import IST, PaperRunner
from nsebot.strategies.baseline import BaselineStrategy

DAYS = pd.bdate_range("2025-01-01", "2026-03-31", name="date")
BENCH = "NB.NS"


def frame(seed, drift):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(DAYS))))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=DAYS)


DATA = {f"S{i:02d}.NS": frame(i, 0.0002 * i) for i in range(25)}
DATA["A.NS"] = frame(100, 0.0005)
_nb = 250 * np.exp(0.0008 * np.arange(len(DAYS)))  # steady uptrend: risk-on at every month end
DATA[BENCH] = pd.DataFrame({"open": _nb, "high": _nb * 1.005, "low": _nb * 0.995, "close": _nb, "volume": 1.0},
                           index=DAYS)


class Loader:
    def __init__(self):
        self.calls = []

    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
        df = DATA[symbol]
        if as_of is not None:
            df = df.loc[: pd.Timestamp(as_of)]
        self.calls.append((symbol, as_of, refresh, df.index.max() if len(df) else None))
        return df


class Uni:
    def members_on(self, d):
        return frozenset(k for k in DATA if k.startswith("S"))

    def ever_members(self, a, b):
        return self.members_on(a)


def cfg_for(tmp_path):
    return Config(symbols=["A.NS"], benchmark=BENCH, data={"start": "2025-01-01"},
                  paths={"state_dir": str(tmp_path / "state"), "reports_dir": str(tmp_path / "rep")})


def at(d):
    d = pd.Timestamp(d)
    return datetime(d.year, d.month, d.day, 16, 30, tzinfo=IST)


def runner(tmp_path, loader):
    cfg = cfg_for(tmp_path)
    uni = Uni()
    return PaperRunner(cfg, loader, {"baseline": BaselineStrategy(cfg.baseline)}, tmp_path / "state",
                       universe=uni, universe_loader=loader, universe_books=v2_universe_books(cfg, uni))


def test_daily_run_creates_and_persists_v2_books(tmp_path):
    L = Loader()
    r = runner(tmp_path, L)
    r.run(at("2026-01-28"))
    r.run(at("2026-02-03"))           # crosses the Jan->Feb month end: v2b decides at the 01-30 close
    books = r.load_books()
    assert {"baseline", "buyhold_niftybees", "v2a_parking", "v2b_momentum"} <= set(books)
    v2a, v2b = books["v2a_parking"], books["v2b_momentum"]
    assert BENCH in v2a.positions                                     # idle cash parked in NIFTYBEES
    buys = [t for t in v2b.trades if t["side"] == "BUY"]
    assert buys and {t["date"] for t in buys} == {"2026-02-02"}      # first trading day of February
    assert v2b.meta.get("last_rebalance_month") == "2026-02"
    assert all(c[3] is None or c[3] <= pd.Timestamp(c[1]) for c in L.calls if c[1] is not None)  # no future rows


def test_same_day_rerun_is_noop_for_v2_books(tmp_path):
    r = runner(tmp_path, Loader())
    r.run(at("2026-01-28"))
    before = (tmp_path / "state" / "portfolio.json").read_text()
    res = r.run(at("2026-01-28"))
    assert res.processed_days == [] and (tmp_path / "state" / "portfolio.json").read_text() == before


def test_replay_includes_v2_books_and_never_refreshes_in_loop(tmp_path):
    L = Loader()
    cfg = cfg_for(tmp_path)
    uni = Uni()
    res = replay(cfg, L, "2026-01-01", "2026-03-31", tmp_path / "replay", universe=uni, universe_loader=L,
                 universe_books=v2_universe_books(cfg, uni))
    assert {"v2a_parking", "v2b_momentum"} <= set(res.final["book"])
    v2b = res.trades[(res.trades.book == "v2b_momentum") & (res.trades.action == "BUY")]
    first = {"2026-01-01", "2026-02-02", "2026-03-02"}
    assert len(v2b) and set(v2b.date) <= first
    in_loop = [c for c in L.calls if c[1] is not None]
    assert in_loop and not any(c[2] for c in in_loop)
    assert all(c[3] <= pd.Timestamp(c[1]) for c in in_loop if c[3] is not None)


def test_new_book_makes_no_fill_on_its_first_session(tmp_path):
    """The bot runs after the close, so on a book's first session it cannot have traded at that morning's open:
    v2a parks its cash at the NEXT open, not the open of the day it was created."""
    L = Loader()
    r = runner(tmp_path, L)
    r.run(at("2026-01-28"))
    v2a = r.load_books()["v2a_parking"]
    assert v2a.trades == [] and v2a.cash == 100_000
    r.run(at("2026-01-29"))
    v2a = r.load_books()["v2a_parking"]
    assert v2a.trades and v2a.trades[0]["date"] == "2026-01-29" and v2a.trades[0]["symbol"] == BENCH
