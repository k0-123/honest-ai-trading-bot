"""nse_holidays in config.yaml must equal NSE's official equity trading-holiday list (weekdays only)."""
import json
from datetime import datetime

import pandas as pd

from nsebot.config import Config, load_config

SNAPSHOT = "data/nse_holidays/nse_trading_holidays_2026.json"


def official_weekday_holidays(path=SNAPSHOT):
    with open(path) as f:
        cm = json.load(f)["CM"]
    days = [pd.to_datetime(e["tradingDate"], format="%d-%b-%Y") for e in cm]
    return sorted(d.strftime("%Y-%m-%d") for d in days if d.day_of_week < 5)


def test_config_matches_nse_official_2026_list():
    cfg = load_config("config.yaml")
    official = official_weekday_holidays()
    assert len(official) == 16
    assert [d for d in cfg.nse_holidays if d.startswith("2026")] == official


def test_holidays_are_valid_sorted_unique_weekdays():
    h = load_config("config.yaml").nse_holidays
    assert h == sorted(set(h))
    assert all(pd.Timestamp(d).day_of_week < 5 for d in h)


def test_paper_runner_skips_a_listed_holiday_even_if_the_feed_has_a_bar(tmp_path):
    """Yahoo shows a zero-volume phantom bar on 2026-01-15 (an NSE holiday): it must not become a session."""
    from nsebot.paper.runner import IST, PaperRunner
    days = pd.bdate_range("2026-01-12", "2026-01-20", name="date")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=days)

    class L:
        def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
            return df.loc[: pd.Timestamp(as_of)] if as_of is not None else df

    cfg = Config(symbols=["A.NS"], benchmark="NB.NS", nse_holidays=["2026-01-15"],
                 paths={"state_dir": str(tmp_path / "s"), "reports_dir": str(tmp_path / "r")})

    class Hold:
        source = "t"

        def generate(self, symbol, history, in_position):
            from nsebot.strategies.base import Signal
            return Signal(history.index[-1], symbol, "HOLD", 0.0, "hold", "t")

    r = PaperRunner(cfg, L(), {"baseline": Hold()}, tmp_path / "s")
    r.run(datetime(2026, 1, 12, 16, 30, tzinfo=IST))
    res = r.run(datetime(2026, 1, 19, 16, 30, tzinfo=IST))
    assert "2026-01-15" not in res.processed_days
    assert res.processed_days == ["2026-01-13", "2026-01-14", "2026-01-16", "2026-01-19"]


def test_missing_holiday_year_is_reported():
    from nsebot.paper.runner import holiday_warning
    cfg = Config(symbols=["A.NS"], nse_holidays=["2026-01-26"])
    assert holiday_warning(cfg, datetime(2026, 5, 1)) is None
    msg = holiday_warning(cfg, datetime(2027, 1, 4))
    assert msg and "2027" in msg
