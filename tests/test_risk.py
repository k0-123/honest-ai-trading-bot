import pandas as pd

from nsebot.config import RiskCfg
from nsebot.risk.manager import RiskManager

rm = RiskManager(RiskCfg())


def test_size_limited_by_risk_per_trade():
    # risk 1% of 100k = 1000; per-share risk 100 -> 10 shares (cap 20% = 20k/1000=20)
    assert rm.size(equity=100_000, cash=100_000, entry_price=1000, stop=900) == 10


def test_size_limited_by_max_position():
    # per-share risk 1 -> 1000 shares by risk, but 20% cap -> 20 shares
    assert rm.size(equity=100_000, cash=100_000, entry_price=1000, stop=999) == 20


def test_size_limited_by_cash():
    assert rm.size(equity=100_000, cash=5_000, entry_price=1000, stop=999) == 4  # leaves room for costs


def test_size_zero_for_bad_stop_or_tiny_equity():
    assert rm.size(100_000, 100_000, 1000, 1000) == 0
    assert rm.size(100_000, 100_000, 1000, 1100) == 0
    assert rm.size(100_000, 100_000, 1000, None) == 0
    assert rm.size(1_000, 1_000, 5000, 4000) == 0


def bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": 1})


def test_stop_breached_on_gap_below():
    assert rm.stop_breached(95, bar(90, 92, 88, 91)) is True


def test_stop_breached_on_intraday_touch():
    assert rm.stop_breached(95, bar(100, 101, 94, 99)) is True
    assert rm.stop_breached(95, bar(100, 101, 95, 99)) is True   # touching the stop counts


def test_stop_not_breached():
    assert rm.stop_breached(95, bar(100, 101, 96, 99)) is False


def test_daily_loss_limit():
    assert rm.daily_loss_breached(100_000, 98_000) is True
    assert rm.daily_loss_breached(100_000, 98_100) is False


def test_drawdown_kill_switch():
    assert rm.drawdown_breached(120_000, 108_000) is True
    assert rm.drawdown_breached(120_000, 108_100) is False
