from __future__ import annotations

import pytest

from nsebot.config import CostCfg
from nsebot.costs import CostBreakdown, apply_slippage, trade_costs


def test_buy_default_costs_hand_computed():
    cfg = CostCfg()
    c = trade_costs("BUY", 10, 1000.0, cfg)
    assert isinstance(c, CostBreakdown)
    turnover = 10000.0
    assert c.brokerage == 0.0
    assert c.stt == pytest.approx(10.0)
    assert c.exchange == pytest.approx(0.297)
    assert c.sebi == pytest.approx(0.01)
    assert c.gst == pytest.approx(0.18 * (0.0 + 0.297 + 0.01))
    assert c.stamp == pytest.approx(1.5)
    assert c.dp == 0.0
    assert c.total == pytest.approx(10.0 + 0.297 + 0.01 + 0.18 * 0.307 + 1.5)
    assert turnover == 10 * 1000.0


def test_sell_default_costs_hand_computed():
    cfg = CostCfg()
    c = trade_costs("SELL", 10, 1000.0, cfg)
    assert c.brokerage == 0.0
    assert c.stt == pytest.approx(10.0)
    assert c.exchange == pytest.approx(0.297)
    assert c.sebi == pytest.approx(0.01)
    assert c.gst == pytest.approx(0.18 * 0.307)
    assert c.stamp == 0.0  # stamp duty is buy-side only
    assert c.dp == pytest.approx(15.93)  # DP charge is sell-side only
    assert c.total == pytest.approx(10.0 + 0.297 + 0.01 + 0.18 * 0.307 + 15.93)


def test_total_is_sum_of_components():
    cfg = CostCfg(brokerage_pct=0.0003)
    for side in ("BUY", "SELL"):
        c = trade_costs(side, 37, 2345.6, cfg)
        parts = c.brokerage + c.stt + c.exchange + c.sebi + c.gst + c.stamp + c.dp
        assert c.total == pytest.approx(parts)


def test_brokerage_percentage_applies_below_cap():
    cfg = CostCfg(brokerage_pct=0.0003, brokerage_cap=20.0)
    c = trade_costs("BUY", 10, 1000.0, cfg)  # turnover 10000 -> 3.0 raw
    assert c.brokerage == pytest.approx(3.0)
    assert c.gst == pytest.approx(0.18 * (3.0 + 0.297 + 0.01))


def test_brokerage_is_capped():
    cfg = CostCfg(brokerage_pct=0.001, brokerage_cap=20.0)
    c = trade_costs("BUY", 1000, 1000.0, cfg)  # turnover 1e6 -> 1000 raw, capped at 20
    assert c.brokerage == pytest.approx(20.0)
    assert c.gst == pytest.approx(0.18 * (20.0 + c.exchange + c.sebi))


def test_zero_qty_all_zero():
    for side in ("BUY", "SELL"):
        c = trade_costs(side, 0, 1000.0, CostCfg())
        assert c.brokerage == c.stt == c.exchange == c.sebi == c.gst == 0.0
        assert c.stamp == c.dp == c.total == 0.0


def test_slippage_direction():
    cfg = CostCfg(slippage_pct=0.0005)
    assert apply_slippage("BUY", 1000.0, cfg) == pytest.approx(1000.5)
    assert apply_slippage("SELL", 1000.0, cfg) == pytest.approx(999.5)


def test_zero_slippage_is_identity():
    cfg = CostCfg(slippage_pct=0.0)
    assert apply_slippage("BUY", 123.4, cfg) == pytest.approx(123.4)
    assert apply_slippage("SELL", 123.4, cfg) == pytest.approx(123.4)
