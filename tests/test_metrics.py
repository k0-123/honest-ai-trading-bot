from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nsebot import metrics as m


def _eq(values) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values), name="date")
    return pd.Series(values, index=idx, dtype=float)


def _trades(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "symbol", "side", "qty", "price", "costs"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---- returns / growth ----------------------------------------------------

def test_daily_returns_simple_pct_change():
    r = m.daily_returns(_eq([100.0, 110.0, 99.0]))
    assert list(r.round(10)) == pytest.approx([0.1, -0.1])


def test_total_return():
    assert m.total_return(_eq([100.0, 110.0, 99.0])) == pytest.approx(-0.01)


def test_cagr_uses_periods_as_bars_per_year():
    # 2 return observations with periods=2 -> 1 year -> CAGR equals total growth
    assert m.cagr(_eq([100.0, 110.0, 121.0]), periods=2) == pytest.approx(0.21)


def test_cagr_two_years():
    # 4 return observations with periods=2 -> 2 years; 100 -> 144 is 20% per year
    assert m.cagr(_eq([100.0, 110.0, 120.0, 130.0, 144.0]), periods=2) == pytest.approx(0.2)


# ---- risk-adjusted ------------------------------------------------------

def test_sharpe_flat_equity_is_zero():
    assert m.sharpe(m.daily_returns(_eq([100.0, 100.0, 100.0, 100.0]))) == 0.0


def test_sharpe_too_few_obs_is_zero():
    assert m.sharpe(pd.Series([0.01])) == 0.0
    assert m.sharpe(pd.Series([], dtype=float)) == 0.0


def test_sharpe_formula():
    r = pd.Series([0.02, 0.00, 0.02, 0.00])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert m.sharpe(r) == pytest.approx(expected)


def test_sharpe_risk_free_rate_is_subtracted_daily():
    r = pd.Series([0.02, 0.00, 0.02, 0.00])
    rf_daily = 0.252 / 252  # 0.001 per day
    expected = (r - rf_daily).mean() / r.std(ddof=1) * math.sqrt(252)
    assert m.sharpe(r, rf_annual=0.252) == pytest.approx(expected)


def test_sortino_no_negative_returns_is_zero_even_with_rf():
    # all returns positive (but below the daily rf hurdle): contract says 0.0 when no returns < 0
    assert m.sortino(pd.Series([0.0005, 0.0006, 0.0004]), rf_annual=0.252) == 0.0


def test_sortino_formula_downside_deviation_over_all_obs():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    dd = math.sqrt(((r.clip(upper=0.0)) ** 2).mean())  # sqrt((1e-4 + 4e-4) / 4)
    expected = r.mean() / dd * math.sqrt(252)
    assert m.sortino(r) == pytest.approx(expected)


def test_sortino_no_negative_returns_is_zero():
    assert m.sortino(pd.Series([0.01, 0.02, 0.0])) == 0.0


def test_max_drawdown_simple():
    assert m.max_drawdown(_eq([100.0, 110.0, 99.0])) == pytest.approx(0.1)


def test_max_drawdown_uses_running_peak():
    assert m.max_drawdown(_eq([100.0, 120.0, 60.0, 130.0])) == pytest.approx(0.5)


def test_max_drawdown_monotonic_is_zero():
    assert m.max_drawdown(_eq([100.0, 101.0, 105.0])) == 0.0


# ---- tail risk ----------------------------------------------------------

def test_var_cvar_fixed_array():
    r = pd.Series([-0.05, -0.02, 0.01, 0.03, -0.10, 0.02, 0.00, -0.01, 0.04, 0.015])
    var, cvar = m.var_cvar(r, level=0.95)
    # losses sorted: ..., 0.01, 0.02, 0.05, 0.10; linear quantile at 0.95 -> 0.05 + 0.55*0.05
    assert var == pytest.approx(0.0775)
    # tail = losses >= VaR = [0.10]
    assert cvar == pytest.approx(0.10)


def test_var_cvar_are_positive_loss_fractions():
    r = pd.Series([-0.03, 0.01, -0.02, 0.02])
    var, cvar = m.var_cvar(r, level=0.5)
    assert var > 0 and cvar >= var


# ---- trades -------------------------------------------------------------

def test_round_trip_fifo_partial_sells():
    t = _trades([
        ["2024-01-01", "A", "BUY", 10, 100.0, 0.0],
        ["2024-01-02", "A", "BUY", 10, 110.0, 0.0],
        ["2024-01-03", "A", "SELL", 4, 120.0, 0.0],
        ["2024-01-04", "A", "SELL", 16, 90.0, 0.0],
    ])
    # sell 4 consumes lot1: (120-100)*4 = 80
    # sell 16 consumes lot1 remainder 6 and lot2 10: (90-100)*6 + (90-110)*10 = -260
    assert m.round_trip_pnls(t) == pytest.approx([80.0, -260.0])


def test_round_trip_nets_costs_proportionally():
    t = _trades([
        ["2024-01-01", "A", "BUY", 10, 100.0, 10.0],   # 1.0 per share
        ["2024-01-02", "A", "SELL", 4, 110.0, 8.0],    # 2.0 per share
        ["2024-01-03", "A", "SELL", 6, 120.0, 6.0],    # 1.0 per share
    ])
    # sell1: 40 - 4*1 - 8 = 28 ; sell2: 120 - 6*1 - 6 = 108
    assert m.round_trip_pnls(t) == pytest.approx([28.0, 108.0])


def test_round_trip_fifo_across_lots_with_costs():
    t = _trades([
        ["2024-01-01", "A", "BUY", 10, 100.0, 10.0],
        ["2024-01-02", "A", "BUY", 10, 110.0, 0.0],
        ["2024-01-03", "A", "SELL", 15, 120.0, 15.0],
    ])
    # lot1 10 sh: (120-100)*10 - 10 - 10 = 180 ; lot2 5 sh: (120-110)*5 - 0 - 5 = 45 ; one sell fill -> 225
    assert m.round_trip_pnls(t) == pytest.approx([225.0])


def test_round_trip_is_fifo_per_symbol():
    t = _trades([
        ["2024-01-01", "A", "BUY", 5, 100.0, 0.0],
        ["2024-01-01", "B", "BUY", 5, 50.0, 0.0],
        ["2024-01-02", "A", "SELL", 5, 110.0, 0.0],
    ])
    # B lot is never sold -> no round trip; A round trip = 50
    assert m.round_trip_pnls(t) == pytest.approx([50.0])


def test_round_trip_empty_frame():
    t = pd.DataFrame(columns=["date", "symbol", "side", "qty", "price", "costs"])
    assert m.round_trip_pnls(t) == []


def test_win_rate():
    assert m.win_rate([100.0, -50.0, 0.0, 25.0]) == pytest.approx(0.5)


def test_win_rate_empty_is_zero():
    assert m.win_rate([]) == 0.0


# ---- summary ------------------------------------------------------------

_SUMMARY_KEYS = {
    "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
    "var95", "cvar95", "win_rate", "n_trades", "total_costs",
}


def test_summarize_keys_and_values_without_trades():
    eq = _eq([100.0, 110.0, 99.0])
    s = m.summarize(eq)
    assert set(s) == _SUMMARY_KEYS
    assert s["total_return"] == pytest.approx(-0.01)
    assert s["max_drawdown"] == pytest.approx(0.1)
    assert s["n_trades"] == 0
    assert s["total_costs"] == 0.0
    assert s["win_rate"] == 0.0


def test_summarize_with_trades():
    eq = _eq([100.0, 110.0, 121.0])
    t = _trades([
        ["2024-01-01", "A", "BUY", 10, 100.0, 10.0],
        ["2024-01-03", "A", "SELL", 10, 110.0, 20.0],
    ])
    s = m.summarize(eq, t)
    assert set(s) == _SUMMARY_KEYS
    assert s["n_trades"] == 2
    assert s["total_costs"] == pytest.approx(30.0)
    assert s["win_rate"] == pytest.approx(1.0)  # one round trip: 100 - 10 - 20 = 70 > 0
    assert s["cagr"] == pytest.approx((121.0 / 100.0) ** (252 / 2) - 1)
    assert math.isfinite(s["sharpe"]) and math.isfinite(s["sortino"])
    assert s["cvar95"] >= s["var95"]  # gain-only curve: both negative, VaR is not clamped


def test_summarize_flat_equity_no_nan():
    s = m.summarize(_eq([100.0, 100.0, 100.0]))
    assert all(np.isfinite(v) for v in s.values())
    assert s["sharpe"] == 0.0
