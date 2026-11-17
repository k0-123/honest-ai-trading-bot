import numpy as np
import pandas as pd

from nsebot.data.quality import audit_symbol, render


def test_audit_flags_unadjusted_split_and_reports_actions():
    idx = pd.bdate_range("2020-01-01", periods=40, name="date")
    c = np.full(40, 1000.0)
    c[20:] = 500.0  # a 2:1 split that was NOT adjusted -> -50% jump
    raw = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    a = audit_symbol("X.NS", raw, [(idx[20].strftime("%Y-%m-%d"), 2.0)])
    assert a.action_max_move[idx[20].strftime("%Y-%m-%d")] >= 0.49
    assert a.big_moves and a.big_moves[0][0] == idx[20].strftime("%Y-%m-%d")
    assert "X.NS" in render([a])


def test_audit_clean_adjusted_series_has_no_big_moves():
    idx = pd.bdate_range("2020-01-01", periods=40, name="date")
    c = np.linspace(100, 110, 40)
    raw = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    a = audit_symbol("Y.NS", raw, [(idx[10].strftime("%Y-%m-%d"), 10.0)])
    assert a.big_moves == [] and a.held_out == [] and a.action_max_move[idx[10].strftime("%Y-%m-%d")] < 0.01
