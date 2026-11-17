from __future__ import annotations

import numpy as np
import pandas as pd

from nsebot.backtest.engine import BacktestResult
from nsebot.metrics import summarize
from nsebot.paper.runner import TRADE_COLS, DailyResult
from nsebot.portfolio import Book, Order, Position
from nsebot.reports.backtest import write_backtest_report
from nsebot.reports.daily import write_daily_report
from nsebot.reports.weekly import write_weekly_report

BANNER = ("CONTAMINATED: LLM results on historical dates may reflect the model's knowledge of what happened. "
          "NOT evidence of skill.")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PERIODS = {"train": ("2024-01-01", "2024-01-12"), "test": ("2024-01-15", "2024-02-09")}
WEEKLY_DISCLAIMER = "Only forward paper-trading results count as evidence. Past performance does not guarantee future results."


def trade_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRADE_COLS)


def row(date, symbol, side, qty, price, costs, book, contaminated=False, source="baseline", reason="r"):
    return {"date": date, "symbol": symbol, "side": side, "qty": qty, "price": price, "costs": costs,
            "reason": reason, "signal_source": source, "contaminated": contaminated, "book": book}


def equity_series(name: str, start: str = "2024-01-01", n: int = 30, base: float = 100000.0) -> pd.Series:
    idx = pd.bdate_range(start, periods=n, name="date")
    i = np.arange(n)
    return pd.Series(base + 300 * i + 500 * np.sin(i), index=idx, name=name, dtype=float)


def bt_result(name: str, contaminated: bool = False, halted: str = "") -> BacktestResult:
    tr = trade_rows([
        row("2024-01-03", "A.NS", "BUY", 10, 100.0, 5.0, name),
        row("2024-01-10", "A.NS", "SELL", 10, 110.0, 6.0, name),
        row("2024-01-15", "B.NS", "BUY", 5, 50.0, 3.0, name),
        row("2024-01-20", "B.NS", "SELL", 5, 40.0, 3.0, name),
    ])
    return BacktestResult(name, equity_series(name), tr, [], halted, contaminated)


# ---------- backtest report ----------

def test_backtest_files_created(tmp_path):
    out = write_backtest_report({"baseline": bt_result("baseline")}, PERIODS, tmp_path)
    assert out == tmp_path / "backtest_report.md" and out.exists()
    assert (tmp_path / "equity_curves.png").read_bytes()[:8] == PNG_MAGIC


def test_backtest_numbers_match_summarize(tmp_path):
    r = bt_result("baseline")
    out = write_backtest_report({"baseline": r}, PERIODS, tmp_path)
    text = out.read_text(encoding="utf-8")
    eq = r.equity.loc["2024-01-15":"2024-02-09"]
    dates = pd.to_datetime(r.trades["date"])
    tr = r.trades[(dates >= "2024-01-15") & (dates <= "2024-02-09")]
    s = summarize(eq, tr)
    assert "## test" in text and "## train" in text and "## full" in text
    assert f"{s['total_return'] * 100:.2f}%" in text
    assert f"{s['cagr'] * 100:.2f}%" in text
    assert f"{s['max_drawdown'] * 100:.2f}%" in text
    assert f"{s['var95'] * 100:.2f}%" in text
    assert f"{s['cvar95'] * 100:.2f}%" in text
    assert f"{s['win_rate'] * 100:.2f}%" in text
    assert f"{s['sharpe']:.2f}" in text
    assert f"{s['sortino']:.2f}" in text
    assert f"₹{s['total_costs']:,.0f}" in text
    full = summarize(r.equity, r.trades)
    assert f"{full['win_rate'] * 100:.2f}%" in text
    assert f"₹{full['total_costs']:,.0f}" in text


def test_banner_only_when_contaminated(tmp_path):
    clean = write_backtest_report({"baseline": bt_result("baseline")}, PERIODS, tmp_path / "a")
    assert "CONTAMINATED" not in clean.read_text(encoding="utf-8")

    dirty = write_backtest_report(
        {"baseline": bt_result("baseline"), "llm": bt_result("llm", contaminated=True)},
        PERIODS, tmp_path / "b")
    text = dirty.read_text(encoding="utf-8")
    lines = text.splitlines()
    banner_idx = next(i for i, ln in enumerate(lines) if BANNER in ln)
    assert banner_idx < 3
    assert text.count("| llm (CONTAMINATED) |") == 3  # train, test, full
    assert "baseline (CONTAMINATED)" not in text


def test_llm_spend_and_net_equity_inr(tmp_path):
    llm = bt_result("llm", contaminated=True)
    out = write_backtest_report({"llm": llm}, PERIODS, tmp_path, llm_spend_usd=2.5, usd_inr=84.0)
    text = out.read_text(encoding="utf-8")
    assert "LLM API spend: $2.50 (₹210)" in text
    net = float(llm.equity.iloc[-1]) - 2.5 * 84.0
    assert f"₹{net:,.0f}" in text


def test_halted_and_notes_listed(tmp_path):
    r = bt_result("baseline", halted="KILL-SWITCH 2024-01-20: drawdown 10.5% >= 10%")
    out = write_backtest_report({"baseline": r}, PERIODS, tmp_path, notes=["Universe: A.NS, B.NS"])
    text = out.read_text(encoding="utf-8")
    assert "KILL-SWITCH 2024-01-20: drawdown 10.5% >= 10%" in text
    assert "- Universe: A.NS, B.NS" in text


# ---------- daily report ----------

def daily_fixture() -> DailyResult:
    book = Book(
        "baseline", cash=50000.0,
        positions={"ABC.NS": Position("ABC.NS", 10, 100.0, 95.0, "2024-01-03", "baseline")},
        pending=[Order("XYZ.NS", "BUY", 5, "BUY: test reason", "baseline", "2024-01-05", stop=180.0)],
        last_prices={"ABC.NS": 110.0},
        last_date="2024-01-05",
        trades=[row("2024-01-03", "ABC.NS", "BUY", 10, 100.0, 4.0, "baseline"),
                row("2024-01-04", "ABC.NS", "BUY", 10, 100.0, 4.0, "baseline", reason="entry day")],
        equity_curve=[("2024-01-04", 100000.0), ("2024-01-05", 101000.0)],
    )
    llm = Book("llm", cash=100000.0, equity_curve=[("2024-01-05", 100000.0)])
    signals = [
        {"date": "2024-01-05", "book": "baseline", "symbol": "XYZ.NS", "action": "BUY", "confidence": 0.6,
         "stop": 180.0, "contaminated": False, "reason": "BUY: test reason"},
        {"date": "2024-01-05", "book": "llm", "symbol": "ABC.NS", "action": "HOLD", "confidence": 0.0,
         "stop": None, "contaminated": True, "reason": "HOLD: model view"},
        {"date": "2024-01-04", "book": "baseline", "symbol": "OLD.NS", "action": "SELL", "confidence": 0.6,
         "stop": None, "contaminated": False, "reason": "old day signal"},
    ]
    return DailyResult(processed_days=["2024-01-04", "2024-01-05"], message="ok",
                       books={"baseline": book, "llm": llm}, signals=signals)


def test_daily_none_when_nothing_processed(tmp_path):
    assert write_daily_report(DailyResult(), tmp_path) is None


def test_daily_report_contents(tmp_path):
    out = write_daily_report(daily_fixture(), tmp_path)
    assert out == tmp_path / "daily" / "2024-01-05.md" and out.exists()
    text = out.read_text(encoding="utf-8")
    assert "PENDING" in text and "not filled" in text
    assert "XYZ.NS" in text and "180" in text
    assert "₹100.00" in text  # unrealised (110-100)*10
    assert "old day signal" not in text  # signals only for the report date
    assert "[CONTAMINATED] HOLD: model view" in text
    assert "BUY: test reason" in text
    assert "entry day" in text  # fill on processed day 2024-01-04
    assert "| 2024-01-03 |" not in text  # fill before the processed days is excluded
    assert "Not halted" in text


def test_daily_halted_status(tmp_path):
    res = daily_fixture()
    res.books["baseline"].halted = True
    res.books["baseline"].halt_reason = "KILL-SWITCH 2024-01-05: drawdown 12.0% >= 10%"
    text = write_daily_report(res, tmp_path).read_text(encoding="utf-8")
    assert "KILL-SWITCH 2024-01-05: drawdown 12.0% >= 10%" in text


def test_daily_date_str_override(tmp_path):
    out = write_daily_report(daily_fixture(), tmp_path, date_str="2024-01-04")
    assert out == tmp_path / "daily" / "2024-01-04.md"


# ---------- weekly report ----------

def write_state(state, n: int, llm_end: float = 112000.0):
    state.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    eq = pd.DataFrame({
        "baseline": np.linspace(100000, 104000, n),
        "llm": np.linspace(100000, llm_end, n),
        "buyhold_niftybees": np.linspace(100000, 101000, n),
    }, index=idx)
    eq.to_csv(state / "equity.csv")
    trades = trade_rows([
        row("2024-01-03", "A.NS", "BUY", 10, 100.0, 5.0, "baseline"),
        row("2024-01-10", "A.NS", "SELL", 10, 110.0, 6.0, "baseline"),
        row("2024-01-04", "A.NS", "BUY", 10, 100.0, 5.0, "llm", source="llm"),
        row("2024-01-12", "A.NS", "SELL", 10, 95.0, 6.0, "llm", source="llm"),
    ])
    trades.to_csv(state / "trades.csv", index=False)
    return eq


def test_weekly_files_and_insufficient_message(tmp_path):
    eq = write_state(tmp_path / "state", 5)
    out = write_weekly_report(tmp_path / "state", tmp_path / "out", llm_spend_usd=0.0, usd_inr=84.0)
    last = eq.index[-1].strftime("%Y-%m-%d")
    assert out == tmp_path / "out" / "weekly" / f"weekly_{last}.md"
    text = out.read_text(encoding="utf-8")
    assert "Insufficient data (5 sessions < 20): metrics are not meaningful yet" in text
    assert text.index("Insufficient data") < text.index(WEEKLY_DISCLAIMER)
    assert WEEKLY_DISCLAIMER in text
    assert "(CONTAMINATED)" not in text


def test_weekly_sufficient_has_no_warning_and_matches_summarize(tmp_path):
    eq = write_state(tmp_path / "state", 25)
    out = write_weekly_report(tmp_path / "state", tmp_path / "out", llm_spend_usd=0.0, usd_inr=84.0)
    text = out.read_text(encoding="utf-8")
    assert "Insufficient data" not in text
    assert WEEKLY_DISCLAIMER in text
    trades = pd.read_csv(tmp_path / "state" / "trades.csv")
    s = summarize(eq["llm"], trades[trades["book"] == "llm"])
    assert f"{s['total_return'] * 100:.2f}%" in text
    assert f"{s['max_drawdown'] * 100:.2f}%" in text
    assert f"{s['win_rate'] * 100:.2f}%" in text
    assert f"{s['sharpe']:.2f}" in text
    assert f"₹{s['total_costs']:,.0f}" in text
    assert "buyhold_niftybees" in text


def test_weekly_ai_net_of_api_spend(tmp_path):
    write_state(tmp_path / "state", 25, llm_end=112000.0)
    out = write_weekly_report(tmp_path / "state", tmp_path / "out", llm_spend_usd=0.5, usd_inr=84.0)
    text = out.read_text(encoding="utf-8")
    spend_inr = 0.5 * 84.0
    net = 112000.0 - spend_inr
    ret = net / 100000.0 - 1.0
    assert "LLM API spend: $0.50 (₹42)" in text
    assert f"₹{net:,.0f}" in text
    assert f"{ret * 100:.2f}%" in text


def test_weekly_asof_overrides_filename(tmp_path):
    write_state(tmp_path / "state", 25)
    out = write_weekly_report(tmp_path / "state", tmp_path / "out", llm_spend_usd=0.0, usd_inr=84.0,
                              asof="2024-01-10")
    assert out.name == "weekly_2024-01-10.md"
