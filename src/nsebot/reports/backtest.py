"""Markdown + PNG report for backtest results. Every number comes from metrics.summarize on the inputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from nsebot.backtest.engine import BacktestResult
from nsebot.metrics import summarize

CONTAM_BANNER = ("⚠️ CONTAMINATED: LLM results on historical dates may reflect the model's knowledge of what "
                 "happened. NOT evidence of skill.")
TABLE_HEADER = ("| Strategy | Total return | Days invested | Trades | Costs (INR) | Max DD | CAGR | Sharpe | "
                "Sortino | VaR95 | CVaR95 | Win rate |")
TABLE_SEP = "|" + "---|" * 12


def _label(name: str, contaminated: bool) -> str:
    return f"{name} (CONTAMINATED)" if contaminated else name


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _inr(x: float) -> str:
    return f"₹{x:,.0f}"


def _slice(result: BacktestResult, period: tuple[str, str] | None) -> tuple[pd.Series, pd.DataFrame]:
    eq, tr = result.equity, result.trades
    if period is None:
        return eq, tr
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    eq_s = eq.loc[start:end]
    dates = pd.to_datetime(tr["date"]) if len(tr) else pd.Series(dtype="datetime64[ns]")
    tr_s = tr[(dates >= start) & (dates <= end)] if len(tr) else tr
    return eq_s, tr_s


def _rebased(eq: pd.Series) -> pd.Series:
    if len(eq) == 0 or eq.iloc[0] == 0:
        return eq
    return eq / eq.iloc[0] * 100.0


def _period_table(results: dict[str, BacktestResult], title: str, period: tuple[str, str] | None) -> list[str]:
    lines = [f"## {title}", ""]
    lines += [TABLE_HEADER, TABLE_SEP]
    for name, r in results.items():
        eq, tr = _slice(r, period)
        # Metrics are scale-invariant, so they are computed on the sliced (not rebased) equity.
        s = summarize(eq, tr)
        inv = r.invested.loc[eq.index.min():eq.index.max()] if len(eq) and len(r.invested) else r.invested
        days_inv = float((inv > 0).mean()) if len(inv) else 0.0
        lines.append(
            f"| {_label(name, r.contaminated)} | {_pct(s['total_return'])} | {_pct(days_inv)} | {s['n_trades']} | "
            f"{_inr(s['total_costs'])} | {_pct(s['max_drawdown'])} | {_pct(s['cagr'])} | {s['sharpe']:.2f} | "
            f"{s['sortino']:.2f} | {_pct(s['var95'])} | {_pct(s['cvar95'])} | {_pct(s['win_rate'])} |"
        )
    lines.append("")
    return lines


def _plot(results: dict[str, BacktestResult], path: Path) -> None:
    fig = Figure(figsize=(10, 5.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    for name, r in results.items():
        if len(r.equity) == 0:
            continue
        ax.plot(r.equity.index, _rebased(r.equity).to_numpy(), label=_label(name, r.contaminated))
    ax.axhline(100.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title("Equity curves (rebased to 100, net of costs)")
    ax.set_ylabel("Rebased equity")
    ax.grid(alpha=0.3)
    if ax.get_lines():
        ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def write_backtest_report(results: dict[str, BacktestResult], periods: dict[str, tuple[str, str]],
                          out_dir: str | Path, llm_spend_usd: float = 0.0, usd_inr: float = 84.0,
                          notes: list[str] | None = None,
                          period_results: dict[str, dict[str, BacktestResult]] | None = None) -> Path:
    """period_results: optional independent runs per period (fresh capital each). When given for a period,
    that table uses them instead of slicing the continuous run, so a halt in one period cannot hide others."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    any_contaminated = any(r.contaminated for r in results.values())

    lines: list[str] = []
    if any_contaminated:
        lines += [f"## {CONTAM_BANNER}", ""]
    lines += ["# Backtest report", "", (
        "Equity and returns are net of simulated trading costs (brokerage, STT, exchange, SEBI fee, GST, "
        "stamp duty, DP charge) and slippage. Costs are in INR."), ""]

    periods = dict(periods)
    period_results = period_results or {}
    for title, period in periods.items():
        if title in period_results:
            lines += _period_table(period_results[title],
                                   f"{title} ({period[0]} to {period[1]}), independent run, fresh capital", None)
        else:
            lines += _period_table(results, f"{title} ({period[0]} to {period[1]})", period)
    if "full" not in periods:
        lines += _period_table(results, "full (all dates)", None)

    llm_results = {n: r for n, r in results.items() if r.contaminated or n == "llm"}
    if llm_results:
        spend_inr = llm_spend_usd * usd_inr
        lines += ["## LLM API spend", "",
                  f"LLM API spend: ${llm_spend_usd:,.2f} (₹{spend_inr:,.0f})", ""]
        for name, r in llm_results.items():
            if len(r.equity) == 0:
                continue
            final = float(r.equity.iloc[-1])
            net = final - spend_inr
            lines.append(f"- {_label(name, r.contaminated)}: final equity {_inr(final)}; "
                         f"net of API spend {_inr(net)}")
        lines.append("")

    halt_lines = [f"- {_label(n, r.contaminated)} (full run): {r.halted}" for n, r in results.items() if r.halted]
    for title, rs in period_results.items():
        halt_lines += [f"- {_label(n, r.contaminated)} ({title} run): {r.halted}" for n, r in rs.items() if r.halted]
    if halt_lines:
        lines += ["## Halts (kill-switch: drawdown >= limit, book flattened and stopped)", "", *halt_lines, ""]

    lines += ["## Equity curves", "", "![Equity curves rebased to 100](equity_curves.png)", ""]

    if notes:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in notes]
        lines.append("")

    report = out / "backtest_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    _plot(results, out / "equity_curves.png")
    return report
