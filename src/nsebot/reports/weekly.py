"""Weekly Markdown report built only from state/equity.csv and state/trades.csv."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nsebot.metrics import summarize
from nsebot.paper.runner import TRADE_COLS

BOOKS = ["baseline", "llm", "buyhold_niftybees"]
DISCLAIMER = "Only forward paper-trading results count as evidence. Past performance does not guarantee future results."


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _inr(x: float) -> str:
    return f"₹{x:,.0f}"


def _read_equity(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col="date", parse_dates=True)


def _read_trades(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=TRADE_COLS)
    return pd.read_csv(path)


def write_weekly_report(state_dir: str | Path, out_dir: str | Path, llm_spend_usd: float, usd_inr: float,
                        min_sessions: int = 20, asof: str | None = None) -> Path:
    state = Path(state_dir)
    eq = _read_equity(state / "equity.csv")
    trades = _read_trades(state / "trades.csv")
    if len(eq):
        eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    if asof is None:
        asof = eq.index.max().strftime("%Y-%m-%d") if len(eq) else date.today().isoformat()
    cutoff = pd.Timestamp(asof)
    eq = eq[eq.index <= cutoff]
    if len(trades):
        trades = trades[pd.to_datetime(trades["date"]) <= cutoff]
    n_sessions = len(eq)

    lines = [f"# Weekly paper-trading report (as of {asof})", ""]
    if n_sessions < min_sessions:
        lines += [(f"> **Insufficient data ({n_sessions} sessions < {min_sessions}): metrics are not meaningful "
                   "yet.** Raw numbers are shown below anyway."), ""]
    if n_sessions:
        lines += [(f"Sessions: {n_sessions} ({eq.index[0].strftime('%Y-%m-%d')} to "
                   f"{eq.index[-1].strftime('%Y-%m-%d')})."), ""]
    else:
        lines += ["No equity data found.", ""]

    lines += ["| Book | Return | Sharpe | Max DD | Win rate | Total costs (INR) | Trades | Final equity (INR) |",
              "|---|---|---|---|---|---|---|---|"]
    book_stats: dict[str, dict] = {}
    for name in BOOKS:
        if name not in eq.columns:
            continue
        series = eq[name].dropna()
        if series.empty:
            continue
        tr = trades[trades["book"] == name] if len(trades) else trades
        s = summarize(series, tr)
        contaminated = bool(len(tr)) and "contaminated" in tr.columns and (
            tr["contaminated"].astype(str).str.lower().eq("true").any())
        label = f"{name} (CONTAMINATED)" if contaminated else name
        book_stats[name] = {"series": series, "summary": s}
        lines.append(f"| {label} | {_pct(s['total_return'])} | {s['sharpe']:.2f} | {_pct(s['max_drawdown'])} | "
                     f"{_pct(s['win_rate'])} | {_inr(s['total_costs'])} | {s['n_trades']} | "
                     f"{_inr(float(series.iloc[-1]))} |")
    lines.append("")

    spend_inr = llm_spend_usd * usd_inr
    lines += [f"LLM API spend: ${llm_spend_usd:,.2f} (₹{spend_inr:,.0f})", ""]
    if "llm" in book_stats:
        series = book_stats["llm"]["series"]
        start, final = float(series.iloc[0]), float(series.iloc[-1])
        net = final - spend_inr
        ret = net / start - 1.0 if start > 0 else 0.0
        lines += [(f"AI net of API spend: {_inr(net)} (final equity {_inr(final)} minus API spend "
                   f"{_inr(spend_inr)}); return on that {_pct(ret)} from starting equity {_inr(start)}."), ""]

    lines += [f"_{DISCLAIMER}_", ""]

    out = Path(out_dir) / "weekly"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"weekly_{asof}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
