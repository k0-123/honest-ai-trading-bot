"""Replay the normal daily paper run for each trading day in a date range, one day at a time.

- Each day calls the real PaperRunner with the clock set to 16:30 IST that day, so the runner's own
  as_of cut-off means it only sees data up to and including that day.
- Prices are downloaded ONCE up front; daily steps never refresh (no network inside the loop).
- Writes to a separate folder (default ./replay) and refuses to touch the real paper state.
- Baseline + NIFTYBEES books only: no LLM, no API cost.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from nsebot.config import Config
from nsebot.paper.runner import IST, PaperRunner
from nsebot.strategies.baseline import BaselineStrategy


class _NoRefresh:
    """Loader wrapper that forbids re-downloading inside the replay loop."""

    def __init__(self, loader):
        self.loader = loader

    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
        return self.loader.get(symbol, start, end, as_of=as_of, refresh=False, clean=clean)


@dataclass
class ReplayResult:
    days: list[str]
    trades: pd.DataFrame        # date, book, action, stock, qty, price
    final: pd.DataFrame         # book, cash, holdings, value, return
    pending: list[dict]
    out_dir: Path


def _check_separate(out: Path, real_state: Path) -> None:
    o, r = out.resolve(), real_state.resolve()
    if o == r or r in o.parents or o in r.parents:
        raise ValueError(f"replay folder {o} overlaps the real paper state {r}; choose another --out")


def replay(cfg: Config, loader, start: str, end: str | None = None, out_dir: str | Path = "replay",
           refresh_data: bool = True, report_fn=None) -> ReplayResult:
    out = Path(out_dir)
    _check_separate(out, Path(cfg.paths.state_dir))
    state, reports = out / "state", out / "reports"
    for d in (state, reports):  # always start from fresh books, inside the replay folder only
        if d.exists():
            shutil.rmtree(d)
    rcfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(
        update={"state_dir": str(state), "reports_dir": str(reports)})})

    syms = cfg.symbols
    if refresh_data:
        for s in syms + [cfg.benchmark]:
            loader.get(s, cfg.data.start, refresh=True)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp(datetime.now(IST).date())
    all_days = sorted(set().union(*[loader.get(s, cfg.data.start, clean=False).index for s in syms]))
    days = [d for d in all_days if pd.Timestamp(start) <= d <= end_ts]

    runner = PaperRunner(rcfg, _NoRefresh(loader),
                         {"baseline": BaselineStrategy(cfg.baseline, cfg.risk.atr_stop_mult)}, state)
    done: list[str] = []
    for d in days:
        res = runner.run(now=datetime.combine(d.date(), time(16, 30), tzinfo=IST))
        done += res.processed_days
        if report_fn is not None:
            report_fn(res, reports)

    books = runner.load_books()
    rows = [{"date": t["date"], "book": name, "action": t["side"], "stock": t["symbol"].removesuffix(".NS"),
             "qty": t["qty"], "price": t["price"]} for name, b in books.items() for t in b.trades]
    trades = pd.DataFrame(rows, columns=["date", "book", "action", "stock", "qty", "price"])
    trades = trades.sort_values(["date", "book"], kind="stable").reset_index(drop=True)
    final = pd.DataFrame([{"book": n, "cash": round(b.cash, 2), "holdings": round(b.equity() - b.cash, 2),
                           "value": round(b.equity(), 2), "return": b.equity() / cfg.capital - 1,
                           "as_of": b.last_date} for n, b in books.items()])
    pending = [{"book": n, "action": o.side, "stock": o.symbol, "qty": o.qty, "decided": o.created}
               for n, b in books.items() for o in b.pending]
    return ReplayResult(done, trades, final, pending, out)


def render(r: ReplayResult, capital: float) -> str:
    lines = [(f"# Paper replay {r.days[0] if r.days else '-'} to {r.days[-1] if r.days else '-'} "
             f"({len(r.days)} sessions)"), "",
             "Each day used only data up to that day. Fills are simulated at the next session's open.", "",
             "## Trades", "", "| Date | Book | Action | Stock | Qty | Price (₹) |", "|---|---|---|---|---|---|"]
    if len(r.trades):
        lines += [f"| {t.date} | {t.book} | {t.action} | {t.stock} | {t.qty} | {t.price:,.2f} |"
                  for t in r.trades.itertuples()]
    else:
        lines.append("| - | - | no trades | - | - | - |")
    lines += ["", f"## Book values at the last close (start ₹{capital:,.0f} each)", "",
              "| Book | Cash (₹) | Holdings (₹) | Value (₹) | Return |", "|---|---|---|---|---|"]
    lines += [f"| {f.book} | {f.cash:,.2f} | {f.holdings:,.2f} | {f.value:,.2f} | {f._5:+.2%} |"
              for f in r.final.itertuples()]
    if r.pending:
        lines += ["", "Pending orders for the next open (NOT filled):", ""]
        lines += [f"- {p['book']}: {p['action']} {p['qty']} {p['stock']} (decided {p['decided']})" for p in r.pending]
    return "\n".join(lines) + "\n"
