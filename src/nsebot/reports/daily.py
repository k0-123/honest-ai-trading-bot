"""Per-day Markdown report for the paper runner. Only logged state is reported; nothing is estimated."""
from __future__ import annotations

from pathlib import Path

from nsebot.paper.runner import DailyResult

CONTAM_PREFIX = "[CONTAMINATED] "


def _inr(x: float, dp: int = 2) -> str:
    return f"₹{x:,.{dp}f}"


def _cell(x) -> str:
    return "" if x is None else str(x).replace("|", "\\|")


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out + [""]


def _book_section(name: str, book, day: str, processed: set[str]) -> list[str]:
    lines = [f"## Book: {name}", ""]
    eq_by_day = dict(book.equity_curve)
    if day in eq_by_day:
        lines.append(f"- Equity at close {day}: {_inr(eq_by_day[day])}")
    else:
        lines.append(f"- Equity at close {day}: not recorded for this day")
    lines.append(f"- Cash: {_inr(book.cash)}")
    if book.halted:
        lines.append(f"- **HALTED**: {book.halt_reason}")
    else:
        lines.append("- Not halted.")
    if book.block_new_entries:
        lines.append("- New entries are blocked.")
    lines.append("")

    lines += ["### Open positions", ""]
    if book.positions:
        rows = []
        for sym, p in sorted(book.positions.items()):
            last = book.last_prices.get(sym, p.entry_price)
            pnl = (last - p.entry_price) * p.qty
            rows.append([sym, p.qty, _inr(p.entry_price), "" if p.stop is None else _inr(p.stop),
                         _inr(last), _inr(pnl)])
        lines += _table(["Symbol", "Qty", "Entry", "Stop", "Last price",
                         "Unrealised P&L (before exit costs)"], rows)
    else:
        lines += ["None.", ""]

    lines += ["### Pending orders for next open (PENDING, not filled)", ""]
    if book.pending:
        rows = [[o.side, o.symbol, o.qty, "" if o.stop is None else _inr(o.stop), o.created, o.source,
                 "yes" if o.contaminated else "no", o.reason] for o in book.pending]
        lines += _table(["Side", "Symbol", "Qty", "Stop", "Decided at close", "Source", "Contaminated",
                         "Reason"], rows)
    else:
        lines += ["None.", ""]

    lines += [f"### Fills on processed days ({', '.join(sorted(processed))})", ""]
    fills = [t for t in book.trades if t["date"] in processed]
    if fills:
        rows = [[t["date"], t["side"], t["symbol"], t["qty"], _inr(t["price"]), _inr(t["costs"]),
                 t["reason"]] for t in fills]
        lines += _table(["Date", "Side", "Symbol", "Qty", "Price", "Costs", "Reason"], rows)
    else:
        lines += ["None.", ""]
    return lines


def write_daily_report(result: DailyResult, out_dir: str | Path, date_str: str | None = None) -> Path | None:
    if not result.processed_days:
        return None
    day = date_str or result.processed_days[-1]
    processed = set(result.processed_days)

    lines = [f"# Paper trading report: {day}", "",
             f"Run message: {result.message}", "",
             ("Orders decided at a close fill at the next session's open. Anything marked PENDING has not "
              "been filled."), ""]
    for name, book in result.books.items():
        lines += _book_section(name, book, day, processed)

    lines += [f"## Signals for {day}", ""]
    sigs = [s for s in result.signals if s["date"] == day]
    if sigs:
        rows = []
        for s in sigs:
            reason = s["reason"]
            if s.get("contaminated") and not reason.startswith(CONTAM_PREFIX.strip()):
                reason = CONTAM_PREFIX + reason
            stop = "" if s.get("stop") is None else _inr(s["stop"])
            rows.append([s["book"], s["symbol"], s["action"], f"{s['confidence']:.2f}", stop, reason])
        lines += _table(["Book", "Symbol", "Action", "Confidence", "Stop", "Reason"], rows)
    else:
        lines += ["No signals for this date.", ""]

    out = Path(out_dir) / "daily"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{day}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
