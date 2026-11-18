"""Daily paper-trading run (after NSE close). No broker connection exists anywhere in this project.

Books (each starts with cfg.capital virtual INR):
  baseline           rule-based strategy
  llm                TradingAgents (only if enabled and the API key env var is set)
  buyhold_niftybees  passive benchmark
Each run processes every trading day after the last processed day, up to the effective date, using the
exact same Simulator as the backtest. Orders decided at a close fill at the NEXT day's open.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nsebot.config import Config
from nsebot.portfolio import Book
from nsebot.sim import Simulator, Strategy

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
DATA_READY_IST = time(15, 45)  # before this, today's daily bar is incomplete
TRADE_COLS = ["date", "symbol", "side", "qty", "price", "costs", "reason", "signal_source", "contaminated", "book"]
BENCH_BOOK = "buyhold_niftybees"


def holiday_warning(cfg: Config, now: datetime | None = None) -> str | None:
    """Warn when config.nse_holidays has no entry for the current year (v2b month-end detection and holiday
    skipping then rely on the data feed alone)."""
    year = str((now or datetime.now(IST)).year)
    if any(h.startswith(year) for h in cfg.nse_holidays):
        return None
    return (f"WARNING: config.yaml nse_holidays has no {year} dates. Add NSE's official {year} trading-holiday list "
            "(see data/nse_holidays/ and README).")


def effective_date(now: datetime | None = None) -> pd.Timestamp:
    """Last date whose daily bar is complete."""
    now = (now or datetime.now(IST)).astimezone(IST)
    d = pd.Timestamp(now.date())
    return d if now.time() >= DATA_READY_IST else d - pd.Timedelta(days=1)


@dataclass
class DailyResult:
    processed_days: list[str] = field(default_factory=list)
    message: str = ""
    books: dict[str, Book] = field(default_factory=dict)
    signals: list[dict] = field(default_factory=list)


class PaperRunner:
    def __init__(self, cfg: Config, loader, strategies: dict[str, Strategy], state_dir: str | Path | None = None,
                 universe=None, universe_loader=None, universe_books: dict | None = None):
        """universe_books: {name: factory(book, calendar_days) -> Simulator} for books trading the point-in-time
        Nifty 50 (v2a, v2b). Their prices come from universe_loader (documented corrections applied)."""
        self.cfg, self.loader, self.strategies = cfg, loader, strategies
        self.universe, self.universe_loader = universe, universe_loader
        self.universe_books = universe_books or {}
        self.state_dir = Path(state_dir or cfg.paths.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ---------- state ----------
    @property
    def portfolio_path(self) -> Path:
        return self.state_dir / "portfolio.json"

    def load_books(self) -> dict[str, Book]:
        if self.portfolio_path.exists():
            raw = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
            return {k: Book.from_dict(v) for k, v in raw["books"].items()}
        return {}

    def _save(self, books: dict[str, Book]) -> None:
        tmp = self.portfolio_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"books": {k: b.to_dict() for k, b in books.items()},
                                   "saved_at_ist": datetime.now(IST).isoformat(timespec="seconds")},
                                  indent=2), encoding="utf-8")
        os.replace(tmp, self.portfolio_path)
        rows = [t for b in books.values() for t in b.trades]
        rows.sort(key=lambda t: (t["date"], t["book"]))
        with open(self.state_dir / "trades.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_COLS)
            w.writeheader()
            w.writerows(rows)
        eq = pd.DataFrame({k: pd.Series(dict(b.equity_curve), dtype=float) for k, b in books.items()})
        eq.index.name = "date"
        eq.sort_index().to_csv(self.state_dir / "equity.csv")
        for name, b in books.items():
            flag = self.state_dir / f"HALTED_{name}"
            if b.halted:
                flag.write_text(b.halt_reason + "\n", encoding="utf-8")

    def _append_signals(self, rows: list[dict]) -> None:
        if not rows:
            return
        p = self.state_dir / "signals.csv"
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["date", "book", "symbol", "action", "confidence", "stop",
                                              "contaminated", "reason"])
            if new:
                w.writeheader()
            w.writerows(rows)

    # ---------- run ----------
    def run(self, now: datetime | None = None) -> DailyResult:
        res = DailyResult()
        upto = effective_date(now)
        syms, bench = self.cfg.symbols, self.cfg.benchmark
        prices = {s: self.loader.get(s, self.cfg.data.start, as_of=upto, refresh=True) for s in syms + [bench]}
        prices = {s: df for s, df in prices.items() if len(df)}
        raw = {s: self.loader.get(s, self.cfg.data.start, as_of=upto, clean=False) for s in prices}
        if not prices:
            res.message = "No price data available; nothing done."
            return res
        books = self.load_books()
        for name in list(self.strategies) + [BENCH_BOOK] + list(self.universe_books):
            books.setdefault(name, Book(name, cash=self.cfg.capital))
        last_done = max((pd.Timestamp(b.last_date) for b in books.values() if b.last_date), default=None)
        days = sorted(set().union(*[df.index for s, df in raw.items() if s in syms]))
        holidays = {pd.Timestamp(h) for h in self.cfg.nse_holidays}
        # an NSE holiday is never a session, even if the feed shows a (phantom, zero-volume) bar for it
        days = [d for d in days if d <= upto and d not in holidays and (last_done is None or d > last_done)]
        if last_done is None:
            days = days[-1:]  # first ever run: start from the latest completed session only
        if not days:
            res.message = (f"Nothing to do: no new completed NSE session up to {upto.date()} "
                           f"(holiday/weekend, data not published yet, or already processed).")
            res.books = books
            return res
        sims = {name: Simulator(books[name], self.cfg, strat, syms) for name, strat in self.strategies.items()}
        sims[BENCH_BOOK] = Simulator(books[BENCH_BOOK], self.cfg, None, [bench], buy_and_hold=True)
        usims, uprices, uraw = {}, {}, {}
        if self.universe_books:
            calendar = sorted(set().union(*[df.index for s, df in raw.items() if s in syms]))  # sessions <= upto only
            held = set().union(*[set(books[n].positions) for n in self.universe_books])
            keys = (set().union(*[set(self.universe.members_on(d)) for d in days]) | held | {bench})
            ustart = (days[0] - pd.Timedelta(days=600)).strftime("%Y-%m-%d")
            for k in sorted(keys):
                df = self.universe_loader.get(k, ustart, as_of=upto, refresh=True)
                if len(df):
                    uprices[k] = df
                    uraw[k] = self.universe_loader.get(k, ustart, as_of=upto, clean=False)
            usims = {n: f(books[n], calendar) for n, f in self.universe_books.items()}
        for d in days:
            today = {s: df.loc[d] for s, df in raw.items() if d in df.index}  # raw: fills + stops
            hist = {s: df.loc[:d] for s, df in prices.items()}
            for name, sim in sims.items():
                if sim.book.last_date is None:
                    sim._today_raw = today        # first session: the bot did not exist at this open -> no fills
                else:
                    sim.on_open(d, today)
                for sg in sim.on_close(d, hist):
                    res.signals.append({"date": d.strftime("%Y-%m-%d"), "book": name, "symbol": sg.symbol,
                                        "action": sg.action, "confidence": round(sg.confidence, 3),
                                        "stop": None if sg.stop is None else round(sg.stop, 2),
                                        "contaminated": sg.contaminated, "reason": sg.reason})
            if usims:
                utoday = {k: df.loc[d] for k, df in uraw.items() if d in df.index}
                uhist = {k: df.loc[:d] for k, df in uprices.items()}
                for name, sim in usims.items():
                    if sim.book.last_date is None:
                        sim._today_raw = utoday   # first session: the bot did not exist at this open -> no fills
                    else:
                        sim.on_open(d, utoday)
                    for sg in sim.on_close(d, uhist):
                        res.signals.append({"date": d.strftime("%Y-%m-%d"), "book": name, "symbol": sg.symbol,
                                            "action": sg.action, "confidence": round(sg.confidence, 3),
                                            "stop": None if sg.stop is None else round(sg.stop, 2),
                                            "contaminated": sg.contaminated, "reason": sg.reason})
            res.processed_days.append(d.strftime("%Y-%m-%d"))
        self._save(books)
        self._append_signals(res.signals)
        res.books = books
        res.message = f"Processed {len(days)} session(s): {', '.join(res.processed_days)}"
        return res
