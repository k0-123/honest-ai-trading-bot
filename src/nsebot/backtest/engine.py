"""Event-driven backtest: loops trading days through the same Simulator the paper runner uses."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nsebot.config import Config
from nsebot.metrics import summarize
from nsebot.portfolio import Book
from nsebot.sim import Simulator, Strategy
from nsebot.strategies.base import Signal


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    signals: list[Signal] = field(default_factory=list)
    halted: str = ""
    contaminated: bool = False

    def summary(self) -> dict:
        return summarize(self.equity, self.trades if len(self.trades) else None)


def run_backtest(prices: dict[str, pd.DataFrame], strategy: Strategy | None, cfg: Config, start: str,
                 end: str | None = None, name: str = "baseline", buy_and_hold: bool = False) -> BacktestResult:
    """prices: full bar history per symbol (warm-up before `start` is used for indicators only)."""
    symbols = list(prices)
    all_days = sorted(set().union(*[df.index for df in prices.values()]))
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end else all_days[-1]
    days = [d for d in all_days if s <= d <= e]
    book = Book(name, cash=cfg.capital)
    sim = Simulator(book, cfg, strategy, symbols, buy_and_hold=buy_and_hold)
    for d in days:
        today = {sym: df.loc[d] for sym, df in prices.items() if d in df.index}
        sim.on_open(d, today)
        sim.on_close(d, {sym: df.loc[:d] for sym, df in prices.items()})
    eq = pd.Series({pd.Timestamp(k): v for k, v in book.equity_curve}, name=name, dtype=float)
    trades = pd.DataFrame(book.trades, columns=["date", "symbol", "side", "qty", "price", "costs", "reason",
                                                 "signal_source", "contaminated", "book"])
    contaminated = any(sg.contaminated for sg in sim.signals)
    return BacktestResult(name, eq, trades, sim.signals, book.halt_reason, contaminated)
