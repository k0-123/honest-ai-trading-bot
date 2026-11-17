"""Virtual book state. Serialisable to JSON so the paper runner survives restarts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    stop: float | None
    entry_date: str
    source: str


@dataclass
class Order:
    symbol: str
    side: str            # BUY | SELL
    qty: int
    reason: str
    source: str
    created: str         # date of the close the decision was made on
    stop: float | None = None
    contaminated: bool = False


@dataclass
class Book:
    name: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    last_prices: dict[str, float] = field(default_factory=dict)
    peak_equity: float = 0.0
    last_equity: float = 0.0
    last_date: str | None = None
    halted: bool = False
    halt_reason: str = ""
    block_new_entries: bool = False
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)

    def equity(self) -> float:
        return self.cash + sum(p.qty * self.last_prices.get(s, p.entry_price) for s, p in self.positions.items())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Book:
        d = dict(d)
        d["positions"] = {k: Position(**v) for k, v in d.get("positions", {}).items()}
        d["pending"] = [Order(**o) for o in d.get("pending", [])]
        d["equity_curve"] = [tuple(x) for x in d.get("equity_curve", [])]
        return cls(**d)
