"""Shared signal type. Every signal carries a plain-English reason."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class Signal:
    date: pd.Timestamp          # the bar close the decision was made on (day t)
    symbol: str
    action: Action
    confidence: float           # 0..1
    reason: str
    source: str                 # "baseline" | "llm"
    stop: float | None = None   # suggested stop for BUY
    contaminated: bool = False  # True for LLM decisions on past dates

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = pd.Timestamp(self.date).strftime("%Y-%m-%d")
        return d
