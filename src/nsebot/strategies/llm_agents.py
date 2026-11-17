"""LLM strategy: wraps TauricResearch/TradingAgents (multi-agent LLM trading pipeline).

Pinned against the public clone at commit 9fa1196fffbcfd9de5146d2dd2869df84ab99838.
TradingAgents is imported lazily inside the default backend only, so importing this
module and running the tests needs neither the package nor any API key.

Cost tracking: token usage is read from a LangChain callback attached to the graph's
LLM clients. If langchain_core is unavailable, usage is counted as 0 and the budget
is still enforced using cfg.est_cost_per_call_usd per call (pessimistic); a warning is logged. TODO: verify usage capture against the real
Google provider before enabling live runs.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nsebot.config import LlmCfg
from nsebot.strategies.base import Action, Signal

logger = logging.getLogger(__name__)

CONTAMINATED_PREFIX = "[CONTAMINATED] "
_IST = timezone(timedelta(hours=5, minutes=30))
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Backend = Callable[[str, str], tuple[str, int, int]]


@dataclass
class LlmDecision:
    action: Action
    confidence: float
    raw: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    called_at_utc: str
    cached: bool
    cost_estimated: bool = False  # True when token usage was unavailable and a fallback estimate was charged


class BudgetExceeded(Exception):
    """Raised before calling the backend when the total LLM budget is spent."""


# ---------- decision parsing ----------

_PROPOSAL_RE = re.compile(r"FINAL\s+TRANSACTION\s+PROPOSAL:\s*\**\s*(BUY|SELL|HOLD)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\b(BUY|SELL|HOLD|OVERWEIGHT|UNDERWEIGHT)\b", re.IGNORECASE)
_RATING_TO_ACTION = {"BUY": "BUY", "OVERWEIGHT": "BUY", "SELL": "SELL", "UNDERWEIGHT": "SELL", "HOLD": "HOLD"}
_CONF_LABEL_RE = re.compile(r"confidence\W{0,10}?(\d+(?:\.\d+)?)\s*(%)?", re.IGNORECASE)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DEFAULT_CONF = {"BUY": 0.6, "SELL": 0.6, "HOLD": 0.0}


def _parse_confidence(text: str) -> float | None:
    m = _CONF_LABEL_RE.search(text)
    if m:
        value, is_pct = float(m.group(1)), bool(m.group(2))
    else:
        m = _PCT_RE.search(text)
        if not m:
            return None
        value, is_pct = float(m.group(1)), True
    if is_pct:
        value /= 100.0
    elif value > 1.0:
        if value > 100.0:
            return None
        value /= 100.0
    return value if 0.0 <= value <= 1.0 else None


def parse_decision(text: str) -> tuple[Action, float]:
    """Map a TradingAgents final text to (BUY|SELL|HOLD, confidence in 0..1).

    The "FINAL TRANSACTION PROPOSAL" line wins. Otherwise the last BUY/SELL/HOLD-type
    word (or Overweight/Underweight rating) is used. No recognisable action gives HOLD 0.0.
    """
    text = text or ""
    m = _PROPOSAL_RE.search(text)
    if m:
        action: str | None = m.group(1).upper()
    else:
        words = _WORD_RE.findall(text)
        action = _RATING_TO_ACTION[words[-1].upper()] if words else None
    if action is None:
        return "HOLD", 0.0
    confidence = _parse_confidence(text)
    if confidence is None:
        confidence = _DEFAULT_CONF[action]
    return action, confidence  # type: ignore[return-value]


# ---------- indicators (local until nsebot.indicators lands) ----------

def _atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---------- default backend (real TradingAgents) ----------

def _usage_from_llm_result(response: Any) -> tuple[int, int]:
    """Sum (input, output) tokens from a LangChain LLMResult-like object (duck-typed)."""
    tin = tout = 0
    found = False
    for gens in getattr(response, "generations", None) or []:
        for gen in gens:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            if meta:
                found = True
                tin += int(meta.get("input_tokens", 0) or 0)
                tout += int(meta.get("output_tokens", 0) or 0)
    if found:
        return tin, tout
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    tin = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    tout = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return tin, tout


def _make_usage_handler() -> Any | None:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        return None

    class _UsageHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__()
            self.lock = threading.Lock()
            self.tokens_in = 0
            self.tokens_out = 0

        def reset(self) -> None:
            with self.lock:
                self.tokens_in = 0
                self.tokens_out = 0

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            tin, tout = _usage_from_llm_result(response)
            with self.lock:
                self.tokens_in += tin
                self.tokens_out += tout

    return _UsageHandler()


def _make_default_backend(cfg: LlmCfg) -> Backend:
    """Return a backend that runs real TradingAgents. The graph is built once, on first call."""
    lock = threading.Lock()
    holder: dict[str, Any] = {"graph": None, "usage": None}

    def _build() -> None:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        state_root = Path(cfg.cache_dir) / "tradingagents"
        conf = dict(DEFAULT_CONFIG)
        conf.update(
            {
                "llm_provider": cfg.provider,
                "deep_think_llm": cfg.model,
                "quick_think_llm": cfg.model,
                "max_debate_rounds": cfg.max_debate_rounds,
                "max_risk_discuss_rounds": cfg.max_debate_rounds,
                "checkpoint_enabled": False,
                "results_dir": str(state_root / "logs"),
                "data_cache_dir": str(state_root / "data_cache"),
                "memory_log_path": str(state_root / "memory" / "trading_memory.md"),
            }
        )
        usage = _make_usage_handler()
        if usage is None:
            logger.warning(
                "LLM cost is UNTRACKED: langchain_core is not importable, so token usage is counted as 0 "
                "and the budget is not enforced. TODO: install langchain_core in the runtime environment."
            )
        holder["usage"] = usage
        holder["graph"] = TradingAgentsGraph(config=conf, callbacks=[usage] if usage is not None else None)

    def backend(symbol: str, date_str: str) -> tuple[str, int, int]:
        api_key_env = cfg.api_key_env
        if not os.environ.get(api_key_env):
            raise RuntimeError(f"LLM backend needs an API key: set the environment variable {api_key_env}")
        with lock:
            if holder["graph"] is None:
                _build()
            usage = holder["usage"]
            if usage is not None:
                usage.reset()
            final_state, _signal = holder["graph"].propagate(symbol, date_str)
            tin, tout = (usage.tokens_in, usage.tokens_out) if usage is not None else (0, 0)
        text = str((final_state or {}).get("final_trade_decision") or "")
        return text, tin, tout

    return backend


# ---------- strategy ----------

def _ist_today() -> date:
    return datetime.now(_IST).date()


def _summary(raw: str, limit: int = 240) -> str:
    collapsed = " ".join((raw or "").split())
    if not collapsed:
        return "no decision text returned"
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


class LlmAgentStrategy:
    source = "llm"

    def __init__(
        self,
        cfg: LlmCfg,
        atr_stop_mult: float = 2.0,
        backend: Backend | None = None,
        today: Callable[[], date] | None = None,
    ):
        self.cfg = cfg
        self.atr_stop_mult = atr_stop_mult
        self._backend: Backend = backend if backend is not None else _make_default_backend(cfg)
        self._today: Callable[[], date] = today or _ist_today
        self._cache_root = Path(cfg.cache_dir)
        self._lock = threading.Lock()
        self.spent_usd: float = self._load_spend()

    # -- persistence --

    def _spend_path(self) -> Path:
        return self._cache_root / "spend.json"

    def _load_spend(self) -> float:
        path = self._spend_path()
        if not path.exists():
            return 0.0
        try:
            return float(json.loads(path.read_text(encoding="utf-8"))["spent_usd"])
        except (ValueError, KeyError, TypeError) as exc:
            # Fail closed: an unreadable spend record must not silently reset the budget.
            raise RuntimeError(f"LLM spend file {path} is unreadable; fix or remove it to continue") from exc

    def _save_spend(self) -> None:
        path = self._spend_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"spent_usd": self.spent_usd}), encoding="utf-8")
        os.replace(tmp, path)

    def _cache_path(self, symbol: str, date_str: str) -> Path:
        if not symbol or symbol in (".", "..") or "/" in symbol or "\\" in symbol:
            raise ValueError(f"unsafe symbol for cache path: {symbol!r}")
        if not _DATE_RE.match(date_str):
            raise ValueError(f"date must be YYYY-MM-DD, got {date_str!r}")
        return self._cache_root / symbol / f"{date_str}.json"

    def _read_cache(self, symbol: str, date_str: str) -> LlmDecision | None:
        path = self._cache_path(symbol, date_str)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return replace(LlmDecision(**data), cached=True, cost_usd=0.0)
        except (ValueError, TypeError) as exc:
            logger.warning("ignoring unreadable LLM cache file %s (%s)", path, type(exc).__name__)
            return None

    def _write_cache(self, symbol: str, date_str: str, decision: LlmDecision) -> None:
        path = self._cache_path(symbol, date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        prices = self.cfg.price_per_mtok_usd
        return (tokens_in * prices.get("input", 0.0) + tokens_out * prices.get("output", 0.0)) / 1e6

    # -- public API --

    def decide(self, symbol: str, date_str: str) -> LlmDecision:
        """Cache first (free), then budget check, then the backend. Raises BudgetExceeded."""
        cached = self._read_cache(symbol, date_str)
        if cached is not None:
            return cached
        with self._lock:
            if self.spent_usd >= self.cfg.budget_usd_total:
                raise BudgetExceeded(
                    f"LLM budget exhausted: spent ${self.spent_usd:.4f} of ${self.cfg.budget_usd_total:.2f} total"
                )
        text, tokens_in, tokens_out = self._backend(symbol, date_str)
        action, confidence = parse_decision(text)
        cost = self._cost(tokens_in, tokens_out)
        estimated = tokens_in == 0 and tokens_out == 0
        if estimated:
            # Fail safe: never let missing usage data switch the budget off. Charge a pessimistic estimate.
            cost = self.cfg.est_cost_per_call_usd
            logger.warning("LLM token usage unavailable; charging estimated $%.3f for this call", cost)
        decision = LlmDecision(
            action=action,
            confidence=confidence,
            raw=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model=self.cfg.model,
            called_at_utc=datetime.now(UTC).isoformat(),
            cached=False,
            cost_estimated=estimated,
        )
        with self._lock:
            self.spent_usd += cost
            self._save_spend()
        self._write_cache(symbol, date_str, decision)
        return decision

    def generate(self, symbol: str, history: pd.DataFrame, in_position: bool) -> Signal:
        bar_date = pd.Timestamp(history.index[-1])
        date_str = bar_date.strftime("%Y-%m-%d")
        close = float(history["close"].iloc[-1])
        contaminated = bar_date.date() < self._today()

        def signal(action: Action, confidence: float, reason: str, stop: float | None = None) -> Signal:
            if contaminated:
                reason = CONTAMINATED_PREFIX + reason
            return Signal(
                date=bar_date,
                symbol=symbol,
                action=action,
                confidence=confidence,
                reason=reason,
                source=self.source,
                stop=stop,
                contaminated=contaminated,
            )

        try:
            decision = self.decide(symbol, date_str)
        except BudgetExceeded:
            return signal("HOLD", 0.0, "LLM budget exhausted")

        model = self.cfg.model
        summary = _summary(decision.raw)
        if decision.action == "BUY":
            if in_position:
                return signal("HOLD", 0.0, f"LLM said BUY but already in position ({model}): {summary}")
            atr = float(_atr(history, 14).iloc[-1])
            if not math.isfinite(atr) or atr <= 0:
                return signal("HOLD", 0.0, "BUY ignored: insufficient history for ATR(14)")
            stop = close - self.atr_stop_mult * atr
            reason = (
                f"LLM BUY ({model}): {summary}; stop {stop:.2f} "
                f"({self.atr_stop_mult:.1f}xATR {atr:.2f})"
            )
            return signal("BUY", decision.confidence, reason, stop=stop)
        if decision.action == "SELL":
            if not in_position:
                return signal("HOLD", 0.0, f"LLM said SELL but flat; nothing to sell ({model}): {summary}")
            return signal("SELL", decision.confidence, f"LLM SELL ({model}): {summary}")
        return signal("HOLD", decision.confidence, f"LLM HOLD ({model}): {summary}")
