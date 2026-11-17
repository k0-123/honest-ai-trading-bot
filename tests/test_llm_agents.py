"""Offline tests for the LLM (TradingAgents) strategy. No network, no keys, fake backend only."""
from __future__ import annotations

import json
import sys
from datetime import date
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nsebot.config import LlmCfg
from nsebot.strategies import llm_agents
from nsebot.strategies.llm_agents import (
    BudgetExceeded,
    LlmAgentStrategy,
    LlmDecision,
    parse_decision,
)

TODAY = date(2026, 11, 17)
BUY_TEXT = "Analysis done.\nFINAL TRANSACTION PROPOSAL: **BUY**"
SELL_TEXT = "Analysis done.\nFINAL TRANSACTION PROPOSAL: **SELL**"
HOLD_TEXT = "FINAL TRANSACTION PROPOSAL: **HOLD**"


class FakeBackend:
    def __init__(self, text: str = BUY_TEXT, tin: int = 1000, tout: int = 100):
        self.text, self.tin, self.tout = text, tin, tout
        self.calls: list[tuple[str, str]] = []

    def __call__(self, symbol: str, date_str: str) -> tuple[str, int, int]:
        self.calls.append((symbol, date_str))
        return self.text, self.tin, self.tout


def make_bars(end: str = "2026-11-17", n: int = 80, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=n, freq="B", name="date")
    close = start + step * np.arange(n) + np.sin(np.arange(n)) * 0.3
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def make_strategy(tmp_path, backend=None, **cfg_overrides) -> tuple[LlmAgentStrategy, FakeBackend]:
    cfg = LlmCfg(cache_dir=str(tmp_path / "llm"), **cfg_overrides)
    fake = backend or FakeBackend()
    strat = LlmAgentStrategy(cfg, backend=fake, today=lambda: TODAY)
    return strat, fake


# ---------- parse_decision ----------

@pytest.mark.parametrize(
    "text, action",
    [
        ("BUY", "BUY"),
        ("**SELL**", "SELL"),
        ("FINAL TRANSACTION PROPOSAL: **HOLD**", "HOLD"),
        ("final transaction proposal: **buy**", "BUY"),
        ("sell", "SELL"),
        ("Rating: Overweight", "BUY"),
        ("Rating: Underweight", "SELL"),
    ],
)
def test_parse_decision_actions(text, action):
    got, _ = parse_decision(text)
    assert got == action


def test_parse_decision_default_confidence_per_action():
    assert parse_decision("BUY") == ("BUY", 0.6)
    assert parse_decision("**SELL**") == ("SELL", 0.6)
    assert parse_decision("FINAL TRANSACTION PROPOSAL: **HOLD**") == ("HOLD", 0.0)


def test_parse_decision_garbage_is_hold_zero():
    assert parse_decision("the market is weird today, no idea") == ("HOLD", 0.0)
    assert parse_decision("") == ("HOLD", 0.0)


def test_parse_decision_proposal_line_wins_over_earlier_words():
    text = "I considered a SELL but the desk ruling is.\nFINAL TRANSACTION PROPOSAL: **BUY**"
    assert parse_decision(text)[0] == "BUY"


def test_parse_decision_parses_decimal_confidence():
    assert parse_decision("confidence: 0.8\nFINAL TRANSACTION PROPOSAL: **BUY**") == ("BUY", 0.8)


def test_parse_decision_parses_percent_confidence():
    assert parse_decision("Confidence 80%. FINAL TRANSACTION PROPOSAL: **SELL**") == ("SELL", 0.8)


# ---------- cache, cost, budget ----------

def test_cache_hit_does_not_call_backend_and_costs_zero(tmp_path):
    strat, fake = make_strategy(tmp_path)
    first = strat.decide("RELIANCE.NS", "2026-11-17")
    second = strat.decide("RELIANCE.NS", "2026-11-17")
    assert len(fake.calls) == 1
    assert first.cached is False and first.cost_usd > 0
    assert second.cached is True and second.cost_usd == 0.0
    assert second.action == first.action == "BUY"
    assert (tmp_path / "llm" / "RELIANCE.NS" / "2026-11-17.json").exists()


def test_cost_is_tokens_times_price_per_mtok(tmp_path):
    cfg = LlmCfg(cache_dir=str(tmp_path / "llm"), price_per_mtok_usd={"input": 0.30, "output": 2.50})
    fake = FakeBackend(tin=1_000_000, tout=100_000)
    strat = LlmAgentStrategy(cfg, backend=fake, today=lambda: TODAY)
    d = strat.decide("RELIANCE.NS", "2026-11-17")
    assert d.tokens_in == 1_000_000 and d.tokens_out == 100_000
    assert d.cost_usd == pytest.approx(0.30 + 0.25)
    assert strat.spent_usd == pytest.approx(0.55)


def test_decision_fields_and_model_recorded(tmp_path):
    strat, _ = make_strategy(tmp_path)
    d = strat.decide("RELIANCE.NS", "2026-11-17")
    assert isinstance(d, LlmDecision)
    assert d.raw == BUY_TEXT
    assert d.model == "gemini-2.5-flash"
    assert d.called_at_utc.endswith("+00:00") or d.called_at_utc.endswith("Z")


def test_spend_persisted_and_reloaded(tmp_path):
    strat, _ = make_strategy(tmp_path)
    strat.decide("RELIANCE.NS", "2026-11-17")
    spend_file = tmp_path / "llm" / "spend.json"
    assert spend_file.exists()
    total = strat.spent_usd
    assert total > 0
    reloaded, _ = make_strategy(tmp_path)
    assert reloaded.spent_usd == pytest.approx(total)


def test_budget_exceeded_blocks_backend_and_generate_holds(tmp_path):
    strat, fake = make_strategy(tmp_path, budget_usd_total=0.0005)
    strat.decide("RELIANCE.NS", "2026-11-10")  # spends 0.00055 >= cap
    assert len(fake.calls) == 1
    with pytest.raises(BudgetExceeded):
        strat.decide("RELIANCE.NS", "2026-11-11")
    assert len(fake.calls) == 1

    sig = strat.generate("RELIANCE.NS", make_bars(), in_position=False)
    assert sig.action == "HOLD"
    assert "LLM budget exhausted" in sig.reason
    assert len(fake.calls) == 1


def test_cached_decision_still_served_when_budget_exhausted(tmp_path):
    strat, fake = make_strategy(tmp_path, budget_usd_total=0.0005)
    strat.decide("RELIANCE.NS", "2026-11-10")
    again = strat.decide("RELIANCE.NS", "2026-11-10")
    assert again.cached is True
    assert len(fake.calls) == 1


# ---------- generate() signal rules ----------

def test_contaminated_past_date_sets_flag_and_prefix(tmp_path):
    strat, _ = make_strategy(tmp_path)
    bars = make_bars(end="2026-01-05")
    sig = strat.generate("RELIANCE.NS", bars, in_position=False)
    assert sig.contaminated is True
    assert sig.reason.startswith("[CONTAMINATED] ")
    assert sig.source == "llm"


def test_current_date_not_contaminated(tmp_path):
    strat, _ = make_strategy(tmp_path)
    sig = strat.generate("RELIANCE.NS", make_bars(end="2026-11-17"), in_position=False)
    assert sig.contaminated is False
    assert not sig.reason.startswith("[CONTAMINATED] ")


def test_buy_while_in_position_is_hold(tmp_path):
    strat, _ = make_strategy(tmp_path, backend=FakeBackend(BUY_TEXT))
    sig = strat.generate("RELIANCE.NS", make_bars(), in_position=True)
    assert sig.action == "HOLD"
    assert sig.stop is None


def test_sell_while_flat_is_hold(tmp_path):
    strat, _ = make_strategy(tmp_path, backend=FakeBackend(SELL_TEXT))
    sig = strat.generate("RELIANCE.NS", make_bars(), in_position=False)
    assert sig.action == "HOLD"


def test_sell_while_in_position_is_sell(tmp_path):
    strat, _ = make_strategy(tmp_path, backend=FakeBackend(SELL_TEXT))
    sig = strat.generate("RELIANCE.NS", make_bars(), in_position=True)
    assert sig.action == "SELL"
    assert sig.stop is None


def test_buy_stop_below_close(tmp_path):
    strat, _ = make_strategy(tmp_path, backend=FakeBackend(BUY_TEXT))
    bars = make_bars()
    sig = strat.generate("RELIANCE.NS", bars, in_position=False)
    assert sig.action == "BUY"
    assert sig.stop is not None
    assert sig.stop < float(bars["close"].iloc[-1])
    assert sig.date == bars.index[-1]
    assert sig.symbol == "RELIANCE.NS"


def test_buy_with_insufficient_history_for_atr_is_hold(tmp_path):
    strat, _ = make_strategy(tmp_path, backend=FakeBackend(BUY_TEXT))
    sig = strat.generate("RELIANCE.NS", make_bars(n=5), in_position=False)
    assert sig.action == "HOLD"


def test_atr_helper_is_wilder_and_nan_until_warm():
    bars = make_bars(n=40)
    atr = llm_agents._atr(bars, 14)
    assert atr.iloc[:13].isna().all()
    assert atr.iloc[-1] > 0


# ---------- secrets never leak ----------

def test_api_key_never_written_or_returned(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "sekrit-test")
    strat, _ = make_strategy(tmp_path, backend=FakeBackend(BUY_TEXT))
    sig = strat.generate("RELIANCE.NS", make_bars(), in_position=False)
    strat.decide("RELIANCE.NS", "2026-11-12")
    assert "sekrit-test" not in sig.reason
    for path in (tmp_path / "llm").rglob("*"):
        if path.is_file():
            assert "sekrit-test" not in path.read_text(encoding="utf-8"), path


# ---------- default backend (real TradingAgents is faked via sys.modules) ----------

def _install_fake_tradingagents(monkeypatch, record: dict, final_text: str = BUY_TEXT):
    pkg = ModuleType("tradingagents")
    dc = ModuleType("tradingagents.default_config")
    dc.DEFAULT_CONFIG = {"llm_provider": "openai", "deep_think_llm": "x", "quick_think_llm": "y"}
    graph_pkg = ModuleType("tradingagents.graph")
    tg = ModuleType("tradingagents.graph.trading_graph")

    class FakeGraph:
        def __init__(self, selected_analysts=None, debug=False, config=None, callbacks=None):
            record["init_count"] = record.get("init_count", 0) + 1
            record["config"] = config
            record["callbacks"] = callbacks

        def propagate(self, company_name, trade_date, asset_type="stock", portfolio=None):
            record.setdefault("propagate", []).append((company_name, trade_date))
            return {"final_trade_decision": final_text}, "Buy"

    tg.TradingAgentsGraph = FakeGraph
    dc.__name__ = "tradingagents.default_config"
    for name, mod in {
        "tradingagents": pkg,
        "tradingagents.default_config": dc,
        "tradingagents.graph": graph_pkg,
        "tradingagents.graph.trading_graph": tg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_default_backend_builds_graph_once_with_cfg_values(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "sekrit-test")
    record: dict = {}
    _install_fake_tradingagents(monkeypatch, record)
    cfg = LlmCfg(cache_dir=str(tmp_path / "llm"), max_debate_rounds=2, model="gemini-2.5-flash")
    strat = LlmAgentStrategy(cfg, today=lambda: TODAY)

    text, tin, tout = strat._backend("RELIANCE.NS", "2026-11-17")
    strat._backend("TCS.NS", "2026-11-17")

    assert text == BUY_TEXT
    assert (tin, tout) == (0, 0)  # fake graph makes no LLM calls, so no usage is counted
    assert record["init_count"] == 1
    conf = record["config"]
    assert conf["llm_provider"] == "google"
    assert conf["deep_think_llm"] == "gemini-2.5-flash"
    assert conf["quick_think_llm"] == "gemini-2.5-flash"
    assert conf["max_debate_rounds"] == 2
    assert conf["max_risk_discuss_rounds"] == 2
    assert record["propagate"] == [("RELIANCE.NS", "2026-11-17"), ("TCS.NS", "2026-11-17")]


def test_default_backend_missing_key_names_env_var_only(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    strat = LlmAgentStrategy(LlmCfg())
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY") as exc:
        strat._backend("RELIANCE.NS", "2026-11-17")
    assert "sekrit" not in str(exc.value)


def test_usage_extraction_from_llm_result_shapes():
    msg = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 4})
    res = SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output=None)
    assert llm_agents._usage_from_llm_result(res) == (10, 4)

    res2 = SimpleNamespace(generations=[[]], llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert llm_agents._usage_from_llm_result(res2) == (7, 3)

    res3 = SimpleNamespace(generations=[[]], llm_output=None)
    assert llm_agents._usage_from_llm_result(res3) == (0, 0)


def test_spend_file_is_valid_json(tmp_path):
    strat, _ = make_strategy(tmp_path)
    strat.decide("RELIANCE.NS", "2026-11-17")
    data = json.loads((tmp_path / "llm" / "spend.json").read_text(encoding="utf-8"))
    assert data["spent_usd"] == pytest.approx(strat.spent_usd)


def test_budget_still_enforced_when_usage_untracked(tmp_path):
    from nsebot.config import LlmCfg
    from nsebot.strategies.llm_agents import BudgetExceeded, LlmAgentStrategy
    cfg = LlmCfg(cache_dir=str(tmp_path), budget_usd_total=0.10, est_cost_per_call_usd=0.05)
    strat = LlmAgentStrategy(cfg, backend=lambda s, d: ("FINAL TRANSACTION PROPOSAL: **BUY**", 0, 0))
    d1 = strat.decide("AAA.NS", "2024-01-01")
    assert d1.cost_estimated and d1.cost_usd == 0.05
    strat.decide("AAA.NS", "2024-01-02")
    import pytest as _pt
    with _pt.raises(BudgetExceeded):
        strat.decide("AAA.NS", "2024-01-03")
