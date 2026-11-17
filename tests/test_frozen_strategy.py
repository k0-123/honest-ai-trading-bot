"""STRATEGY FREEZE (tag strategy-v1). The baseline rules and risk limits are frozen so that every result
from here on is measured on the same, unchanged strategy. If this test fails you changed the strategy:
that requires a NEW version tag, a README disclosure, and treating all prior results as a different system."""
import hashlib
from pathlib import Path

from nsebot.config import load_config

FROZEN_VERSION = "strategy-v1"
BASELINE_SHA256 = "5ca1655976f8a89ac842186ce6c71772b70905122baf3abc2729df659886b428"
FROZEN_BASELINE = {"sma_fast": 50, "sma_slow": 200, "rsi_period": 14, "rsi_entry": 35.0, "rsi_exit": 70.0,
                   "atr_period": 14, "exit_below_fast_days": 0}
FROZEN_RISK = {"risk_per_trade_pct": 0.01, "max_position_pct": 0.20, "daily_loss_limit_pct": 0.02,
               "max_drawdown_pct": 0.10, "atr_stop_mult": 2.0}


def test_baseline_strategy_code_is_frozen():
    h = hashlib.sha256(Path("src/nsebot/strategies/baseline.py").read_bytes()).hexdigest()
    assert h == BASELINE_SHA256, f"baseline.py changed after the {FROZEN_VERSION} freeze"


def test_strategy_and_risk_parameters_are_frozen():
    cfg = load_config("config.yaml")
    assert cfg.baseline.model_dump() == FROZEN_BASELINE
    assert cfg.risk.model_dump() == FROZEN_RISK
