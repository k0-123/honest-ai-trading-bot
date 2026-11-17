import pytest
import yaml
from pydantic import ValidationError

from nsebot.config import Config, load_config


def test_repo_config_loads():
    cfg = load_config("config.yaml")
    assert cfg.capital == 100000
    assert "RELIANCE.NS" in cfg.symbols
    assert cfg.risk.max_drawdown_pct == 0.10


def test_rejects_empty_symbols():
    with pytest.raises(ValidationError):
        Config(symbols=[])


def test_rejects_absurd_risk(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"symbols": ["X.NS"], "risk": {"risk_per_trade_pct": 0.5}}))
    with pytest.raises(ValidationError):
        load_config(p)


def test_config_has_no_secret_values():
    text = open("config.yaml").read().lower()
    assert "api_key:" not in text and "sk-" not in text
