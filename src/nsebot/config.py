"""Load and validate config.yaml. API keys are NEVER stored here, only env var names."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DataCfg(BaseModel):
    start: str = "2015-01-01"
    cache_dir: str = "cache/prices"


class CostCfg(BaseModel):
    brokerage_pct: float = Field(0.0, ge=0, le=0.01)
    brokerage_cap: float = Field(20.0, ge=0)
    stt_pct: float = Field(0.001, ge=0, le=0.01)
    exchange_txn_pct: float = Field(0.0000297, ge=0, le=0.001)
    sebi_fee_pct: float = Field(0.000001, ge=0, le=0.001)
    gst_pct: float = Field(0.18, ge=0, le=0.5)
    stamp_duty_buy_pct: float = Field(0.00015, ge=0, le=0.01)
    dp_charge_per_sell: float = Field(15.93, ge=0)
    slippage_pct: float = Field(0.0005, ge=0, le=0.01)


class RiskCfg(BaseModel):
    risk_per_trade_pct: float = Field(0.01, gt=0, le=0.05)
    max_position_pct: float = Field(0.20, gt=0, le=1.0)
    daily_loss_limit_pct: float = Field(0.02, gt=0, le=0.5)
    max_drawdown_pct: float = Field(0.10, gt=0, le=0.9)
    atr_stop_mult: float = Field(2.0, gt=0)


class BaselineCfg(BaseModel):
    sma_fast: int = Field(50, gt=1)
    sma_slow: int = Field(200, gt=1)
    rsi_period: int = Field(14, gt=1)
    rsi_entry: float = Field(35, gt=0, lt=100)
    rsi_exit: float = Field(70, gt=0, lt=100)
    atr_period: int = Field(14, gt=1)
    exit_below_fast_days: int = Field(0, ge=0)  # 0 disables the SMA-fast exit


class LlmCfg(BaseModel):
    enabled: bool = True
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GOOGLE_API_KEY"
    max_debate_rounds: int = Field(1, ge=1)
    cache_dir: str = "cache/llm"
    budget_usd_total: float = Field(5.0, ge=0)
    budget_usd_per_day: float = Field(0.5, ge=0)
    usd_inr: float = Field(84.0, gt=0)
    est_cost_per_call_usd: float = Field(0.05, ge=0)  # charged when token usage cannot be read
    price_per_mtok_usd: dict[str, float] = {"input": 0.30, "output": 2.50}


class SplitCfg(BaseModel):
    train_end: str = "2020-12-31"
    val_end: str = "2022-12-31"


class BacktestCfg(BaseModel):
    splits: SplitCfg = SplitCfg()


class PathsCfg(BaseModel):
    state_dir: str = "state"
    reports_dir: str = "reports_out"


class Config(BaseModel):
    symbols: list[str]
    benchmark: str = "NIFTYBEES.NS"
    capital: float = Field(100000, gt=0)
    data: DataCfg = DataCfg()
    costs: CostCfg = CostCfg()
    risk: RiskCfg = RiskCfg()
    baseline: BaselineCfg = BaselineCfg()
    llm: LlmCfg = LlmCfg()
    backtest: BacktestCfg = BacktestCfg()
    paths: PathsCfg = PathsCfg()
    # Weekday NSE trading holidays (YYYY-MM-DD) from NSE's published calendar. Used only in live paper mode, where
    # tomorrow's bar does not exist yet, to find the last session of a month (v2b rebalance). Keep it up to date.
    nse_holidays: list[str] = []

    @field_validator("symbols")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols must not be empty")
        return v


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
