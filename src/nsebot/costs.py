"""Indian equity trading costs (paper-trading model) and slippage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nsebot.config import CostCfg


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    gst: float
    stamp: float
    dp: float
    total: float


def apply_slippage(side: Literal["BUY", "SELL"], price: float, cfg: CostCfg) -> float:
    """Fill price after slippage: BUY pays more, SELL receives less."""
    if side == "BUY":
        return price * (1 + cfg.slippage_pct)
    return price * (1 - cfg.slippage_pct)


def trade_costs(side: Literal["BUY", "SELL"], qty: int, price: float, cfg: CostCfg) -> CostBreakdown:
    """Costs for one fill. Zero quantity costs nothing."""
    if qty == 0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    turnover = qty * price
    brokerage = min(turnover * cfg.brokerage_pct, cfg.brokerage_cap)
    stt = turnover * cfg.stt_pct
    exchange = turnover * cfg.exchange_txn_pct
    sebi = turnover * cfg.sebi_fee_pct
    gst = cfg.gst_pct * (brokerage + exchange + sebi)
    stamp = turnover * cfg.stamp_duty_buy_pct if side == "BUY" else 0.0
    dp = cfg.dp_charge_per_sell if side == "SELL" else 0.0
    total = brokerage + stt + exchange + sebi + gst + stamp + dp
    return CostBreakdown(brokerage, stt, exchange, sebi, gst, stamp, dp, total)
