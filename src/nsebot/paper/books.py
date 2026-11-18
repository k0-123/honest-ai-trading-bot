"""Factories for the v2 paper books (point-in-time Nifty 50 universe). Each factory: (book, calendar_days) -> Simulator."""
from __future__ import annotations

from nsebot.config import Config
from nsebot.sim_momentum import MomentumSimulator
from nsebot.sim_parking import ParkingSimulator
from nsebot.strategies.baseline import BaselineStrategy
from nsebot.strategies.momentum import MomentumCfg

V2A_BOOK, V2B_BOOK = "v2a_parking", "v2b_momentum"


def v2_universe_books(cfg: Config, universe) -> dict:
    return {
        V2A_BOOK: lambda book, cal: ParkingSimulator(book, cfg, universe,
                                                     BaselineStrategy(cfg.baseline, cfg.risk.atr_stop_mult), cal),
        V2B_BOOK: lambda book, cal: MomentumSimulator(book, cfg, universe, MomentumCfg(), cal,
                                                      holidays=cfg.nse_holidays),
    }
