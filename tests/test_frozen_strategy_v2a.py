"""STRATEGY FREEZE (tag strategy-v2a). The v2a rules ("v1 rules + NIFTYBEES parking") are frozen. If this test fails
you changed v2a (directly, via the frozen v1 baseline, or via the shared Simulator): that needs a NEW version tag, a
README disclosure, and treating earlier v2a results as a different system. Bug fixes only."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from nsebot.backtest.engine import run_backtest
from nsebot.config import Config
from nsebot.sim_parking import PARK_BUFFER, ParkingSimulator
from nsebot.strategies.baseline import BaselineStrategy

FROZEN_VERSION = "strategy-v2a"
SIM_PARKING_SHA256 = "6a3068a1f491b2aae26072925c2594769ad8bb93f6f5357e1d5af8473e6b160d"
BENCH = "NB.NS"


class _AllMembers:
    def __init__(self, keys):
        self.keys = frozenset(keys)

    def members_on(self, d):
        return self.keys


def _golden_market():
    days = pd.bdate_range("2019-01-01", periods=700, name="date")
    rng = np.random.default_rng(11)
    prices = {}
    for i in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, 700)))
        prices[f"S{i:02d}.NS"] = pd.DataFrame({"open": c * (1 + rng.normal(0, 0.003, 700)), "high": c * 1.012,
                                               "low": c * 0.988, "close": c, "volume": 1.0}, index=days)
    nb = 250 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 700)))
    prices[BENCH] = pd.DataFrame({"open": nb, "high": nb * 1.005, "low": nb * 0.995, "close": nb, "volume": 1.0},
                                 index=days)
    return prices


def _run(kill_switch):
    p = _golden_market()
    cfg = Config(symbols=["S00.NS"], benchmark=BENCH)
    uni = _AllMembers(k for k in p if k != BENCH)
    return run_backtest(p, None, cfg, "2019-12-01", None, "v2a", sim_factory=lambda b, d: ParkingSimulator(
        b, cfg, uni, BaselineStrategy(cfg.baseline, cfg.risk.atr_stop_mult), d, kill_switch=kill_switch))


def _sig(res):
    return hashlib.sha256(res.trades[["date", "symbol", "side", "qty"]].to_csv(index=False).encode()).hexdigest()


def test_v2a_code_and_parameters_are_frozen():
    assert hashlib.sha256(Path("src/nsebot/sim_parking.py").read_bytes()).hexdigest() == SIM_PARKING_SHA256, \
        f"sim_parking.py changed after {FROZEN_VERSION}"
    assert PARK_BUFFER == 1_000.0


def test_v2a_behaviour_is_frozen_including_shared_code():
    r = _run(True)
    assert (len(r.trades), round(float(r.equity.iloc[-1]), 2), r.halted[:22], _sig(r)) == (43, 89839.13, 'KILL-SWITCH 2020-07-14', 'ebc71d419573e00dab9211ba0c49075fe71dfb7201d71a8338a6568062174682')
    d = _run(False)
    assert (len(d.trades), round(float(d.equity.iloc[-1]), 2), d.halted[:22], _sig(d)) == (130, 106855.34, '', '91cc30bfdc4ed9a459a5b7886c9bd280d5ef9bb490bc1c3d90dde1f0aeedd39c')
