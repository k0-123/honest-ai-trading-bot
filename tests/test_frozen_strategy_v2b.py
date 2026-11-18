"""STRATEGY FREEZE (tag strategy-v2b). The v2b momentum rules are frozen so every result from here on is measured on
the same strategy. If this test fails you changed v2b (directly, or via the shared Simulator it builds on): that needs a
NEW version tag, a README disclosure, and treating earlier v2b results as a different system. Bug fixes only."""
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_momentum import BENCH, make_market, run_v2b

from nsebot.strategies.momentum import MomentumCfg

FROZEN_VERSION = "strategy-v2b"
MOMENTUM_SHA256 = "ac64cded55eaa83d38192bcc0f1197a8d3ba7ba962f33094e2185194795f67a2"
SIM_MOMENTUM_SHA256 = "4a4f0319180e6a659291079f6bb7758a143ec7ae026cd735524e4fdfc72233e6"
FROZEN_CFG = {"top_n": 10, "sell_rank": 20, "lookback": 252, "skip": 21, "weight": 0.10, "regime_sma": 200}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_v2b_code_is_frozen():
    assert _sha("src/nsebot/strategies/momentum.py") == MOMENTUM_SHA256, f"momentum.py changed after {FROZEN_VERSION}"
    assert _sha("src/nsebot/sim_momentum.py") == SIM_MOMENTUM_SHA256, f"sim_momentum.py changed after {FROZEN_VERSION}"


def test_v2b_parameters_are_frozen():
    assert asdict(MomentumCfg()) == FROZEN_CFG


def _golden_market():
    days, prices = make_market(n=700, seed=3)
    for i, k in enumerate(sorted(prices)):
        if k != BENCH and i % 2 == 0:
            prices[k].loc[days[450]:, ["open", "high", "low", "close"]] *= 0.75
    prices[BENCH].loc[days[460]:days[560], ["open", "high", "low", "close"]] *= 0.8
    return prices


def _sig(res):
    return hashlib.sha256(res.trades[["date", "symbol", "side", "qty"]].to_csv(index=False).encode()).hexdigest()


def test_v2b_behaviour_is_frozen_including_shared_simulator():
    """Golden run: catches behaviour changes coming from the shared Simulator base class too."""
    p = _golden_market()
    r = run_v2b(p, "2020-01-01")
    assert (len(r.trades), round(float(r.equity.iloc[-1]), 2)) == (20, 112943.04)
    assert r.halted.startswith("KILL-SWITCH 2020-09-22")
    assert _sig(r) == "9a7f503ec5c3fad084a1768d6361b207402d2a0a34c6728dd8ca4dc7bcd6868e"
    d = run_v2b(p, "2020-01-01", kill_switch=False)
    assert (len(d.trades), round(float(d.equity.iloc[-1]), 2)) == (16, 160635.42)
    assert _sig(d) == "3c4e374b3ee1b414ac92c5c63f1c2ebf92c1b0049870e6297e59e6f4f80b02c4"
