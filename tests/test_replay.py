import pandas as pd
import pytest

from nsebot.config import Config
from nsebot.paper.replay import render, replay

DAYS = pd.bdate_range("2026-01-01", periods=260, name="date")


def frame(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(DAYS))))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0}, index=DAYS)


DATA = {"A.NS": frame(1), "NB.NS": frame(2)}


class RecordingLoader:
    def __init__(self):
        self.calls = []

    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True):
        df = DATA[symbol]
        if as_of is not None:
            df = df.loc[: pd.Timestamp(as_of)]
        self.calls.append((symbol, as_of, refresh, df.index.max()))
        return df


def cfg_for(tmp_path):
    return Config(symbols=["A.NS"], benchmark="NB.NS",
                  paths={"state_dir": str(tmp_path / "state"), "reports_dir": str(tmp_path / "reports_out")})


def test_replay_uses_only_data_up_to_each_day_and_never_refreshes_in_loop(tmp_path):
    loader = RecordingLoader()
    res = replay(cfg_for(tmp_path), loader, "2026-10-01", "2026-10-30", tmp_path / "replay")
    assert res.days == [d.strftime("%Y-%m-%d") for d in DAYS if "2026-10-01" <= d.strftime("%Y-%m-%d") <= "2026-10-30"]
    in_loop = [c for c in loader.calls if c[1] is not None]
    assert in_loop and all(c[3] <= pd.Timestamp(c[1]) for c in in_loop)   # nothing after that day
    assert not any(c[2] for c in in_loop)                                 # no downloads inside the loop
    assert set(res.final["book"]) == {"baseline", "buyhold_niftybees"}     # no LLM book
    assert "| Book |" in render(res, 100000)


def test_replay_leaves_real_state_untouched_and_starts_fresh(tmp_path):
    real = tmp_path / "state"
    real.mkdir()
    (real / "portfolio.json").write_text("REAL BOOKS")
    cfg = cfg_for(tmp_path)
    replay(cfg, RecordingLoader(), "2026-10-01", "2026-10-15", tmp_path / "replay")
    again = replay(cfg, RecordingLoader(), "2026-10-01", "2026-10-15", tmp_path / "replay")
    assert (real / "portfolio.json").read_text() == "REAL BOOKS"
    assert again.days[0] == "2026-10-01"   # second replay did not continue from the first one


def test_replay_refuses_to_use_the_real_state_folder(tmp_path):
    cfg = cfg_for(tmp_path)
    with pytest.raises(ValueError):
        replay(cfg, RecordingLoader(), "2026-10-01", "2026-10-15", tmp_path / "state")
    with pytest.raises(ValueError):
        replay(cfg, RecordingLoader(), "2026-10-01", "2026-10-15", tmp_path)
