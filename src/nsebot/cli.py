"""Command line: nsebot daily | backtest | report-weekly | crosscheck | reset-halt | live"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from nsebot.config import Config, load_config

log = logging.getLogger("nsebot")

SEBI_REMINDER = """LIVE TRADING IS NOT BUILT, ON PURPOSE.
Under SEBI's retail algo framework (in force from April 2026), retail algorithmic orders must go through a
broker-approved API, from a registered static IP, with exchange-issued Algo-ID tagging. Talk to your broker
first. This project only paper-trades; nothing here can place a real order."""


def _loader(cfg: Config):
    from nsebot.data.loader import PriceLoader
    return PriceLoader(cfg.data.cache_dir)


def _llm_spend(cache_dir: str) -> float:
    p = Path(cache_dir) / "spend.json"
    try:
        return float(json.loads(p.read_text())["spent_usd"])
    except (OSError, ValueError, KeyError):
        return 0.0


def _strategies(cfg: Config, with_llm: bool, llm_cache_dir: str | None = None) -> dict:
    from nsebot.strategies.baseline import BaselineStrategy
    strats: dict = {"baseline": BaselineStrategy(cfg.baseline, cfg.risk.atr_stop_mult)}
    if with_llm and cfg.llm.enabled:
        if not os.environ.get(cfg.llm.api_key_env):
            log.warning("LLM book skipped: environment variable %s is not set", cfg.llm.api_key_env)
        else:
            from nsebot.strategies.llm_agents import LlmAgentStrategy
            llm_cfg = cfg.llm.model_copy(update={"cache_dir": llm_cache_dir} if llm_cache_dir else {})
            strats["llm"] = LlmAgentStrategy(llm_cfg, cfg.risk.atr_stop_mult)
    return strats


def cmd_daily(cfg: Config, args) -> int:
    from nsebot.paper.runner import PaperRunner
    from nsebot.reports.daily import write_daily_report
    from nsebot.reports.weekly import write_weekly_report
    kw = {}
    if not args.no_v2:
        from nsebot.paper.books import v2_universe_books
        from nsebot.universe.nifty50 import Nifty50Universe
        uni = Nifty50Universe()
        kw = {"universe": uni, "universe_loader": _universe_loader(cfg), "universe_books": v2_universe_books(cfg, uni)}
    runner = PaperRunner(cfg, _loader(cfg), _strategies(cfg, with_llm=not args.no_llm), **kw)
    res = runner.run()
    print(res.message)
    path = write_daily_report(res, cfg.paths.reports_dir)
    if path:
        print(f"Daily report: {path}")
    if res.processed_days and (args.weekly or pd.Timestamp(res.processed_days[-1]).dayofweek == 4):
        print("Weekly report:", write_weekly_report(cfg.paths.state_dir, cfg.paths.reports_dir,
                                                    _llm_spend(cfg.llm.cache_dir), cfg.llm.usd_inr))
    for name, b in res.books.items():
        if b.halted:
            print(f"!! {name} HALTED: {b.halt_reason}")
    return 0


V2A_BOOK = "v2a_parking"
V2A_DIAG = "v2a_parking DIAGNOSTIC (no kill-switch)"
V2B_BOOK = "v2b_momentum"
V2B_DIAG = "v2b_momentum DIAGNOSTIC (no kill-switch)"


def _universe_loader(cfg: Config):
    from nsebot.data.crosscheck import jugaad_fetcher
    from nsebot.data.loader import PriceLoader
    from nsebot.universe.nifty50 import UniversePriceLoader
    return UniversePriceLoader(PriceLoader(cfg.data.cache_dir),
                               PriceLoader(str(Path(cfg.data.cache_dir).parent / "prices_nse"), fetcher=jugaad_fetcher))


def _v2b_inputs(cfg: Config, start: str, end: str, refresh: bool = False):
    """Point-in-time Nifty 50 universe + prices for every member in [start, end] (+400 days warm-up).
    Raises MissingPriceData rather than silently dropping a member."""
    from nsebot.universe.nifty50 import Nifty50Universe, check_prices
    uni, ul = Nifty50Universe(), _universe_loader(cfg)
    data_start = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    keys = sorted(uni.ever_members(start, end))
    prices = {k: ul.get(k, data_start, end, refresh=refresh) for k in keys}
    check_prices(uni, ul, start, end)
    raw = {k: ul.get(k, data_start, end, clean=False) for k in keys}
    prices[cfg.benchmark] = ul.get(cfg.benchmark, data_start, end)
    raw[cfg.benchmark] = ul.get(cfg.benchmark, data_start, end, clean=False)
    return uni, prices, raw


def _run_v2b(cfg: Config, uni, prices, raw, start, end, name, kill_switch=True):
    from nsebot.backtest.engine import run_backtest
    from nsebot.sim_momentum import MomentumSimulator
    from nsebot.strategies.momentum import MomentumCfg
    holder = {}

    def factory(book, all_days):
        holder["sim"] = MomentumSimulator(book, cfg, uni, MomentumCfg(), all_days, kill_switch=kill_switch)
        return holder["sim"]
    res = run_backtest(prices, None, cfg, start, end, name, raw_prices=raw, sim_factory=factory)
    res.sim = holder["sim"]
    return res


def _append_corrections_section(cfg: Config, path: Path, periods, v2b, period_results) -> None:
    """Report section: v2b BEFORE vs AFTER the documented data corrections, and every correction with evidence."""
    from nsebot.data.crosscheck import jugaad_fetcher
    from nsebot.data.loader import PriceLoader
    from nsebot.universe.adjustments import load_adjustments, load_bar_overrides
    from nsebot.universe.nifty50 import UniversePriceLoader
    uni = v2b[0]
    adj, ov = load_adjustments(), load_bar_overrides()
    raw_loader = UniversePriceLoader(PriceLoader(cfg.data.cache_dir),
                                     PriceLoader(str(Path(cfg.data.cache_dir).parent / "prices_nse"), fetcher=jugaad_fetcher),
                                     adjustments=adj.iloc[0:0], overrides=ov.iloc[0:0])
    keys = list(v2b[1])
    ds = min(df.index.min() for df in v2b[1].values() if len(df)).strftime("%Y-%m-%d")
    pb = {k: raw_loader.get(k, ds) for k in keys}
    rb = {k: raw_loader.get(k, ds, clean=False) for k in keys}
    lines = ["", "## v2b data corrections: results BEFORE vs AFTER", "",
             ("Same rules, same code; only the price data differs. Corrections come from a full Yahoo-vs-NSE audit "
             "(see data/nifty50/). They were applied whether they helped or hurt v2b."), "",
             "| Period | Book | Before | After | Difference (pp) |", "|---|---|---|---|---|"]
    for title, (ps, pe) in periods.items():
        for name, ks in ((V2B_BOOK, True), (V2B_DIAG, False)):
            before = _run_v2b(cfg, uni, pb, rb, ps, pe, name, kill_switch=ks)
            after = period_results[title][name]
            b, a = before.equity.iloc[-1] / cfg.capital - 1, after.equity.iloc[-1] / cfg.capital - 1
            lines.append(f"| {title} | {name} | {b:+.2%} | {a:+.2%} | {(a - b) * 100:+.2f} |")
    lines += ["", f"### Factor adjustments ({len(adj)})", "", "| Stock | Date | Factor | Kind | Evidence |",
              "|---|---|---|---|---|"]
    lines += [f"| {r.key} | {r.date:%Y-%m-%d} | {r.factor:.4f} | {r.kind} | {r.evidence} |" for r in adj.itertuples()]
    lines += ["", f"### Bar overrides: stale or bad Yahoo bars replaced by NSE's real bar ({len(ov)})", "",
              "| Stock | Date | Evidence |", "|---|---|---|"]
    lines += [f"| {r.key} | {r.date:%Y-%m-%d} | {r.evidence} |" for r in ov.itertuples()]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _run_v2a(cfg: Config, uni, prices, raw, start, end, name, kill_switch=True):
    from nsebot.backtest.engine import run_backtest
    from nsebot.sim_parking import ParkingSimulator
    from nsebot.strategies.baseline import BaselineStrategy
    holder = {}

    def factory(book, all_days):
        holder["sim"] = ParkingSimulator(book, cfg, uni, BaselineStrategy(cfg.baseline, cfg.risk.atr_stop_mult),
                                         all_days, kill_switch=kill_switch)
        return holder["sim"]
    res = run_backtest(prices, None, cfg, start, end, name, raw_prices=raw, sim_factory=factory)
    res.sim = holder["sim"]
    return res


def cmd_backtest(cfg: Config, args) -> int:
    from nsebot.backtest.engine import run_backtest
    from nsebot.backtest.splits import split_periods
    from nsebot.reports.backtest import write_backtest_report
    loader = _loader(cfg)
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = {s: loader.get(s, cfg.data.start, end, refresh=args.refresh) for s in cfg.symbols}
    bench = loader.get(cfg.benchmark, cfg.data.start, end, refresh=args.refresh)
    raw = {s: loader.get(s, cfg.data.start, end, clean=False) for s in cfg.symbols}
    raw_bench = {cfg.benchmark: loader.get(cfg.benchmark, cfg.data.start, end, clean=False)}
    # first date where every symbol has enough history for SMA200 (warm-up is never traded)
    warm = max(df.index[min(cfg.baseline.sma_slow + 5, len(df) - 1)] for df in prices.values())
    start = max(pd.Timestamp(args.start), warm).strftime("%Y-%m-%d") if args.start else warm.strftime("%Y-%m-%d")
    strats = _strategies(cfg, with_llm=False)
    results = {
        "baseline": run_backtest(prices, strats["baseline"], cfg, start, end, "baseline", raw_prices=raw),
        "buyhold_niftybees": run_backtest({cfg.benchmark: bench}, None, cfg, start, end, "buyhold_niftybees",
                                          buy_and_hold=True, raw_prices=raw_bench),
    }
    notes = [f"Universe: {', '.join(cfg.symbols)}; capital Rs {cfg.capital:,.0f}; long-only delivery.",
             "Fills at next day's open with slippage; all Indian delivery costs deducted.",
             "Prices are yfinance split/dividend-adjusted (small known bias).",
             ("Baseline rules were changed twice AFTER seeing backtest results (logic fixes, not parameter "
             "tuning); see README 'Strategy changes'. Treat val/test numbers as no longer pristine."),
             "Train/val/test tables are independent runs with fresh capital; 'full' is one continuous run."]
    spend = 0.0
    if args.llm_days:
        llm_dir = str(Path(cfg.llm.cache_dir).parent / "llm_backtest")
        llm = _strategies(cfg, with_llm=True, llm_cache_dir=llm_dir).get("llm")
        if llm is None:
            notes.append(f"LLM sample NOT run: {cfg.llm.api_key_env} not set.")
        else:
            days = sorted(set().union(*[df.index for df in prices.values()]))
            llm_start = days[-args.llm_days].strftime("%Y-%m-%d")
            results["llm (sample)"] = run_backtest(prices, llm, cfg, llm_start, end, "llm (sample)", raw_prices=raw)
            results["baseline (same window)"] = run_backtest(prices, strats["baseline"], cfg, llm_start, end,
                                                             "baseline (same window)", raw_prices=raw)
            spend = _llm_spend(llm_dir)
            notes.append(f"LLM sample: last {args.llm_days} sessions x {len(cfg.symbols)} stocks, "
                         f"budget cap ${cfg.llm.budget_usd_total:.2f}.")
    sp = cfg.backtest.splits
    periods = split_periods(start, end, sp.train_end, sp.val_end)
    v2b = None
    if not args.no_v2:
        v2b = _v2b_inputs(cfg, start, end, refresh=args.refresh)
        results[V2A_BOOK] = _run_v2a(cfg, *v2b, start, end, V2A_BOOK)
        results[V2A_DIAG] = _run_v2a(cfg, *v2b, start, end, V2A_DIAG, kill_switch=False)
        results[V2B_BOOK] = _run_v2b(cfg, *v2b, start, end, V2B_BOOK)
        results[V2B_DIAG] = _run_v2b(cfg, *v2b, start, end, V2B_DIAG, kill_switch=False)
        notes += [("v2b_momentum (strategy-v2b): point-in-time Nifty 50 (official NSE Indices changes, see "
                  "data/nifty50/), 12-1 momentum, top 10, monthly. Momentum is a well-known historical effect, so "
                  "a good backtest here is NOT proof it will continue. DIAGNOSTIC = same rules without the 10% "
                  "kill-switch, to show the strategy itself; it is not a tradable book."),
                  "HDFC Ltd, Cairn India, Tata Motors DVR and IDFC prices come from NSE (not dividend-adjusted).",
                  ("v2a_parking (strategy-v2a): the frozen v1 rules on the point-in-time Nifty 50, idle cash parked in "
                  "NIFTYBEES (sold just enough to fund buys, proceeds re-parked). Same costs and risk rules incl. the "
                  "10% kill-switch; DIAGNOSTIC = without the kill-switch.")]
    period_results = {}
    for title, (ps, pe) in periods.items():
        period_results[title] = {
            "baseline": run_backtest(prices, strats["baseline"], cfg, ps, pe, "baseline", raw_prices=raw),
            "buyhold_niftybees": run_backtest({cfg.benchmark: bench}, None, cfg, ps, pe, "buyhold_niftybees",
                                              buy_and_hold=True, raw_prices=raw_bench),
        }
        if v2b is not None:
            period_results[title][V2A_BOOK] = _run_v2a(cfg, *v2b, ps, pe, V2A_BOOK)
            period_results[title][V2A_DIAG] = _run_v2a(cfg, *v2b, ps, pe, V2A_DIAG, kill_switch=False)
            period_results[title][V2B_BOOK] = _run_v2b(cfg, *v2b, ps, pe, V2B_BOOK)
            period_results[title][V2B_DIAG] = _run_v2b(cfg, *v2b, ps, pe, V2B_DIAG, kill_switch=False)
    path = write_backtest_report(results, periods, cfg.paths.reports_dir, spend, cfg.llm.usd_inr, notes,
                                 period_results)
    if v2b is not None:
        _append_corrections_section(cfg, Path(path), periods, v2b, period_results)
    print(f"Backtest report: {path}")
    for name, r in results.items():
        s = r.summary()
        print(f"  {name:24s} return {s['total_return']:+.1%}  sharpe {s['sharpe']:.2f}  "
              f"maxDD {s['max_drawdown']:.1%}  trades {s['n_trades']}  costs Rs {s['total_costs']:,.0f}")
    return 0


def cmd_weekly(cfg: Config, args) -> int:
    from nsebot.reports.weekly import write_weekly_report
    print(write_weekly_report(cfg.paths.state_dir, cfg.paths.reports_dir, _llm_spend(cfg.llm.cache_dir),
                              cfg.llm.usd_inr))
    return 0


def cmd_crosscheck(cfg: Config, args) -> int:
    from nsebot.data.crosscheck import cross_check, jugaad_fetcher
    loader = _loader(cfg)
    start = (pd.Timestamp.today() - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")
    bad = 0
    for s in cfg.symbols:
        try:
            nse = jugaad_fetcher(s, start, None)
        except Exception as e:  # noqa: BLE001  NSE often blocks datacenter IPs; any failure = skip
            print(f"{s}: NSE fetch failed ({type(e).__name__}); skipped")
            continue
        r = cross_check(loader.get(s, start, refresh=True), nse, s)
        bad += r.n_mismatch
        print(f"{s}: compared {r.n_compared} days, mismatches {r.n_mismatch}, "
              f"max return diff {r.max_abs_pct_diff:.3%}, missing yahoo {len(r.missing_in_primary)}, "
              f"missing nse {len(r.missing_in_secondary)}")
    return 1 if bad else 0


def cmd_datacheck(cfg: Config, args) -> int:
    from nsebot.data.crosscheck import jugaad_fetcher
    from nsebot.data.quality import audit_symbol, render, yahoo_splits
    loader = _loader(cfg)
    audits = []
    for s in cfg.symbols + [cfg.benchmark]:
        raw = loader.get(s, cfg.data.start, clean=False)
        audits.append(audit_symbol(s, raw, yahoo_splits(s), None if args.no_nse else jugaad_fetcher))
    out = Path(cfg.paths.reports_dir) / "data_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(audits), encoding="utf-8")
    print(out.read_text())
    return 0


def cmd_replay(cfg: Config, args) -> int:
    from nsebot.paper.replay import render, replay
    from nsebot.reports.daily import write_daily_report
    kw = {}
    if not args.no_v2:
        from nsebot.paper.books import v2_universe_books
        from nsebot.universe.nifty50 import Nifty50Universe
        uni = Nifty50Universe()
        kw = {"universe": uni, "universe_loader": _universe_loader(cfg), "universe_books": v2_universe_books(cfg, uni)}
    res = replay(cfg, _loader(cfg), args.start, args.end, args.out,
                 report_fn=lambda r, d: write_daily_report(r, d), **kw)
    text = render(res, cfg.capital)
    (res.out_dir / "replay_summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Files: {res.out_dir}/state (trades.csv, equity.csv, portfolio.json), {res.out_dir}/reports/daily/")
    return 0


def cmd_reset_halt(cfg: Config, args) -> int:
    from nsebot.paper.runner import PaperRunner
    if not args.confirm:
        print("Refusing: pass --confirm after reviewing why the book halted.")
        return 2
    runner = PaperRunner(cfg, None, {})
    books = runner.load_books()
    b = books.get(args.book)
    if b is None:
        print(f"No book named {args.book}")
        return 2
    print(f"Resetting halt on {args.book} (was: {b.halt_reason}). Peak equity re-based to current equity.")
    b.halted, b.halt_reason, b.peak_equity = False, "", b.equity()
    runner._save(books)
    (runner.state_dir / f"HALTED_{args.book}").unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="nsebot", description="Honest NSE paper-trading bot (no real money).")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("daily", help="run after market close (16:30 IST)")
    d.add_argument("--no-llm", action="store_true")
    d.add_argument("--no-v2", action="store_true", help="skip the v2a/v2b universe books")
    d.add_argument("--weekly", action="store_true", help="also write the weekly report")
    b = sub.add_parser("backtest")
    b.add_argument("--start")
    b.add_argument("--end")
    b.add_argument("--refresh", action="store_true", help="re-download recent prices")
    b.add_argument("--no-v2", action="store_true", help="skip the v2 books (faster)")
    b.add_argument("--llm-days", type=int, default=0, help="also run a CONTAMINATED LLM sample on last N sessions")
    sub.add_parser("report-weekly")
    c = sub.add_parser("crosscheck", help="compare Yahoo vs NSE (jugaad-data) returns")
    c.add_argument("--days", type=int, default=90)
    dc = sub.add_parser("datacheck", help="split/bonus + bad-bar audit for every symbol")
    dc.add_argument("--no-nse", action="store_true")
    rp = sub.add_parser("replay", help="replay the daily paper run day by day into a separate folder")
    rp.add_argument("--start", default="2026-10-01")
    rp.add_argument("--end", help="default: today")
    rp.add_argument("--out", default="replay", help="separate folder (never the real state dir)")
    rp.add_argument("--no-v2", action="store_true", help="skip the v2a/v2b universe books")
    r = sub.add_parser("reset-halt")
    r.add_argument("--book", required=True)
    r.add_argument("--confirm", action="store_true")
    sub.add_parser("live", help="explains why live trading is not available")
    args = p.parse_args(argv)
    if args.cmd == "live":
        print(SEBI_REMINDER)
        return 3
    cfg = load_config(args.config)
    return {"daily": cmd_daily, "backtest": cmd_backtest, "report-weekly": cmd_weekly,
            "crosscheck": cmd_crosscheck, "datacheck": cmd_datacheck, "replay": cmd_replay,
            "reset-halt": cmd_reset_halt}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
