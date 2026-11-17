# Plan: Honest AI Paper-Trading Bot (NSE)

Status: **DRAFT — waiting for "go"**. Nothing is built until approved.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Universe | RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS; benchmark NIFTYBEES.NS |
| Product | Long-only **delivery (CNC)**. No shorting, no intraday, no leverage |
| Capital | ₹1,00,000 virtual |
| LLM | TradingAgents with **Google Gemini Flash** (model name in config; key from `GOOGLE_API_KEY` only) |
| Code home | New GitHub repo (name TBD by you), work on branch `feat/paper-bot` |
| Daily run | Your VPS, cron at 16:30 IST, Mon–Fri, skips NSE holidays |
| LLM backtest | ~20 dates × 5 stocks, hard cap **$5**, labelled **CONTAMINATED** |
| Live trading | **Not built.** Any request triggers the SEBI retail-algo reminder (broker-approved API, static IP, Algo-ID) |

## Layout

```
nse-paper-bot/
  config.yaml
  pyproject.toml            # Python >=3.12, pytest, ruff
  src/nsebot/
    config.py               # load + validate config.yaml (pydantic)
    data/loader.py          # yfinance -> parquet cache, holiday/missing-day handling
    data/calendar.py        # NSE trading days (from data, plus holiday list for "is today a session?")
    strategies/base.py      # Signal dataclass: date, symbol, action, confidence, stop, reason, source
    strategies/baseline.py  # SMA50/200 trend + RSI pullback
    strategies/llm_agents.py# TradingAgents wrapper, disk cache, cost tracking, budget cap
    costs.py                # Indian delivery cost model
    risk/manager.py         # sizing, stops, daily-loss halt, drawdown kill-switch
    backtest/engine.py      # event-driven, next-bar-open fills
    backtest/splits.py      # train/val/test + walk-forward windows
    metrics.py              # return, CAGR, Sharpe, Sortino, max DD, VaR/CVaR 95%, win rate
    paper/runner.py         # daily run; portfolio.json, trades.csv, equity.csv
    reports/daily.py, reports/weekly.py, reports/backtest.py
    cli.py                  # `nsebot daily | backtest | report-weekly | reset-halt`
  tests/                    # all offline, synthetic data, no network, no API keys
  state/                    # portfolio.json, trades.csv, equity.csv, HALTED (gitignored)
  cache/                    # parquet + LLM json outputs (gitignored)
  reports_out/              # generated reports
  scripts/cron.example, scripts/run_daily.sh
```

## Key design rules

### No look-ahead (enforced by tests)
- A strategy gets `history = bars[:t]` (up to and including the close at day *t*) and never sees the full frame.
- Orders made at the close of day *t* fill at the **open of day t+1**, plus slippage.
- Test: the signal for day *t* is the same whether or not bars after *t* exist.
- yfinance `auto_adjust=True` uses split/dividend-adjusted prices. This is a known minor bias, documented in the README.

### Baseline strategy (designed up front, then frozen; no curve fitting)
- **Trend filter:** close > SMA200 **and** SMA50 > SMA200.
- **Entry:** trend filter on, and RSI(14) crosses back up through 35 (a pullback that is now recovering).
- **Exit:** RSI(14) > 70, **or** close < SMA50 for 2 consecutive closes, **or** stop hit.
- **Stop:** entry − 2 × ATR(14), fixed at entry.
- Every signal carries text like: "BUY: uptrend (close 2,950 > SMA200 2,780, SMA50 > SMA200); RSI pullback recovered 33→38; stop 2,810 (2×ATR)".
- Parameters stay fixed. The backtest reports train/val/test periods separately so you can see whether results hold up out of sample. No parameter search unless you ask for one, and then only on train, picked on val, and reported once on test.

### Cost model (delivery; every number configurable; verify against your broker)
| Item | Default |
|---|---|
| Brokerage | ₹0 (discount broker delivery); optional % with ₹20 cap |
| STT | 0.1% buy + 0.1% sell |
| NSE exchange txn | 0.00297% |
| SEBI fee | ₹10 / crore |
| GST | 18% on (brokerage + exchange + SEBI) |
| Stamp duty | 0.015% buy only |
| DP charge | ₹15.93 per scrip per sell day |
| Slippage | 0.05% each side (range 0.05–0.10%) |

### Risk manager
- Size = min(1% of equity ÷ (entry − stop), 20% of equity ÷ entry, available cash), rounded down to whole shares. Skip the trade if size is 0.
- Every position has a stop. On the next bar: if it opens below the stop, fill at the open (gap); if the low touches the stop, fill at the stop minus slippage.
- **Daily loss limit 2%:** if mark-to-market equity drops ≥2% vs the previous close, no new entries for the next session. Exits and stops still run.
- **Kill-switch 10%:** if drawdown from peak equity reaches ≥10%, close all positions at the next open and write `state/HALTED`. The bot refuses to run until `nsebot reset-halt --confirm`.
- The backtest and paper runner use the **same** RiskManager and fill code, so the backtest and the paper bot behave the same way.

### LLM wrapper (TradingAgents)
- Install from the TauricResearch/TradingAgents git repo, pinned to a commit. Configure for Google/Gemini Flash with the minimum debate rounds.
- One call per (symbol, date). Cache path: `cache/llm/{symbol}/{date}.json`, holding the raw output, the parsed decision, a UTC timestamp, the model, tokens and cost in USD.
- Map the output to BUY/SELL/HOLD plus a confidence score in [0, 1]. The confidence is parsed from the text if present, otherwise derived from how firm the decision wording is. The rule is documented.
- LLM BUY uses the same stop rule (2×ATR) and the same risk manager as the baseline.
- Cost is tracked from token-usage callbacks × a price table in config. A **budget guard** stops making calls once the cap is reached.
- Any LLM result for a date earlier than the date the call was made is stamped `CONTAMINATED` in the cache, the trades and the reports.
- Unit tests use a fake TradingAgents object, so no key and no network are needed.

### Paper runner (`nsebot daily`)
1. Exit if today isn't an NSE session or `HALTED` exists.
2. Refresh data up to today's close.
3. Fill yesterday's pending orders at **today's open** (simulated; recorded with today's date). Check stops against today's bar.
4. Update equity; check daily-loss and drawdown limits.
5. Make signals from both sources (baseline, and LLM if a key is set and budget remains) and size them. Save them as pending orders for tomorrow's open.
6. Write `portfolio.json`, append to `trades.csv` (date, symbol, side, qty, price, costs, reason, signal_source), append to `equity.csv`, write the daily summary.
- Three separate virtual books, each with ₹1L: `baseline`, `llm`, and `buyhold_niftybees`. This gives a fair comparison side by side.
- Only fills that have actually happened are reported. Pending orders are shown as pending.

### Reports
- **Daily:** positions, pending orders, today's fills, equity per book, risk status.
- **Weekly:** AI vs baseline vs NIFTYBEES buy-and-hold: return, Sharpe, max DD, win rate, total costs, LLM API spend (USD and ₹), and **AI net of API spend**. Forward-only data. Says "insufficient data" until there are at least 20 sessions.
- **Backtest:** baseline (clean) per split plus the full period, and the equity curve PNG. The LLM sample section carries a large CONTAMINATED banner.

## Tasks (TDD: failing test → code → green → commit)

1. Scaffold: pyproject, package, config.yaml + pydantic validation, pytest/ruff, .gitignore. *Test:* config loads; bad values are rejected.
2. Cost model. *Test:* known buy/sell examples worked out by hand; GST base; stamp on buy only.
3. Data loader + cache, using a fake yfinance in tests. *Test:* cache round-trip; missing days/NaNs dropped; holidays absent; no rows after `as_of`.
4. Indicators (SMA, RSI-Wilder, ATR). *Test:* match reference values; no future values used.
5. Baseline strategy. *Test:* each rule fires and stays silent on crafted series; reason text present; **look-ahead invariance test**.
6. Metrics. *Test:* Sharpe/Sortino/DD/VaR/CVaR on known series.
7. Risk manager. *Test:* sizing caps (1%, 20%, cash); gap-through-stop fill; daily-loss halt; kill-switch flattens and persists HALTED.
8. Backtest engine + splits. *Test:* next-open fills; costs deducted; equity accounting identity (cash + positions = equity); walk-forward windows don't overlap.
9. LLM wrapper with a fake TA backend. *Test:* output mapping; cache hit = no call; budget guard; CONTAMINATED stamp; key never appears in logs/cache.
10. Paper runner. *Test:* a simulated 3-day sequence (signal → next-open fill → stop-out) with a frozen clock; state files are correct; restart-safe (running twice on the same day does nothing).
11. Reports + CLI. *Test:* the weekly report computes from fixture logs and shows nothing it can't support from them.
12. Real-data run (needs Yahoo network access): baseline backtest 2015→today, with the report.
13. Real LLM sample (needs `GOOGLE_API_KEY`): ~20 dates × 5 stocks, $5 cap, report marked CONTAMINATED.
14. README, cron + Task Scheduler instructions, `scripts/run_daily.sh`.
15. Self code review of risk/backtest/runner, fix findings, full test run + backtest, push branch.

## Things I need from you during the build
- **GitHub repo name** (create it empty; I'll push `feat/paper-bot`).
- Approve **network access** for Yahoo Finance hosts (prompt will appear) — needed for task 12.
- **Gemini API key** via the secure credential form (never in chat) — needed for task 13. TradingAgents may also need news/data hosts allowed; I'll ask if blocked.

## Known limitations (will be in README)
- Daily bars only. Stops are simulated against the next day's high/low, not tick data.
- yfinance is unofficial and can break or have gaps.
- The LLM historical sample is small and contaminated, so it is not evidence of skill.
- Forward paper results need months before they mean anything statistically.
