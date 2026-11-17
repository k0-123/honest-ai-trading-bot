# Honest AI Paper-Trading Bot (NSE)

A **paper-trading** bot (virtual money only, no broker connection) for 5 liquid NSE stocks. It compares:

| Book | What it does |
|---|---|
| `baseline` | Rule-based: 50/200-day trend filter plus an RSI(14) pullback entry, with a 2×ATR stop |
| `llm` | [TradingAgents](https://github.com/TauricResearch/TradingAgents) multi-agent LLM (Gemini Flash), optional |
| `buyhold_niftybees` | Passive benchmark: buy NIFTYBEES once and hold |

Each book starts with ₹1,00,000 of virtual cash. All results are **net of Indian delivery costs and slippage**. LLM results are also shown **net of API spend**.

> **Success here means honest measurement, not profit.** So far the baseline does **not** beat buy-and-hold (see below).

---

## 1. Setup (VPS / Linux; Python ≥ 3.12)

```bash
git clone git@github.com:k0-123/honest-ai-trading-bot.git
cd honest-ai-trading-bot
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # core + tests
.venv/bin/pip install -e ".[llm]"          # optional: TradingAgents for the LLM book
.venv/bin/pytest                           # all tests run offline
```

API keys come **only from environment variables**. Put them in a private file that is never committed:

```bash
printf 'GOOGLE_API_KEY=your-key-here\n' > ~/.nsebot.env && chmod 600 ~/.nsebot.env
```

If `GOOGLE_API_KEY` is missing, the LLM book is skipped and everything else still runs.

## 2. Commands

| Command | What it does |
|---|---|
| `nsebot daily` | **Daily paper run** (after market close). Add `--no-llm` to skip the LLM; `--weekly` forces the weekly report |
| `nsebot backtest` | Baseline vs buy-and-hold backtest, written to `reports_out/backtest_report.md` |
| `nsebot backtest --llm-days 20` | Also runs a small **CONTAMINATED** LLM sample on the last 20 sessions (budget-capped) |
| `nsebot report-weekly` | Weekly comparison from the paper logs |
| `nsebot crosscheck` | Compares Yahoo returns with NSE's own data (jugaad-data) |
| `nsebot reset-halt --book baseline --confirm` | Restarts a book after the kill-switch has halted it |
| `nsebot live` | Explains why live trading is not built (SEBI rules) |

### One-line daily run

```bash
./scripts/run_daily.sh
```

## 3. Scheduling at ~4:30 PM IST

**Linux/VPS (cron):** run `crontab -e` and add **one** of these lines:

```cron
# server clock set to IST
30 16 * * 1-5  /home/YOU/honest-ai-trading-bot/scripts/run_daily.sh
# server clock in UTC (16:30 IST = 11:00 UTC)
0 11 * * 1-5   /home/YOU/honest-ai-trading-bot/scripts/run_daily.sh
```

Check the server clock with `timedatectl`. Logs are written to `logs/daily_YYYY-MM.log`.

**Windows Task Scheduler:** Create Basic Task → Daily, 16:30, Mon–Fri → Start a program:
- Program: `C:\path\to\honest-ai-trading-bot\.venv\Scripts\nsebot.exe`
- Arguments: `daily`
- Start in: `C:\path\to\honest-ai-trading-bot`

Set `GOOGLE_API_KEY` as a *user* environment variable (System → Environment Variables), not in the task.

**Safe by design:**
- Holidays and weekends are skipped automatically (no new bar means nothing happens).
- A second run on the same day does nothing.
- If the VPS was down, the next run catches up the missed sessions in order.
- A run before 15:45 IST only processes up to yesterday, because today's daily bar isn't complete.

## 4. How it works (and why it is honest)

- **No look-ahead.** A decision at day *t*'s close uses only bars up to *t*, and the order fills at day *t+1*'s **open** with slippage. Tests check that adding future data never changes a past signal or a past backtest result.
- **The backtest and the paper bot use the same simulator** (`src/nsebot/sim.py`), so they can't drift apart.
- **Costs (delivery, `config.yaml`):** brokerage (default ₹0, the usual delivery rate at discount brokers), STT 0.1% on buy and sell, NSE exchange fee 0.00297%, SEBI fee ₹10/crore, GST 18% on brokerage + exchange + SEBI fees, stamp duty 0.015% on buys, DP charge ₹15.93 per sell, slippage 0.05% each side. **Check these against your broker's current charge sheet.**
- **Risk rules:**
  - Each trade risks at most 1% of equity, and the stop is set when the order is placed.
  - At most 20% of equity goes into any one stock.
  - A loss of 2% or more in a day blocks new entries for the next session.
  - A drawdown of **10% or more triggers the kill-switch**: the book sells everything at the next open, `state/HALTED_<book>` is written, and the book stops until you run `reset-halt`.
  - The NIFTYBEES benchmark is exempt from all of these.
- **LLM honesty:**
  - Any LLM decision for a date before the day the model was called is labelled `[CONTAMINATED]`, because the model may already "know" what happened.
  - Every LLM output is cached with a timestamp, the model name, tokens and cost.
  - A hard budget cap stops calls once it's reached. If token usage can't be read, each call is charged a pessimistic estimate so the cap still holds.
- **Data:**
  - Yahoo Finance (adjusted prices) is cached as parquet.
  - Short "spike and revert" glitches are dropped and logged. One real example: two NIFTYBEES rows in Dec 2019 were in pre-split units.
  - `nsebot crosscheck` compares daily returns against NSE's own data via jugaad-data. NSE often blocks datacenter IPs; if it fails on your VPS, the check is skipped.

## 5. Reading the outputs

| File | Contents |
|---|---|
| `state/trades.csv` | Every simulated fill: `date, symbol, side, qty, price, costs, reason, signal_source, contaminated, book` |
| `state/signals.csv` | Every daily signal with its plain-English reason, including HOLDs |
| `state/equity.csv` | Equity at each close, one column per book |
| `state/portfolio.json` | Full state: cash, positions, stops, pending orders, halt status |
| `reports_out/daily/YYYY-MM-DD.md` | Positions, **PENDING** orders (not yet filled), fills, signals |
| `reports_out/weekly/weekly_*.md` | AI vs baseline vs buy-and-hold: return, Sharpe, max drawdown, win rate, costs, LLM spend, AI net of spend |
| `reports_out/backtest_report.md` | Backtest tables for train/val/test/full, with an equity curve PNG |

The weekly report shows "Insufficient data" until there are at least 20 sessions. Meaningful conclusions need **months** of forward data.

## 6. Backtest results (baseline, real data, Oct 2015 – Nov 2026)

These are independent runs with fresh capital in each period. All figures are net of costs and slippage.

| Period | Baseline | Max DD | Trades | Buy-and-hold NIFTYBEES | Max DD |
|---|---|---|---|---|---|
| Train 2015-10 → 2020-12 | **−10.2%** (kill-switch on 2020-03-06) | 10.6% | 58 | +81.1% | 36.1% |
| Val 2021 → 2022 | +1.9% | 3.4% | 19 | +31.2% | 16.1% |
| Test 2023 → 2026-11 | **−4.3%** | 6.8% | 64 | +43.3% | 15.2% |

**Verdict:** the baseline is low-risk but clearly loses to buy-and-hold. It sits in cash most of the time, and its win rate is low (17–44%). The drawdown control works as designed: the kill-switch caught the COVID crash.

### Strategy changes (full disclosure)

The baseline rules were changed **twice after seeing backtest output**. These were logic fixes, not parameter tuning, but they mean the val/test numbers are **no longer pristine out-of-sample evidence**:

1. **v0 (as planned):** exit after 2 closes below SMA50. This caused buy-today/sell-tomorrow churn. Before the first backtest it was changed to 5 closes plus a falling SMA50.
2. **v0.1** (5 closes): **−4.6%** over the full period, 170 trades. Gross P&L before costs was only +₹156, so costs (₹4,774) caused the whole loss. Cause: the entry and exit rules overlapped. Fix: never BUY on a bar where the SELL rule would fire.
3. **v1 (current):** RSI dips almost always sit under a flattening SMA50, so the SMA50 exit is **off by default** (`exit_below_fast_days: 0`). Exits are now RSI > 70, close < SMA200, or the stop.

### LLM backtest

Not run yet (needs `GOOGLE_API_KEY`). Any LLM backtest on historical dates is **CONTAMINATED** by design and is not evidence of skill. **Only forward paper-trading results count.**

## 7. Not included (on purpose)

- **Live trading.** Under SEBI's retail algo framework (in force from April 2026), retail algo orders need a **broker-approved API, a registered static IP and an exchange Algo-ID**. Nothing here places real orders. Ask explicitly before any live work starts.
- News and FinBERT sentiment are planned for later.

## 8. Known limitations

- Daily bars only. Stops are simulated against the next day's high/low (gaps fill at the open).
- yfinance is unofficial; it can break, revise data or have gaps.
- Survivorship bias: today's large caps were chosen with hindsight.
- TradingAgents is US-centric (its data tools target US markets). Its NSE analysis quality is unproven.
- None of this is investment advice.
