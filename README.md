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
| `nsebot datacheck` | Split/bonus and bad-bar audit for every symbol, written to `reports_out/data_check.md` |
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

## 6. Backtest results (frozen `strategy-v1`, real data, Oct 2015 – Nov 2026)

Each period is an independent run starting with fresh ₹1,00,000. All figures are net of costs and slippage. "Days invested" is the share of sessions with any open position.

| Period | Strategy | Return | Days invested | Trades | Costs | Max DD |
|---|---|---|---|---|---|---|
| Train 2015-10 → 2020-12 | Baseline | **−10.2%** | 25.3% | 60 | ₹1,586 | 10.5% (kill-switch) |
| | NIFTYBEES buy-and-hold | +81.1% | 99.9% | 1 | ₹117 | 36.1% |
| Val 2021 → 2022 | Baseline | +2.3% | 41.5% | 19 | ₹523 | 3.0% |
| | NIFTYBEES buy-and-hold | +31.2% | 99.8% | 1 | ₹118 | 16.1% |
| Test 2023 → 2026-11 | Baseline | **−4.4%** | 38.5% | 64 | ₹1,841 | 6.7% |
| | NIFTYBEES buy-and-hold | +43.3% | 99.9% | 1 | ₹118 | 15.2% |

**Verdict:** the baseline loses to buy-and-hold in every period. It is in the market only 25–42% of the time, wins 17–44% of its trades, and its edge before costs is roughly zero. The risk controls work. Return is the problem.

### What the kill-switch did in the train period

- Equity peaked at ₹100,345 in **Oct 2016**, then drifted down for four years.
- The COVID crash took it to a 9.8% drawdown on 6 Mar 2020, just under the 10% limit.
- The limit was hit on **2 Nov 2020** (−10.5%). The book was already flat, so nothing needed selling.
- It then **sat 100% in cash for the last 40 sessions to 31 Dec 2020**: equity stayed at exactly ₹89,799, with zero trades.
- In the continuous "full" run (in `reports_out/backtest_report.md`), the same halt leaves the book in cash from Nov 2020 to Nov 2026. That's why its days-invested figure is only 11.9%, and why the per-period independent runs are the fairer view.

### Data check (`nsebot datacheck`, all 5 stocks + NIFTYBEES)

- **Splits and bonuses Yahoo has on record since 2015:**
  - RELIANCE 2017 and 2024 (1:1)
  - TCS 2018
  - HDFCBANK 2019 (split) and 2025 (bonus)
  - INFY 2015 and 2018
  - ICICIBANK 2017 (1:10 bonus)
- **Adjusted prices are clean around every event.** The largest move within ±3 sessions is 8.9% (HDFCBANK, Sep 2019, the corporate tax-cut rally, a real move).
- **Compared against NSE's own (unadjusted) prices,** each event shows **exactly one** return mismatch, on the event day itself, as expected. Two extra ~1.1% differences on HDFCBANK in Sep 2019 come from how each source defines the closing price.
- **Moves over 15%:** INFY 22 Oct 2019 (−16.2%, whistleblower news) and ICICIBANK 23 Mar 2020 (−17.8%, COVID). Both are real.
- **Only bad data found:** two NIFTYBEES rows (19–20 Dec 2019) in pre-split units. The spike filter holds them out.
- **Bug found by this check:** jugaad-data returns IST dates as UTC 18:30 of the previous day, which shifted every NSE bar one day early. Fixed and covered by a regression test.

### How prices are used

- **Signals and mark-to-market** use the cleaned series. The spike filter is *causal*: whether a bar is kept depends only on that bar and earlier ones, and a test checks this.
- **Fills and stop checks** use the **raw** open, high and low, so a real crash triggers the stop the same day.
- **Trade-off:** a data glitch in the raw feed could cause a fake stop-out. `nsebot crosscheck` is the guard against that.

### Strategy freeze

The rules were frozen at git tag **`strategy-v1`**. `tests/test_frozen_strategy.py` fails if the strategy code or its rule/risk parameters change. Any future change needs a new version tag and a disclosure here.

### Strategy changes before the freeze (full disclosure)

The baseline rules were changed **twice after seeing backtest output**. These were logic fixes, not parameter tuning, but they mean the val/test numbers are **not pristine out-of-sample evidence**:

1. **v0 (as planned):** exit after 2 closes below SMA50. This caused buy-today/sell-tomorrow churn. Before the first backtest it was changed to 5 closes plus a falling SMA50.
2. **v0.1** (5 closes): **−4.6%** over the full period, 170 trades. Gross P&L before costs was +₹156, so costs (₹4,774) caused the whole loss. Fix: never BUY on a bar where the SELL rule would fire.
3. **v1 (frozen):** the SMA50 exit is off (`exit_below_fast_days: 0`). Exits are RSI > 70, close < SMA200, or the 2×ATR stop.

### Engine fixes from code review (no rule changes)

- The 1% risk and 20% cap are now re-checked at the actual fill price.
- The spike filter was made causal, and the cache now stores raw data.
- Fills and stops use raw prices.
- A BUY that can't fill at the next session's open expires.
- These moved the train kill-switch from 6 Mar 2020 to 2 Nov 2020, and changed returns by under 0.5 percentage points.

**Halting is per book.** A halted book places no new BUYs and only sells until flat (tested). Other books keep running.

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
