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
| `nsebot replay --start 2026-10-01` | Quick test: replays the daily paper run day by day up to today in a separate `replay/` folder (baseline + NIFTYBEES only, no AI cost), with each day seeing only data up to that day. Prints a trades table and the final book values |
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
- **Costs (delivery, `config.yaml`): based on Zerodha's equity delivery charges**, chosen as a typical discount-broker reference (you don't have a broker yet). Brokerage ₹0 on delivery; STT 0.1% on buy and sell; NSE exchange fee 0.00297%; SEBI fee ₹10/crore; GST 18% on brokerage + exchange + SEBI fees; stamp duty 0.015% on buys; Zerodha DP charge ₹15.93 per scrip per sell day (₹13.50 + 18% GST); slippage 0.05% each side. STT, exchange, SEBI, GST and stamp duty are statutory and the same at every broker; brokerage and DP charges differ. **When you open an account, update `costs:` in `config.yaml` from that broker's charge sheet.**
- **Risk rules:**
  - Each trade risks at most 1% of equity, and the stop is set when the order is placed.
  - **Stops are checked after the close, like the real bot.** If a day's low touches or breaks the stop (including a gap below it), the position is sold at the **next session's open**. The bot runs once a day and has no order waiting at a broker, so it can't sell at the stop price itself.
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
| Train 2015-10 → 2020-12 | Baseline | **−8.7%** | 26.0% | 58 | ₹1,552 | 10.5% (kill-switch) |
| | NIFTYBEES buy-and-hold | +81.1% | 99.9% | 1 | ₹117 | 36.1% |
| Val 2021 → 2022 | Baseline | +2.3% | 41.7% | 19 | ₹523 | 3.2% |
| | NIFTYBEES buy-and-hold | +31.2% | 99.8% | 1 | ₹118 | 16.1% |
| Test 2023 → 2026-11 | Baseline | **−4.2%** | 39.1% | 64 | ₹1,846 | 5.6% |
| | NIFTYBEES buy-and-hold | +43.3% | 99.9% | 1 | ₹118 | 15.2% |

**Verdict:** the baseline loses to buy-and-hold in every period. It is in the market only 25–42% of the time, wins 17–44% of its trades, and its edge before costs is roughly zero. The risk controls work. Return is the problem.

### What the kill-switch did in the train period

- Equity peaked at ₹102,026 in **Jun 2019**, then drifted down.
- The COVID crash hit the 10% limit on **6 Mar 2020** (−10.2%). The one open position (ICICIBANK) was sold at the next open on 9 Mar.
- The book then **sat 100% in cash for the remaining 203 sessions to 31 Dec 2020**: equity stayed at exactly ₹91,280, with zero trades.
- In the continuous "full" run (in `reports_out/backtest_report.md`), the same halt leaves the book in cash from Mar 2020 to Nov 2026. That's why its days-invested figure is only 12.2%, and why the per-period independent runs are the fairer view.

### Stops sell at the next open (what that costs)

- Across 23 stop exits in the three periods, the next open was better than the stop price 15 times and worse 8 times. On average it was 0.16% below the stop.
- Worst case: **INFY, 23 Oct 2019**, the day after the whistleblower crash, filled **15.3% below its stop**.
- This is the real risk of a once-a-day bot with no resting stop order. Before this change, the simulator assumed an exit at the stop price during the day, which flattered results. The train kill-switch moved from 2 Nov 2020 back to 6 Mar 2020.
- Live trading could fix this with broker GTT stop orders (see section 7).

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
- **Fills and stop checks** use the **raw** open and low. A real crash is detected on the day it happens (even while the spike filter is holding that bar out) and sold at the next raw open.
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
- **Later engine change:** stop breaches now sell at the next open instead of at the stop price during the day (described above). This moved the train kill-switch back to 6 Mar 2020. Train went from −10.2% to −8.7% and test from −4.4% to −4.2%.

**Halting is per book.** A halted book places no new BUYs and only sells until flat (tested). Other books keep running.

### LLM backtest

Not run yet (needs `GOOGLE_API_KEY`). Any LLM backtest on historical dates is **CONTAMINATED** by design and is not evidence of skill. **Only forward paper-trading results count.**

## 6b. Strategy v2b: Momentum (frozen at tag `strategy-v2b`)

A separate ₹1,00,000 paper book (`v2b_momentum`). v1 is unchanged and keeps running.

**Universe: the point-in-time Nifty 50.** Each date uses only the stocks that were in the index on that date, rebuilt from NSE's official current list plus every NIFTY 50 change in NSE Indices' official press releases since 2014 (`data/nifty50/`, with the source file for each change). The reconstruction:
- has exactly 50 members at every step, except Apr 2016 – Sep 2017, when Tata Motors DVR was a 51st security (Wikipedia's change list misses this);
- leaves out temporary demerger inclusions that lasted only days or weeks (Jio Financial 2023, ITC Hotels 2025, TML Commercial Vehicles 2025);
- uses NSE prices for HDFC Ltd, Cairn India, Tata Motors DVR and IDFC, which have no Yahoo history (not dividend-adjusted, a small bias);
- refuses to run if any member is missing price history.

**Rules (fixed before the first backtest; commit `5e010f5`):**
- **Rebalance:** decided at the close of the last session of each month, filled at the open of the first trading day of the next month.
- **Ranking:** Nifty 50 members by 12-month return excluding the latest month (close 21 sessions ago ÷ close 252 sessions ago − 1).
- **Holdings:** the top 10, about 10% each. A holding is sold only when its rank drops below 20 (or it leaves the index); freed slots go to the best-ranked stocks not held. No trimming of winners.
- **Risk-off:** if NIFTYBEES closes below its 200-day average at the decision, hold **cash**: no new stocks that month, and sells still happen. *(Rule fixed by you before any v2b result was seen; the first draft parked cash in NIFTYBEES.)*
- **Risk and costs:** the same as v1. That means a 2×ATR stop on every position, sold at the next open when breached; sizing of min(10%, the 1% risk rule, the 20% cap); the 2% daily-loss gate; and the **10% kill-switch**. "DIAGNOSTIC" runs remove only the kill-switch to show the strategy itself; they are not a tradable book.

**Honest label:** momentum is a well-known historical effect, and its long-run record is widely published. A backtest like this is **not proof** that it will work from here. Only forward paper trading counts.

### Results (net of costs; a fresh ₹1L in each period; corrected data)

| Period | Book | Return | Days invested | Trades | Costs | Max DD |
|---|---|---|---|---|---|---|
| Train 2015-10 → 2020-12 | v1 baseline | −8.72% | 26.0% | 58 | ₹1,552 | 10.5% |
| | **v2b momentum** | **+3.57%** (kill-switch 2016-12-20) | 11.6% | 64 | ₹1,223 | 10.2% |
| | v2b DIAGNOSTIC | +14.45% | 81.6% | 292 | ₹5,711 | 25.8% |
| | NIFTYBEES buy-and-hold | +81.09% | 99.9% | 1 | ₹117 | 36.1% |
| Val 2021 → 2022 | v1 baseline | +2.32% | 41.7% | 19 | ₹523 | 3.2% |
| | **v2b momentum** | **+31.83%** (kill-switch 2021-11-22) | 40.1% | 44 | ₹854 | 10.7% |
| | v2b DIAGNOSTIC | +22.95% | 96.0% | 89 | ₹1,669 | 25.0% |
| | NIFTYBEES buy-and-hold | +31.20% | 99.8% | 1 | ₹118 | 16.1% |
| Test 2023 → 2026-11 | v1 baseline | −4.16% | 39.1% | 64 | ₹1,846 | 5.6% |
| | **v2b momentum** | **+37.54%** (kill-switch 2024-06-04) | 34.5% | 78 | ₹1,471 | 10.5% |
| | v2b DIAGNOSTIC | +23.75% | 83.9% | 160 | ₹3,165 | 24.1% |
| | NIFTYBEES buy-and-hold | +43.31% | 99.9% | 1 | ₹118 | 15.2% |

**Verdict:**
- v2b beats v1 in every period.
- It does **not** beat buy-and-hold NIFTYBEES in any period.
- The kill-switch fired in all three periods, so most of the "v2b" numbers are a few good months followed by sitting in cash.

### Why DIAGNOSTIC made only +14% in 2015–2020 (NIFTYBEES +81%)

- **Risk-off months:** 14 of 63 rebalances (Oct 2015 – Mar 2016, Dec 2016, Oct 2018, Aug 2019, Feb – Jun 2020). The book was 100% cash on 18% of days and 68% invested on average.
- **2018–19:** −9.3% vs NIFTYBEES +19.8%, while 77% invested. Momentum picks did badly, with 107 stop-outs over the period in total.
- **2020:** it was out at the bottom. All positions were stopped out in the March crash (−10.5% vs NIFTYBEES −29.0% for Q1). It then stayed fully in cash through Apr – Jul 2020 (risk-off decisions Mar – Jun), so it **missed most of the recovery**: Apr – Dec +13.2% vs NIFTYBEES +66.8%. It only re-bought in Aug 2020.

### Why the kill-switch book beat DIAGNOSTIC in 2021–22 and 2023–26

- **Val:** the kill-switch fired on **22 Nov 2021** (−10.2%), and the book was flat at the next open. Afterwards NIFTYBEES ended +5.0% (range −11.5% to +9.0%). DIAGNOSTIC peaked at ₹1,39,127 on 17 Jan 2022, then fell 11.6%, ending −7.2% from the halt.
- **Test:** the kill-switch fired on **4 Jun 2024** (election-results day, −10.5%). Afterwards NIFTYBEES rose +14.8%, while DIAGNOSTIC peaked at ₹1,59,338 on 1 Aug 2024 and then fell 22.3%, ending −8.9% from the halt.
- **So:** the market went up after both halts, but momentum went down. Sitting in cash beat staying in momentum. It did **not** beat simply holding the index.

### Data corrections (full Yahoo-vs-NSE audit)

**The audit:** all 77 Yahoo-priced members plus NIFTYBEES were compared day by day with NSE's own prices over the dates that matter: 154,131 stock-days. Of 335 days where the two disagree by more than 1.5%, 299 are correct Yahoo adjustments (210 dividends, 89 splits/bonuses). Every corporate action since 2014 was also checked against NSE's corporate-action records.

**Real errors corrected, each with evidence in `data/nifty50/`:**
- **73 bar overrides** (`bar_overrides.csv`): stale Yahoo bars that repeat the previous close, replaced with NSE's real bar:
  - **53 stocks on 18 Mar 2025**
  - Diwali Muhurat sessions (2018, 2021, 2022, 2024)
  - a few others
  - the NIFTYBEES Dec 2019 split glitch
- **4 factor adjustments** (`adjustments.csv`):
  - Trent: the 1:2 bonus (ex-date 4 Jun 2026) was adjusted by Yahoo only from 1 Jan 2026.
  - Adani Enterprises: the 3:25 rights issue (ex-date 17 Nov 2025) was adjusted only from 10 Nov.
  - **Unadjusted demergers:** Tata Motors (14 Oct 2025, factor 0.605) and Hindustan Unilever (5 Dec 2025).
- **Known and not corrected:** Britannia's Aug 2019 debenture bonus. NSE's record gives no value, so no factor can be computed from records.

**Effect:**
- v2b changed by at most 0.26 percentage points; the before/after table is in `reports_out/backtest_report.md`, which lists every correction. The same picks were made, with different share counts.
- **v1 re-run on corrected data: identical**, with zero trades different. v1 still uses its original data path.

### Independent cross-check (vectorbt)

`scripts/crosscheck_v2b_vectorbt.py` re-implements the v2b rules from the spec without importing the simulator, and lets vectorbt do the accounting (fills, slippage, buy/sell fees, DP charge).
- **Result:** all 6 runs (3 periods × with/without kill-switch) match **exactly**: identical trades, identical returns, costs within ₹0.01.
- **Limit:** it can't catch a spec misreading shared by both implementations, or errors in the shared data (that's what the audit is for).

## 7. Not included (on purpose)

- **Live trading.** Under SEBI's retail algo framework (in force from April 2026), retail algo orders need a **broker-approved API, a registered static IP and an exchange Algo-ID**. Nothing here places real orders. Ask explicitly before any live work starts.
  - **Stops in live trading could use broker GTT (Good Till Triggered) stop-loss orders**, placed with the broker right after each entry. The stop would then trigger at the exchange during market hours instead of waiting for the next open. Paper mode deliberately does *not* assume this, because nothing like it exists here yet.
- News and FinBERT sentiment are planned for later.

## 8. Known limitations

- Daily bars only. A stop breach is detected from the day's low after the close and sold at the next open, so overnight gaps and crash days can fill well below the stop.
- yfinance is unofficial; it can break, revise data or have gaps.
- Survivorship bias: today's large caps were chosen with hindsight.
- TradingAgents is US-centric (its data tools target US markets). Its NSE analysis quality is unproven.
- None of this is investment advice.
