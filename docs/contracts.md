# Module contracts (workers MUST follow these exactly)

Common rules
- Python 3.12+, type hints, `from __future__ import annotations`. Package root: `src/nsebot/`.
- Venv: `source /home/user/nse-paper-bot/.venv/bin/activate`. Run tests: `pytest tests/<your_file> -q`.
- TDD: write the failing test file FIRST, run it (see it fail), then implement until green.
- Tests are offline: no network, no API keys, synthetic data only.
- Config models live in `nsebot.config` (CostCfg, RiskCfg, BaselineCfg, LlmCfg, Config). Do not edit config.py.
- `Signal` lives in `nsebot.strategies.base`. Do not edit it.
- Price DataFrame convention ("bars"): DatetimeIndex named `date`, tz-naive, ascending, unique;
  float columns `open, high, low, close, volume` (lowercase).
- Never print or log API keys. Read keys only from `os.environ[cfg.api_key_env]`.
- Do NOT git commit. Do NOT edit files outside your assignment.

## nsebot/costs.py
```python
@dataclass(frozen=True)
class CostBreakdown: brokerage: float; stt: float; exchange: float; sebi: float; gst: float; stamp: float; dp: float; total: float
def apply_slippage(side: Literal["BUY","SELL"], price: float, cfg: CostCfg) -> float   # BUY*(1+s), SELL*(1-s)
def trade_costs(side, qty: int, price: float, cfg: CostCfg) -> CostBreakdown
```
turnover = qty*price. brokerage = min(turnover*brokerage_pct, brokerage_cap). stt = turnover*stt_pct (both sides).
exchange = turnover*exchange_txn_pct. sebi = turnover*sebi_fee_pct. gst = gst_pct*(brokerage+exchange+sebi).
stamp = turnover*stamp_duty_buy_pct on BUY only. dp = dp_charge_per_sell on SELL only (qty>0). qty==0 → all zero.

## nsebot/indicators.py
```python
def sma(s: pd.Series, n: int) -> pd.Series            # rolling mean, NaN until n values
def rsi(close: pd.Series, n: int = 14) -> pd.Series   # Wilder smoothing (ewm alpha=1/n, adjust=False, min_periods=n); all-gains → 100
def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series # Wilder ATR of true range; NaN until warm
```
Value at index t must depend only on data at <= t.

## nsebot/metrics.py
Inputs: `equity: pd.Series` (daily, DatetimeIndex). Returns are simple daily pct changes.
```python
def daily_returns(equity) -> pd.Series
def total_return(equity) -> float
def cagr(equity, periods: int = 252) -> float
def sharpe(returns, rf_annual: float = 0.0, periods: int = 252) -> float   # 0.0 if std==0 or <2 obs
def sortino(returns, rf_annual: float = 0.0, periods: int = 252) -> float  # downside dev of returns<0; 0.0 if none
def max_drawdown(equity) -> float          # positive fraction, e.g. 0.12
def var_cvar(returns, level: float = 0.95) -> tuple[float, float]   # historical, positive loss fractions
def win_rate(pnls: Sequence[float]) -> float  # share of round-trip pnls > 0; 0.0 if empty
def round_trip_pnls(trades: pd.DataFrame) -> list[float]  # trades cols: date,symbol,side,qty,price,costs; FIFO per symbol, pnl net of costs
def summarize(equity, trades: pd.DataFrame | None = None) -> dict  # keys: total_return,cagr,sharpe,sortino,max_drawdown,var95,cvar95,win_rate,n_trades,total_costs
```

## nsebot/data/loader.py
```python
Fetcher = Callable[[str, str, str | None], pd.DataFrame]   # (symbol, start, end) -> raw yfinance-style frame
def yfinance_fetcher(symbol, start, end) -> pd.DataFrame   # yf.download(..., auto_adjust=True, progress=False); flatten MultiIndex cols
def normalize(raw: pd.DataFrame) -> pd.DataFrame   # → bars convention; drop rows with NaN OHLC, non-positive prices, duplicate dates
class PriceLoader:
    def __init__(self, cache_dir: str | Path, fetcher: Fetcher | None = None): ...
    def get(self, symbol: str, start: str, end: str | None = None, as_of: str | pd.Timestamp | None = None,
            refresh: bool = False) -> pd.DataFrame
```
Cache = `{cache_dir}/{symbol}.parquet`. refresh=True fetches only from (last cached date - 5 days) and merges
(new rows win). Result is filtered to [start, end] and then to `<= as_of` (no future rows ever). Holidays simply
have no row (never forward-fill a missing day into a fake bar).

## nsebot/data/calendar.py
```python
def is_weekday(d) -> bool
def has_session(bars: pd.DataFrame, d) -> bool  # True iff a bar exists for date d (data-driven holiday handling)
```

## nsebot/strategies/baseline.py
```python
class BaselineStrategy:
    source = "baseline"
    def __init__(self, cfg: BaselineCfg, atr_stop_mult: float = 2.0): ...
    def generate(self, symbol: str, history: pd.DataFrame, in_position: bool) -> Signal
```
`history` = bars up to and including day t (last row). Uses ONLY history.
- Not enough data (< sma_slow + 2 rows) → HOLD, reason "insufficient history".
- trend = close > SMA_slow and SMA_fast > SMA_slow.
- BUY (only if not in_position): trend and RSI crossed up through rsi_entry: rsi[t-1] < rsi_entry <= rsi[t].
  stop = close - atr_stop_mult*ATR. confidence 0.6.
- SELL (only if in_position): rsi[t] > rsi_exit, OR close < SMA_fast for `exit_below_fast_days` (5) consecutive closes AND SMA_fast[t] < SMA_fast[t-k],
  OR close < SMA_slow (trend broken). confidence 0.6.
- else HOLD (confidence 0.0).
- reason: plain English with numbers, e.g.
  "BUY: uptrend (close 2950.0 > SMA200 2780.1, SMA50 2850.3 > SMA200); RSI(14) recovered 33.1->38.2 through 35; stop 2810.4 (2.0xATR 69.8)".

## nsebot/strategies/llm_agents.py
```python
@dataclass
class LlmDecision: action: Action; confidence: float; raw: str; tokens_in: int; tokens_out: int; cost_usd: float; model: str; called_at_utc: str; cached: bool
class BudgetExceeded(Exception)
def parse_decision(text: str) -> tuple[Action, float]   # map TradingAgents final text → BUY/SELL/HOLD + confidence
class LlmAgentStrategy:
    source = "llm"
    def __init__(self, cfg: LlmCfg, atr_stop_mult: float = 2.0, backend: Callable[[str, str], tuple[str, int, int]] | None = None,
                 today: Callable[[], date] | None = None): ...
    # backend(symbol, date_str) -> (final_decision_text, tokens_in, tokens_out); default = real TradingAgents
    def decide(self, symbol: str, date_str: str) -> LlmDecision        # cache first, then budget check, then backend
    def generate(self, symbol: str, history: pd.DataFrame, in_position: bool) -> Signal
    spent_usd: float  # running total, persisted in {cache_dir}/spend.json
```
Cache file `{cache_dir}/{symbol}/{date}.json`. A cached hit costs 0 and never calls backend.
contaminated = (date < today()). Signal.contaminated set accordingly and reason starts with "[CONTAMINATED] " when true.
BUY while in_position → HOLD; SELL while flat → HOLD. BUY stop = close - atr_stop_mult*ATR(14).
BudgetExceeded if spent_usd >= cfg.budget_usd_total before calling backend; generate() catches it → HOLD "LLM budget exhausted".
