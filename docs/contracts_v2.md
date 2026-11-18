# strategy-v2 contracts (workers MUST follow exactly)

Common rules: same as docs/contracts.md (TDD, offline tests, no network in tests, no git commits, don't edit
files outside your assignment, venv at /home/user/nse-paper-bot/.venv). Do NOT touch src/nsebot/strategies/baseline.py,
tests/test_frozen_strategy.py, or any v1 behaviour.

## src/nsebot/universe/nifty50.py  (+ src/nsebot/universe/__init__.py)

Data: `data/nifty50/current_2026-11-17.csv` (cols symbol, company) and `data/nifty50/changes.csv`
(cols effective_date, action IN|OUT, symbol, source). Read data/nifty50/README.md.

```python
SNAPSHOT_DATE = "2026-11-17"
# official symbol in releases -> symbol used for prices today (same company, renamed)
CANONICAL = {"MCDOWELL-N": "UNITDSPR", "INFRATEL": "INDUSTOWER", "IBULHSGFIN": "SAMMAANCAP",
             "ZOMATO": "ETERNAL", "TATAMOTORS": "TMPV", "LTIM": "LTM"}
# no Yahoo history: prices come from NSE via jugaad-data (EQ series only)
NSE_SOURCED = {"HDFC.NS", "CAIRN.NS", "TATAMTRDVR.NS", "IDFC.NS"}

def price_key(symbol: str) -> str            # CANONICAL.get(symbol, symbol) + ".NS"

class MissingPriceData(Exception)

class Nifty50Universe:
    def __init__(self, data_dir: str | Path = "data/nifty50"): ...
    def members_on(self, date) -> frozenset[str]
        # price keys of the index on `date`: start from the snapshot and UNDO every change whose
        # effective_date > date (undo = remove IN symbols, add back OUT symbols). Changes with
        # effective_date <= date are applied. date > SNAPSHOT_DATE -> snapshot. Uses only the data files.
    def ever_members(self, start, end) -> frozenset[str]   # union of members_on over [start, end] (changes only move on effective dates)
    def membership_spans(self, start, end) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]  # inclusive spans per key, clipped to [start,end]

class UniversePriceLoader:
    """Same .get() signature as PriceLoader; routes NSE_SOURCED keys to the NSE loader, everything else to Yahoo."""
    def __init__(self, yahoo_loader, nse_loader): ...
    def get(self, symbol, start, end=None, as_of=None, refresh=False, clean=True) -> pd.DataFrame

def check_prices(universe: Nifty50Universe, loader, start: str, end: str, tolerance_days: int = 7) -> None
    # For every key and every membership span within [start,end]: loader.get(key, start-400 days, end, clean=False)
    # must be non-empty, its first bar <= span_start + tolerance and its last bar >= span_end - tolerance.
    # Otherwise raise MissingPriceData listing every offending key with its span. Never skip silently.
```

## src/nsebot/data/crosscheck.py: bug fix in jugaad_fetcher
NSE returns other series (e.g. HDFC warrants "W2") mixed into the EQ request. After lower-casing columns, keep only
rows with `series == "EQ"` (when a `series` column exists) BEFORE building the index. Add a regression test with a
fake stock_df that mixes EQ and W2 rows on the same dates.

## Tests (write FIRST): tests/test_universe.py
- members_on gives exactly 50 keys on many sample dates, and 51 between 2016-04-01 and 2017-09-28 (DVR period).
- For EVERY row in changes.csv on date D: an IN key is absent on D-1 calendar day and present on D; an OUT key is
  present on D-1 and absent on D. (This proves only constituents known on that date are used.)
- Specific: HDFC.NS member on 2023-07-12 not 2023-07-13; LTM.NS member 2023-07-13..2024-09-29 only;
  TATAMTRDVR.NS member 2016-06-01, not 2017-10-03; BSE.NS not on 2026-09-29, yes on 2026-09-30;
  JIOFIN.NS not a member on 2023-08-01 (temporary demerger inclusion excluded); ETERNAL.NS member 2025-03-28.
- members_on(today-ish date after snapshot) == snapshot.
- UniversePriceLoader routes HDFC.NS to the NSE loader and RELIANCE.NS to Yahoo (fake loaders recording calls).
- check_prices: passes on complete fake data; raises MissingPriceData naming the key when a member's data is
  missing, starts too late, or ends too early.
