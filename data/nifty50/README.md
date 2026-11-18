# Nifty 50 point-in-time membership

- `current_2026-11-17.csv`: official NSE list (nsearchives.nseindia.com/content/indices/ind_nifty50list.csv), downloaded 2026-11-17.
- `changes.csv`: every NIFTY 50 constituent change since 2014, one row per stock, each with the official NSE Indices
  press release it comes from. "Effective date" is the first session the change applies to (NSE: "effective from D,
  close of D-1"). Symbols are exactly as printed in the release (e.g. MCDOWELL-N, INFRATEL, ZOMATO, LTIM).
- Rebuilt by starting from the current list and undoing changes backwards. Exactly 50 members at every step except
  2016-04-01 to 2017-09-28, when Tata Motors DVR (TATAMTRDVR) was a 51st security, as the official releases show.
- Cross-checked against Wikipedia's "Index changes" table: all 25 change dates match; Wikipedia omits the DVR period.
- NOT included (by decision): temporary demerger inclusions that sat in the index for days/weeks
  (Jio Financial Jul-Sep 2023, ITC Hotels Jan-Feb 2025, TML Commercial Vehicles Oct-Nov 2025).
