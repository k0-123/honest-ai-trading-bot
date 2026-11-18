"""Independent cross-check of the v2b (momentum) backtest with vectorbt.

Layer 1: the v2b rules re-implemented from the written spec, WITHOUT importing nsebot.sim, nsebot.sim_momentum or
         nsebot.strategies.momentum. It produces a list of share orders (date, key, signed qty).
Layer 2: vectorbt.Portfolio.from_orders turns those orders into cash/fees/equity with its own engine, using the same
         open-price fills, slippage and Indian delivery costs (buy %, sell %, fixed DP charge on sells).

Compares (a) layer-1 trades vs the simulator's trades (rule logic) and (b) vectorbt equity vs simulator equity
(accounting + costs), per period. Usage:  python scripts/crosscheck_v2b_vectorbt.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import vectorbt as vbt

from nsebot.cli import _run_v2b, _v2b_inputs
from nsebot.config import load_config


# ------------------------------------------------------------------ costs (from config, not from nsebot.costs)
def cost_rates(c):
    exch_sebi = c.exchange_txn_pct + c.sebi_fee_pct
    buy = c.stt_pct + exch_sebi + c.gst_pct * exch_sebi + c.stamp_duty_buy_pct   # brokerage 0 in config
    sell = c.stt_pct + exch_sebi + c.gst_pct * exch_sebi
    assert c.brokerage_pct == 0, "cross-check assumes zero brokerage (the config default)"
    return buy, sell, c.dp_charge_per_sell


def cost(side, qty, px, rates):
    buy, sell, dp = rates
    return qty * px * buy if side > 0 else qty * px * sell + (dp if qty > 0 else 0.0)


def wilder_atr(h: pd.DataFrame, n=14) -> float:
    pc = h["close"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(), (h["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().iloc[-1])


# ------------------------------------------------------------------ layer 1: rules from the spec
def layer1_orders(cfg, uni, P, R, start, end, kill_switch=True):
    c, r = cfg.costs, cfg.risk
    rates = cost_rates(c)
    slip = c.slippage_pct
    keys = [k for k in P if k != cfg.benchmark]
    cal = sorted(set().union(*[R[k].index for k in keys + [cfg.benchmark]]))
    cal = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    nxt = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    cash, pos, last_px = cfg.capital, {}, {}          # pos: key -> [qty, stop]
    peak, prev_eq, halted, block = 0.0, None, False, False
    pending, orders, trades, equity = [], [], [], {}

    def eq_now():
        return cash + sum(q * last_px[k] for k, (q, _) in pos.items())

    for d in cal:
        # ---- open: sells first, then buys
        for o in sorted(pending, key=lambda o: o["side"] > 0):
            k, bar = o["key"], R[o["key"]].loc[d] if d in R[o["key"]].index else None
            if bar is None:
                if o["side"] < 0:
                    o["carry"] = True
                continue
            if o["side"] < 0 and k in pos:
                q = pos.pop(k)[0]; px = bar["open"] * (1 - slip)
                cash += q * px - cost(-1, q, px, rates); last_px[k] = px
                orders.append((d, k, -q)); trades.append((d, k, "SELL", q))
            elif o["side"] > 0 and k not in pos:
                if bar["open"] <= o["stop"]:
                    continue
                px = bar["open"] * (1 + slip)
                e = eq_now()
                q = min(o["qty"], math.floor(min(e * r.risk_per_trade_pct / (px - o["stop"]),
                                                 e * r.max_position_pct / px, max(cash, 0) / (px * 1.003))))
                while q > 0 and q * px + cost(1, q, px, rates) > cash:
                    q -= 1
                if q <= 0:
                    continue
                cash -= q * px + cost(1, q, px, rates); pos[k] = [q, o["stop"]]; last_px[k] = px
                orders.append((d, k, q)); trades.append((d, k, "BUY", q))
        pending = [o for o in pending if o.get("carry")]
        for o in pending:
            o.pop("carry", None)
        # ---- close: mark, breakers, stops
        for k in list(last_px):
            if d in P[k].index:
                last_px[k] = float(P[k].loc[d, "close"])
        e = eq_now(); peak = max(peak, e); equity[d] = e
        pe = prev_eq if prev_eq else e; prev_eq = e
        if halted:
            continue
        if kill_switch and 1 - e / peak >= r.max_drawdown_pct - 1e-9:
            halted = True
            pending = [{"key": k, "side": -1} for k in pos]
            continue
        block = (e / pe - 1) <= -r.daily_loss_limit_pct + 1e-9
        for k, (q, stp) in pos.items():
            if d in R[k].index and min(R[k].loc[d, "low"], R[k].loc[d, "open"]) <= stp and \
                    not any(o["key"] == k and o["side"] < 0 for o in pending):
                pending.append({"key": k, "side": -1})
        for k in list(pos):  # data ended (merger/delisting): exit at last close
            if (d - P[k].index[-1]).days >= 7 and P[k].index[-1] < d:
                q = pos.pop(k)[0]; px = float(P[k]["close"].iloc[-1]) * (1 - slip)
                cash += q * px - cost(-1, q, px, rates)
                orders.append((d, k, -q)); trades.append((d, k, "SELL", q))
                pending = [o for o in pending if o["key"] != k]
        # ---- month-end decision: next session in a new month
        if d not in nxt or nxt[d].month == d.month:
            continue
        mem = uni.members_on(d)
        sc = {}
        for k in mem:
            s = P.get(k)
            if s is None:
                continue
            cl = s["close"].loc[:d]
            if len(cl) >= 253 and cl.index[-1] == d:
                sc[k] = cl.iloc[-22] / cl.iloc[-253] - 1
        ranked = sorted(sc, key=lambda k: (-sc[k], k))
        rank = {k: i + 1 for i, k in enumerate(ranked)}
        sells = [k for k in pos if rank.get(k, 999) > 20]
        keep = set(pos) - set(sells)
        buys = [k for k in ranked if k not in keep][: max(10 - len(keep), 0)]
        est = cash
        for k in sells:
            est += pos[k][0] * last_px[k] * 0.997
            if not any(o["key"] == k and o["side"] < 0 for o in pending):
                pending.append({"key": k, "side": -1})
        nb = P[cfg.benchmark]["close"].loc[:d]
        if len(nb) >= 200 and nb.iloc[-1] < nb.tail(200).mean():
            continue                                   # risk-off: hold cash
        if block:
            continue
        for k in buys:
            h = P[k].loc[:d].tail(400)
            cl = float(h["close"].iloc[-1]); stp = cl - r.atr_stop_mult * wilder_atr(h)
            if stp <= 0 or stp >= cl:
                continue
            q = min(math.floor(e * 0.10 / cl), math.floor(min(e * r.risk_per_trade_pct / (cl - stp),
                    e * r.max_position_pct / cl, max(est, 0) / (cl * 1.003))))
            if q > 0:
                est -= q * cl * 1.003
                pending.append({"key": k, "side": 1, "qty": q, "stop": stp})
    return orders, pd.DataFrame(trades, columns=["date", "symbol", "side", "qty"]), pd.Series(equity)


# ------------------------------------------------------------------ layer 2: vectorbt accounting
def vbt_equity(cfg, R, P, orders, start, end):
    if not orders:
        return None
    keys = sorted({k for _, k, _ in orders})
    cal = sorted(set().union(*[R[k].index for k in keys] + [R[cfg.benchmark].index]))
    cal = pd.DatetimeIndex([d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)])
    opn = pd.DataFrame({k: R[k]["open"] for k in keys}).reindex(cal)
    cls = pd.DataFrame({k: P[k]["close"] for k in keys}).reindex(cal).ffill()
    last_close = {k: float(P[k]["close"].iloc[-1]) for k in keys}
    size = pd.DataFrame(0.0, index=cal, columns=keys)
    price = opn.copy()
    for d, k, q in orders:
        size.loc[d, k] += q
        if pd.isna(opn.loc[d, k]):                     # data-ended exit at last close
            price.loc[d, k] = last_close[k]
    buy, sell, dp = cost_rates(cfg.costs)
    fees = pd.DataFrame(np.where(size > 0, buy, np.where(size < 0, sell, 0.0)), index=cal, columns=keys)
    fixed = pd.DataFrame(np.where(size < 0, dp, 0.0), index=cal, columns=keys)
    pf = vbt.Portfolio.from_orders(
        close=cls, size=size.replace(0.0, np.nan), price=price.ffill(), size_type="amount",
        fees=fees, fixed_fees=fixed, slippage=cfg.costs.slippage_pct, init_cash=cfg.capital,
        cash_sharing=True, group_by=True, call_seq="auto", freq="1D")
    return pf


def main() -> int:
    cfg = load_config()
    end = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.today().strftime("%Y-%m-%d")
    periods = {"train": ("2015-10-30", "2020-12-31"), "val": ("2021-01-01", "2022-12-31"), "test": ("2023-01-01", end)}
    uni, P, R = _v2b_inputs(cfg, "2015-10-30", end)
    rows = []
    for ks in (True, False):
        for name, (s, e) in periods.items():
            sim = _run_v2b(cfg, uni, P, R, s, e, "v2b", kill_switch=ks)
            orders, l1, l1eq = layer1_orders(cfg, uni, P, R, s, e, kill_switch=ks)
            pf = vbt_equity(cfg, R, P, orders, s, e)
            vb_ret = float(pf.total_return()) if pf is not None else 0.0
            st = sim.trades[["date", "symbol", "side", "qty"]].copy()
            st["date"] = pd.to_datetime(st["date"])
            m = st.merge(l1, on=["date", "symbol", "side", "qty"], how="outer", indicator=True)
            rows.append({"run": ("v2b" if ks else "v2b DIAGNOSTIC") + f" {name}",
                         "simulator": sim.equity.iloc[-1] / cfg.capital - 1, "layer1": l1eq.iloc[-1] / cfg.capital - 1,
                         "vectorbt": vb_ret, "trades sim": len(st), "trades layer1": len(l1),
                         "trade mismatches": int((m["_merge"] != "both").sum()),
                         "vbt fees": float(pf.orders.records_readable["Fees"].sum()) if pf is not None else 0.0,
                         "sim costs": float(sim.trades["costs"].sum())})
    out = pd.DataFrame(rows)
    out["sim - vbt (pp)"] = (out["simulator"] - out["vectorbt"]) * 100
    pd.set_option("display.width", 220)
    fmt = out.copy()
    for col in ("simulator", "layer1", "vectorbt"):
        fmt[col] = fmt[col].map(lambda x: f"{x:+.2%}")
    print(fmt.round(2).to_string(index=False))
    out.to_csv("reports_out/crosscheck_v2b_vectorbt.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
