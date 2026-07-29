"""S1 step 1 -- validate measures.py against hand-verified anchor stock-days.

Two stock-days were computed by hand during preparation and are used here as
mechanical tripwires: 7203 on a 1-yen grid, and 8604 on the 0.1-yen grid where
floating-point digit arithmetic goes wrong. The reference numbers came from an
independent implementation with a slightly looser gate (it required only that
both sides were quoted, not that both quotes were *ordinary*), so the comparison
runs on that same looser gate. The paper-faithful gate is reported alongside, and
the gap between them is itself informative.

Ground truth is the paper's magnitudes, not these numbers -- the anchors catch
mechanical slips (an inverted direction map, a float modulo), while the paper's
15%/11%/+1bp catch conceptual ones.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import measures as M
from tse_tick import read_ticks

OUT = os.path.join(C.RESULTS, "s1_pilot")
RAW = C.RAW_ANCHORS
TOL_PP = 0.3       # percentage points
TOL_COUNT = 25

ANCHORS = [
    {"ticker": "7203", "date": "20240401", "name": "Toyota",
     "tick_yen": 1.0, "n_zaraba": 35318,
     "m0_all": 18.00, "m_b_large0": 15.66, "m_s_large0": 20.72,
     "m_b_small0": 9.95, "m_s_small0": 10.64},
    {"ticker": "8604", "date": "20240201", "name": "Nomura",
     "tick_yen": 0.1, "n_zaraba": 42975,
     "m0_all": 31.58, "m_b_large0": 42.18, "m_s_large0": 22.39,
     "m_b_small0": 11.40, "m_s_small0": 9.92},
]

CELLS = ["m0_all", "m_b_large0", "m_s_large0", "m_b_small0", "m_s_small0"]


def main() -> int:
    tee = C.Tee("s1_step1_anchor_measures")
    fails, payload = [], []
    try:
        print("=== S1 step 1: anchor validation ===\n")
        t500 = C.load_topix500()

        for a in ANCHORS:
            d = dt.datetime.strptime(a["date"], "%Y%m%d").date()
            print(f"--- {a['ticker']} {a['name']} {a['date']} ---")
            t0 = time.perf_counter()
            df = read_ticks(RAW, ticker_filter={a["ticker"]}, date=a["date"], language="en")
            t_read = time.perf_counter() - t0

            # Trade-row discriminators must agree. `Execution Price` is populated on
            # every row, so filtering on it silently keeps the whole file -- the
            # single easiest way to get this wrong.
            n_et = int(df["Execution Type"].is_not_null().sum())
            n_tm = int((df["Execution Time"].is_not_null()
                        & (df["Execution Time"] != "")).sum())
            ok_disc = n_et == n_tm
            print(f"rows {df.height:,}  read {t_read:.1f}s   "
                  f"trade rows: by Execution Type {n_et:,}, by Execution Time {n_tm:,} "
                  f"[{'ok' if ok_disc else 'MISMATCH'}]")
            if not ok_disc:
                fails.append(f"{a['ticker']}: trade-row discriminators disagree")

            t0 = time.perf_counter()
            n = M.normalize_day(df, d, wide=True)
            emp = M.infer_tick10(n)
            px = n.filter(pl.col("is_trade") & (pl.col("Volume") > 0))
            tbl = C.day_tick_constant10(int(round(float(px["Execution Price"].min()) * 10)),
                                        int(round(float(px["Execution Price"].max()) * 10)),
                                        a["ticker"] in t500)
            tick10, src = M.resolve_tick10(tbl, emp)
            ok_tick = abs(tick10 / 10.0 - a["tick_yen"]) < 1e-9
            print(f"tick: table {tbl/10 if tbl else None} yen, tape {emp/10 if emp else None} yen"
                  f" -> {tick10/10} yen ({src}) [{'ok' if ok_tick else 'FAIL'}]")
            if not ok_tick:
                fails.append(f"{a['ticker']}: tick {tick10/10} != expected {a['tick_yen']}")

            # Drill gate (both sides quoted) -- the like-for-like comparison.
            got = M.m_measures(n, tick10, gate="zaraba_2s")
            n_zar = int(n["zaraba_2s"].sum())
            # Paper gate (both sides quoted *and* ordinary).
            got_ord = M.m_measures(n, tick10, gate="zaraba_ord")
            n_ord = int(n["zaraba_ord"].sum())
            t_meas = time.perf_counter() - t0

            dn = abs(n_zar - a["n_zaraba"])
            ok_n = dn <= TOL_COUNT
            print(f"zaraba trades: drill gate {n_zar:,} (expected {a['n_zaraba']:,}, "
                  f"diff {dn}) [{'ok' if ok_n else 'FAIL'}]   paper gate {n_ord:,} "
                  f"({100*n_ord/max(n_zar,1):.1f}% of drill gate)")
            if not ok_n:
                fails.append(f"{a['ticker']}: zaraba count off by {dn}")

            print(f"{'cell':<12} {'expected':>9} {'drill gate':>11} {'diff':>7}   "
                  f"{'paper gate':>11}")
            rec = {"ticker": a["ticker"], "date": a["date"], "tick_yen": tick10 / 10,
                   "n_zaraba_drill": n_zar, "n_zaraba_paper": n_ord,
                   "read_sec": round(t_read, 2), "measure_sec": round(t_meas, 2),
                   "rows": df.height, "cells": {}}
            for c in CELLS:
                exp = a[c]
                v = got.get(c)
                vo = got_ord.get(c)
                if v is None:
                    fails.append(f"{a['ticker']}: {c} is null")
                    print(f"{c:<12} {exp:>9.2f} {'null':>11}")
                    continue
                v_pp, vo_pp = 100 * v, (100 * vo if vo is not None else float('nan'))
                diff = v_pp - exp
                ok = abs(diff) <= TOL_PP
                if not ok:
                    fails.append(f"{a['ticker']}: {c} {v_pp:.2f} vs expected {exp:.2f}")
                print(f"{c:<12} {exp:>9.2f} {v_pp:>11.2f} {diff:>+7.2f} "
                      f"{'ok ' if ok else 'FAIL'} {vo_pp:>11.2f}")
                rec["cells"][c] = {"expected": exp, "drill_gate": v_pp,
                                   "paper_gate": vo_pp, "diff": diff, "ok": ok}

            # Negative control: the closing auction is one enormous print. If we
            # wrongly let it in, the daily measure must visibly move -- proof that
            # the auction exclusion is doing real work rather than nothing.
            with_auction = n.with_columns(
                zaraba_loose=pl.col("is_trade") & (pl.col("Volume") > 0)
                & pl.col("prev_two_sided"))
            m_loose = M.m_measures(with_auction, tick10, gate="zaraba_loose")
            shift = 100 * (m_loose["m0_all"] - got["m0_all"])
            print(f"negative control: including auctions moves M0 by {shift:+.2f} pp "
                  f"({100*got['m0_all']:.2f} -> {100*m_loose['m0_all']:.2f})")
            rec["auction_control_shift_pp"] = shift

            # Unmapped execution types would silently drop trades.
            unk = n.filter(pl.col("is_trade") & pl.col("sign").is_null())
            share = 100 * unk.height / max(n_et, 1)
            print(f"unmapped Execution Type: {unk.height} rows ({share:.4f}%)"
                  + (f" -> {sorted(set(unk['Execution Type'].to_list()))}" if unk.height else ""))
            if share > 0.1:
                fails.append(f"{a['ticker']}: {share:.2f}% unmapped execution types")
            rec["unmapped_share_pct"] = share
            rec["digit_dist"] = {f"d{i}": got.get(f"m{i}_all") for i in range(10)}

            ds = sum(v for v in rec["digit_dist"].values() if v is not None)
            print(f"digit distribution sums to {ds:.6f}")
            if abs(ds - 1.0) > 1e-6:
                fails.append(f"{a['ticker']}: digit distribution sums to {ds}")
            print(f"timing: read {t_read:.1f}s, measures {t_meas:.1f}s\n")
            payload.append(rec)

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "anchors.json"),
                      {"tolerance_pp": TOL_PP, "anchors": payload, "fails": fails})

        if fails:
            print(f"GATE FAILED ({len(fails)}):")
            for f in fails:
                print("  -", f)
            return 1
        print("GATE PASSED -- measures.py reproduces both anchor stock-days")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
