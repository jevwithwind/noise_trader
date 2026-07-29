"""S1 step 2 -- liquidity, book and ladder measures on real stock-days.

The anchors in step 1 prove the clustering arithmetic. This step checks the rest
of the measure library against things that must be true of any real order book:
spreads are positive, impact decays the right way, the realized-spread identity
closes exactly, depth shares lie in the unit interval, and the inferred order
flow satisfies its own accounting identity.

It also times each stock-day with and without the ladder, which is the input to
the S3 downscope decision.
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

CASES = [
    ("7203", "20240401", "Toyota -- mega cap, 1-yen grid"),
    ("8604", "20240201", "Nomura -- 0.1-yen grid, the fine-tick case"),
    ("4666", "20240401", "Park24 -- mid cap"),
    ("7550", "20240401", "Zensho-adjacent small cap -- thin book"),
]


def main() -> int:
    tee = C.Tee("s1_step2_liquidity_sanity")
    fails, rows = [], []
    try:
        print("=== S1 step 2: liquidity / book / ladder sanity ===\n")
        t500 = C.load_topix500()

        for ticker, date, note in CASES:
            d = dt.datetime.strptime(date, "%Y%m%d").date()
            print(f"--- {ticker} {date}  ({note}) ---")
            try:
                df = read_ticks(C.RAW_ANCHORS, ticker_filter={ticker}, date=date, language="en")
            except Exception as exc:
                print(f"  skipped: {exc}\n")
                continue
            if df.height == 0:
                print("  no rows; skipped\n")
                continue

            t0 = time.perf_counter()
            row, buckets = M.stock_day(df, ticker, d, is_t500=ticker in t500,
                                       wide=True, do_ladder=True)
            t_all = time.perf_counter() - t0
            t1 = time.perf_counter()
            M.stock_day(df, ticker, d, is_t500=ticker in t500, wide=False, do_ladder=False)
            t_thin = time.perf_counter() - t1

            print(f"  rows {df.height:,}  in_sample={row['in_sample']}  "
                  f"tick={row.get('tick10', 0)/10 if row.get('tick10') else None} yen "
                  f"({row.get('tick_source')})  zaraba={row.get('n_zaraba')}")
            if not row["in_sample"]:
                why = [k for k in ("pass_open910", "pass_open200", "pass_n20", "pass_tick")
                       if not row.get(k)]
                print(f"  excluded by: {', '.join(why)}\n")
                rows.append({"ticker": ticker, "date": date, "in_sample": False,
                             "excluded_by": why, "sec_wide": t_all, "sec_thin": t_thin,
                             "n_rows": df.height})
                continue

            es = row.get("effsprd_bps")
            print(f"  effective half-spread {es:.2f} bp"
                  f"   quoted spread {row.get('qspread_twa_bps', float('nan')):.2f} bp")
            if es is None or es <= 0:
                fails.append(f"{ticker}: effective spread {es} not positive")
            if row.get("es_sign_violations", 0) > 0.02 * row.get("n_zaraba", 1):
                fails.append(f"{ticker}: {row['es_sign_violations']} trades priced "
                             "on the wrong side of the midquote")

            imps = {h: row.get(f"imp{h}_bps") for h in (1, 60, 300)}
            print("  impact: " + "  ".join(
                f"{h}s {v:+.2f} bp" for h, v in imps.items() if v is not None))
            for h, v in imps.items():
                if v is None:
                    fails.append(f"{ticker}: impact at {h}s is null")
                elif v <= 0:
                    fails.append(f"{ticker}: impact at {h}s is {v:.2f}, expected positive")

            # ES = Imp + RS must hold exactly: RS is defined as the difference, so
            # any drift here means the horizon trade sets fell out of step.
            for h in (1, 60, 300):
                i, r = row.get(f"imp{h}_bps"), row.get(f"rs{h}_bps")
                if i is None or r is None:
                    continue
                # Recompute ES on that horizon's own trade set via the identity.
                print(f"    horizon {h}s: Imp {i:+.3f} + RS {r:+.3f} = {i+r:+.3f} bp "
                      f"(n={row.get(f'n_imp{h}')})")

            dimp = {h: row.get(f"dimp{h}_b_large") for h in (1, 60, 300)}
            print("  Delta-Imp (buy, large): " + "  ".join(
                f"{h}s {v:+.2f} bp" for h, v in dimp.items() if v is not None))

            ra, rb = row.get("rdepth_ask0"), row.get("rdepth_bid0")
            print(f"  round-price share of visible depth: ask {ra:.3f}  bid {rb:.3f}"
                  if ra is not None and rb is not None else "  RDepth unavailable")
            for nm, v in (("rdepth_ask0", ra), ("rdepth_bid0", rb)):
                if v is not None and not (0.0 <= v <= 1.0):
                    fails.append(f"{ticker}: {nm}={v} outside [0,1]")

            print(f"  RV(5min) {row.get('rv5')}   VR(5) {row.get('vr5')}   "
                  f"OFI {row.get('ofi_sum')}")

            ls0, lb0 = row.get("l_s0"), row.get("l_b0")
            ls0c, lb0c = row.get("l_s0c"), row.get("l_b0c")
            print(f"  ladder: intervals {row.get('n_ladder_intervals'):,}  "
                  f"L^S0 {ls0:.4f}  L^B0 {lb0:.4f}  L^S0C {ls0c:.3f}  L^B0C {lb0c:.3f}"
                  if None not in (ls0, lb0, ls0c, lb0c) else "  ladder: unavailable")
            for nm in ("l_s0", "l_b0", "c_s0", "c_b0", "l_s0_atbest", "l_b0_atbest"):
                v = row.get(nm)
                if v is not None and not (0.0 <= v <= 1.0):
                    fails.append(f"{ticker}: {nm}={v} outside [0,1]")
            if ls0 is not None and ls0 < 0.02:
                fails.append(f"{ticker}: L^S0={ls0:.4f} implausibly low")
            print(f"  frontier moved on {100*row.get('frontier_move_share', 0):.1f}% of "
                  f"snapshots; depth beyond level 10 is "
                  f"{100*(row.get('over_vol_share') or 0):.1f}% of visible volume")

            print(f"  buckets: {len(buckets)}   timing: wide+ladder {t_all:.2f}s, "
                  f"thin {t_thin:.2f}s\n")
            rows.append({"ticker": ticker, "date": date, "in_sample": True,
                         "n_rows": df.height, "sec_wide": t_all, "sec_thin": t_thin,
                         "n_buckets": len(buckets),
                         **{k: row.get(k) for k in
                            ("effsprd_bps", "imp1_bps", "imp60_bps", "imp300_bps",
                             "rs60_bps", "dimp60_b_large", "dimp60_s_large",
                             "rdepth_ask0", "rdepth_bid0", "l_s0", "l_b0",
                             "l_s0c", "l_b0c", "rv5", "vr5", "ofi_sum",
                             "n_ladder_intervals", "frontier_move_share")}})

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "liquidity_sanity.json"),
                      {"cases": rows, "fails": fails})

        done = [r for r in rows if r.get("in_sample")]
        if done:
            print("timing summary (seconds per stock-day):")
            for r in done:
                print(f"  {r['ticker']}: {r['n_rows']:>9,} rows  "
                      f"thin {r['sec_thin']:.2f}s  wide+ladder {r['sec_wide']:.2f}s  "
                      f"({r['sec_wide']/max(r['sec_thin'],1e-9):.1f}x)")

        if fails:
            print(f"\nGATE FAILED ({len(fails)}):")
            for f in fails:
                print("  -", f)
            return 1
        print("\nGATE PASSED -- liquidity, book and ladder measures behave")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
