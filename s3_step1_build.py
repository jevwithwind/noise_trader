"""S3 step 1 -- build the stock-day and intraday panels from the Parquet store.

One read per (date, ticker) file produces the daily row, the 30-minute bucket
rows, and -- for the subsample carrying the ladder inference -- the limit-order
submission and cancellation aggregates. Checkpointed per date, so an interrupted
overnight run resumes where it stopped.

A stock-day that fails Ohta's admission filters still produces a row, carrying
the reason. The sample-construction waterfall in the report is then a count of
what happened rather than an inference from what is missing.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s3_common as S3

OUT = S3.OUT
CKPT = os.path.join(OUT, "build_ckpt.json")


def write_date(date: str, rows: list[dict], buckets: list[dict]) -> None:
    C.ensure_dir(os.path.join(OUT, "daily"))
    C.ensure_dir(os.path.join(OUT, "intraday"))
    if rows:
        p = os.path.join(OUT, "daily", f"{date}.parquet")
        tmp = p + ".tmp"
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(C.write_guard(tmp))
        os.replace(tmp, p)
    if buckets:
        p = os.path.join(OUT, "intraday", f"{date}.parquet")
        tmp = p + ".tmp"
        pl.DataFrame(buckets, infer_schema_length=None).write_parquet(C.write_guard(tmp))
        os.replace(tmp, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=f"{C.YEAR}0101")
    ap.add_argument("--end", default=f"{C.YEAR}1231")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-wide", action="store_true",
                    help="skip the ten-level book measures (RDepth, slope, quoted spread)")
    ap.add_argument("--ladder-tickers", default="",
                    help="path to a one-per-line ticker list to run ladder inference on")
    ap.add_argument("--max-dates", type=int, default=0)
    ap.add_argument("--gate-after", type=int, default=5,
                    help="project total runtime after this many dates")
    ap.add_argument("--gate-hours", type=float, default=24.0)
    args = ap.parse_args()

    tee = C.Tee("s3_step1_build")
    try:
        print("=== S3 step 1: panel build ===\n")
        wide = not args.no_wide
        ladder: set[str] = set()
        if args.ladder_tickers and os.path.exists(args.ladder_tickers):
            with open(args.ladder_tickers, encoding="utf-8") as fh:
                ladder = {l.strip().zfill(4) for l in fh if l.strip()}
        print(f"wide book measures: {wide}   ladder tickers: {len(ladder)}   "
              f"workers: {args.workers}")

        cal = pl.read_csv(C.CALENDAR_CSV)
        usable = set(cal.filter(pl.col("status") == "ok")["date"].cast(pl.Utf8).to_list())
        dates = [d for d in S3.store_dates()
                 if args.start <= d <= args.end and d in usable]
        if args.max_dates:
            dates = dates[:args.max_dates]
        print(f"dates in store and usable: {len(dates)}")
        if not dates:
            print("nothing to do")
            return 1

        t500 = C.load_topix500()
        units = S3.load_units()
        print(f"trading units loaded for {len(units):,} stock-days"
              + (f"; distinct values {sorted(set(units.values()))[:5]}" if units else ""))

        ck = C.read_json(CKPT, {"done": [], "errors": []})
        done = set(ck.get("done", []))
        todo = [d for d in dates if d not in done]
        print(f"already done: {len(done)}   to do: {len(todo)}\n")

        t_start = time.perf_counter()
        n_processed = 0
        errors: list[str] = list(ck.get("errors", []))

        for i, date in enumerate(todo, 1):
            files = S3.date_files(date)
            if not files:
                done.add(date)
                continue
            tasks = [(tk, path, date, tk in t500, units.get((date, tk), 100),
                      wide, tk in ladder) for tk, path, _ in files]
            rows, buckets = [], []
            t0 = time.perf_counter()
            with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
                for row, bk, err in ex.map(S3.process_one, tasks, chunksize=8):
                    if err:
                        errors.append(err)
                    if row is not None:
                        rows.append(row)
                    buckets.extend(bk)
            write_date(date, rows, buckets)
            el = time.perf_counter() - t0
            n_processed += 1
            done.add(date)
            C.atomic_json(CKPT, {"done": sorted(done), "errors": errors[-200:]})

            n_in = sum(1 for r in rows if r.get("in_sample"))
            print(f"[{i}/{len(todo)}] {date}: {len(files):>5,} tickers, "
                  f"{n_in:>4,} in sample, {len(buckets):>5,} buckets, "
                  f"{el:>6.1f}s" + (f", {len(errors)} errors" if errors else ""))

            if n_processed == args.gate_after and len(todo) > args.gate_after:
                per = (time.perf_counter() - t_start) / n_processed
                proj = per * len(todo) / 3600
                print(f"\n  projection: {per:.1f}s per date -> {proj:.1f} hours "
                      f"for {len(todo)} dates")
                if proj > args.gate_hours:
                    msg = (f"projected {proj:.1f}h exceeds the {args.gate_hours}h gate")
                    print(f"  GATE TRIPPED: {msg}")
                    C.atomic_json(os.path.join(OUT, "downscope_decision.json"), {
                        "tripped": True, "projected_hours": round(proj, 1),
                        "gate_hours": args.gate_hours, "sec_per_date": round(per, 1),
                        "wide": wide, "n_ladder_tickers": len(ladder),
                        "advice": "rerun with --no-wide, or shrink the ladder list",
                    })
                    print("  stopping so the decision is made explicitly, not silently.")
                    return 2
                print()

        el_total = time.perf_counter() - t_start
        C.atomic_json(os.path.join(OUT, "build_summary.json"), {
            "dates_done": len(done), "dates_processed": n_processed,
            "seconds": round(el_total, 1), "wide": wide,
            "n_ladder_tickers": len(ladder), "n_errors": len(errors),
            "errors_tail": errors[-20:],
        })
        print(f"\nprocessed {n_processed} dates in {el_total/3600:.2f} h"
              f"   errors: {len(errors)}")
        if errors:
            print("first few errors:")
            for e in errors[:3]:
                print("  ", e.splitlines()[0])
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
