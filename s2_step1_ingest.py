"""S2 step 1 -- two-shot Stage 1: ingest the tape into the Parquet store.

Stage 1 of tse_tick's two-stage workflow, run in code-ordered ticker batches.

Why batches. A full-frame ingest concatenates every part of a trading day before
writing, which for 2024's message rates needs far more than this machine's 32 GB
and does in fact fail. The library's streaming path -- each part goes straight to
its ticker's Parquet writer and is dropped -- holds only one part at a time, and
it engages when the request carries a ticker filter. Batching is therefore not a
workaround but the supported route, and the library's coverage markers are built
to accumulate across successive filtered ingests of the same date.

Why the batches are in code order. NEEDS writes each day's parts in ascending
stock-code order and the library prunes a filtered request to the contiguous run
of parts spanning its codes, so a code-ordered batch decompresses roughly its own
slice of the tape rather than all of it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s2_ingest")
TICK_ROOT = os.path.join(C.RAW_ROOT, "個別株式2024", "TICST120")
SUMMARY_ROOT = os.path.join(C.RAW_ROOT, "個別株式2024", "TICSS110")
CKPT = os.path.join(OUT, "ingest_ckpt.json")


def store_stats() -> tuple[int, float]:
    n, b = 0, 0
    root = os.path.join(C.STORE, "individual_stock")
    if not os.path.isdir(root):
        return 0, 0.0
    for d in os.scandir(root):
        if not d.is_dir():
            continue
        for f in os.scandir(d.path):
            if f.name.endswith(".parquet"):
                n += 1
                b += f.stat().st_size
    return n, b / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="2024")
    ap.add_argument("--workers", default="2")
    ap.add_argument("--batches", default="all",
                    help="'all' or comma-separated batch indices, e.g. 0,1")
    ap.add_argument("--max-stream", type=int, default=512)
    ap.add_argument("--summary", action="store_true",
                    help="also ingest the daily-summary product")
    args = ap.parse_args()
    workers = args.workers if args.workers == "auto" else int(args.workers)

    tee = C.Tee(f"s2_step1_ingest_{args.period}")
    try:
        import tse_tick
        import tse_tick.ingest as TI

        # Raise the streaming threshold so a 400-code batch streams rather than
        # concatenating the day. The constant bounds how many Parquet writers are
        # open at once, not how much data is held: under streaming the library's
        # own worker-memory estimate is a flat 3 GB regardless of ticker count,
        # because each part is written and dropped. Several hundred open writers
        # is comfortable here.
        old = TI._MAX_STREAM_TICKERS
        TI._MAX_STREAM_TICKERS = args.max_stream
        print(f"=== S2 step 1: ingest (period={args.period}, workers={args.workers}) ===")
        print(f"streaming threshold raised {old} -> {TI._MAX_STREAM_TICKERS}\n")

        C.ensure_dir(C.STORE)
        C.ensure_dir(OUT)
        log = C.read_json(CKPT, {"runs": []})

        if args.summary:
            print("--- daily summary (TICSS110) ---")
            t0 = time.perf_counter()
            tse_tick.ingest_period(input_root=SUMMARY_ROOT, output_dir=C.STORE,
                                   period="2024", data_type="stock_summary",
                                   language="en", resume=True, max_workers=workers,
                                   compression="zstd")
            print(f"    {time.perf_counter()-t0:.0f}s\n")

        bdir = os.path.join(OUT, "batches")
        files = sorted(f for f in os.listdir(bdir) if f.endswith(".txt"))
        if args.batches != "all":
            want = {int(x) for x in args.batches.split(",")}
            files = [f for f in files if int(f[6:8]) in want]
        print(f"batches to ingest: {len(files)}\n")

        n0, gb0 = store_stats()
        t_all = time.perf_counter()
        for bf in files:
            with open(os.path.join(bdir, bf), encoding="utf-8") as fh:
                tickers = {l.strip() for l in fh if l.strip()}
            print(f"--- {bf}: {len(tickers)} codes "
                  f"({min(tickers)}..{max(tickers)}) ---", flush=True)
            t0 = time.perf_counter()
            res = tse_tick.ingest_period(
                input_root=TICK_ROOT, output_dir=C.STORE, period=args.period,
                data_type="individual_stock", language="en", resume=True,
                max_workers=workers, ticker_filter=tickers, compression="zstd")
            el = time.perf_counter() - t0
            n1, gb1 = store_stats()
            rows = sum(r.get("rows", 0) for r in res if isinstance(r, dict))
            dates = len([r for r in res if isinstance(r, dict) and r.get("rows")])
            print(f"    {el/60:.1f} min, {dates} dates, {rows:,} rows, "
                  f"+{n1-n0:,} files, +{gb1-gb0:.1f} GB "
                  f"(store {n1:,} files, {gb1:.1f} GB)", flush=True)
            log["runs"].append({"batch": bf, "codes": len(tickers), "minutes": round(el/60, 1),
                                "dates": dates, "rows": rows, "store_files": n1,
                                "store_gb": round(gb1, 2)})
            C.atomic_json(CKPT, log)
            n0, gb0 = n1, gb1

        el = time.perf_counter() - t_all
        n1, gb1 = store_stats()
        print(f"\ntotal {el/3600:.2f} h   store: {n1:,} files, {gb1:.1f} GB")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
