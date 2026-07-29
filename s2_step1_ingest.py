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
TICK_ROOT = C.RAW_TICKS
SUMMARY_ROOT = C.RAW_SUMMARY
CKPT_TMPL = os.path.join(OUT, "ingest_ckpt{tag}.json")


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
    ap.add_argument("--period", default=str(C.YEAR))
    ap.add_argument("--workers", default="2")
    ap.add_argument("--batches", default="all",
                    help="'all' or comma-separated batch indices, e.g. 0,1")
    ap.add_argument("--max-stream", type=int, default=0,
                    help="raise the library's streaming threshold (0 = leave alone). "
                         "Almost always leave this at 0: see the note in main().")
    ap.add_argument("--summary", action="store_true",
                    help="also ingest the daily-summary product")
    ap.add_argument("--by-month", action="store_true",
                    help="ingest the whole universe month by month, so every "
                         "finished date is complete and the panel can be built "
                         "incrementally while the rest still runs")
    ap.add_argument("--months", default="",
                    help="comma-separated YYYYMM to restrict --by-month")
    args = ap.parse_args()
    workers = args.workers if args.workers == "auto" else int(args.workers)

    tee = C.Tee(f"s2_step1_ingest_{args.period}")
    try:
        import tse_tick
        import tse_tick.ingest as TI

        # Do NOT raise the streaming threshold. Ingest workers are *spawned*
        # processes that re-import the library, so a constant patched here never
        # reaches them: they would keep concatenating the whole day while the
        # parent, seeing the patched value, sized their memory as if they were
        # streaming and started too many of them. That combination is what
        # exhausted memory. Keeping batches at or below the library's own
        # threshold makes the workers stream natively, which bounds each of them
        # at roughly one part and lets several run at once.
        print(f"=== S2 step 1: ingest (period={args.period}, workers={args.workers}) ===")
        if args.max_stream:
            old = TI._MAX_STREAM_TICKERS
            TI._MAX_STREAM_TICKERS = args.max_stream
            print(f"streaming threshold raised {old} -> {TI._MAX_STREAM_TICKERS} "
                  f"(parent only -- see the comment above)")
        print(f"library streaming threshold: {TI._MAX_STREAM_TICKERS} codes per request\n")

        C.ensure_dir(C.STORE)
        C.ensure_dir(OUT)
        tag = "_" + (args.months.replace(",", "-") if args.months else args.period)
        ckpt = CKPT_TMPL.format(tag=tag)
        log = C.read_json(ckpt, {"runs": []})

        if args.summary:
            print("--- daily summary (TICSS110) ---")
            t0 = time.perf_counter()
            tse_tick.ingest_period(input_root=SUMMARY_ROOT, output_dir=C.STORE,
                                   period=str(C.YEAR), data_type="stock_summary",
                                   language="en", resume=True, max_workers=workers,
                                   compression="zstd")
            print(f"    {time.perf_counter()-t0:.0f}s\n")

        bdir = os.path.join(OUT, "batches")
        if args.by_month:
            # One unit of work per calendar month, covering the whole universe, so
            # a finished month is finished for every stock. Splitting by ticker
            # instead would leave every date partially written until the last
            # batch, and nothing downstream could start.
            all_t: set[str] = set()
            for f in sorted(os.listdir(bdir)):
                if f.endswith(".txt"):
                    with open(os.path.join(bdir, f), encoding="utf-8") as fh:
                        all_t |= {l.strip() for l in fh if l.strip()}
            months = ([m.strip() for m in args.months.split(",") if m.strip()]
                      or list(C.MONTHS) or [f"{C.YEAR}{m:02d}" for m in range(1, 13)])
            units = [(m, all_t) for m in months]
            print(f"universe: {len(all_t)} codes; ingesting {len(units)} months\n")
        else:
            files = sorted(f for f in os.listdir(bdir) if f.endswith(".txt"))
            if args.batches != "all":
                want = {int(x) for x in args.batches.split(",")}
                files = [f for f in files if int(f[6:8]) in want]
            # Periods first, batches within them. Running several processes over
            # disjoint month ranges is how this parallelises beyond one process's
            # worker cap, and disjoint months means no two processes ever write
            # the same date partition or race on its coverage marker.
            periods = ([m.strip() for m in args.months.split(",") if m.strip()]
                       or [args.period])
            units = []
            for p in periods:
                for bf in files:
                    with open(os.path.join(bdir, bf), encoding="utf-8") as fh:
                        units.append((p, {l.strip() for l in fh if l.strip()}))
            print(f"units to ingest: {len(units)} "
                  f"({len(periods)} period(s) x {len(files)} batches)\n")

        n0, gb0 = store_stats()
        t_all = time.perf_counter()
        for period, tickers in units:
            print(f"--- {period}: {len(tickers)} codes "
                  f"({min(tickers)}..{max(tickers)}) ---", flush=True)
            t0 = time.perf_counter()
            res = tse_tick.ingest_period(
                input_root=TICK_ROOT, output_dir=C.STORE, period=period,
                data_type="individual_stock", language="en", resume=True,
                max_workers=workers, ticker_filter=tickers, compression="zstd")
            el = time.perf_counter() - t0
            n1, gb1 = store_stats()
            rows = sum(r.get("rows", 0) for r in res if isinstance(r, dict))
            dates = len([r for r in res if isinstance(r, dict) and r.get("rows")])
            print(f"    {el/60:.1f} min, {dates} dates, {rows:,} rows, "
                  f"+{n1-n0:,} files, +{gb1-gb0:.1f} GB "
                  f"(store {n1:,} files, {gb1:.1f} GB)", flush=True)
            log["runs"].append({"unit": period, "codes": len(tickers),
                                "minutes": round(el / 60, 1), "dates": dates,
                                "rows": rows, "store_files": n1,
                                "store_gb": round(gb1, 2)})
            C.atomic_json(ckpt, log)
            n0, gb0 = n1, gb1

        el = time.perf_counter() - t_all
        n1, gb1 = store_stats()
        print(f"\ntotal {el/3600:.2f} h   store: {n1:,} files, {gb1:.1f} GB")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
