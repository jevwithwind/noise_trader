"""S2 step 0 -- select the sample universe the way Ohta (2026) does, before ingest.

Ohta's sample is TSE First Section / Prime **common stocks** whose trading days
satisfy four conditions on more than half the year's trading days: the first trade
by 9:10, an opening price above 200 yen, more than twenty continuous-session
trades, and a tick size that is a power of ten all day.

Those conditions are almost entirely decidable from the daily-summary product,
which is small and already ingested. Applying them first means the tick ingest --
the expensive part -- runs only on stocks that can actually enter the sample, and
it means the universe is Ohta's rather than "whatever fits".

The screen is deliberately generous where the summary is coarser than the tick
data (its execution count includes auction prints and trades against one-sided
quotes, which the tick-level filter excludes). A stock that scrapes through here
is re-tested properly in S3 against the real definition.
"""
from __future__ import annotations

import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s2_ingest")
SUMMARY = os.path.join(C.STORE, "stock_summary")

EXEC_COLS = ["AM Execution Count", "PM Execution Count"]
NEED = ["Data Date", "Exchange Code", "Security Type", "Stock Code", "Trading Unit",
        "Issued Shares", "AM Opening Price", "AM Opening Time", "AM High Price",
        "AM Low Price", "PM High Price", "PM Low Price", "AM Close Price",
        "PM Close Price", "Daily VWAP", "AM Total Volume", "PM Total Volume"] + EXEC_COLS


def load_summary(dates: set[str]) -> pl.DataFrame:
    frames = []
    for e in sorted(os.scandir(SUMMARY), key=lambda x: x.name):
        if not (e.is_dir() and e.name.startswith("date=")):
            continue
        date = e.name[5:]
        if date not in dates:
            continue
        for f in os.scandir(e.path):
            if f.name.endswith(".parquet"):
                sch = pl.read_parquet_schema(f.path)
                cols = [c for c in NEED if c in sch]
                frames.append(pl.read_parquet(f.path, columns=cols)
                              .with_columns(date=pl.lit(date)))
    return pl.concat(frames, how="vertical_relaxed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-day-fraction", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=400)
    args = ap.parse_args()

    tee = C.Tee("s2_step0_universe")
    try:
        print("=== S2 step 0: sample universe (Ohta's selection) ===\n")
        cal = pl.read_csv(os.path.join(C.RESULTS, "s0_inst", "calendar_2024.csv"))
        usable = set(cal.filter(pl.col("status") == "ok")["date"].cast(pl.Utf8).to_list())
        print(f"usable trading days: {len(usable)}")

        df = load_summary(usable)
        print(f"summary rows loaded: {df.height:,}")
        print("\nExchange Code values:")
        print(df["Exchange Code"].value_counts().sort("count", descending=True).head(8))
        print("\nSecurity Type values:")
        print(df["Security Type"].value_counts().sort("count", descending=True).head(12))

        df = df.with_columns(
            ticker=pl.col("Stock Code").cast(pl.Utf8).str.strip_chars(),
            n_exec=pl.sum_horizontal([pl.col(c).fill_null(0) for c in EXEC_COLS]),
        )

        # Ohta studies TSE First Section (Prime from April 2022) common stocks. The
        # feed still carries the legacy section labels, so First Section is the
        # Prime population; Second Section and Mothers are today's Standard and
        # Growth, which are outside his sample. Regional venues are dropped too --
        # a Nagoya print is a different book.
        n0 = df.height
        df = df.filter(pl.col("Exchange Code").str.contains("Tokyo Stock Exchange")
                       & (pl.col("Security Type") == "First Section"))
        print(f"\nTSE First Section only: {n0:,} -> {df.height:,} stock-day rows")

        # The same code can appear for more than one venue; keep the busiest row
        # per stock-day, which is the primary listing.
        before = df.height
        df = (df.sort("n_exec", descending=True)
                .unique(subset=["date", "ticker"], keep="first"))
        print(f"\nde-duplicated venue rows: {before:,} -> {df.height:,}")

        t500 = C.load_topix500()

        # Day price range, for the tick-constancy test.
        lo = pl.min_horizontal([pl.col(c) for c in ["AM Low Price", "PM Low Price"]])
        hi = pl.max_horizontal([pl.col(c) for c in ["AM High Price", "PM High Price"]])
        df = df.with_columns(
            day_lo=pl.when(lo > 0).then(lo).otherwise(None),
            day_hi=pl.when(hi > 0).then(hi).otherwise(None),
            open_px=pl.col("AM Opening Price"),
            open_time=pl.col("AM Opening Time").cast(pl.Utf8).str.strip_chars(),
            is_t500=pl.col("ticker").is_in(list(t500)),
        )

        # Tick constancy has to be evaluated row by row: it depends on both ends of
        # the day's range and on index membership.
        rows = df.select("ticker", "day_lo", "day_hi", "is_t500").to_dicts()
        ticks = []
        for r in rows:
            if r["day_lo"] is None or r["day_hi"] is None or r["day_lo"] <= 0:
                ticks.append(None)
                continue
            t = C.day_tick_constant10(int(round(r["day_lo"] * 10)),
                                      int(round(r["day_hi"] * 10)), r["is_t500"])
            ticks.append(t)
        df = df.with_columns(tick10=pl.Series("tick10", ticks, dtype=pl.Int64))

        df = df.with_columns(
            f_unit=pl.col("Trading Unit") == 100,
            f_open910=(pl.col("open_time").str.len_chars() >= 6)
            & (pl.col("open_time").str.slice(0, 6) <= "091000")
            & (pl.col("open_time").str.slice(0, 6) >= "000001"),
            f_price=pl.col("open_px") > 200.0,
            f_trades=pl.col("n_exec") > 20,
            f_tick=pl.col("tick10").is_in(sorted(C.POWER_OF_TEN_TICKS10)),
        )
        df = df.with_columns(
            qualifies=pl.col("f_unit") & pl.col("f_open910") & pl.col("f_price")
            & pl.col("f_trades") & pl.col("f_tick"))

        print("\nstock-day filter pass rates (screen level):")
        for f in ("f_unit", "f_open910", "f_price", "f_trades", "f_tick", "qualifies"):
            print(f"  {f:12s} {100*df[f].mean():5.1f}%")

        n_days = len(usable)
        per = (df.group_by("ticker").agg(
            n_days_seen=pl.len(),
            n_qual=pl.col("qualifies").sum(),
            unit_mode=pl.col("Trading Unit").mode().first(),
            med_px=pl.col("Daily VWAP").median(),
            med_exec=pl.col("n_exec").median(),
            med_vol=(pl.col("AM Total Volume").fill_null(0)
                     + pl.col("PM Total Volume").fill_null(0)).median(),
            issued=pl.col("Issued Shares").median(),
            is_t500=pl.col("is_t500").any(),
        ).with_columns(
            qual_frac=pl.col("n_qual") / n_days,
            med_yenvol=pl.col("med_px") * pl.col("med_vol"),
            mktcap=pl.col("med_px") * pl.col("issued"),
        ).sort("ticker"))

        keep = per.filter(pl.col("qual_frac") > args.min_day_fraction)
        print(f"\nstocks seen: {per.height:,}")
        print(f"stocks qualifying on > {args.min_day_fraction:.0%} of "
              f"{n_days} days: {keep.height:,}")
        print(f"  of which TOPIX500 constituents: {int(keep['is_t500'].sum())}")

        q = keep["med_yenvol"].quantile
        print(f"  median daily turnover: p10 {q(0.1)/1e6:.1f}M, "
              f"median {q(0.5)/1e6:.1f}M, p90 {q(0.9)/1e6:.1f}M yen")

        for t in ("7203", "8604", "8306", "4666"):
            row = keep.filter(pl.col("ticker") == t)
            seen = per.filter(pl.col("ticker") == t)
            status = ("in universe" if row.height else
                      (f"excluded (qualifies {float(seen['qual_frac'][0]):.0%} of days)"
                       if seen.height else "not seen"))
            print(f"  {t}: {status}")

        C.ensure_dir(OUT)
        keep.write_csv(C.write_guard(os.path.join(OUT, "universe.csv")))
        per.write_csv(C.write_guard(os.path.join(OUT, "universe_all_candidates.csv")))

        # Ingest batches. NEEDS writes each day's parts in ascending stock-code
        # order and tse_tick prunes to the contiguous run of parts covering a
        # filter's codes, so batching *in code order* means each batch decompresses
        # only its own slice of the tape rather than the whole day.
        # The anchor stock-days are ingested whether or not they make the sample,
        # so the panel can be checked against the pilot end to end. 7203 fails the
        # >50%-of-days rule because its price crosses 3,000 yen during 2024, where
        # the fine grid switches to the 0.5-yen tick that filter (d) excludes.
        anchors = ["7203", "8604", "8306", "4666"]
        tickers = sorted(set(keep["ticker"].to_list())
                         | {a for a in anchors if a in set(per["ticker"].to_list())})
        extra = [a for a in anchors if a not in set(keep["ticker"].to_list())]
        if extra:
            print(f"  plus anchors kept for validation only: {', '.join(extra)}")
        batches = [tickers[i:i + args.batch_size]
                   for i in range(0, len(tickers), args.batch_size)]
        bdir = C.ensure_dir(os.path.join(OUT, "batches"))
        # Clear stale batch files: a re-run with different criteria produces fewer
        # batches, and a leftover file would quietly ingest the previous universe.
        for f in os.listdir(bdir):
            if f.startswith("batch_") and f.endswith(".txt"):
                os.remove(C.write_guard(os.path.join(bdir, f)))
        for i, b in enumerate(batches):
            with open(C.write_guard(os.path.join(bdir, f"batch_{i:02d}.txt")), "w") as fh:
                fh.write("\n".join(b))
        print(f"\nwrote {len(batches)} ingest batches of up to {args.batch_size} "
              f"codes ({batches[0][0]}..{batches[-1][-1]})")

        C.atomic_json(os.path.join(OUT, "universe_summary.json"), {
            "n_days": n_days, "n_candidates": per.height, "n_universe": keep.height,
            "n_topix500": int(keep["is_t500"].sum()),
            "min_day_fraction": args.min_day_fraction,
            "batch_size": args.batch_size, "n_batches": len(batches),
            "screen_pass_rates": {f: float(df[f].mean())
                                  for f in ("f_unit", "f_open910", "f_price",
                                            "f_trades", "f_tick", "qualifies")},
        })
        print("\nGATE PASSED -- universe selected")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
