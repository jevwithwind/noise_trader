"""S3 step 2 -- assemble the per-date outputs into the analysis panels.

Applies the stock-level rule (Ohta keeps a stock when it qualifies on more than
half the year's trading days), attaches reference data, and writes the sample
construction waterfall that the report's data section is built from.
"""
from __future__ import annotations

import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s3_common as S3

OUT = S3.OUT


def read_dir(sub: str) -> pl.DataFrame | None:
    d = os.path.join(OUT, sub)
    if not os.path.isdir(d):
        return None
    fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet"))
    if not fs:
        return None
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-day-fraction", type=float, default=0.5)
    args = ap.parse_args()

    tee = C.Tee("s3_step2_assemble")
    try:
        print("=== S3 step 2: assemble panels ===\n")
        daily = read_dir("daily")
        if daily is None:
            print("no daily outputs; run s3_step1_build.py first")
            return 1
        print(f"stock-day rows: {daily.height:,}")

        n_days = daily["date"].n_unique()
        print(f"dates: {n_days}   distinct tickers: {daily['ticker'].n_unique():,}")

        # ---- sample construction waterfall
        print("\nsample construction:")
        total = daily.height
        skipped = daily.filter(pl.col("skip_reason").is_not_null())
        print(f"  stock-days attempted            {total:>9,}")
        for r in (skipped.group_by("skip_reason").len()
                  .sort("len", descending=True).iter_rows(named=True)):
            print(f"    pre-screened out: {r['skip_reason']:<14s} {r['len']:>9,}")
        scored = daily.filter(pl.col("skip_reason").is_null())
        print(f"  fully evaluated                 {scored.height:>9,}")
        wf = {"attempted": total, "evaluated": scored.height}
        for f, label in (("pass_open910", "first trade by 9:10"),
                         ("pass_open200", "open above 200 yen"),
                         ("pass_n20", "more than 20 zaraba trades"),
                         ("pass_tick", "tick a power of ten all day")):
            if f in scored.columns:
                n = int(scored[f].fill_null(False).sum())
                print(f"    passes {label:<30s} {n:>9,}")
                wf[f] = n
        in_s = scored.filter(pl.col("in_sample"))
        print(f"  passing all four filters        {in_s.height:>9,}")
        wf["in_sample"] = in_s.height

        # ---- stock-level rule
        per = (in_s.group_by("ticker").agg(n_qual=pl.len())
               .with_columns(frac=pl.col("n_qual") / n_days))
        keep = set(per.filter(pl.col("frac") > args.min_day_fraction)["ticker"].to_list())
        daily = daily.with_columns(
            in_sample_final=pl.col("in_sample").fill_null(False)
            & pl.col("ticker").is_in(list(keep)))
        final = daily.filter(pl.col("in_sample_final"))
        print(f"  stocks qualifying on > {args.min_day_fraction:.0%} of days "
              f"{len(keep):>6,}")
        print(f"  FINAL stock-day sample          {final.height:>9,}")
        wf["stocks_final"] = len(keep)
        wf["stock_days_final"] = final.height

        # ---- reference data
        t500 = C.load_topix500()
        shares = S3.load_shares_outstanding()
        if shares:
            key = pl.concat_str([pl.col("date").cast(pl.Utf8).str.replace_all("-", ""),
                                 pl.lit("|"), pl.col("ticker")])
            sh = pl.DataFrame({
                "k": [f"{d}|{t}" for (d, t) in shares.keys()],
                "issued": list(shares.values())})
            daily = daily.with_columns(k=key).join(sh, on="k", how="left").drop("k")
            daily = daily.with_columns(
                mktcap=pl.when(pl.col("issued").is_not_null()
                               & pl.col("close_px").is_not_null())
                .then(pl.col("issued") * pl.col("close_px")).otherwise(None))
            print(f"\nshares outstanding matched for "
                  f"{100*daily['issued'].is_not_null().mean():.1f}% of stock-days")
        daily = daily.with_columns(topix500=pl.col("ticker").is_in(list(t500)))

        # Tick-regime composition -- the caveat that shapes the regression sample.
        print("\ntick regime among fully evaluated stock-days:")
        comp = (scored.group_by(["tick10", "topix500"]).len()
                .sort("len", descending=True).head(10))
        for r in comp.iter_rows(named=True):
            t = r["tick10"]
            print(f"  tick {('%.1f' % (t/10)) if t else 'n/a':>6} yen  "
                  f"TOPIX500={str(r['topix500']):<5s} {r['len']:>9,}"
                  + ("" if t is None or t in C.POWER_OF_TEN_TICKS10 else "   [excluded by filter d]"))

        p = os.path.join(OUT, "panel_stockday.parquet")
        daily.write_parquet(C.write_guard(p))
        print(f"\nwrote {p}  ({daily.height:,} rows x {daily.width} cols)")

        intr = read_dir("intraday")
        if intr is not None:
            intr = intr.with_columns(
                in_sample_final=pl.col("ticker").is_in(list(keep)))
            p2 = os.path.join(OUT, "panel_intraday.parquet")
            intr.write_parquet(C.write_guard(p2))
            print(f"wrote {p2}  ({intr.height:,} rows x {intr.width} cols)")
            wf["intraday_rows"] = intr.height

        pl.DataFrame({"ticker": sorted(keep)}).write_csv(
            C.write_guard(os.path.join(OUT, "sample_stocks.csv")))
        C.atomic_json(os.path.join(OUT, "waterfall.json"), wf)

        # ---- data-quality flags worth surfacing
        if "n_unmapped_exec_type" in daily.columns:
            unm = int(daily["n_unmapped_exec_type"].fill_null(0).sum())
            print(f"\nunmapped execution-type rows across the panel: {unm:,}")
        if "tick_mismatch" in daily.columns:
            mm = daily.filter(pl.col("tick_mismatch"))
            print(f"tick table/tape disagreements: {mm.height:,} "
                  f"({100*mm.height/max(scored.height,1):.2f}% of evaluated stock-days)")
            print("  These are resolved in the tape's favour wherever it is "
                  "informative, so a disagreement is a diagnostic of index "
                  "membership drift rather than an error. What would be an error "
                  "is a stock-day whose digits collapse onto one value:")
            if "m0_all" in daily.columns:
                bad = daily.filter(pl.col("in_sample") & (pl.col("m0_all") > 0.6))
                print(f"  stock-days with M0 above 60%: {bad.height:,}"
                      + ("  <- inspect: this is what a wrong tick looks like"
                         if bad.height else "  (none)"))
        if "tick_source" in daily.columns:
            print("tick source: " + ", ".join(
                f"{r['tick_source']}={r['len']:,}" for r in
                daily.group_by("tick_source").len().sort("len", descending=True)
                .iter_rows(named=True)))
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
