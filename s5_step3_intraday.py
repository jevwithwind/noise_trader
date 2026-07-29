"""S5 step 3 -- the "or shorter" half of the paper's sentence.

Ohta's conclusion asks for the relationship between noise-trader activity and
liquidity "at a daily or shorter frequency". Daily is step 1. This is shorter:
thirty-minute buckets, with stock-day fixed effects so that everything about the
day -- the stock, the news, the market state -- is absorbed, and bucket fixed
effects so that the familiar intraday U-shape in spreads and volume is absorbed
too. What is left is purely within-day, bucket-to-bucket variation.

That is a demanding design, and the coefficients should be read as small by
construction. It is also the only design here that cannot be explained by
slow-moving stock characteristics.
"""
from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4
import s5_common as S5

OUT = S5.OUT


def main() -> int:
    tee = C.Tee("s5_step3_intraday")
    try:
        print("=== S5 step 3: intraday (30-minute buckets) ===\n")
        if not os.path.exists(S4.PANEL_INTRADAY):
            print("no intraday panel")
            return 1
        df = pl.read_parquet(S4.PANEL_INTRADAY)
        if "in_sample_final" in df.columns:
            df = df.filter(pl.col("in_sample_final"))
        print(f"bucket rows: {df.height:,}   "
              f"stock-days: {df.select(['date','ticker']).n_unique():,}")

        df = df.with_columns(
            stockday=pl.concat_str([pl.col("ticker"), pl.lit("_"),
                                    pl.col("date").cast(pl.Utf8)]),
            bucket_s=pl.col("bucket").cast(pl.Utf8),
        ).sort(["ticker", "date", "bucket"])

        # Lags within a stock-day only: the first bucket of a day has no
        # predecessor, since the overnight gap is not a thirty-minute interval.
        same = (pl.col("stockday") == pl.col("stockday").shift(1))
        for c in ["m0", "m_b_large0", "effsprd_bps", "imp60_bps", "ofi",
                  "ret_mid_bps", "yenvol"]:
            if c in df.columns:
                df = df.with_columns(
                    **{f"{c}_l1": pl.when(same).then(pl.col(c).shift(1)).otherwise(None)})
        if "effsprd_bps" in df.columns:
            df = df.with_columns(
                ln_effsprd=pl.when(pl.col("effsprd_bps") > 0)
                .then(pl.col("effsprd_bps").log()).otherwise(None))
            df = df.with_columns(
                ln_effsprd_l1=pl.when(same).then(pl.col("ln_effsprd").shift(1))
                .otherwise(None))
        if "yenvol" in df.columns:
            df = df.with_columns(
                ln_yenvol=pl.when(pl.col("yenvol") > 0)
                .then(pl.col("yenvol").log()).otherwise(None))

        results, rows = {}, []
        for y, lab in (("ln_effsprd", "Effective spread (log)"),
                       ("imp60_bps", "Price impact, 60s (bp)")):
            if y not in df.columns:
                continue
            for xname, xlab in (("m0_l1", "$M^{0}_{b-1}$"),
                                ("m_b_large0_l1", "$M^{BLarge0}_{b-1}$")):
                if xname not in df.columns:
                    continue
                xs = [xname, f"{y}_l1", "ln_yenvol"]
                xs = [c for c in xs if c in df.columns]
                r = S5.fit(df, y, xs, entity="stockday", time="bucket_s")
                b, se = S5.cell(r, xname)
                t = r.get("t", {}).get(xname, float("nan"))
                print(f"{lab:<28} {xlab:<22} {b:<14} {se:<12} t={t:>6.2f}  "
                      f"n={r.get('n',0):,}"
                      + (f"  [{r['error'][:40]}]" if "error" in r else ""))
                rows.append([lab, xlab, b, se, f"{r.get('n',0):,}"])
                results[f"{y}|{xname}"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}

        # Does the price move more per unit of order flow when clustering is high?
        # This is the execution-relevant version of the impact result.
        print("\n--- order-flow sensitivity, conditional on clustering ---")
        if all(c in df.columns for c in ("ret_mid_bps", "ofi", "m0")):
            d = df.drop_nulls(["ret_mid_bps", "ofi", "m0"])
            med = d.select(pl.col("m0").median()).item()
            d = d.with_columns(
                high_m=(pl.col("m0") > med).cast(pl.Float64),
                ofi_z=(pl.col("ofi") - pl.col("ofi").mean()) / pl.col("ofi").std())
            d = d.with_columns(ofi_x_high=pl.col("ofi_z") * pl.col("high_m"))
            r = S5.fit(d, "ret_mid_bps", ["ofi_z", "ofi_x_high", "high_m"],
                       entity="stockday", time="bucket_s")
            for v in ("ofi_z", "ofi_x_high"):
                b, se = S5.cell(r, v, nd=4)
                t = r.get("t", {}).get(v, float("nan"))
                print(f"  {v:<14} {b:<14} {se:<12} t={t:>6.2f}")
            results["ofi_interaction"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}
            print(f"  n={r.get('n',0):,}   (median M^0 split at {100*med:.2f}%)")
            print("  A positive interaction means the same order-flow imbalance "
                  "moves the midquote further on high-clustering days -- i.e. the "
                  "book is easier to push.")

        if rows:
            S4.latex_table(
                os.path.join(S4.TABLES, "t_intraday.tex"),
                "Clustering and liquidity within the trading day",
                "tab:intraday",
                ["Outcome", "Regressor", "Coefficient", "SE", "Bucket-rows"], rows,
                notes="Thirty-minute buckets. Stock-day and bucket-of-day fixed "
                      "effects, so all daily variation and the average intraday "
                      "pattern are absorbed and only within-day, "
                      "bucket-to-bucket variation identifies the coefficient. "
                      "Standard errors clustered by stock and by date. "
                      "Each specification includes the outcome's own lagged "
                      "bucket and log turnover. $^{*}p<0.1$, $^{**}p<0.05$, "
                      "$^{***}p<0.01$.")

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "intraday.json"), results)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
