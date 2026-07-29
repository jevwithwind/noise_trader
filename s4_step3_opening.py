"""S4 step 3 -- the opening-digit contamination the paper warns about.

Ohta's Table 2 records a mechanical distortion in the clustering measure: a day
that opens at a price ending far from zero and then barely moves can never print
a round price, so its measure is low for a reason that has nothing to do with
noise traders. The fix is to control for the interaction of the opening digit
with a low-volatility indicator, and this step confirms the distortion is present
in the sample and that those controls are therefore needed.
"""
from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4

OUT = os.path.join(C.RESULTS, "s4_rq1")


def main() -> int:
    tee = C.Tee("s4_step3_opening")
    try:
        print("=== S4 step 3: opening-digit contamination ===\n")
        df = S4.load_panel(final_only=True)

        # Low volatility means below the cross-sectional median that day, so the
        # split is relative to the market rather than to an absolute threshold.
        df = df.drop_nulls(["open_digit", "m0_all"]).with_columns(
            lowvol=pl.col("rv5") < pl.col("rv5").median().over("date"))
        print(f"stock-days: {df.height:,}")

        print("\ndistribution of the opening price's last digit:")
        od = (df.group_by("open_digit").len().sort("open_digit")
                .with_columns(share=pl.col("len") / df.height))
        for r in od.iter_rows(named=True):
            print(f"  digit {r['open_digit']}: {100*r['share']:5.2f}%  "
                  + "#" * int(round(200 * r["share"])))
        print("  (clustering is present in opening prices too: digit 0 is well "
              "above the 10% a uniform grid would give)")

        print("\nM0 by opening digit and previous-day volatility:")
        print(f"{'digit':>6} {'low vol':>10} {'high vol':>10} {'gap':>8} {'n low':>9}")
        rows, payload = [], {}
        for d in range(10):
            lo = df.filter((pl.col("open_digit") == d) & pl.col("lowvol"))
            hi = df.filter((pl.col("open_digit") == d) & ~pl.col("lowvol"))
            if lo.height < 50 or hi.height < 50:
                continue
            a, _, _ = S4.twoway_cluster_mean(lo, "m0_all")
            b, _, _ = S4.twoway_cluster_mean(hi, "m0_all")
            print(f"{d:>6} {100*a:>10.2f} {100*b:>10.2f} {100*(a-b):>+8.2f} "
                  f"{lo.height:>9,}")
            rows.append([str(d), f"{100*a:.2f}", f"{100*b:.2f}", f"{100*(a-b):+.2f}",
                         f"{lo.height:,}"])
            payload[f"digit{d}"] = {"lowvol": a, "highvol": b, "n_low": lo.height}

        S4.latex_table(
            os.path.join(S4.TABLES, "t_opening.tex"),
            "Opening-price digit, volatility, and measured clustering",
            "tab:opening",
            ["Opening digit", "Low volatility (\\%)", "High volatility (\\%)",
             "Difference", "Stock-days, low vol."],
            rows,
            notes="Mean $M^{0}$ by the last digit of the opening price, split on "
                  "whether the \\emph{same day's} realised variance was below "
                  "that day's cross-sectional median. On quiet days a stock that "
                  "opens far from a round price may never reach one, which "
                  "depresses the measure mechanically. A median split is a "
                  "diluted version of the paper's stuck-price notion -- the "
                  "median day still moves through many grid points -- so "
                  "Table~\\ref{tab:openingstuck} repeats the comparison on the "
                  "bottom decile of the day's price range, where the mechanism "
                  "has to show if it is real. The regressions control for the "
                  "opening-digit by low-volatility interactions with the timing "
                  "matched to the regressor.")

        # A compact summary: how much of the measure's variation the contamination
        # alone can account for.
        near = df.filter(pl.col("open_digit").is_in([9, 0, 1]))
        far = df.filter(pl.col("open_digit").is_in([4, 5, 6]))
        for lab, mask in (("low volatility", True), ("high volatility", False)):
            n_a, _, _ = S4.twoway_cluster_mean(near.filter(pl.col("lowvol") == mask), "m0_all")
            f_a, _, _ = S4.twoway_cluster_mean(far.filter(pl.col("lowvol") == mask), "m0_all")
            print(f"\nopening near a round price (9,0,1) vs far (4,5,6), {lab}: "
                  f"{100*n_a:.2f}% vs {100*f_a:.2f}%  (gap {100*(n_a-f_a):+.2f} pp)")
            payload[f"nearfar_{mask}"] = {"near": n_a, "far": f_a}

        # ---- the paper's actual conditioning: stuck prices, not below-median
        # volatility. Ohta's Table 2 mechanism is a day whose price barely
        # moves, and a median split waters that down -- half the market still
        # traverses plenty of grid points. The bottom decile of the day's
        # price range is where the contamination has to show if it is real:
        # a stuck day that opened at digit 0 can print almost nothing else,
        # and a stuck day that opened at digits 4--7 can print almost no
        # round price at all.
        stuck_df = df.filter(pl.col("pmax").is_not_null()
                             & pl.col("pmin").is_not_null()
                             & (pl.col("open_px") > 0))
        stuck_df = stuck_df.with_columns(
            rng=(pl.col("pmax") - pl.col("pmin")) / pl.col("open_px"))
        stuck_df = stuck_df.with_columns(
            stuck=pl.col("rng") <= pl.col("rng").quantile(0.1).over("date"))
        print("\nM0 by opening digit on stuck days (bottom decile of daily "
              "range) versus mobile days:")
        print(f"{'digit':>6} {'stuck':>10} {'mobile':>10} {'gap':>8} {'n stuck':>9}")
        srows = []
        for d in range(10):
            lo = stuck_df.filter((pl.col("open_digit") == d) & pl.col("stuck"))
            hi = stuck_df.filter((pl.col("open_digit") == d) & ~pl.col("stuck"))
            if lo.height < 50 or hi.height < 50:
                continue
            a, _, _ = S4.twoway_cluster_mean(lo, "m0_all")
            b = float(hi["m0_all"].drop_nulls().mean())
            print(f"{d:>6} {100*a:>10.2f} {100*b:>10.2f} {100*(a-b):>+8.2f} "
                  f"{lo.height:>9,}")
            srows.append([str(d), f"{100*a:.2f}", f"{100*b:.2f}",
                          f"{100*(a-b):+.2f}", f"{lo.height:,}"])
            payload[f"stuck_digit{d}"] = {"stuck": a, "mobile": b,
                                          "n_stuck": lo.height}
        if srows:
            S4.latex_table(
                os.path.join(S4.TABLES, "t_opening_stuck.tex"),
                "The contamination on genuinely stuck days",
                "tab:openingstuck",
                ["Opening digit", "Stuck days (\\%)", "Mobile days (\\%)",
                 "Difference", "Stock-days, stuck"],
                srows,
                notes="Mean $M^{0}$ by the last digit of the opening price. "
                      "Stuck means the day's high--low range, scaled by the "
                      "opening price, falls in the bottom decile of that day's "
                      "cross-section -- the closest available analogue of "
                      "Ohta's Table 2 conditioning. The signature of the "
                      "mechanical contamination is a positive difference at "
                      "digit zero and negative differences in the middle "
                      "digits: a stuck day prints mostly its opening "
                      "neighbourhood, whatever digit that happens to be.")

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "opening.json"), payload)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
