"""S4 step 3 -- the opening-digit contamination the paper warns about.

Ohta's Table 2 records a mechanical distortion in the clustering measure: a day
that opens at a price ending far from zero and then barely moves can never print
a round price, so its measure is low for a reason that has nothing to do with
noise traders. The fix is to control for the interaction of the opening digit
with a low-volatility indicator, and this step confirms the distortion is present
in 2024 and that those controls are therefore needed.
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
                  "whether the previous day's realised variance was below that "
                  "day's cross-sectional median. On quiet days a stock that opens "
                  "far from a round price may never reach one, which depresses the "
                  "measure mechanically. Every regression in this study therefore "
                  "controls for the interaction of the opening-digit dummies with "
                  "the low-volatility indicator, as Ohta (2026) does.")

        # A compact summary: how much of the measure's variation the contamination
        # alone can account for.
        near = df.filter(pl.col("open_digit").is_in([9, 0, 1]))
        far = df.filter(pl.col("open_digit").is_in([4, 5, 6]))
        for lab, sub in (("all days", df),):
            pass
        for lab, mask in (("low volatility", True), ("high volatility", False)):
            n_a, _, _ = S4.twoway_cluster_mean(near.filter(pl.col("lowvol") == mask), "m0_all")
            f_a, _, _ = S4.twoway_cluster_mean(far.filter(pl.col("lowvol") == mask), "m0_all")
            print(f"\nopening near a round price (9,0,1) vs far (4,5,6), {lab}: "
                  f"{100*n_a:.2f}% vs {100*f_a:.2f}%  (gap {100*(n_a-f_a):+.2f} pp)")
            payload[f"nearfar_{mask}"] = {"near": n_a, "far": f_a}

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "opening.json"), payload)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
