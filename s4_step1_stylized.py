"""S4 step 1 -- does 2024 look like the paper's 1997-2022?

The paper's sample ends in 2022. This is the first look at its measures two years
later, and it is also the last chance to catch a pipeline that is quietly wrong:
if the 2024 tape does not reproduce the paper's stylized facts, the explanation
is far more likely to be a bug here than a change in the market.
"""
from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4

OUT = os.path.join(C.RESULTS, "s4_rq1")

# Post-2010 benchmarks from Ohta (2026), Table 1 and Section 3.
BENCH = {"m_b_large0": 0.148, "m_s_large0": 0.143,
         "m_b_small0": 0.111, "m_s_small0": 0.109, "m0_all": 0.135}


def main() -> int:
    tee = C.Tee("s4_step1_stylized")
    try:
        print("=== S4 step 1: 2024 stylized facts ===\n")
        df = S4.load_panel(final_only=True)
        df = S4.size_quintiles(df)
        print(f"stock-days in sample: {df.height:,}   "
              f"stocks: {df['ticker'].n_unique():,}   dates: {df['date'].n_unique()}")

        # ---- headline measures, with two-way clustered standard errors
        print("\nclustering measures (volume-weighted shares at the round price):")
        print(f"{'measure':<14} {'2024 mean':>10} {'SE':>8} {'paper':>8} "
              f"{'diff':>8} {'n':>10}")
        rows, payload = [], {}
        for m in ["m0_all", "m_b_large0", "m_s_large0", "m_b_small0", "m_s_small0"]:
            mu, se, n = S4.twoway_cluster_mean(df, m)
            b = BENCH[m]
            print(f"{m:<14} {100*mu:>10.2f} {100*se:>8.3f} {100*b:>8.1f} "
                  f"{100*(mu-b):>+8.2f} {n:>10,}")
            rows.append([f"$M^{{{_tex(m)}}}$", f"{100*mu:.2f}", f"({100*se:.3f})",
                         f"{100*b:.1f}", f"{100*(mu-b):+.2f}", f"{n:,}"])
            payload[m] = {"mean": mu, "se": se, "n": n, "paper": b}

        S4.latex_table(
            os.path.join(S4.TABLES, "t_stylized.tex"),
            "Price clustering on the Tokyo Stock Exchange in 2024, against the "
            "published 2010--2022 benchmarks",
            "tab:stylized",
            ["Measure", "2024 mean (\\%)", "SE", "Ohta (\\%)", "Difference", "Stock-days"],
            rows,
            notes="Volume-weighted share of trading executing at a last price digit "
                  "of zero, by trade initiator and trade size. Large means a trade "
                  "larger than one trading unit. Standard errors are clustered by "
                  "stock and by day. The benchmark column reports Ohta's (2026) "
                  "2010--2022 means. Under a uniform digit distribution every "
                  "measure would be 10\\%.")

        # ---- the ordering the mechanism predicts
        print("\nlarge versus small trades (the mechanism's central prediction):")
        for side, tag in (("buy-initiated", "b"), ("sell-initiated", "s")):
            lg, _, _ = S4.twoway_cluster_mean(df, f"m_{tag}_large0")
            sm, _, _ = S4.twoway_cluster_mean(df, f"m_{tag}_small0")
            d = df.drop_nulls([f"m_{tag}_large0", f"m_{tag}_small0"]).with_columns(
                gap=pl.col(f"m_{tag}_large0") - pl.col(f"m_{tag}_small0"))
            g, gse, gn = S4.twoway_cluster_mean(d, "gap")
            t = g / gse if gse else float("nan")
            print(f"  {side:<16} large {100*lg:.2f}%  small {100*sm:.2f}%  "
                  f"gap {100*g:+.2f} pp (t={t:.1f}){S4.stars(t)}")
            payload[f"gap_{tag}"] = {"gap": g, "se": gse, "t": t, "n": gn}

        # ---- the digit distribution, which is where a bug would show
        print("\nvolume-weighted last-digit distribution:")
        dist = []
        for d in range(10):
            mu, se, _ = S4.twoway_cluster_mean(df, f"m{d}_all")
            dist.append((d, mu, se))
            bar = "#" * int(round(200 * mu))
            print(f"  digit {d}: {100*mu:6.2f}%  {bar}")
        tot = sum(m for _, m, _ in dist)
        print(f"  sum {100*tot:.4f}%   (must be 100)")
        payload["digit_dist"] = [{"digit": d, "mean": m, "se": s} for d, m, s in dist]

        # ---- tick regime: finer grids cluster more
        print("\nby tick size:")
        trows = []
        for t in sorted(x for x in df["tick10"].unique().to_list() if x is not None):
            sub = df.filter(pl.col("tick10") == t)
            if sub.height < 200:
                continue
            mu, se, n = S4.twoway_cluster_mean(sub, "m_b_large0")
            m0, _, _ = S4.twoway_cluster_mean(sub, "m0_all")
            print(f"  {t/10:>6.1f} yen: M0 {100*m0:6.2f}%  M^BLarge0 {100*mu:6.2f}% "
                  f"({100*se:.3f})  n={n:,}  stocks={sub['ticker'].n_unique()}")
            trows.append([f"{t/10:g}", f"{100*m0:.2f}", f"{100*mu:.2f}",
                          f"({100*se:.3f})", f"{n:,}", f"{sub['ticker'].n_unique():,}"])
            payload[f"tick_{t}"] = {"m0": m0, "m_b_large0": mu, "se": se, "n": n}
        S4.latex_table(
            os.path.join(S4.TABLES, "t_tick.tex"),
            "Price clustering by tick size", "tab:tick",
            ["Tick (\\textyen)", "$M^{0}$ (\\%)", "$M^{BLarge0}$ (\\%)", "SE",
             "Stock-days", "Stocks"], trows,
            notes="Ohta's sample filter admits only tick sizes that are powers of "
                  "ten, since the last-digit construction is otherwise undefined. "
                  "Harris (1991) predicts more clustering on finer grids.")

        # ---- size: clustering is a small-cap phenomenon
        print("\nby size quintile (1 = smallest):")
        qrows = []
        for q in range(1, 6):
            sub = df.filter(pl.col("size_q") == q)
            if sub.height < 200:
                continue
            mu, se, n = S4.twoway_cluster_mean(sub, "m_b_large0")
            m0, _, _ = S4.twoway_cluster_mean(sub, "m0_all")
            print(f"  Q{q}: M0 {100*m0:6.2f}%  M^BLarge0 {100*mu:6.2f}% "
                  f"({100*se:.3f})  n={n:,}")
            qrows.append([f"Q{q}", f"{100*m0:.2f}", f"{100*mu:.2f}",
                          f"({100*se:.3f})", f"{n:,}"])
            payload[f"size_q{q}"] = {"m0": m0, "m_b_large0": mu, "se": se, "n": n}
        S4.latex_table(
            os.path.join(S4.TABLES, "t_size.tex"),
            "Price clustering by market-capitalisation quintile", "tab:size",
            ["Quintile", "$M^{0}$ (\\%)", "$M^{BLarge0}$ (\\%)", "SE", "Stock-days"],
            qrows,
            notes="Quintiles are formed once per stock on its median market "
                  "capitalisation over 2024, so a stock does not drift between "
                  "quintiles as its price moves. Q1 is the smallest.")

        # ---- round-price depth: the measure the paper's data could not support
        if "rdepth_ask0" in df.columns and df["rdepth_ask0"].is_not_null().any():
            print("\nround-price share of visible ten-level depth "
                  "(10% under a uniform grid):")
            for c, lab in (("rdepth_ask0", "ask side"), ("rdepth_bid0", "bid side")):
                mu, se, n = S4.twoway_cluster_mean(df, c)
                print(f"  {lab}: {100*mu:.2f}% ({100*se:.3f})  n={n:,}")
                payload[c] = {"mean": mu, "se": se, "n": n}

        # ---- persistence: how much within-stock variation there is to work with
        d = df.sort(["ticker", "date"]).with_columns(
            lag=pl.col("m_b_large0").shift(1).over("ticker"),
            same=pl.col("ticker") == pl.col("ticker").shift(1))
        d = d.filter(pl.col("same")).drop_nulls(["m_b_large0", "lag"])
        rho = (float(d.select(pl.corr("m_b_large0", "lag")).item())
               if d.height > 30 else float("nan"))
        print(f"\nfirst-order autocorrelation of M^BLarge0 within stock: {rho:.3f}")
        print("  (high persistence means the identifying variation in a "
              "stock-and-day fixed-effects panel is smaller than the row count "
              "suggests -- reported so the reader can judge the power)")
        payload["ar1_m_b_large0"] = rho

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "stylized.json"), payload)
        return 0
    finally:
        tee.close()


def _tex(m: str) -> str:
    return {"m0_all": "0", "m_b_large0": "BLarge0", "m_s_large0": "SLarge0",
            "m_b_small0": "BSmall0", "m_s_small0": "SSmall0"}[m]


if __name__ == "__main__":
    raise SystemExit(main())
