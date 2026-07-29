"""S4 step 4 -- figures for the stylized-facts section."""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4

OUT = os.path.join(C.RESULTS, "s4_rq1")


def main() -> int:
    tee = C.Tee("s4_step4_figures")
    try:
        print("=== S4 step 4: figures ===\n")
        plt = S4.setup_mpl()
        C.ensure_dir(S4.FIGURES)
        df = S4.load_panel(final_only=True)
        df = S4.size_quintiles(df)

        # ---- F1: the digit distribution, by tick regime
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        regimes = [(10, "\\textyen 1 tick", S4.BLUE), (1, "\\textyen 0.1 tick", S4.ORANGE)]
        width = 0.38
        for i, (t, lab, col) in enumerate(regimes):
            sub = df.filter(pl.col("tick10") == t)
            if sub.height < 200:
                continue
            vals = [S4.twoway_cluster_mean(sub, f"m{d}_all")[0] for d in range(10)]
            ax.bar(np.arange(10) + (i - 0.5) * width, [100 * v for v in vals],
                   width=width, label=lab.replace("\\textyen ", "¥"), color=col)
        ax.axhline(10, color="0.35", lw=1, ls="--", zorder=0)
        ax.annotate("uniform benchmark", xy=(6.6, 10), xytext=(6.6, 11.6),
                    fontsize=8, color="0.35")
        ax.set_xticks(range(10))
        ax.set_xlabel("Last price digit")
        ax.set_ylabel("Share of traded volume (\\%)".replace("\\%", "%"))
        ax.legend(frameon=False, fontsize=9)
        fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_digits.pdf")))
        plt.close(fig)
        print("wrote f_digits.pdf")

        # ---- F2: the measures through the year
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        daily = (df.group_by("date").agg(
            b=pl.col("m_b_large0").mean(), s=pl.col("m_s_large0").mean(),
            bs=pl.col("m_b_small0").mean(), n=pl.len()).sort("date"))
        x = daily["date"].to_list()
        ax.plot(x, [100 * v for v in daily["b"]], color=S4.BLUE, lw=1.1,
                label="large, buy-initiated")
        ax.plot(x, [100 * v for v in daily["s"]], color=S4.ORANGE, lw=1.1,
                ls="--", label="large, sell-initiated")
        ax.plot(x, [100 * v for v in daily["bs"]], color="0.45", lw=1.0,
                ls=":", label="small, buy-initiated")
        ax.axhline(10, color="0.35", lw=0.8, ls="--", zorder=0)
        # The afternoon session lengthened on 2024-11-05. Mark it only when the
        # panel actually spans that date.
        import datetime as dt
        ext = C.TSE_CLOSE_EXTENSION
        if x and min(x) <= ext <= max(x):
            ax.axvline(ext, color="0.6", lw=0.8)
            ax.annotate("session extended to 15:30", xy=(ext, ax.get_ylim()[1]),
                        xytext=(-6, -12), textcoords="offset points", fontsize=7,
                        color="0.4", ha="right")
        ax.set_ylabel("Cross-sectional mean (%)")
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
        fig.autofmt_xdate()
        fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_timeseries.pdf")))
        plt.close(fig)
        print("wrote f_timeseries.pdf")

        # ---- F3: the impact premium with confidence intervals
        imp = C.read_json(os.path.join(OUT, "impact.json"), {})
        if imp:
            fig, ax = plt.subplots(figsize=(6.4, 3.0))
            labs, mus, ses, cols = [], [], [], []
            for h in (1, 60, 300):
                for tag, side, col in (("b", "buy", S4.BLUE), ("s", "sell", S4.ORANGE)):
                    k = f"dimp{h}_{tag}"
                    if k in imp and np.isfinite(imp[k]["pooled"]):
                        labs.append(f"{h}s\n{side}")
                        mus.append(imp[k]["pooled"])
                        ses.append(imp[k]["se"])
                        cols.append(col)
            xs = np.arange(len(mus))
            ax.bar(xs, mus, yerr=[1.96 * s for s in ses], color=cols, width=0.62,
                   error_kw=dict(lw=1, capsize=3, ecolor="0.3"))
            ax.axhline(0, color="0.3", lw=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels(labs, fontsize=8)
            ax.set_ylabel("Round-price impact premium (bp)")
            fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_dimp.pdf")))
            plt.close(fig)
            print("wrote f_dimp.pdf")

        # ---- F4: round-price resting depth against measured clustering
        if "rdepth_ask0" in df.columns and df["rdepth_ask0"].is_not_null().any():
            d = df.drop_nulls(["rdepth_ask0", "m_b_large0"])
            if d.height > 500:
                fig, ax = plt.subplots(figsize=(6.4, 3.2))
                q = d.with_columns(
                    bin=(pl.col("rdepth_ask0").rank("ordinal") * 20
                         // (d.height + 1)).cast(pl.Int32))
                g = q.group_by("bin").agg(x=pl.col("rdepth_ask0").mean(),
                                          y=pl.col("m_b_large0").mean(),
                                          n=pl.len()).sort("bin")
                ax.scatter([100 * v for v in g["x"]], [100 * v for v in g["y"]],
                           s=22, color=S4.BLUE, zorder=3)
                xs = np.array([100 * v for v in g["x"]])
                ys = np.array([100 * v for v in g["y"]])
                if len(xs) > 2:
                    b1, b0 = np.polyfit(xs, ys, 1)
                    ax.plot(xs, b0 + b1 * xs, color=S4.ORANGE, lw=1.2,
                            label=f"slope {b1:.2f}")
                    ax.legend(frameon=False, fontsize=9)
                ax.set_xlabel("Round-price share of visible ask depth (%)")
                ax.set_ylabel("$M^{BLarge0}$ (%)")
                fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_rdepth.pdf")))
                plt.close(fig)
                print("wrote f_rdepth.pdf")

        # ---- F6: the intraday shape of clustering, against the spread
        # Ohta (2006) reports that TSE clustering is greatest at the open and
        # decays through the day, which the price-resolution hypothesis predicts:
        # uncertainty about value is highest when trading starts, and a round
        # number is the cheapest substitute for an estimate. This is that curve,
        # computed from a sample two decades later.
        if os.path.exists(S4.PANEL_INTRADAY):
            idf = pl.read_parquet(S4.PANEL_INTRADAY)
            if "in_sample_final" in idf.columns:
                idf = idf.filter(pl.col("in_sample_final"))
            g = (idf.drop_nulls("m0").group_by("bucket")
                 .agg(m0=pl.col("m0").mean(),
                      es=pl.col("effsprd_bps").mean(), n=pl.len())
                 .sort("bucket"))
            if g.height >= 6:
                fig, ax = plt.subplots(figsize=(6.4, 3.2))
                b = g["bucket"].to_list()
                ax.plot(b, [100 * v for v in g["m0"]], color=S4.BLUE, lw=1.4,
                        marker="o", ms=3.5, label="round-price share (left)")
                ax.axhline(10, color="0.35", lw=0.8, ls="--", zorder=0)
                ax.set_xlabel("30-minute bucket (0 = 09:00, 10 = 15:00--15:30)")
                ax.set_ylabel("$M^{0}$ (%)", color=S4.BLUE)
                ax.tick_params(axis="y", labelcolor=S4.BLUE)
                ax2 = ax.twinx()
                ax2.plot(b, g["es"].to_list(), color=S4.ORANGE, lw=1.2, ls="--",
                         marker="s", ms=3, label="effective spread (right)")
                ax2.set_ylabel("Effective half-spread (bp)", color=S4.ORANGE)
                ax2.tick_params(axis="y", labelcolor=S4.ORANGE)
                ax2.spines["top"].set_visible(False)
                ax2.grid(False)
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
                fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_intraday.pdf")))
                plt.close(fig)
                print("wrote f_intraday.pdf")
                C.atomic_json(os.path.join(OUT, "intraday_shape.json"), {
                    "buckets": [{"bucket": r["bucket"], "m0": r["m0"],
                                 "effsprd_bps": r["es"], "n": r["n"]}
                                for r in g.iter_rows(named=True)]})

        # ---- F5: clustering across size quintiles
        fig, ax = plt.subplots(figsize=(6.4, 2.9))
        qs, vals, errs = [], [], []
        for q in range(1, 6):
            sub = df.filter(pl.col("size_q") == q)
            if sub.height < 200:
                continue
            mu, se, _ = S4.twoway_cluster_mean(sub, "m_b_large0")
            qs.append(f"Q{q}")
            vals.append(100 * mu)
            errs.append(196 * se)
        ax.bar(qs, vals, yerr=errs, color=S4.BLUE, width=0.6,
               error_kw=dict(lw=1, capsize=3, ecolor="0.3"))
        ax.axhline(10, color="0.35", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("Market-capitalisation quintile (Q1 smallest)")
        ax.set_ylabel("$M^{BLarge0}$ (%)")
        fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_size.pdf")))
        plt.close(fig)
        print("wrote f_size.pdf")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
