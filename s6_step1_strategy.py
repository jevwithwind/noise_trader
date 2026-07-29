"""S6 -- a strategy demonstration, and what it is not.

This exists because the internship task is to build a *rebalancing strategy* on
this indicator, and the gap between "the indicator predicts liquidity" and "here
is a tradable rule" is exactly where such projects fail. So the machinery is
built and exercised end to end: form a tradable signal from information available
at the time, sort on it, rebalance, and charge realistic costs.

It is a demonstration of plumbing, not an alpha claim. One year, one market, no
out-of-sample period, and returns computed from raw closing prices that carry no
adjustment for splits or dividends. The honest output is the cost accounting and
the turnover, not the Sharpe ratio.

The signal is deliberately size-neutralised. Clustering is strongest in small,
retail-heavy stocks, so an unneutralised sort is mostly a small-cap bet wearing a
microstructure costume.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4
import s5_common as S5

OUT = os.path.join(C.RESULTS, "s6_strategy")
MAX_ABS_RET = 0.25   # a daily move beyond this is a corporate action, not a return


def residual_signal(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional residual of clustering on everything that mechanically drives it.

    Run separately each day, so nothing from the future enters. The controls are
    the same ones the panel regressions use: the opening-digit contamination, the
    relative tick, size, turnover, volatility and spread.
    """
    feats = [c for c in ["rel_tick", "ln_yenvol_l1", "ln_rv5_l1", "ln_effsprd_l1",
                         "ln_mktcap", "ret_overnight"] if c in df.columns]
    dig = [c for c in df.columns if c.startswith("d") and c.endswith("_lowvol")]
    feats += dig
    out = []
    for (d,), g in df.group_by(["date"], maintain_order=True):
        g = g.drop_nulls(["m_b_large0"] + feats)
        if g.height < 40:
            continue
        X = np.column_stack([np.ones(g.height)]
                            + [g[c].to_numpy().astype(float) for c in feats])
        y = g["m_b_large0"].to_numpy().astype(float)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
        except Exception:
            continue
        sd = r.std()
        out.append(g.select("date", "ticker", "close_px", "effsprd_bps", "imp60_bps")
                    .with_columns(signal=pl.Series("signal", r / sd if sd > 0 else r)))
    return pl.concat(out) if out else pl.DataFrame()


def main() -> int:
    tee = C.Tee("s6_step1_strategy")
    try:
        print("=== S6: strategy demonstration ===\n")
        raw = S4.load_panel(final_only=True, drop_fine_tick=True)
        df = S5.build_frame(raw, winsorize=False)
        _, df = S5.opening_digit_dummies(df)
        sig = residual_signal(df)
        if sig.is_empty():
            print("no signal could be formed")
            return 1
        print(f"signal formed on {sig.height:,} stock-days, "
              f"{sig['date'].n_unique()} days")

        # Forward return, within stock, with corporate actions screened out.
        sig = sig.sort(["ticker", "date"]).with_columns(
            nxt=pl.col("close_px").shift(-1).over("ticker"),
            same=(pl.col("ticker") == pl.col("ticker").shift(-1)))
        sig = sig.with_columns(
            fwd=pl.when(pl.col("same") & (pl.col("close_px") > 0))
            .then(pl.col("nxt") / pl.col("close_px") - 1.0).otherwise(None))
        n_before = sig.height
        sig = sig.drop_nulls("fwd").filter(pl.col("fwd").abs() <= MAX_ABS_RET)
        print(f"usable stock-day returns: {sig.height:,} "
              f"(dropped {n_before - sig.height:,} for missing or extreme moves)")

        # Quintile sort, rebalanced daily, equal weight, long the top and short the
        # bottom. The half-spread is charged on every unit of turnover on both legs.
        sig = sig.with_columns(
            q=(pl.col("signal").rank("ordinal").over("date") * 5
               // (pl.len().over("date") + 1) + 1).cast(pl.Int8))
        by_q = (sig.group_by(["date", "q"]).agg(
            r=pl.col("fwd").mean(), n=pl.len(),
            hs=pl.col("effsprd_bps").mean()).sort(["date", "q"]))

        print("\nnext-day return by signal quintile (Q5 = most clustered):")
        rows = []
        for q in range(1, 6):
            g = by_q.filter(pl.col("q") == q)
            r = g["r"].drop_nulls().to_numpy()
            if len(r) < 3:
                print(f"  Q{q}: only {len(r)} day(s); skipped")
                continue
            mu = float(r.mean()) * 1e4
            se = float(r.std(ddof=1)) / np.sqrt(len(r)) * 1e4
            t = mu / se if se > 0 else float("nan")
            print(f"  Q{q}: {mu:+7.2f} bp/day  (t={t:5.2f})  "
                  f"avg names {g['n'].mean():.0f}  eff. half-spread "
                  f"{g['hs'].mean():.2f} bp")
            rows.append([f"Q{q}", f"{mu:+.2f}", f"({se:.2f})", f"{t:.2f}",
                         f"{g['n'].mean():.0f}", f"{g['hs'].mean():.2f}"])

        wide = by_q.pivot(values="r", index="date", on="q").sort("date")
        qcols = [c for c in wide.columns if c != "date"]
        if "5" in qcols and "1" in qcols and wide.height >= 5:
            ls = (wide["5"] - wide["1"]).to_numpy().astype(float)
            ls = ls[np.isfinite(ls)]
        else:
            ls = np.array([])
        if len(ls) >= 5:
            mu, sd = float(ls.mean()), float(ls.std(ddof=1))
            se = sd / np.sqrt(len(ls))
            ann = mu * 245 * 100
            sharpe = mu / sd * np.sqrt(245) if sd > 0 else float("nan")

            # Turnover: how much of each leg is replaced from one day to the next.
            memb = sig.select("date", "ticker", "q").sort("date")
            days = memb["date"].unique().sort().to_list()
            turns = []
            prev = {}
            for d in days:
                cur = {}
                gg = memb.filter(pl.col("date") == d)
                for qq in (1, 5):
                    cur[qq] = set(gg.filter(pl.col("q") == qq)["ticker"].to_list())
                if prev:
                    t = []
                    for qq in (1, 5):
                        if prev.get(qq):
                            t.append(len(cur[qq] - prev[qq]) / max(len(prev[qq]), 1))
                    if t:
                        turns.append(float(np.mean(t)))
                prev = cur
            turnover = float(np.mean(turns)) if turns else float("nan")
            hs_bp = float(sig["effsprd_bps"].mean())
            # Both legs, both sides of the round trip.
            cost_bp = 2 * turnover * hs_bp
            net_bp = mu * 1e4 - cost_bp

            print(f"\nlong-short (Q5 minus Q1), rebalanced daily:")
            print(f"  gross      {mu*1e4:+7.2f} bp/day   t={mu/se:.2f}   "
                  f"annualised {ann:+.1f}%   Sharpe {sharpe:.2f}")
            print(f"  turnover   {100*turnover:.1f}% of each leg per day")
            print(f"  cost       {cost_bp:.2f} bp/day at a {hs_bp:.2f} bp "
                  f"effective half-spread")
            print(f"  net        {net_bp:+7.2f} bp/day")
            print("\n  The cost line is the point of this exercise. A signal that "
                  "must be refreshed daily in names this wide pays its gross "
                  "return away several times over; any real use of the indicator "
                  "has to either hold longer or trade it inside a rebalance that "
                  "was going to happen anyway.")

            rows.append(["Q5 - Q1 gross", f"{mu*1e4:+.2f}", f"({se*1e4:.2f})",
                         f"{mu/se:.2f}", "--", f"{hs_bp:.2f}"])
            rows.append(["Q5 - Q1 net of costs", f"{net_bp:+.2f}", "--", "--",
                         "--", "--"])

            C.ensure_dir(OUT)
            C.atomic_json(os.path.join(OUT, "strategy.json"), {
                "gross_bp_day": mu * 1e4, "se_bp": se * 1e4, "t": mu / se,
                "annualised_pct": ann, "sharpe": sharpe, "turnover": turnover,
                "cost_bp_day": cost_bp, "net_bp_day": net_bp,
                "half_spread_bp": hs_bp, "n_days": len(ls),
                "max_abs_ret_filter": MAX_ABS_RET})

            # Cumulative gross and net paths.
            plt = S4.setup_mpl()
            fig, ax = plt.subplots(figsize=(6.4, 3.2))
            dts = wide["date"].to_list()[:len(ls)]
            ax.plot(dts, 100 * np.cumsum(ls), color=S4.BLUE, lw=1.2, label="gross")
            ax.plot(dts, 100 * np.cumsum(ls - cost_bp / 1e4), color=S4.ORANGE,
                    lw=1.2, ls="--", label="net of estimated costs")
            ax.axhline(0, color="0.35", lw=0.8)
            ax.set_ylabel("Cumulative return (%)")
            ax.legend(frameon=False, fontsize=9)
            fig.autofmt_xdate()
            fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_strategy.pdf")))
            plt.close(fig)
            print("\nwrote f_strategy.pdf")

        S4.latex_table(
            os.path.join(S4.TABLES, "t_strategy.tex"),
            "Next-day returns to a size-neutralised clustering signal",
            "tab:strategy",
            ["Portfolio", "Return (bp/day)", "SE", "$t$", "Names", "Eff. half-spread (bp)"],
            rows,
            notes="The signal is the daily cross-sectional residual of "
                  "$M^{BLarge0}$ on the opening-digit contamination controls, "
                  "relative tick size, log market capitalisation, lagged log "
                  "turnover, lagged volatility, lagged spread and the overnight "
                  "return, standardised within the day. Portfolios are equally "
                  "weighted and rebalanced daily. Costs charge the measured "
                  "effective half-spread on both legs of the observed turnover. "
                  "Stock-days with an absolute move above 25\\% are dropped, since "
                  "closing prices here carry no adjustment for splits or "
                  "dividends. This is a demonstration that the pipeline runs end "
                  "to end, not evidence of a profitable strategy: one year, one "
                  "market, and no out-of-sample period.")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
