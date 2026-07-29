"""S5 step 4 -- robustness, and one test that could have killed the result.

The placebo is the important one. If the round-price share predicts liquidity
because round prices are where stale noise-trader orders sit, then the share of
volume at digit 7 should predict nothing. If digit 7 works just as well, the
finding is an artefact of how the measure is built -- something about volume
concentration or price paths -- and not about round numbers at all.
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

OUT = S5.OUT
Y = "ln_effsprd"


def main() -> int:
    tee = C.Tee("s5_step4_robustness")
    try:
        print("=== S5 step 4: robustness ===\n")
        raw_all = S4.load_panel(final_only=True, drop_fine_tick=False)
        raw = raw_all.filter(~pl.col("fine_tick_day"))
        df = S5.build_frame(raw)
        dums, df = S5.opening_digit_dummies(df)
        controls = [c for c in S5.CONTROLS if c in df.columns] + dums
        df = df.with_columns(fine_tick_day=pl.col("fine_tick_day").cast(pl.Float64))
        results = {}

        base = S5.fit(df, Y, [f"{S5.PRIMARY_X}_l1", f"{Y}_l1"] + controls)
        bb, bse = S5.cell(base, f"{S5.PRIMARY_X}_l1")
        print(f"baseline (dynamic, log effective spread): {bb} {bse}  "
              f"n={base.get('n',0):,}\n")
        results["baseline"] = {k: base.get(k) for k in ("coef", "se", "t", "n")}

        # ---- placebo digits
        print("--- placebo: the same specification with digits 1-9 ---")
        prows, pvals = [], []
        for d in range(10):
            col = f"m_b_large{d}_l1" if d else f"{S5.PRIMARY_X}_l1"
            if col not in df.columns:
                continue
            r = S5.fit(df, Y, [col, f"{Y}_l1"] + controls)
            b = r.get("coef", {}).get(col, float("nan"))
            se = r.get("se", {}).get(col, float("nan"))
            t = r.get("t", {}).get(col, float("nan"))
            flag = "  <- round price" if d == 0 else ("  (semi-focal)" if d == 5 else "")
            print(f"  digit {d}: beta={b:>9.4f} se={se:>8.4f} t={t:>6.2f}"
                  f"{S4.stars(t):<3}{flag}")
            prows.append([str(d), f"{b:.4f}{S4.stars(t)}", f"({se:.4f})", f"{t:.2f}"])
            pvals.append((d, b, se, t))
            results[f"placebo_d{d}"] = {"beta": b, "se": se, "t": t, "n": r.get("n")}
        S4.latex_table(
            os.path.join(S4.TABLES, "t_placebo.tex"),
            "Placebo test: the same regression run on every last price digit",
            "tab:placebo",
            ["Last digit", "Coefficient", "SE", "$t$"], prows,
            notes="Outcome is the log effective spread in the dynamic "
                  "specification of Table~\\ref{tab:rq2}, with the round-price "
                  "share replaced by the share of buy-initiated large-trade volume "
                  "at each last digit in turn. If the round-price result reflects "
                  "stale limit orders at psychologically salient prices, digits "
                  "other than zero should carry no comparable information. Digit "
                  "five is marked as semi-focal because half-way points attract "
                  "some of the same behaviour.")

        # ---- fine-tick days, excluded from the main sample, examined on their own
        print("\n--- 0.1-yen-tick days (excluded from the main regressions) ---")
        fine = S5.build_frame(raw_all.filter(pl.col("fine_tick_day")))
        if fine.height > 1000:
            dums_f, fine = S5.opening_digit_dummies(fine)
            cf = [c for c in S5.CONTROLS if c in fine.columns and c != "fine_tick_day"] + dums_f
            r = S5.fit(fine, Y, [f"{S5.PRIMARY_X}_l1", f"{Y}_l1"] + cf)
            b, se = S5.cell(r, f"{S5.PRIMARY_X}_l1")
            print(f"  {b} {se}  n={r.get('n',0):,}")
            results["fine_tick_only"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}
        else:
            print(f"  only {fine.height:,} stock-days; skipped")

        # ---- size heterogeneity
        print("\n--- by size quintile ---")
        dq = S4.size_quintiles(raw)
        srows = []
        for q in range(1, 6):
            sub = S5.build_frame(dq.filter(pl.col("size_q") == q))
            if sub.height < 2000:
                continue
            dq_dums, sub = S5.opening_digit_dummies(sub)
            cq = [c for c in S5.CONTROLS if c in sub.columns] + dq_dums
            sub = sub.with_columns(fine_tick_day=pl.col("fine_tick_day").cast(pl.Float64))
            r = S5.fit(sub, Y, [f"{S5.PRIMARY_X}_l1", f"{Y}_l1"] + cq)
            b, se = S5.cell(r, f"{S5.PRIMARY_X}_l1")
            t = r.get("t", {}).get(f"{S5.PRIMARY_X}_l1", float("nan"))
            print(f"  Q{q}: {b:<14} {se:<12} t={t:>6.2f}  n={r.get('n',0):,}")
            srows.append([f"Q{q}", b, se, f"{r.get('n',0):,}"])
            results[f"size_q{q}"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}

        # ---- sub-period stability, halves derived from the sample itself.
        # Never a hardcoded calendar date: the one time this was hardcoded the
        # study window later moved, H1 silently became empty, H2 became the
        # whole sample, and the table reported a check that no longer checked
        # anything. sample_halves() asserts a genuine partition.
        print("\n--- sub-period stability (split at the median trading date) ---")
        hrows = []
        for lab, sub in S5.sample_halves(df):
            r = S5.fit(sub, Y, [f"{S5.PRIMARY_X}_l1", f"{Y}_l1"] + controls)
            b, se = S5.cell(r, f"{S5.PRIMARY_X}_l1")
            print(f"  {lab}: {b:<14} {se:<12} n={r.get('n',0):,}")
            hrows.append([lab, b, se, f"{r.get('n',0):,}"])
            key = "half_H1" if lab.startswith("H1") else "half_H2"
            results[key] = {k: r.get(k) for k in ("coef", "se", "t", "n")}
            results[key]["label"] = lab

        # ---- event window: the same estimate with the wildest week removed.
        # The window is a fact about this sample (the April 2025 tariff shock)
        # and is applied only when it intersects the panel's dates, so the
        # code stays year-agnostic.
        xrows = []
        ds_all = df["date"].cast(pl.Utf8)
        for lo, hi in [("2025-04-03", "2025-04-11")]:
            if not ((ds_all >= lo) & (ds_all <= hi)).any():
                continue
            sub = df.filter(~ds_all.is_between(lo, hi))
            r = S5.fit(sub, Y, [f"{S5.PRIMARY_X}_l1", f"{Y}_l1"] + controls)
            b, se = S5.cell(r, f"{S5.PRIMARY_X}_l1")
            print(f"  excluding {lo}..{hi}: {b:<14} {se:<12} n={r.get('n',0):,}")
            xrows.append([f"Excluding {lo} to {hi}", b, se, f"{r.get('n',0):,}"])
            results["ex_crash"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}
            results["ex_crash"]["window"] = [lo, hi]

        # ---- deeper dynamics: five lags of the outcome instead of one.
        # "Adds information beyond the outcome's own history" should not hinge
        # on history meaning exactly one day.
        deep = df.sort(["ticker", "date"])
        deep_lags = []
        for k in range(2, 6):
            cname = f"{Y}_l{k}"
            deep = deep.with_columns(**{cname: pl.col(Y).shift(k).over("ticker")})
            deep_lags.append(cname)
        r = S5.fit(deep, Y, [f"{S5.PRIMARY_X}_l1", f"{Y}_l1"] + deep_lags + controls)
        b, se = S5.cell(r, f"{S5.PRIMARY_X}_l1")
        t = r.get("t", {}).get(f"{S5.PRIMARY_X}_l1", float("nan"))
        print(f"\n  five lags of the outcome: {b:<14} {se:<12} t={t:>6.2f}  "
              f"n={r.get('n',0):,}")
        xrows.append(["Five lags of the outcome", b, se, f"{r.get('n',0):,}"])
        results["deep_lags"] = {k2: r.get(k2) for k2 in ("coef", "se", "t", "n")}

        # ---- Fama-MacBeth as an estimator cross-check
        print("\n--- Fama-MacBeth cross-check (per-day cross-sections) ---")
        fm = df.select(["date", Y, f"{S5.PRIMARY_X}_l1", f"{Y}_l1"]).drop_nulls()
        betas = []
        for (d,), g in fm.group_by(["date"], maintain_order=True):
            if g.height < 40:
                continue
            X = np.column_stack([np.ones(g.height),
                                 g[f"{S5.PRIMARY_X}_l1"].to_numpy(),
                                 g[f"{Y}_l1"].to_numpy()])
            yv = g[Y].to_numpy()
            try:
                betas.append(float(np.linalg.lstsq(X, yv, rcond=None)[0][1]))
            except Exception:
                continue
        if len(betas) > 20:
            b = np.array(betas)
            mu = b.mean()
            # Newey-West with five lags, for the autocorrelation in daily betas.
            e = b - mu
            T = len(b)
            s = (e @ e) / T
            for L in range(1, 6):
                c = (e[L:] @ e[:-L]) / T
                s += 2 * (1 - L / 6) * c
            se = float(np.sqrt(max(s, 0) / T))
            print(f"  mean beta {mu:.4f}  NW(5) se {se:.4f}  t={mu/se:.2f}"
                  f"{S4.stars(mu/se)}  over {T} cross-sections")
            results["fama_macbeth"] = {"beta": mu, "se": se, "t": mu / se, "T": T}

        rrows = ([["Baseline (dynamic)", bb, bse, f"{base.get('n',0):,}"]]
                 + [["Size " + r[0], r[1], r[2], r[3]] for r in srows]
                 + [["Period " + r[0], r[1], r[2], r[3]] for r in hrows]
                 + xrows)
        if "fine_tick_only" in results:
            r = results["fine_tick_only"]
            k = f"{S5.PRIMARY_X}_l1"
            if k in r.get("coef", {}):
                rrows.append(["0.1-yen-tick days only",
                              f"{r['coef'][k]:.4f}{S4.stars(r['t'][k])}",
                              f"({r['se'][k]:.4f})", f"{r['n']:,}"])
        S4.latex_table(
            os.path.join(S4.TABLES, "t_robust.tex"),
            "Robustness of the clustering--liquidity relationship",
            "tab:robust",
            ["Specification", "Coefficient", "SE", "Stock-days"], rrows,
            notes="Each row re-estimates the dynamic specification of "
                  "Table~\\ref{tab:rq2} with the log effective spread as the "
                  "outcome, on the stated subsample or with the stated change. "
                  "Size quintiles are formed once per stock on median market "
                  "capitalisation. The period rows split the panel at its "
                  "median trading date, derived from the sample rather than "
                  "from a calendar. The event-window row removes the named "
                  "week. The five-lag row replaces the single lag of the "
                  "outcome with five, so the coefficient reads against a "
                  "richer version of the outcome's own history. "
                  "$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.")

        # ---- placebo figure
        if pvals:
            plt = S4.setup_mpl()
            fig, ax = plt.subplots(figsize=(6.4, 3.0))
            ds = [d for d, _, _, _ in pvals]
            bs = [b for _, b, _, _ in pvals]
            es = [1.96 * s for _, _, s, _ in pvals]
            cols = [S4.ORANGE if d == 0 else "0.55" for d in ds]
            ax.bar(ds, bs, yerr=es, color=cols, width=0.65,
                   error_kw=dict(lw=1, capsize=2.5, ecolor="0.3"))
            ax.axhline(0, color="0.3", lw=0.9)
            ax.set_xticks(range(10))
            ax.set_xlabel("Last price digit used to build the measure")
            ax.set_ylabel("Coefficient on log effective spread")
            fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_placebo.pdf")))
            plt.close(fig)
            print("\nwrote f_placebo.pdf")

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "robustness.json"), results)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
