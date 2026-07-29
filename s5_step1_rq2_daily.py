"""S5 step 1 -- does clustering carry information about tomorrow's liquidity?

The paper establishes that clustering measures noise-trader activity. Its
conclusion asks the next question: what does that activity do to price formation
and liquidity? This is that regression, at daily frequency.

Three specifications per outcome, and the differences between them matter:

  contemporaneous  y_t on M_t          -- descriptive association only
  predictive       y_t on M_{t-1}      -- ordering fixed, but confounded by
                                          the persistence of both series
  dynamic          y_t on M_{t-1}, y_{t-1}  -- the headline

The dynamic specification is the one the report leads with, because it is the
only one that answers a question worth asking: does yesterday's clustering say
anything about today's liquidity that yesterday's liquidity did not already say?
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

OUTCOMES = [
    ("ln_effsprd", "Effective spread (log)", True),
    ("imp60_bps", "Price impact, 60s (bp)", True),
    ("rs60_bps", "Realized spread, 60s (bp)", False),
    ("ln_depth_best", "Depth at best quote (log)", False),
    ("ln_rv5", "Realised variance (log)", False),
    ("vr_absdev", "Variance-ratio deviation", False),
    ("ln_amihud", "Amihud illiquidity (log)", False),
]


def main() -> int:
    tee = C.Tee("s5_step1_rq2_daily")
    try:
        print("=== S5 step 1: clustering and next-day liquidity ===\n")
        raw = S4.load_panel(final_only=True, drop_fine_tick=True)
        print(f"stock-days after dropping 0.1-yen-tick days: {raw.height:,}")
        df = S5.build_frame(raw)
        dums, df = S5.opening_digit_dummies(df)
        controls = [c for c in S5.CONTROLS if c in df.columns] + dums
        df = df.with_columns(fine_tick_day=pl.col("fine_tick_day").cast(pl.Float64))

        x = S5.PRIMARY_X
        xl = f"{x}_l1"
        print(f"primary regressor: {xl}   controls: {len(controls)}\n")

        results, rows = {}, []
        for y, label, primary in OUTCOMES:
            if y not in df.columns:
                continue
            yl = f"{y}_l1"
            specs = {
                "contemporaneous": [x] + controls,
                "predictive": [xl] + controls,
                "dynamic": [xl, yl] + controls,
            }
            print(f"--- {label} ---")
            row = [label + ("" if primary else "$^{\\dagger}$")]
            for name, xs in specs.items():
                r = S5.fit(df, y, xs)
                key = x if name == "contemporaneous" else xl
                b, se = S5.cell(r, key)
                t = r.get("t", {}).get(key, float("nan"))
                print(f"  {name:<16} beta={b:<14} se={se:<12} n={r.get('n', 0):>8,}"
                      + (f"  [{r['error'][:50]}]" if "error" in r else ""))
                results[f"{y}|{name}"] = {k: r.get(k) for k in ("coef", "se", "t", "n", "r2")}
                if name == "dynamic":
                    row += [b, se, f"{r.get('n', 0):,}"]
                elif name == "predictive":
                    row.insert(1, b)
                    row.insert(2, se)
            rows.append(row)
            print()

        S4.latex_table(
            os.path.join(S4.TABLES, "t_rq2.tex"),
            "Price clustering and next-day liquidity",
            "tab:rq2",
            ["Outcome", "Predictive", "SE", "Dynamic", "SE", "Stock-days"],
            rows,
            notes="Each cell is the coefficient on $M^{BLarge0}_{t-1}$, the "
                  "round-price share of buy-initiated large-trade volume. All "
                  "specifications carry stock and day fixed effects and the full "
                  "control set (relative tick size, lagged log turnover, lagged "
                  "log realised variance, lagged log effective spread, the "
                  "overnight return, and the opening-digit by low-volatility "
                  "interactions). The dynamic column adds the outcome's own lag, "
                  "so its coefficient is the incremental predictive content of "
                  "clustering given what yesterday's liquidity already said. "
                  "Standard errors are clustered by stock and by day. Days on the "
                  "0.1-yen grid are excluded, following Ohta (2026). "
                  "$^{\\dagger}$ secondary outcome, not pre-specified. "
                  "$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$. "
                  "These are predictive associations in a single year, not "
                  "causal estimates.")

        # Other clustering measures as the regressor, to see whether the result is
        # specific to large trades the way the mechanism says it should be.
        print("--- substituting other clustering measures (outcome: log effective spread) ---")
        srows = []
        for alt in ["m_b_large0", "m_s_large0", "m_b_small0", "m_s_small0", "m0_all"]:
            al = f"{alt}_l1"
            if al not in df.columns:
                continue
            r = S5.fit(df, "ln_effsprd", [al, "ln_effsprd_l1"] + controls)
            b, se = S5.cell(r, al)
            t = r.get("t", {}).get(al, float("nan"))
            print(f"  {alt:<14} beta={b:<14} se={se:<12} t={t:>6.2f}  n={r.get('n',0):,}")
            srows.append([f"$M^{{{alt}}}$".replace("m_b_large0", "BLarge0")
                          .replace("m_s_large0", "SLarge0")
                          .replace("m_b_small0", "BSmall0")
                          .replace("m_s_small0", "SSmall0")
                          .replace("m0_all", "0"), b, se, f"{r.get('n',0):,}"])
            results[f"alt|{alt}"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}
        S4.latex_table(
            os.path.join(S4.TABLES, "t_rq2_alt.tex"),
            "Which clustering measure carries the information?",
            "tab:rq2alt",
            ["Regressor (lagged)", "Coefficient", "SE", "Stock-days"], srows,
            notes="Outcome is the log effective spread; the dynamic specification "
                  "of Table~\\ref{tab:rq2} with the clustering measure replaced. "
                  "The mechanism concerns large trades executing against stale "
                  "round-price limit orders, so the large-trade measures are where "
                  "an effect should appear.")

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "rq2_daily.json"), results)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
