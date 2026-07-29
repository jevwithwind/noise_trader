"""S5 step 2 -- the book-level anatomy: Ohta's equations (1) and (2), rebuilt.

Ohta explains clustering by an order-flow story: noise traders leave round-price
limit orders standing, those orders are cancelled slowly, and fast participants
take them with large market orders. He tests it with limit-order submission and
cancellation shares plus two proxies for individual activity -- margin balances
and ownership -- neither of which is available here.

What is available is better in one respect. This study can see the standing
round-price inventory directly, as a share of visible ten-level depth, where the
paper could only see it after it had been executed against. So the equations are
re-estimated with the book's own variables in place of the individual-activity
proxies:

  eq (1) analogue:  M^{BLarge0}_t  ~ L^{S0}_{t-1} + L^{S0C}_{t-1} + RDepth^{ask0}_{t-1}
  eq (2) analogue:  DImp^{BLarge}_t ~ the same

The hypothesis predicts a positive coefficient on submissions at round prices, a
negative one on their cancellation ratio (orders that are not cancelled are the
ones that get picked off), and a positive one on standing round-price depth.
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
    tee = C.Tee("s5_step2_rq3_book")
    try:
        print("=== S5 step 2: book anatomy (equations 1 and 2 analogues) ===\n")
        raw = S4.load_panel(final_only=True, drop_fine_tick=True)
        df = S5.build_frame(raw)
        dums, df = S5.opening_digit_dummies(df)
        controls = [c for c in S5.CONTROLS if c in df.columns] + dums
        df = df.with_columns(fine_tick_day=pl.col("fine_tick_day").cast(pl.Float64))

        have_ladder = ("l_s0_l1" in df.columns
                       and df["l_s0_l1"].is_not_null().sum() > 500)
        print(f"ladder variables available: {have_ladder}")
        if have_ladder:
            print(f"  stock-days with L^S0: {df['l_s0_l1'].is_not_null().sum():,}")

        results, rows = {}, []
        # Buy side: sell limit orders are what a buy market order consumes, so the
        # sell-side book variables are the relevant ones -- as in the paper.
        specs = [
            ("m_b_large0", "s", "ask", "$M^{BLarge0}$"),
            ("dimp60_b_large", "s", "ask", "$\\Delta\\mathit{Imp}^{BLarge}$"),
            ("m_s_large0", "b", "bid", "$M^{SLarge0}$"),
            ("dimp60_s_large", "b", "bid", "$\\Delta\\mathit{Imp}^{SLarge}$"),
        ]
        for y, side, dside, label in specs:
            if y not in df.columns:
                continue
            xs = []
            if have_ladder:
                xs += [f"l_{side}0_l1", f"l_{side}0c_l1"]
            rd = f"rdepth_{dside}0_l1"
            if rd in df.columns and df[rd].is_not_null().sum() > 500:
                xs.append(rd)
            if not xs:
                print(f"--- {label}: no book regressors available, skipped")
                continue
            r = S5.fit(df, y, xs + controls)
            print(f"--- {label} ---   n={r.get('n', 0):,}"
                  + (f"  [{r['error'][:60]}]" if "error" in r else ""))
            row = [label]
            for v in xs:
                b, se = S5.cell(r, v)
                t = r.get("t", {}).get(v, float("nan"))
                print(f"  {v:<16} {b:<14} {se:<12} t={t:>6.2f}")
                row += [b, se]
            row.append(f"{r.get('n', 0):,}")
            rows.append(row)
            results[y] = {k: r.get(k) for k in ("coef", "se", "t", "n", "r2")}
            print()

        if rows:
            # Different dependent variables can end up with different regressor
            # sets when a book variable is missing on one side, which would make
            # the table ragged and LaTeX refuse it. Pad every row to the widest.
            width = max(len(r) for r in rows)
            rows = [r[:-1] + [""] * (width - len(r)) + [r[-1]] for r in rows]
            ncoef = (width - 2) // 2
            header = ["Dependent variable"]
            names = (["$L^{0}$", "SE", "$L^{0C}$", "SE", "RDepth", "SE"]
                     if have_ladder else ["RDepth", "SE"])
            names = (names + ["", "SE"] * ncoef)[:2 * ncoef]
            header += names + ["Stock-days"]
            S4.latex_table(
                os.path.join(S4.TABLES, "t_rq3.tex"),
                "Order-book anatomy of price clustering",
                "tab:rq3", header, rows,
                notes="Analogues of Ohta's (2026) equations (1) and (2), with "
                      "book-derived variables replacing the margin-trading and "
                      "ownership proxies, which are not available here. $L^{0}$ is "
                      "the round-price share of limit-order volume submitted "
                      "within the visible book, $L^{0C}$ the ratio of cancelled to "
                      "submitted volume at round prices, and RDepth the round-price "
                      "share of visible ten-level resting depth. Sell-side book "
                      "variables are paired with buy-initiated outcomes, since a "
                      "buy market order executes against sell limit orders. All "
                      "regressors are lagged one day. Stock and day fixed effects, "
                      "standard errors clustered by stock and by day. The "
                      "hypothesis predicts a positive coefficient on $L^{0}$ and "
                      "RDepth and a negative one on $L^{0C}$. "
                      "$^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.")

        # Does standing round-price depth predict liquidity directly?
        print("--- round-price resting depth as a liquidity predictor ---")
        lrows = []
        for y, lab in (("ln_effsprd", "Effective spread (log)"),
                       ("imp60_bps", "Price impact, 60s (bp)"),
                       ("ln_depth_best", "Depth at best (log)")):
            if y not in df.columns or "rdepth_ask0_l1" not in df.columns:
                continue
            r = S5.fit(df, y, ["rdepth_ask0_l1", "rdepth_bid0_l1", f"{y}_l1"] + controls)
            ba, sea = S5.cell(r, "rdepth_ask0_l1")
            bb, seb = S5.cell(r, "rdepth_bid0_l1")
            print(f"  {lab:<28} ask {ba:<14} bid {bb:<14} n={r.get('n',0):,}")
            lrows.append([lab, ba, sea, bb, seb, f"{r.get('n',0):,}"])
            results[f"rdepth|{y}"] = {k: r.get(k) for k in ("coef", "se", "t", "n")}
        if lrows:
            S4.latex_table(
                os.path.join(S4.TABLES, "t_rdepth.tex"),
                "Standing round-price depth and next-day liquidity",
                "tab:rdepth",
                ["Outcome", "RDepth ask", "SE", "RDepth bid", "SE", "Stock-days"],
                lrows,
                notes="RDepth is the share of visible ten-level resting volume "
                      "sitting at round prices, time-averaged over the day and "
                      "lagged one day. Dynamic specification, controls and fixed "
                      "effects as in Table~\\ref{tab:rq2}.")

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "rq3_book.json"), results)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
