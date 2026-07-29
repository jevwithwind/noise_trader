"""S5 step 5 -- stage report for the regression results."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s5_reg")


def main() -> int:
    rq2 = C.read_json(os.path.join(OUT, "rq2_daily.json"), {})
    rq3 = C.read_json(os.path.join(OUT, "rq3_book.json"), {})
    intr = C.read_json(os.path.join(OUT, "intraday.json"), {})
    rob = C.read_json(os.path.join(OUT, "robustness.json"), {})

    def cell(d, key, coefname):
        r = d.get(key, {})
        b = (r.get("coef") or {}).get(coefname)
        t = (r.get("t") or {}).get(coefname)
        n = r.get("n")
        if b is None:
            return "--", "--", (f"{n:,}" if n else "--")
        return f"{b:.4f}", f"{t:.2f}" if t is not None else "--", f"{n:,}"

    x = "m_b_large0_l1"
    lines = ["# S5 -- clustering, liquidity and the order book", "",
             "All specifications carry stock and day fixed effects with standard "
             "errors clustered on both margins. Days on the 0.1-yen grid are "
             "excluded, following the paper. Nothing here is causal: these are "
             "predictive associations within a single calendar year.", "",
             "## Next-day liquidity (dynamic specification)", "",
             "The dynamic specification includes the outcome's own lag, so the "
             "coefficient is the incremental content of clustering given what the "
             "outcome's own history already said.", "",
             "| Outcome | Coefficient | t | Stock-days |", "|---|---|---|---|"]
    for y, lab in (("ln_effsprd", "Effective spread (log)"),
                   ("imp60_bps", "Price impact, 60s"),
                   ("rs60_bps", "Realized spread, 60s"),
                   ("ln_depth_best", "Depth at best (log)"),
                   ("ln_rv5", "Realised variance (log)"),
                   ("vr_absdev", "Variance-ratio deviation"),
                   ("ln_amihud", "Amihud illiquidity (log)")):
        b, t, n = cell(rq2, f"{y}|dynamic", x)
        lines.append(f"| {lab} | {b} | {t} | {n} |")

    lines += ["", "## Order-book anatomy", "",
              "| Dependent variable | Regressor | Coefficient | t |",
              "|---|---|---|---|"]
    for y, lab in (("m_b_large0", "M^BLarge0"),
                   ("dimp60_b_large", "Delta-Imp^BLarge")):
        for v, vlab in (("l_s0_l1", "L^S0"), ("l_s0c_l1", "L^S0C"),
                        ("rdepth_ask0_l1", "RDepth^ask0")):
            b, t, _ = cell(rq3, y, v)
            if b != "--":
                lines.append(f"| {lab} | {vlab} | {b} | {t} |")

    lines += ["", "## Within the day (30-minute buckets)", "",
              "Stock-day and bucket-of-day fixed effects, so only within-day "
              "variation identifies these.", "",
              "| Outcome | Regressor | Coefficient | t | Rows |", "|---|---|---|---|---|"]
    for k in intr:
        if "|" not in k:
            continue
        y, v = k.split("|")
        b, t, n = cell(intr, k, v)
        lines.append(f"| {y} | {v} | {b} | {t} | {n} |")
    oi = intr.get("ofi_interaction", {})
    if oi.get("coef"):
        b = (oi["coef"] or {}).get("ofi_x_high")
        t = (oi.get("t") or {}).get("ofi_x_high")
        if b is not None:
            lines += ["", f"Order-flow sensitivity interaction: {b:.4f} "
                          f"(t={t:.2f}). A positive value means the same order-flow "
                          "imbalance moves the midquote further on high-clustering "
                          "days."]

    lines += ["", "## Placebo across last digits", "",
              "If a share of volume at any digit predicted liquidity equally well, "
              "the round-price result would be an artefact of the measure's "
              "construction rather than a fact about round numbers.", "",
              "| Digit | Coefficient | t |", "|---|---|---|"]
    n_sig = 0
    for d in range(10):
        r = rob.get(f"placebo_d{d}", {})
        b, t = r.get("beta"), r.get("t")
        if b is None:
            continue
        mark = " **(round price)**" if d == 0 else (" (semi-focal)" if d == 5 else "")
        lines.append(f"| {d}{mark} | {b:.4f} | {t:.2f} |")
        if d not in (0, 5) and t is not None and abs(t) > 1.96:
            n_sig += 1
    lines += ["", f"Placebo digits (excluding 0 and the semi-focal 5) reaching "
                  f"conventional significance: **{n_sig} of 8**.", ""]

    fm = rob.get("fama_macbeth", {})
    if fm:
        lines += ["## Estimator cross-check", "",
                  f"Fama--MacBeth over {fm.get('T', 0)} daily cross-sections with "
                  f"Newey--West standard errors: {fm.get('beta', 0):.4f} "
                  f"(t={fm.get('t', 0):.2f}). Reported because it makes different "
                  "assumptions about the correlation structure than two-way "
                  "clustering does.", ""]

    lines += ["## Verdict", "",
              "The dynamic specification is the one to read: it asks whether "
              "clustering adds information beyond the outcome's own history, in a "
              "design that absorbs every stock-level and market-wide effect. The "
              "placebo is what separates the finding from an artefact.", ""]

    path = os.path.join(OUT, "s5_report.md")
    with open(C.write_guard(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
