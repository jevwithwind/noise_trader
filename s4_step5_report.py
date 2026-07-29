"""S4 step 5 -- stage report, and the stop rule.

Three of the paper's findings are treated as necessary conditions rather than
results. They are not delicate effects, so a pipeline that fails to reproduce
them is far more likely to be broken than to have discovered something. This
stage halts rather than letting later stages build on a broken measurement.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s4_rq1")


def main() -> int:
    st = C.read_json(os.path.join(OUT, "stylized.json"), {})
    im = C.read_json(os.path.join(OUT, "impact.json"), {})

    def g(d, k, f, default=None):
        return (d.get(k) or {}).get(f, default)

    checks = []

    # 1. Clustering concentrates in large trades.
    gap = g(st, "gap_b", "gap")
    gap_t = g(st, "gap_b", "t")
    checks.append((
        "clustering is higher in large trades than small ones",
        gap is not None and gap > 0 and (gap_t or 0) > 2,
        f"gap {100*(gap or 0):+.2f} pp (t={gap_t:.1f})" if gap is not None else "not computed"))

    # 2. Round-price trades carry a positive impact premium.
    d_b = g(im, "dimp60_b", "pooled")
    d_s = g(im, "dimp60_s", "pooled")
    ok_imp = (d_b is not None and d_b > 0) or (d_s is not None and d_s > 0)
    checks.append((
        "trades at round prices carry a positive price-impact premium",
        ok_imp,
        f"buy {d_b:+.2f} bp, sell {d_s:+.2f} bp"
        if None not in (d_b, d_s) else "not computed"))

    # 3. Clustering is elevated on the finest grid.
    fine = g(st, "tick_1", "m0")
    one = g(st, "tick_10", "m0")
    ok_tick = fine is not None and one is not None and fine > one
    checks.append((
        "clustering is higher on the 0.1-yen grid than the 1-yen grid",
        ok_tick,
        f"{100*(fine or 0):.2f}% vs {100*(one or 0):.2f}%"
        if None not in (fine, one) else "one regime absent from this sample"))

    # 4. Magnitudes are in the neighbourhood of the published ones.
    mbl = g(st, "m_b_large0", "mean")
    checks.append((
        "the large-trade measure is within 5 points of the published 14.8%",
        mbl is not None and abs(100 * mbl - 14.8) < 5.0,
        f"{100*(mbl or 0):.2f}%" if mbl is not None else "not computed"))

    print("=== S4 step 5: stop rule ===\n")
    for name, ok, detail in checks:
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}\n         {detail}")

    failed = [c for c in checks if not c[1]]

    lines = ["# S4 -- replication of the paper's stylized facts on 2024", "",
             "## Stop rule", "",
             "These are treated as necessary conditions, not results. Each is a "
             "robust, repeatedly documented feature of the data; failing one "
             "indicates a defect in this pipeline rather than a discovery.", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += ["", "## Measures against the published benchmarks", "",
              "| Measure | 2024 | SE | Ohta 2010--2022 |", "|---|---|---|---|"]
    for k, lab, b in (("m0_all", "M0", 13.5), ("m_b_large0", "M^BLarge0", 14.8),
                      ("m_s_large0", "M^SLarge0", 14.3),
                      ("m_b_small0", "M^BSmall0", 11.1),
                      ("m_s_small0", "M^SSmall0", 10.9)):
        mu, se = g(st, k, "mean"), g(st, k, "se")
        if mu is not None:
            lines.append(f"| `{lab}` | {100*mu:.2f}% | {100*(se or 0):.3f} | {b}% |")
    rda, rdb = g(st, "rdepth_ask0", "mean"), g(st, "rdepth_bid0", "mean")
    if rda is not None:
        lines += ["", "## Round-price resting depth", "",
                  f"- Ask side: {100*rda:.2f}% of visible ten-level depth",
                  f"- Bid side: {100*(rdb or 0):.2f}%",
                  "- Uniform benchmark: 10%", ""]
    lines += ["## Verdict", "",
              ("All stop-rule checks pass; the measurement reproduces the "
               "literature on a year the paper did not cover, and later stages "
               "may proceed."
               if not failed else
               f"{len(failed)} check(s) FAILED. The most likely causes, in order: "
               "an inverted trade-direction map, floating-point digit arithmetic, "
               "or a mis-assigned tick size. Later stages must not run."), ""]

    path = os.path.join(OUT, "s4_report.md")
    with open(C.write_guard(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {path}")

    if failed:
        print(f"\nSTOP RULE TRIPPED: {len(failed)} check(s) failed")
        return 1
    print("\nSTOP RULE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
