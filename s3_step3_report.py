"""S3 step 3 -- stage report: what the panel contains and what it cost."""
from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s3_common as S3
import s4_common as S4

OUT = S3.OUT


def main() -> int:
    wf = C.read_json(os.path.join(OUT, "waterfall.json"), {})
    bs = C.read_json(os.path.join(OUT, "build_summary.json"), {})
    df = pl.read_parquet(S4.PANEL)
    final = df.filter(pl.col("in_sample_final"))

    lines = [
        "# S3 -- the analysis panels",
        "",
        f"- Stock-day rows attempted: **{wf.get('attempted', 0):,}**",
        f"- Fully evaluated (passed the cheap pre-screen): {wf.get('evaluated', 0):,}",
        f"- Passing all four of Ohta's stock-day filters: {wf.get('in_sample', 0):,}",
        f"- Stocks qualifying on more than half the year: "
        f"**{wf.get('stocks_final', 0):,}**",
        f"- Final stock-day sample: **{wf.get('stock_days_final', 0):,}**",
        f"- Intraday bucket rows: {wf.get('intraday_rows', 0):,}",
        "",
        f"Build time {bs.get('seconds', 0)/3600:.2f} hours over "
        f"{bs.get('dates_processed', 0)} dates, {bs.get('n_errors', 0)} errors.",
        "",
        "## Sample construction",
        "",
        "| Step | Stock-days |",
        "|---|---|",
        f"| Attempted | {wf.get('attempted', 0):,} |",
        f"| Survived the pre-screen | {wf.get('evaluated', 0):,} |",
        f"| First trade by 9:10 | {wf.get('pass_open910', 0):,} |",
        f"| Opening price above 200 yen | {wf.get('pass_open200', 0):,} |",
        f"| More than 20 continuous-session trades | {wf.get('pass_n20', 0):,} |",
        f"| Tick a power of ten all day | {wf.get('pass_tick', 0):,} |",
        f"| All four | {wf.get('in_sample', 0):,} |",
        f"| Stock qualifies on more than half the year | "
        f"{wf.get('stock_days_final', 0):,} |",
        "",
        "The pre-screen reads four columns instead of forty and rejects stock-days "
        "that cannot possibly qualify. It is what makes a full-market panel "
        "affordable, and it changes nothing: every day it rejects would have "
        "failed the twenty-trade or two-hundred-yen filter anyway.",
        "",
    ]
    if final.height:
        lines += [
            "## Coverage",
            "",
            f"- Dates: {final['date'].n_unique()}",
            f"- Stocks: {final['ticker'].n_unique():,}",
            f"- Median stock-days per stock: "
            f"{final.group_by('ticker').len()['len'].median():.0f}",
            "",
        ]
        if "tick_source" in df.columns:
            # A stock-day with no resolvable tick carries a null source, which
            # cannot be sorted against the string keys.
            src = {(r["tick_source"] or "unresolved"): r["len"] for r in
                   df.group_by("tick_source").len().iter_rows(named=True)}
            lines += [
                "## Tick-size resolution",
                "",
                "The coded JPX schedule is primary; the tape is consulted as a "
                "cross-check. A tape *finer* than the table always wins (index "
                "membership moved). A tape *coarser* than the table wins only "
                "when the day shows enough distinct prices, all on the coarser "
                "lattice, for that to be evidence of the real grid rather than "
                "sparsity -- forcing the table's finer tick onto such a day "
                "reads as 50-100% clustering, which is the failure this rule "
                "exists to prevent.",
                "",
            ] + [f"- `{k}`: {v:,} stock-days" for k, v in sorted(src.items())] + [""]
        if "n_unmapped_exec_type" in df.columns:
            lines += [
                f"Unmapped execution-type rows across the whole panel: "
                f"**{int(df['n_unmapped_exec_type'].fill_null(0).sum()):,}**. "
                "The stop-quote variants of the trade-direction field decode to "
                "an unknown label and would otherwise be dropped silently, "
                "removing exactly the trades that occur on the most volatile "
                "days.",
                "",
            ]
    lines += [
        "## Verdict",
        "",
        f"The panel holds {wf.get('stock_days_final', 0):,} stock-days across "
        f"{wf.get('stocks_final', 0):,} stocks under Ohta's own admission rules, "
        "with every rejection counted rather than inferred from absence.",
        "",
    ]
    path = os.path.join(OUT, "s3_report.md")
    with open(C.write_guard(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
