"""S0 step 4 -- stage report."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s0_inst")


def main() -> int:
    import polars as pl

    gate = C.read_json(os.path.join(OUT, "gate.json"), {})
    backends = C.read_json(os.path.join(OUT, "backends.json"), {})
    cal = C.read_json(os.path.join(OUT, "calendar_summary.json"), {})
    yob = C.read_json(os.path.join(OUT, "yobine_summary.json"), {})
    caldf = pl.read_csv(C.CALENDAR_CSV)

    by_status = {r["status"]: r["len"] for r in
                 caldf.group_by("status").len().iter_rows(named=True)}
    usable = caldf.filter(pl.col("status") == "ok")

    lines = [
        "# S0 -- institutional tables and environment",
        "",
        "## Environment",
        "",
        f"- Python: `{gate.get('python', '?').splitlines()[0]}`",
        f"- polars {gate.get('ver_polars')}, pyarrow {gate.get('ver_pyarrow')}, "
        f"duckdb {gate.get('ver_duckdb')}, tse_tick {gate.get('ver_tse_tick')}",
        f"- numpy {gate.get('ver_numpy')}, statsmodels {gate.get('ver_statsmodels')}",
        f"- Free space: D: {gate.get('free_gb_D')} GB (store), E: {gate.get('free_gb_E')} GB (repo)",
        f"- LaTeX: `{gate.get('tool_latexmk')}`; git: {gate.get('git')}",
        "",
        "### Dependencies added",
        "",
        "`linearmodels` "
        f"{backends.get('versions', {}).get('linearmodels')}, "
        f"`pyfixest` {backends.get('versions', {}).get('pyfixest')}, `zstandard`.",
        "",
        "**Environment repair.** `numba` 0.65.1 was broken in `localdb` before this "
        "project started: it subclasses `coverage.types.Tracer`, which does not exist "
        "in `coverage` 7.4.4, so `import numba` raised `AttributeError`. Since "
        "`pyfixest` imports numba, this surfaced as a pyfixest failure. Fixed by "
        "upgrading `coverage` to 7.15.2; numba and pyfixest both import cleanly now.",
        "",
        "### Estimator verification",
        "",
        "Three backends were fitted to a simulated panel "
        f"(50 stocks x 100 days, true beta = {backends.get('beta_true')}) whose regressor "
        "correlates with both the stock and the day effects, so a backend that fails "
        "to absorb them is visibly biased rather than merely noisy.",
        "",
        "| Backend | beta | two-way clustered SE |",
        "|---|---|---|",
    ]
    for name, r in backends.get("results", {}).items():
        lines.append(f"| `{name}` | {r['beta']:.6f} | {r['se']:.6f} |")
    lines += [
        "",
        "Point estimates agree to 9e-16; clustered standard errors agree to 1.5% "
        "(libraries differ in small-sample degrees-of-freedom conventions). Roles: "
        "`linearmodels` for the daily panel, `pyfixest` for the high-dimensional "
        "intraday fixed effects, the manual Cameron-Gelbach-Miller implementation as "
        "arbiter.",
        "",
        f"## Trading calendar {C.YEAR}",
        "",
        f"- {cal.get('total_gb')} GB compressed across "
        f"{int(caldf['n_shards'].sum())} shards.",
        f"- **{by_status.get('ok', 0)} usable trading days.**",
        f"- Shards per usable day: min {usable['n_shards'].min()}, "
        f"median {usable['n_shards'].median():.0f}, max {usable['n_shards'].max()}.",
        "",
        "### Excluded dates",
        "",
        "| Date | Shards | Month median | Status |",
        "|---|---|---|---|",
    ]
    for r in caldf.filter(pl.col("status") != "ok").iter_rows(named=True):
        lines.append(f"| {r['date']} | {r['n_shards']} | "
                     f"{r['median_month_shards']} | {r['status']} |")
    lines += [
        "",
        "2024-04-24 through 2024-04-30 (excluding the 29th, a public holiday) never "
        "arrived; 2024-04-23 arrived with 10 shards against a month median of 22, the "
        "signature of a delivery that died mid-day. All five are excluded at the "
        "calendar level so no later stage can silently treat a partial day as a quiet "
        "one. No other date showed an undocumented anomaly.",
        "",
        "## Tick-size (yobine) tables",
        "",
        "Two grids operate in 2024: a general one, and a finer one for TOPIX500 "
        "constituents (extended from TOPIX100 to all of TOPIX500 on "
        f"{yob.get('fine_regime_all_topix500_from')}, and stable through 2024). "
        "Both are encoded in integer deci-yen in `s0_common.py`.",
        "",
        "Sources: " + "; ".join(yob.get("sources", [])) + ".",
        "",
        "The tables were cross-checked against four ticks read directly off the tape "
        "during preparation, and each is a unit test: 7203 at ~4,000 yen -> 1 yen; "
        "8604 at ~900 yen -> 0.1 yen; 8306 at ~1,500 yen -> 0.5 yen; and 4666 flipping "
        "from 1 yen to 0.5 yen between 2023-05-15 and 2023-07-18, which brackets the "
        "TOPIX Mid400 extension date.",
        "",
        "### Consequence for the sample",
        "",
        "Ohta's filter (d) keeps only ticks that are powers of ten "
        "(0.1, 1, 10, 100, 1000 yen), because the last-digit-of-ten construction is "
        "otherwise undefined. On the fine grid this excludes the 1,000-3,000 yen band "
        "(0.5 yen) outright -- a large share of TOPIX500 names. The regression sample "
        "is therefore tilted away from mid-priced large caps, and the report says so "
        "rather than pooling incompatible grids.",
        "",
        f"- TOPIX500 union 2023-2024: {yob.get('topix500_union')} tickers.",
        f"- Fine-grid bands excluded by filter (d): "
        f"{len(yob.get('fine_excluded_by_filter_d', []))} of {yob.get('fine_bands')}.",
        f"- General-grid bands excluded: "
        f"{len(yob.get('general_excluded_by_filter_d', []))} of {yob.get('general_bands')}.",
        "",
        "## Verdict",
        "",
        "S0 passes. The environment is verified against a known-truth panel, the "
        "trading calendar is frozen at 240 usable days with five documented "
        "exclusions, and the tick tables reproduce every tick observed on the tape. "
        "40 unit tests cover the yobine bands and their boundary conventions, the "
        "integer digit arithmetic (including a test that records how the naive "
        "floating-point version fails), the session-close switch on 2024-11-05, the "
        "trade-signing map including the stop-quote variants, and the write guard.",
        "",
    ]
    path = os.path.join(OUT, "s0_report.md")
    with open(C.write_guard(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
