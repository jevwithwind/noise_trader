"""S3 step 0 -- gate before the panel build.

Runs the full unit-test suite and checks the store is present and complete
enough to build a year-long panel from.
"""
from __future__ import annotations

import os
import subprocess
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s3_common as S3


def main() -> int:
    tee = C.Tee("s3_step0_gate")
    fails = []
    try:
        print("=== S3 step 0: gate ===\n")

        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
             "-p", "no:cacheprovider", "-o", "addopts="],
            cwd=C.PROJ, capture_output=True, text=True)
        last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1]
        print(f"unit tests: {last}")
        if r.returncode != 0:
            fails.append(f"unit tests failed: {last}")

        dates = S3.store_dates()
        print(f"store dates: {len(dates)}"
              + (f"  ({dates[0]}..{dates[-1]})" if dates else ""))
        if not dates:
            fails.append("store is empty")

        cal = pl.read_csv(C.CALENDAR_CSV)
        usable = set(cal.filter(pl.col("status") == "ok")["date"].cast(pl.Utf8).to_list())
        have = set(dates) & usable
        missing = sorted(usable - set(dates))
        print(f"usable days present in store: {len(have)} of {len(usable)}")
        if missing:
            print(f"  missing: {len(missing)}"
                  + (f" (first few: {missing[:5]})" if missing else ""))

        uni = os.path.join(C.RESULTS, "s2_ingest", "universe.csv")
        if os.path.exists(uni):
            u = pl.read_csv(uni)
            n_uni = u.height
            print(f"universe: {n_uni:,} stocks")

            # The ladder inference costs several times the base measures, because
            # it expands every book snapshot into twenty price-level rows before
            # differencing them. It runs on a size-stratified subsample so the
            # cost is bounded and the coverage is still representative; the
            # decision is written down rather than left implicit.
            n_per = 60
            u = u.drop_nulls("mktcap").sort("mktcap")
            if u.height:
                u = u.with_columns(
                    q=(pl.col("mktcap").rank("ordinal") * 5 // (u.height + 1) + 1)
                    .cast(pl.Int8))
                picks = []
                for q in range(1, 6):
                    sub = u.filter(pl.col("q") == q)
                    if sub.height == 0:
                        continue
                    step = max(1, sub.height // n_per)
                    picks += sub["ticker"].to_list()[::step][:n_per]
                picks = sorted(set(str(p).zfill(4) for p in picks))
                p = os.path.join(C.RESULTS, "s3_panel", "ladder_tickers.txt")
                C.ensure_dir(os.path.dirname(p))
                with open(C.write_guard(p), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(picks))
                print(f"ladder subsample: {len(picks)} stocks, evenly spread across "
                      f"market-capitalisation quintiles -> {p}")
            if dates:
                n_files = len(S3.date_files(dates[len(dates) // 2]))
                print(f"tickers in a mid-year date partition: {n_files:,}")
                if n_files < 0.5 * n_uni:
                    fails.append(f"only {n_files} tickers in a date partition vs "
                                 f"universe {n_uni} -- ingest may be incomplete")
        else:
            fails.append("universe.csv missing; run s2_step0_universe.py")

        units = S3.load_units()
        print(f"trading units loaded: {len(units):,} stock-days")
        if not units:
            fails.append("no trading units; the daily-summary product is not ingested")
        else:
            vals = {}
            for v in units.values():
                vals[v] = vals.get(v, 0) + 1
            top = sorted(vals.items(), key=lambda kv: -kv[1])[:4]
            print("  unit distribution: " + ", ".join(f"{k}:{v:,}" for k, v in top))

        print()
        if fails:
            print(f"GATE FAILED ({len(fails)}):")
            for f in fails:
                print("  -", f)
            return 1
        print("GATE PASSED")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
