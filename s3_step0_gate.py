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

        cal = pl.read_csv(os.path.join(C.RESULTS, "s0_inst", "calendar_2024.csv"))
        usable = set(cal.filter(pl.col("status") == "ok")["date"].cast(pl.Utf8).to_list())
        have = set(dates) & usable
        missing = sorted(usable - set(dates))
        print(f"usable days present in store: {len(have)} of {len(usable)}")
        if missing:
            print(f"  missing: {len(missing)}"
                  + (f" (first few: {missing[:5]})" if missing else ""))

        uni = os.path.join(C.RESULTS, "s2_ingest", "universe.csv")
        if os.path.exists(uni):
            n_uni = pl.read_csv(uni).height
            print(f"universe: {n_uni:,} stocks")
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
