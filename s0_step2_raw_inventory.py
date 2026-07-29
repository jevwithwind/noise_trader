"""S0 step 2 -- inventory the raw feed and freeze the trading calendar.

Directory metadata only; no zip is opened. Produces the calendar every later stage
trusts, with any delivery gaps marked so they are excluded once, here, and
never rediscovered downstream.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s0_inst")
FNAME = re.compile(r"^HTICST120\.(\d{8})\.(\d+)\.zip$", re.IGNORECASE)

# A day whose shard count falls this far below its month's median is treated as a
# partial delivery rather than a genuinely quiet session.
TRUNCATION_RATIO = 0.55


def main() -> int:
    tee = C.Tee("s0_step2_raw_inventory")
    try:
        print(f"=== S0 step 2: raw feed inventory ({C.YEAR}) ===\n")
        shards: dict[str, list[int]] = defaultdict(list)
        nbytes: dict[str, int] = defaultdict(int)
        months: dict[str, set[str]] = defaultdict(set)

        month_dirs = sorted((d.name, d.path) for d in os.scandir(C.RAW_TICKS) if d.is_dir())
        for mname, mpath in month_dirs:
            for e in os.scandir(mpath):
                m = FNAME.match(e.name)
                if not m:
                    continue
                date, shard = m.group(1), int(m.group(2))
                shards[date].append(shard)
                nbytes[date] += e.stat().st_size
                months[mname].add(date)

        dates = sorted(shards)
        print(f"dates with data: {len(dates)}  ({dates[0]} .. {dates[-1]})")
        total_gb = sum(nbytes.values()) / 1024 ** 3
        print(f"total compressed: {total_gb:.1f} GB across "
              f"{sum(len(v) for v in shards.values())} shards\n")

        # Median shards per month, to spot partial deliveries.
        med: dict[str, float] = {}
        for mon, ds in months.items():
            counts = sorted(len(shards[d]) for d in ds)
            med[mon] = counts[len(counts) // 2]

        rows, flagged = [], []
        for d in dates:
            mon = d[:6]
            n = len(shards[d])
            contiguous = sorted(shards[d]) == list(range(1, n + 1))
            status = "ok"
            if d in C.KNOWN_MISSING_DATES:
                status = "missing"
            elif d in C.KNOWN_TRUNCATED_DATES:
                status = "truncated"
            elif n < TRUNCATION_RATIO * med[mon]:
                status = "truncated_detected"
                flagged.append((d, n, med[mon]))
            elif not contiguous:
                status = "noncontiguous"
                flagged.append((d, n, med[mon]))
            rows.append({"date": d, "n_shards": n, "bytes": nbytes[d],
                         "median_month_shards": med[mon],
                         "contiguous": contiguous, "status": status})

        # Dates we expected but that never arrived at all.
        for d in sorted(C.KNOWN_MISSING_DATES):
            if d not in shards:
                rows.append({"date": d, "n_shards": 0, "bytes": 0,
                             "median_month_shards": med.get(d[:6], 0),
                             "contiguous": False, "status": "missing"})
        rows.sort(key=lambda r: r["date"])

        import polars as pl
        df = pl.DataFrame(rows)
        C.ensure_dir(OUT)
        path = C.CALENDAR_CSV
        df.write_csv(C.write_guard(path))

        by_status = df.group_by("status").len().sort("status")
        print("calendar by status:")
        for r in by_status.iter_rows(named=True):
            print(f"  {r['status']:20s} {r['len']:4d}")

        usable = df.filter(pl.col("status") == "ok")
        print(f"\nusable trading days: {usable.height}")
        print(f"shards/day: min {usable['n_shards'].min()}  "
              f"median {usable['n_shards'].median():.0f}  "
              f"max {usable['n_shards'].max()}")

        fails = []
        # Documented gaps are expected damage. Anything else is news.
        unexpected = [f for f in flagged if f[0] not in C.EXCLUDED_DATES]
        if unexpected:
            print("\nNEW anomalies not in the documented gap list:")
            for d, n, m in unexpected:
                print(f"  {d}: {n} shards vs month median {m}")
            fails.append(f"{len(unexpected)} undocumented partial/noncontiguous days")
        if not (235 <= usable.height <= 245):
            fails.append(f"usable day count {usable.height} outside plausible 235-245")

        C.atomic_json(os.path.join(OUT, "calendar_summary.json"), {
            "n_dates_with_data": len(dates), "n_usable": usable.height,
            "total_gb": round(total_gb, 2),
            "excluded": sorted(C.EXCLUDED_DATES),
            "unexpected_anomalies": [list(x) for x in unexpected],
            "fails": fails,
        })

        print()
        if fails:
            print("GATE FAILED:")
            for f in fails:
                print("  -", f)
            return 1
        print(f"GATE PASSED -- {usable.height} usable days, {total_gb:.1f} GB, "
              f"{len(C.EXCLUDED_DATES)} documented exclusions")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
