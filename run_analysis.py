"""Run everything downstream of the store: panel, analysis, report.

Separate from the ingest so it can be re-run whenever more of the tape has
landed. Each stage's gate still applies; the stop rule in S4 halts the chain if
the measurement fails to reproduce the literature.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

PY = sys.executable

STAGES = [
    ("s3_step2_assemble.py", [], True),
    ("s3_step3_report.py", [], False),
    ("s4_step1_stylized.py", [], True),
    ("s4_step2_impact.py", [], True),
    ("s4_step3_opening.py", [], False),
    ("s4_step4_figures.py", [], False),
    ("s4_step5_report.py", [], True),        # stop rule
    ("s5_step1_rq2_daily.py", [], False),
    ("s5_step2_rq3_book.py", [], False),
    ("s5_step3_intraday.py", [], False),
    ("s5_step4_robustness.py", [], False),
    ("s5_step5_report.py", [], False),
    ("s6_step1_strategy.py", [], False),
    ("s7_step1_write_chapters.py", [], False),
    ("s7_step2_build.py", [], True),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", default="",
                    help="skip stages before this script name")
    ap.add_argument("--build-panel", default="",
                    help="date range YYYYMMDD-YYYYMMDD to build first")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume-panel", action="store_true",
                    help="keep any existing per-date panel outputs instead of "
                         "rebuilding them")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.build_panel:
        a, b = args.build_panel.split("-")
        print(f"### building panel {a}..{b}\n", flush=True)
        cmd = [PY, "s3_step1_build.py", "--start", a, "--end", b,
               "--workers", str(args.workers), "--gate-after", "999"]
        lad = os.path.join(C.RESULTS, "s3_panel", "ladder_tickers.txt")
        if os.path.exists(lad):
            cmd += ["--ladder-tickers", lad]
        # Rebuild from scratch by default: the store gains tickers between runs,
        # and a resumed build would keep whatever thinner version it wrote first.
        if not args.resume_panel:
            cmd.append("--fresh")
        r = subprocess.run(cmd, cwd=C.PROJ)
        if r.returncode != 0:
            print(f"panel build failed with {r.returncode}")
            return r.returncode

    started = not args.from_stage
    for script, extra, critical in STAGES:
        if not started:
            if script == args.from_stage:
                started = True
            else:
                continue
        print(f"\n{'='*70}\n### {script}\n{'='*70}", flush=True)
        r = subprocess.run([PY, script] + extra, cwd=C.PROJ)
        if r.returncode != 0:
            msg = f"{script} exited {r.returncode}"
            if critical:
                print(f"\nHALTED: {msg} (critical stage)")
                return r.returncode
            print(f"\nWARNING: {msg} (non-critical, continuing)")

    print(f"\n{'='*70}\nanalysis chain complete in "
          f"{(time.perf_counter()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
