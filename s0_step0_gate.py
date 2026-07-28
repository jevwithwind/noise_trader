"""S0 step 0 -- environment gate. Nothing downstream runs until this passes."""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s0_inst")

REQUIRED_IMPORTS = ["polars", "pandas", "pyarrow", "duckdb", "numpy", "scipy",
                    "statsmodels", "matplotlib", "tse_tick"]
MIN_FREE_GB = {"D": 350.0, "E": 15.0}


def free_gb(drive: str) -> float:
    total, used, free = shutil.disk_usage(f"{drive}:\\")
    return free / 1024 ** 3


def main() -> int:
    tee = C.Tee("s0_step0_gate")
    fails: list[str] = []
    info: dict = {}
    try:
        print("=== S0 step 0: environment gate ===\n")

        # --- raw feed readable
        ok_raw = os.path.isdir(C.RAW_2024)
        print(f"[{'ok' if ok_raw else 'FAIL'}] raw feed readable: {C.RAW_2024}")
        if not ok_raw:
            fails.append("raw feed not readable")
        else:
            months = sorted(d.name for d in os.scandir(C.RAW_2024) if d.is_dir())
            info["raw_months"] = months
            print(f"       months present: {len(months)} ({months[0]}..{months[-1]})")
            if len(months) != 12:
                fails.append(f"expected 12 month folders, found {len(months)}")

        # --- TOPIX500 reference
        try:
            t500 = C.load_topix500()
            info["topix500_union_2023_2024"] = len(t500)
            print(f"[ok] TOPIX500 membership loaded: {len(t500)} tickers (2023 u 2024)")
        except Exception as exc:
            fails.append(f"TOPIX500 membership: {exc}")
            print(f"[FAIL] TOPIX500 membership: {exc}")

        # --- disk
        for drive, need in MIN_FREE_GB.items():
            got = free_gb(drive)
            info[f"free_gb_{drive}"] = round(got, 1)
            ok = got >= need
            print(f"[{'ok' if ok else 'FAIL'}] {drive}: free {got:.1f} GB (need {need})")
            if not ok:
                fails.append(f"{drive}: only {got:.1f} GB free, need {need}")

        # --- write guard actually guards
        try:
            C.write_guard(r"C:\Windows\system32\nope.txt")
            fails.append("write_guard did NOT block an outside path")
            print("[FAIL] write_guard did not block C:\\Windows")
        except AssertionError:
            print("[ok] write_guard blocks paths outside the project/store")

        # --- imports
        for mod in REQUIRED_IMPORTS:
            try:
                m = importlib.import_module(mod)
                v = getattr(m, "__version__", "?")
                info[f"ver_{mod}"] = v
                print(f"[ok] import {mod:12s} {v}")
            except Exception as exc:
                fails.append(f"import {mod}: {exc}")
                print(f"[FAIL] import {mod}: {exc}")

        # --- LaTeX
        for tool in ("latexmk", "pdflatex", "biber"):
            path = shutil.which(tool)
            ok = path is not None
            info[f"tool_{tool}"] = path
            print(f"[{'ok' if ok else 'FAIL'}] {tool}: {path}")
            if not ok:
                fails.append(f"{tool} not on PATH")

        # --- git
        try:
            v = subprocess.run(["git", "--version"], capture_output=True, text=True,
                               check=True).stdout.strip()
            info["git"] = v
            print(f"[ok] {v}")
        except Exception as exc:
            fails.append(f"git: {exc}")

        info["python"] = sys.version
        info["fails"] = fails
        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "gate.json"), info)

        print()
        if fails:
            print(f"GATE FAILED ({len(fails)} problems):")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("GATE PASSED")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
