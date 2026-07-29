"""S7 step 3 -- publish the repository to a private GitHub remote.

Refuses to push if anything that looks like data is tracked. The NEEDS licence is
institutional, and a derived Parquet file is still derived from licensed data;
figures and aggregates in the report are the only data products that travel.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

REPO = "jevwithwind/noise_trader"
DATA_EXT = re.compile(r"\.(parquet|csv|pkl|zip|feather|arrow|h5|db|sqlite)$", re.I)
DESCRIPTION = ("Noise-trader activity, price formation and liquidity on the TSE: "
               "a prototype implementing the future work proposed in Ohta (2026)")


def run(cmd: list[str], **kw):
    return subprocess.run(cmd, cwd=C.PROJ, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tee = C.Tee("s7_step3_publish")
    try:
        print(f"=== S7 step 3: publish to {args.repo} (private) ===\n")

        tracked = run(["git", "ls-files"]).stdout.split()
        bad = [f for f in tracked if DATA_EXT.search(f)]
        if bad:
            print(f"REFUSING TO PUBLISH: {len(bad)} data file(s) are tracked:")
            for f in bad[:20]:
                print("  -", f)
            return 1
        size = sum(os.path.getsize(os.path.join(C.PROJ, f))
                   for f in tracked if os.path.exists(os.path.join(C.PROJ, f)))
        print(f"[ok] {len(tracked)} tracked files, {size/1024**2:.2f} MB, "
              f"no data files")

        pdf = os.path.join(C.REPORT, "main.pdf")
        print(f"[{'ok' if os.path.exists(pdf) else 'WARN'}] report PDF "
              f"{'present' if os.path.exists(pdf) else 'MISSING'}")

        status = run(["git", "status", "--porcelain"]).stdout.strip()
        if status:
            print(f"[note] {len(status.splitlines())} uncommitted change(s); "
                  "committing before publish")
            if not args.dry_run:
                run(["git", "add", "-A"])
                run(["git", "-c", "user.name=Kevin Lee",
                     "-c", "user.email=kevinlee.tokyo@gmail.com",
                     "commit", "-q", "-m", "Final results and report"])

        if args.dry_run:
            print("\ndry run: stopping before any remote action")
            return 0

        exists = run(["gh", "repo", "view", args.repo]).returncode == 0
        if not exists:
            print(f"creating private repo {args.repo}")
            r = run(["gh", "repo", "create", args.repo, "--private",
                     "--description", DESCRIPTION, "--source", ".",
                     "--remote", "origin"])
            print(r.stdout.strip() or r.stderr.strip())
            if r.returncode != 0:
                return 1
        else:
            print(f"repo {args.repo} already exists")
            if run(["git", "remote", "get-url", "origin"]).returncode != 0:
                run(["git", "remote", "add", "origin",
                     f"https://github.com/{args.repo}.git"])

        branch = run(["git", "branch", "--show-current"]).stdout.strip() or "master"
        print(f"pushing {branch} ...")
        r = run(["git", "push", "-u", "origin", branch])
        print((r.stdout + r.stderr).strip()[-600:])
        if r.returncode != 0:
            return 1

        v = run(["gh", "repo", "view", args.repo,
                 "--json", "name,visibility,url,defaultBranchRef"])
        print("\n" + v.stdout.strip())
        print("\nGATE PASSED -- published")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
