"""Drop panel dates that were built before the store held every ticker.

The build checkpoint records that a date was processed, not how much of the store
existed at the time. A date built during ingest therefore looks finished while
holding a fraction of the universe. This compares each built date against the
current store and removes any whose ticker count falls short, so the next build
regenerates them.
"""
from __future__ import annotations

import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s3_common as S3

OUT = S3.OUT
CKPT = os.path.join(OUT, "build_ckpt.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.98,
                    help="keep a date whose panel covers at least this share of "
                         "the tickers now in the store for that date")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tee = C.Tee("s3_step0b_drop_stale")
    try:
        print("=== S3 step 0b: drop panel dates built from a partial store ===\n")
        ck = C.read_json(CKPT, {"done": [], "errors": []})
        done = list(ck.get("done", []))
        if not done:
            print("no checkpoint; nothing to do")
            return 0

        stale = []
        for date in sorted(done):
            p = os.path.join(OUT, "daily", f"{date}.parquet")
            in_store = len(S3.date_files(date))
            if not os.path.exists(p):
                stale.append((date, 0, in_store))
                continue
            try:
                n = pl.read_parquet(p, columns=["ticker"])["ticker"].n_unique()
            except Exception:
                stale.append((date, -1, in_store))
                continue
            if in_store and n < args.tolerance * in_store:
                stale.append((date, n, in_store))

        print(f"dates in checkpoint: {len(done)}")
        if not stale:
            print("all built dates match the store; nothing to drop")
            return 0

        print(f"stale dates (panel tickers vs store tickers):")
        for date, n, s in stale:
            print(f"  {date}: {n} of {s}")
        if args.dry_run:
            print("\ndry run: nothing removed")
            return 0

        for date, _, _ in stale:
            for sub in ("daily", "intraday"):
                p = os.path.join(OUT, sub, f"{date}.parquet")
                if os.path.exists(p):
                    os.remove(C.write_guard(p))
            done.remove(date)
        ck["done"] = sorted(done)
        C.atomic_json(CKPT, ck)
        print(f"\ndropped {len(stale)} date(s); {len(done)} remain built")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
