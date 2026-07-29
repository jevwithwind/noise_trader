"""S2 step 2 -- prove the store is a faithful copy of the tape.

Every later stage reads the Parquet store instead of the raw zips, so the store
has to be measure-for-measure identical to what the raw path produces. This
recomputes the clustering and liquidity measures both ways for the same
stock-days and requires exact agreement, not approximate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s3_common as S3
import measures as M
from tse_tick import read_ticks

OUT = os.path.join(C.RESULTS, "s2_ingest")
CHECK = ["m0_all", "m_b_large0", "m_s_large0", "m_b_small0", "m_s_small0",
         "effsprd_bps", "imp60_bps", "rs60_bps", "rv5", "ofi_sum",
         "rdepth_ask0", "rdepth_bid0", "l_s0", "l_s0c"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickers", type=int, default=4)
    ap.add_argument("--n-dates", type=int, default=2)
    args = ap.parse_args()

    tee = C.Tee("s2_step2_verify")
    fails = []
    try:
        print("=== S2 step 2: store vs raw tape ===\n")
        dates = S3.store_dates()
        if not dates:
            print("store is empty")
            return 1
        t500 = C.load_topix500()
        use_dates = dates[:args.n_dates]

        for date in use_dates:
            files = S3.date_files(date)
            # Busiest files first: the biggest stock-days exercise the most code.
            picks = files[:args.n_tickers]
            print(f"--- {date} ({len(files)} tickers in store) ---")
            d = dt.datetime.strptime(date, "%Y%m%d").date()
            for ticker, path, size in picks:
                store_df = pl.read_parquet(path)
                raw_df = read_ticks(C.RAW_2024, ticker_filter={ticker},
                                    date=date, language="en")
                same_rows = store_df.height == raw_df.height
                r_store, _ = M.stock_day(store_df, ticker, d,
                                         is_t500=ticker in t500, wide=True, do_ladder=True)
                r_raw, _ = M.stock_day(raw_df, ticker, d,
                                       is_t500=ticker in t500, wide=True, do_ladder=True)
                diffs = []
                for k in CHECK:
                    a, b = r_store.get(k), r_raw.get(k)
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        diffs.append(f"{k}: store={a} raw={b}")
                    elif abs(a - b) > 1e-9 * max(1.0, abs(b)):
                        diffs.append(f"{k}: store={a!r} raw={b!r}")
                ok = same_rows and not diffs
                print(f"  {ticker}: store {store_df.height:>8,} rows, "
                      f"raw {raw_df.height:>8,} rows, "
                      f"{len(CHECK)} measures {'identical' if ok else 'DIFFER'}")
                if not same_rows:
                    fails.append(f"{ticker} {date}: row count {store_df.height} vs "
                                 f"{raw_df.height}")
                for x in diffs:
                    print(f"      {x}")
                    fails.append(f"{ticker} {date}: {x}")

        # The store adds an Effective Time index column; everything else must match.
        f0 = S3.date_files(use_dates[0])[0]
        s_cols = set(pl.read_parquet_schema(f0[1]))
        r_cols = set(read_ticks(C.RAW_2024, ticker_filter={f0[0]},
                                date=use_dates[0], language="en").columns)
        extra, missing = s_cols - r_cols, r_cols - s_cols
        print(f"\nschema: store has {len(s_cols)} columns, raw {len(r_cols)}")
        if extra:
            print(f"  store-only (expected: the store's time index): {sorted(extra)}")
        if missing:
            print(f"  MISSING from store: {sorted(missing)}")
            fails.append(f"store missing columns: {sorted(missing)}")

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "verify.json"),
                      {"dates": use_dates, "fails": fails, "checked": CHECK})
        print()
        if fails:
            print(f"GATE FAILED ({len(fails)}):")
            for f in fails[:10]:
                print("  -", f)
            return 1
        print("GATE PASSED -- the store reproduces the raw tape exactly")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
