"""Shared machinery for the panel build: store access and the worker function.

Kept in its own module because Windows spawns worker processes rather than
forking them, so anything a worker touches has to be importable at module level.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import measures as M

STORE_TICKS = os.path.join(C.STORE, "individual_stock")
STORE_SUMMARY = os.path.join(C.STORE, "stock_summary")
OUT = os.path.join(C.RESULTS, "s3_panel")

#: Columns read when only the clustering / liquidity / volatility measures are wanted.
COLS_THIN = M.THIN_COLS
#: Columns read when the ten-level book measures are wanted too.
COLS_WIDE = M.WIDE_COLS + ["Sell Quote Vol OVER", "Buy Quote Vol UNDER"]


def store_dates() -> list[str]:
    if not os.path.isdir(STORE_TICKS):
        return []
    out = []
    for e in os.scandir(STORE_TICKS):
        if e.is_dir() and e.name.startswith("date="):
            out.append(e.name[5:])
    return sorted(out)


def date_files(date: str) -> list[tuple[str, str, int]]:
    """(ticker, path, bytes) for one date, largest file first.

    Scheduling the big stocks first keeps the tail of a date from being one slow
    worker while the others idle, and bounds peak memory earlier in the run.
    """
    d = os.path.join(STORE_TICKS, f"date={date}")
    if not os.path.isdir(d):
        return []
    out = []
    for e in os.scandir(d):
        if e.is_file() and e.name.startswith("ticker=") and e.name.endswith(".parquet"):
            out.append((e.name[7:-8], e.path, e.stat().st_size))
    out.sort(key=lambda t: -t[2])
    return out


def load_units() -> dict[tuple[str, str], int]:
    """Trading unit per (date, ticker) from the daily summary product.

    Trading units were unified at 100 shares in October 2018, so 2024 should be
    uniform -- but the large/small split is defined against the unit, so it is
    read rather than assumed.
    """
    if not os.path.isdir(STORE_SUMMARY):
        return {}
    files = []
    for root, _, fs in os.walk(STORE_SUMMARY):
        files.extend(os.path.join(root, f) for f in fs if f.endswith(".parquet"))
    if not files:
        return {}
    frames = []
    for f in files:
        try:
            frames.append(pl.read_parquet(f, columns=["Data Date", "Stock Code",
                                                      "Trading Unit"]))
        except Exception:
            continue
    if not frames:
        return {}
    df = pl.concat(frames, how="vertical_relaxed")
    df = df.with_columns(
        date=pl.col("Data Date").dt.strftime("%Y%m%d"),
        ticker=pl.col("Stock Code").cast(pl.Utf8).str.strip_chars(),
        unit=pl.col("Trading Unit").cast(pl.Int64),
    ).drop_nulls(["date", "ticker", "unit"]).filter(pl.col("unit") > 0)
    return {(r["date"], r["ticker"]): int(r["unit"])
            for r in df.select("date", "ticker", "unit").iter_rows(named=True)}


def load_shares_outstanding() -> dict[tuple[str, str], float]:
    """Shares outstanding per (date, ticker), for the market-capitalisation control."""
    if not os.path.isdir(STORE_SUMMARY):
        return {}
    files = []
    for root, _, fs in os.walk(STORE_SUMMARY):
        files.extend(os.path.join(root, f) for f in fs if f.endswith(".parquet"))
    out: dict[tuple[str, str], float] = {}
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["Data Date", "Stock Code", "Issued Shares"])
        except Exception:
            continue
        df = df.with_columns(
            date=pl.col("Data Date").dt.strftime("%Y%m%d"),
            ticker=pl.col("Stock Code").cast(pl.Utf8).str.strip_chars(),
            sh=pl.col("Issued Shares").cast(pl.Float64),
        ).drop_nulls(["date", "ticker", "sh"]).filter(pl.col("sh") > 0)
        for r in df.select("date", "ticker", "sh").iter_rows(named=True):
            out[(r["date"], r["ticker"])] = float(r["sh"])
    return out


# Phase-1 columns: enough to reject a stock-day without paying for the wide read.
COLS_PROBE = ["Execution Type", "Execution Price", "Execution Time", "Volume"]


def probe_skip(path: str) -> str | None:
    """Cheap pre-check. Returns a reason to skip, or None to proceed.

    Reads four columns instead of forty-plus. Most tickers on most days are
    illiquid enough to fail Ohta's twenty-trade minimum, and rejecting them here
    is what makes a full-market panel affordable.
    """
    try:
        df = pl.read_parquet(path, columns=COLS_PROBE)
    except Exception as exc:
        return f"read_error:{type(exc).__name__}"
    if df.height == 0:
        return "empty"
    tr = df.filter(pl.col("Execution Type").is_not_null() & (pl.col("Volume") > 0))
    if tr.height <= 20:
        return "few_trades"
    # Filter (b): the opening price must exceed 200 yen. Using the day's maximum
    # here is deliberately generous -- a stock that never traded above 200 cannot
    # possibly pass, and anything else is decided properly downstream.
    if float(tr["Execution Price"].max()) <= 200.0:
        return "low_price"
    return None


def process_one(args: tuple) -> tuple[dict | None, list[dict], str | None]:
    """Compute one stock-day. Runs inside a worker process."""
    ticker, path, date_str, is_t500, unit, wide, do_ladder = args
    d = _dt.datetime.strptime(date_str, "%Y%m%d").date()
    try:
        reason = probe_skip(path)
        if reason:
            return ({"date": d, "ticker": ticker, "in_sample": False,
                     "skip_reason": reason}, [], None)
        cols = COLS_WIDE if wide else COLS_THIN
        have = pl.read_parquet_schema(path)
        df = pl.read_parquet(path, columns=[c for c in cols if c in have])
        row, buckets = M.stock_day(df, ticker, d, is_t500=is_t500, unit=unit,
                                   wide=wide, do_ladder=do_ladder)
        row["skip_reason"] = None
        return row, buckets, None
    except Exception as exc:
        import traceback
        return (None, [], f"{ticker} {date_str}: {type(exc).__name__}: {exc}\n"
                          + traceback.format_exc(limit=3))
