"""Shared constants, guards and institutional logic for every stage.

Imported by all s*_step*.py scripts and by measures.py. Contains nothing that
depends on a particular stage: paths, write guards, atomic checkpoints, the TSE
session calendar, the trade-signing map, and the JPX yobine (tick size) tables.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from typing import Iterable

# --------------------------------------------------------------------------- paths
PROJ = r"E:\MTEC\prototype"
STORE = r"D:\MTEC_tick_store"
RAW_ROOT = r"G:\needs"
RAW_2024 = r"G:\needs\個別株式2024\TICST120"
RESULTS = os.path.join(PROJ, "results")
LOGS = os.path.join(PROJ, "logs")
REPORT = os.path.join(PROJ, "report")

# Reference assets borrowed read-only from the thesis project.
TOPIX500_CSV = r"G:\flash_crash\topix\topix500_membership_by_year.csv"

YEAR = 2024


def write_guard(path: str) -> str:
    """Refuse to write anywhere except the project tree or the declared store.

    The single most important safety rule in this repo: the raw NEEDS feed is
    irreplaceable and read-only, and it sits on the same machine.
    """
    ap = os.path.abspath(path)
    allowed = (os.path.abspath(PROJ), os.path.abspath(STORE))
    if not any(ap.startswith(a + os.sep) or ap == a for a in allowed):
        raise AssertionError(f"WRITE OUTSIDE ALLOWED TREES: {ap}")
    return ap


def ensure_dir(path: str) -> str:
    write_guard(path)
    os.makedirs(path, exist_ok=True)
    return path


def atomic_json(path: str, obj) -> None:
    """Checkpoint write that can never leave a half-written file behind."""
    write_guard(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Tee:
    """Mirror stdout to a timestamped log file (pattern from the thesis repo)."""

    def __init__(self, name: str):
        ensure_dir(LOGS)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(LOGS, f"{name}_{ts}.log")
        self.fh = open(write_guard(self.path), "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def close(self):
        sys.stdout = self.stdout
        self.fh.close()


# --------------------------------------------------------------------- session calendar
# The TSE extended the afternoon session close from 15:00 to 15:30 on 2024-11-05.
TSE_CLOSE_EXTENSION = _dt.date(2024, 11, 5)

MORNING_OPEN = 9 * 3600
MORNING_CLOSE = 11 * 3600 + 30 * 60
AFTERNOON_OPEN = 12 * 3600 + 30 * 60


def session_close_sec(d: _dt.date) -> int:
    return 15 * 3600 + 30 * 60 if d >= TSE_CLOSE_EXTENSION else 15 * 3600


def session_ranges(d: _dt.date) -> list[tuple[int, int]]:
    """Continuous-session (zaraba) bounds in seconds from midnight.

    No edge buffer: Ohta's measures use the whole continuous session. (The
    flash-crash thesis trims 5 minutes off each edge for a different purpose --
    do not copy that here.)
    """
    return [(MORNING_OPEN, MORNING_CLOSE), (AFTERNOON_OPEN, session_close_sec(d))]


def in_session(sec: int, d: _dt.date) -> bool:
    return (MORNING_OPEN <= sec <= MORNING_CLOSE) or (
        AFTERNOON_OPEN <= sec <= session_close_sec(d)
    )


# 30-minute buckets. Morning 09:00-11:30 = 5 buckets, afternoon 12:30-15:00 = 5,
# plus bucket 10 = 15:00-15:30, which exists only from 2024-11-05.
BUCKET_EDGES = [
    (0, MORNING_OPEN, MORNING_OPEN + 1800),
    (1, MORNING_OPEN + 1800, MORNING_OPEN + 3600),
    (2, MORNING_OPEN + 3600, MORNING_OPEN + 5400),
    (3, MORNING_OPEN + 5400, MORNING_OPEN + 7200),
    (4, MORNING_OPEN + 7200, MORNING_CLOSE),
    (5, AFTERNOON_OPEN, AFTERNOON_OPEN + 1800),
    (6, AFTERNOON_OPEN + 1800, AFTERNOON_OPEN + 3600),
    (7, AFTERNOON_OPEN + 3600, AFTERNOON_OPEN + 5400),
    (8, AFTERNOON_OPEN + 5400, AFTERNOON_OPEN + 7200),
    (9, AFTERNOON_OPEN + 7200, 15 * 3600),
    (10, 15 * 3600, 15 * 3600 + 1800),
]
N_BUCKETS = len(BUCKET_EDGES)


# ------------------------------------------------------------------- trade signing
# 約定種別. Verified empirically on ~76k trades: a trade tagged "At Sell Quote"
# printed at the prevailing best ASK 100% of the time (buyer-initiated); "At Buy
# Quote" printed at the best BID 100% of the time (seller-initiated). The naming
# refers to the quote consumed, not the initiator -- a standing trap.
#
# tse_tick's decode table covers raw codes 0/1/16/32/48/64 only. The stop-high /
# stop-low variants (1xx, 2xx) fall through as "Unknown (NNN)" strings and must
# be mapped here or volatile-day trades are silently dropped.
SIGN_BUY = 1      # buyer-initiated: market buy lifted the ask
SIGN_SELL = -1    # seller-initiated: market sell hit the bid
SIGN_NONE = 0     # unsigned: auctions, between/outside quotes, unknown

EXEC_TYPE_MAP: dict[str, int] = {
    "At Sell Quote": SIGN_BUY,
    "Unknown (148)": SIGN_BUY,    # stop-high variant of 48
    "Unknown (248)": SIGN_BUY,    # stop-low variant of 48
    "At Buy Quote": SIGN_SELL,
    "Unknown (116)": SIGN_SELL,   # stop-high variant of 16
    "Unknown (216)": SIGN_SELL,   # stop-low variant of 16
    "Between Quotes": SIGN_NONE,
    "Unknown (132)": SIGN_NONE,
    "Unknown (232)": SIGN_NONE,
    "Outside Quotes": SIGN_NONE,
    "Unknown (164)": SIGN_NONE,
    "Unknown (264)": SIGN_NONE,
    "Opening": SIGN_NONE,
    "Unknown (101)": SIGN_NONE,
    "Unknown (201)": SIGN_NONE,
    "Other": SIGN_NONE,
    "Unknown (100)": SIGN_NONE,
    "Unknown (200)": SIGN_NONE,
}

# Execution types that mark an auction print rather than a continuous-session trade.
AUCTION_EXEC_TYPES = {"Opening", "Unknown (101)", "Unknown (201)", "Other",
                      "Unknown (100)", "Unknown (200)"}

# Quote flags arrive as raw integers (tse_tick decodes Ayumi and Execution Type to
# strings but leaves the per-level quote flags numeric). Ohta requires both sides to
# be showing *ordinary* quotes (一般気配); those are exactly 128 "Regular Quote" and
# 131 "Regular (Improving)". Everything else -- 0 no quote, 32/33 special quote,
# 64 market-order quote, 112 pre-opening, 130 final quote -- is excluded.
ORDINARY_QUOTE_FLAGS = frozenset({128, 131})

# Ayumi (歩み値) states that count as continuous-session trading. Trades carrying any
# other state -- the closing prints, call auctions, halts, circuit breakers,
# reference prices -- are excluded from every measure.
ZARABA_AYUMI_FLAGS = frozenset({"Regular", "Discontinuous"})


# ------------------------------------------------------------------- yobine tables
# JPX tick-size (呼値の単位) schedule, in integer DECI-YEN so all downstream
# arithmetic stays exact. Verified 2026-07-29 against two independent brokerage
# transcriptions of the JPX table (Rakuten Securities notice 2023-05-29 and Matsui
# Securities rules page) and cross-checked against four facts read off the tape
# itself: 7203 @ ~4,000 yen -> 1 yen; 8604 @ ~900 yen -> 0.1 yen; 8306 @ ~1,500
# yen -> 0.5 yen; 4666 flipping 1 yen -> 0.5 yen between 2023-05-15 and 2023-07-18.
#
# Bands are "price <= upper" with the upper bound INCLUSIVE (JPX wording: 以下),
# so a stock priced at exactly 3,000 yen is in the 3,000-and-below band.
# Entries are (upper_bound_deci_yen, tick_deci_yen); None = no upper bound.
YOBINE_GENERAL: list[tuple[int | None, int]] = [
    (30_000, 10),                 # <=      3,000 yen -> 1 yen
    (50_000, 50),                 # <=      5,000 yen -> 5 yen
    (300_000, 100),               # <=     30,000 yen -> 10 yen
    (500_000, 500),               # <=     50,000 yen -> 50 yen
    (3_000_000, 1_000),           # <=    300,000 yen -> 100 yen
    (5_000_000, 5_000),           # <=    500,000 yen -> 500 yen
    (30_000_000, 10_000),         # <=  3,000,000 yen -> 1,000 yen
    (50_000_000, 50_000),         # <=  5,000,000 yen -> 5,000 yen
    (300_000_000, 100_000),       # <= 30,000,000 yen -> 10,000 yen
    (500_000_000, 500_000),       # <= 50,000,000 yen -> 50,000 yen
    (None, 1_000_000),            # above          -> 100,000 yen
]

# TOPIX500 constituents (TOPIX100 from 2014-07-22 / 2015-09-24; extended to
# TOPIX Mid400 -- hence all of TOPIX500 -- on 2023-06-05, stable through 2024).
YOBINE_TOPIX500: list[tuple[int | None, int]] = [
    (10_000, 1),                  # <=      1,000 yen -> 0.1 yen
    (30_000, 5),                  # <=      3,000 yen -> 0.5 yen
    (100_000, 10),                # <=     10,000 yen -> 1 yen
    (300_000, 50),                # <=     30,000 yen -> 5 yen
    (1_000_000, 100),             # <=    100,000 yen -> 10 yen
    (3_000_000, 500),             # <=    300,000 yen -> 50 yen
    (10_000_000, 1_000),          # <=  1,000,000 yen -> 100 yen
    (30_000_000, 5_000),          # <=  3,000,000 yen -> 500 yen
    (100_000_000, 10_000),        # <= 10,000,000 yen -> 1,000 yen
    (300_000_000, 50_000),        # <= 30,000,000 yen -> 5,000 yen
    (None, 100_000),              # above          -> 10,000 yen
]

# The date the fine grid was extended from TOPIX100 to all of TOPIX500.
FINE_TICK_ALL_TOPIX500 = _dt.date(2023, 6, 5)

# Ohta's sample filter (d): the last-digit-of-ten construction only makes sense
# when the tick is a power of ten. Deci-yen: 1 = 0.1 yen, 10 = 1 yen, ...
POWER_OF_TEN_TICKS10 = {1, 10, 100, 1_000, 10_000}


def tick_for10(p10: int, is_topix500: bool) -> int:
    """Tick size in deci-yen for a price given in deci-yen."""
    table = YOBINE_TOPIX500 if is_topix500 else YOBINE_GENERAL
    for upper, tick in table:
        if upper is None or p10 <= upper:
            return tick
    raise AssertionError("unreachable: yobine table has no open-ended band")


def day_tick_constant10(pmin10: int, pmax10: int, is_topix500: bool) -> int | None:
    """Tick for the day if constant across the day's price range, else None.

    Both tables are monotone non-decreasing in price, so checking the endpoints
    is sufficient -- if the low and the high map to the same tick, every price in
    between does too. Ohta's filter (d) requires exactly this constancy.
    """
    lo = tick_for10(pmin10, is_topix500)
    hi = tick_for10(pmax10, is_topix500)
    return lo if lo == hi else None


def load_topix500(years: Iterable[int] = (2023, 2024)) -> set[str]:
    """Point-in-time TOPIX500 membership, as a union over the given years.

    Membership is a December snapshot per year (the TOPIX rebalance lands in late
    October). A stock that joined or left mid-2024 could carry either grid for
    part of the year, so we take the union as the "possibly on the fine grid" set
    and let the empirical tick inference arbitrate per stock-day.
    """
    import csv

    want = {int(y) for y in years}
    out: set[str] = set()
    with open(TOPIX500_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if int(row["year"]) in want:
                out.add(str(row["ticker"]).strip().zfill(4))
    if not out:
        raise AssertionError(f"no TOPIX500 rows for {sorted(want)} in {TOPIX500_CSV}")
    return out


# --------------------------------------------------------------------- data calendar
# Verified by direct inventory of G:\needs on 2026-07-29.
KNOWN_MISSING_DATES = {"20240424", "20240425", "20240426", "20240430"}
# 2024-04-23 delivered only 10 shards where adjacent days carry 20-29: a download
# that died mid-day. Excluded as incomplete rather than trusted as a short day.
KNOWN_TRUNCATED_DATES = {"20240423"}
EXCLUDED_DATES = KNOWN_MISSING_DATES | KNOWN_TRUNCATED_DATES
