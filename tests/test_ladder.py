"""Ladder inference tested on a hand-built tape with known order flow.

Every number asserted here was worked out by hand from the tape below, so the
test checks the algorithm against the truth rather than against itself. The three
scenarios are the three ways this inference goes wrong in practice:

  1. an execution mistaken for a cancellation (netting),
  2. a book that shifts a level, so index-matching invents a submission and a
     cancellation that never happened (price matching),
  3. a price entering the visible window from beyond level ten, whose pre-existing
     depth would look like a huge submission (observability).
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import measures as M
import s0_common as C

DATE = dt.date(2024, 6, 3)
TICK10 = 10          # 1-yen grid, so digit 0 means a whole ten yen
REGULAR = 128


def _row(t_us: int, asks: list[tuple[float, int]], bids: list[tuple[float, int]],
         *, exec_type=None, price=0.0, vol=0) -> dict:
    """One tape row: a book snapshot, optionally carrying a trade print."""
    r: dict = {
        "Update Time": _fmt(t_us),
        "Execution Time": "" if exec_type is None else _fmt(t_us)[:6],
        "Execution Price": price, "Execution Type": exec_type,
        "Ayumi Flag": None if exec_type is None else "Regular",
        "Volume": vol, "Session": "Morning / Day",
    }
    for i in range(10):
        ap, av = asks[i] if i < len(asks) else (0.0, 0)
        bp, bv = bids[i] if i < len(bids) else (0.0, 0)
        r[M.ASK_PX[i]], r[M.ASK_VOL[i]], r[M.ASK_FLAG[i]] = ap, av, REGULAR
        r[M.BID_PX[i]], r[M.BID_VOL[i]], r[M.BID_FLAG[i]] = bp, bv, REGULAR
    r["Sell Quote Vol OVER"] = 0
    r["Buy Quote Vol UNDER"] = 0
    return r


def _fmt(t_us: int) -> str:
    s, us = divmod(t_us, 1_000_000)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}{m:02d}{sec:02d}{us:06d}"


T0 = 10 * 3600 * 1_000_000  # 10:00:00, comfortably inside the morning session


def build_tape() -> pl.DataFrame:
    """A six-row tape whose order flow is known exactly.

    Ask side starts 1000/1001/1002, bid side 999/998/997. On the 1-yen grid the
    last digit of 1000 is 0, so 1000 is the round price and every flow below is
    deliberately concentrated there.
    """
    bids = [(999.0, 400), (998.0, 300), (997.0, 100)]
    rows = [
        # seq 0 -- opening state. No previous snapshot, so no interval.
        _row(T0 + 0, [(1000.0, 500), (1001.0, 300), (1002.0, 200)], bids),
        # seq 1 -- nothing changes. Interval must produce no flow at all.
        _row(T0 + 1_000_000, [(1000.0, 500), (1001.0, 300), (1002.0, 200)], bids),
        # seq 2 -- 300 shares added at the round price: a submission.
        _row(T0 + 2_000_000, [(1000.0, 800), (1001.0, 300), (1002.0, 200)], bids),
        # seq 3 -- a buyer lifts 500 at 1000. Depth drops 800 -> 300, but the drop
        # is entirely explained by the print, so neither a submission nor a
        # cancellation should be inferred.
        _row(T0 + 3_000_000, [(1000.0, 300), (1001.0, 300), (1002.0, 200)], bids,
             exec_type="At Sell Quote", price=1000.0, vol=500),
        # seq 4 -- 200 shares vanish at 1000 with no print: a cancellation.
        _row(T0 + 4_000_000, [(1000.0, 100), (1001.0, 300), (1002.0, 200)], bids),
        # seq 5 -- 1000 empties (100 cancelled) and the book shifts up, so 1003
        # enters from beyond the old window. The 100 must be counted; the depth
        # arriving at 1003 must not.
        _row(T0 + 5_000_000, [(1001.0, 300), (1002.0, 200), (1003.0, 150)], bids),
    ]
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def lad() -> dict:
    df = build_tape()
    n = M.normalize_day(df, DATE, wide=True)
    return M.ladder_lc(n, DATE, TICK10)


def test_five_valid_intervals(lad):
    """Six snapshots, five intervals -- the first row has no predecessor."""
    assert lad["n_ladder_intervals"] == 5


def test_submitted_volume_is_the_300_added_at_the_round_price(lad):
    assert lad["sub_vol_s"] == 300


def test_cancelled_volume_is_200_plus_100(lad):
    """The 1003 depth arriving from outside the window must not be counted."""
    assert lad["canc_vol_s"] == 300


def test_execution_is_not_mistaken_for_a_cancellation(lad):
    """Without netting the print, seq 3 would add 500 of phantom cancellation."""
    assert lad["canc_vol_s"] == 300, "500-share execution leaked into cancellations"


def test_round_price_shares_are_one(lad):
    """All flow was placed at 1000, whose last digit is 0."""
    assert lad["l_s0"] == pytest.approx(1.0)
    assert lad["c_s0"] == pytest.approx(1.0)


def test_cancellation_and_execution_ratios(lad):
    # 300 cancelled against 300 submitted at the round price.
    assert lad["l_s0c"] == pytest.approx(1.0)
    # 500 executed against 300 submitted.
    assert lad["l_s0e"] == pytest.approx(500 / 300)


def test_bid_side_saw_no_flow(lad):
    """The bid ladder never moved, so both of its aggregates must be zero."""
    assert lad["sub_vol_b"] == 0
    assert lad["canc_vol_b"] == 0


def test_level_shift_does_not_invent_flow():
    """Index matching would report a submission and a cancellation at every level.

    Between seq 4 and seq 5 the ask ladder shifts: level 1 goes 1000 -> 1001,
    level 2 goes 1001 -> 1002, level 3 goes 1002 -> 1003. Matching by level index
    would see three changed levels; matching by price sees one cancellation.
    """
    df = build_tape().slice(4, 2)
    n = M.normalize_day(df, DATE, wide=True)
    out = M.ladder_lc(n, DATE, TICK10)
    assert out["n_ladder_intervals"] == 1
    assert out["canc_vol_s"] == 100
    assert out["sub_vol_s"] == 0


def test_quiet_interval_produces_nothing():
    df = build_tape().slice(0, 2)
    n = M.normalize_day(df, DATE, wide=True)
    out = M.ladder_lc(n, DATE, TICK10)
    assert out["n_ladder_intervals"] == 1
    assert out["sub_vol_s"] == 0 and out["canc_vol_s"] == 0
    assert out["sub_vol_b"] == 0 and out["canc_vol_b"] == 0


def test_distance_class_puts_the_round_price_at_the_best_quote(lad):
    """1000 is the best ask throughout the flow, so all of it is 'atbest'."""
    assert lad["l_s0_atbest"] == pytest.approx(1.0)


def test_non_ordinary_quote_breaks_the_interval():
    """A special quote in the middle must not be bridged across."""
    rows = build_tape().to_dicts()
    rows[3][M.ASK_FLAG[0]] = 32          # special quote
    rows[3][M.BID_FLAG[0]] = 32
    n = M.normalize_day(pl.DataFrame(rows), DATE, wide=True)
    out = M.ladder_lc(n, DATE, TICK10)
    # One bad row kills two intervals -- the one ending at it and the one starting
    # from it -- so 5 - 2 = 3 survive.
    assert out["n_ladder_intervals"] == 3
    # Those two carried the 500-share execution and the 200-share cancellation, so
    # only the 100 cancelled at seq 5 remains. Dropping real flow is the right
    # trade: the alternative is bridging across a state where the displayed book
    # does not mean what it usually means.
    assert out["canc_vol_s"] == 100
    assert out["sub_vol_s"] == 300


# ------------------------------------------------------------------ measures
def test_digit_zero_is_the_round_price_on_a_one_yen_grid():
    assert M.digit_expr(pl.lit(10000, dtype=pl.Int64), 10).eq(0) is not None
    got = pl.select(M.digit_expr(pl.lit(10000, dtype=pl.Int64), 10)).item()
    assert got == 0
    got = pl.select(M.digit_expr(pl.lit(10015, dtype=pl.Int64), 5)).item()
    assert got == 3      # 1001.5 yen on a 0.5-yen grid -> digit 3


def test_normalize_orders_a_shuffled_tape():
    """A stray pre-open row at the end of the file must not corrupt the sequence."""
    rows = build_tape().to_dicts()
    stray = _row(8 * 3600 * 1_000_000, [(999.0, 100)], [(998.0, 100)])
    n = M.normalize_day(pl.DataFrame(rows + [stray]), DATE, wide=True)
    ts = n["t_us"].to_list()
    assert ts == sorted(ts)
    assert not n["in_session"][0]        # the 08:00 row sorts to the front
