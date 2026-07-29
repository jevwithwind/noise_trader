"""Unit tests for the institutional layer: yobine tables, sessions, signing.

The tick-table cases are not invented -- each one is a fact read off the tape
during the preparation drill, so these tests check the coded tables against
observed market behaviour, not against themselves.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s0_common as C


# --------------------------------------------------------------------- yobine
@pytest.mark.parametrize("yen,is_t500,want_yen,why", [
    # Tape-verified anchors from the drill.
    (4000.0, True, 1.0, "7203 Toyota ~4,000 yen, TOPIX500 fine table 3,000-10,000 band"),
    (3600.0, True, 1.0, "7203 low end of its 2024-04-01 range"),
    (4500.0, True, 1.0, "7203 high end of its 2024-04-01 range"),
    (900.0, True, 0.1, "8604 Nomura ~900 yen -> the 0.1 yen grid"),
    (1500.0, True, 0.5, "8306 MUFG ~1,500 yen -> the 0.5 yen grid (excluded regime)"),
    (1800.0, False, 1.0, "4666 Park24 pre-2023-06-05, general table"),
    (1800.0, True, 0.5, "4666 post-2023-06-05, fine table"),
    # Band edges: JPX wording is 'X 以下' so the upper bound belongs to the band.
    (1000.0, True, 0.1, "exactly 1,000 yen is still in the 0.1 yen band"),
    (1000.1, True, 0.5, "just above 1,000 yen moves to 0.5 yen"),
    (3000.0, True, 0.5, "exactly 3,000 yen is still 0.5 yen"),
    (3000.1, True, 1.0, "just above 3,000 yen moves to 1 yen"),
    (3000.0, False, 1.0, "general table: 3,000 yen is still 1 yen"),
    (3000.1, False, 5.0, "general table: above 3,000 yen jumps to 5 yen"),
    (5000.0, False, 5.0, "general table: 5,000 yen is 5 yen"),
    (5000.1, False, 10.0, "general table: above 5,000 yen is 10 yen"),
    (10000.0, True, 1.0, "fine table: 10,000 yen is still 1 yen"),
    (10000.1, True, 5.0, "fine table: above 10,000 yen is 5 yen"),
])
def test_tick_for10(yen, is_t500, want_yen, why):
    got = C.tick_for10(round(yen * 10), is_t500) / 10.0
    assert got == want_yen, f"{why}: got {got}, want {want_yen}"


def test_tables_monotone_nondecreasing():
    """Endpoint-only day-constancy checks are valid only if the tables are monotone."""
    for table in (C.YOBINE_GENERAL, C.YOBINE_TOPIX500):
        ticks = [t for _, t in table]
        assert ticks == sorted(ticks), "yobine table is not monotone in price"


def test_fine_table_is_never_coarser():
    """The TOPIX500 grid exists to be finer -- it must never be coarser anywhere."""
    for p10 in [1, 5_000, 9_999, 10_000, 30_000, 100_000, 1_000_000, 10_000_000]:
        assert C.tick_for10(p10, True) <= C.tick_for10(p10, False)


def test_day_tick_constant():
    # 7203's whole 2024-04-01 range sits inside one band -> constant.
    assert C.day_tick_constant10(36_000, 45_000, True) == 10
    # A range straddling the 3,000 yen boundary on the fine grid is not constant.
    assert C.day_tick_constant10(29_000, 31_000, True) is None
    # Same straddle on the general grid: 1 yen either side of 3,000 -> not constant.
    assert C.day_tick_constant10(29_000, 31_000, False) is None


def test_power_of_ten_filter_excludes_half_yen():
    """Ohta's filter (d): only powers of ten make the last-digit construction work."""
    assert C.tick_for10(15_000, True) == 5      # 0.5 yen
    assert 5 not in C.POWER_OF_TEN_TICKS10
    assert C.tick_for10(9_000, True) == 1       # 0.1 yen
    assert 1 in C.POWER_OF_TEN_TICKS10
    assert C.tick_for10(40_000, True) == 10     # 1 yen
    assert 10 in C.POWER_OF_TEN_TICKS10


# --------------------------------------------------------------------- digits
def digit(p10: int, tick10: int) -> int:
    return (p10 % (10 * tick10)) // tick10


@pytest.mark.parametrize("yen,tick_yen,want", [
    (1010.0, 1.0, 0), (1011.0, 1.0, 1), (1019.0, 1.0, 9),
    # The 0.1 yen grid is where float arithmetic goes wrong: 900.0 must be digit 0
    # and 900.1 digit 1, exactly.
    (900.0, 0.1, 0), (900.1, 0.1, 1), (900.9, 0.1, 9), (901.0, 0.1, 0),
    (899.9, 0.1, 9),
    (10100.0, 10.0, 0), (10110.0, 10.0, 1),
])
def test_digit_integer_arithmetic(yen, tick_yen, want):
    assert digit(round(yen * 10), round(tick_yen * 10)) == want


def test_float_mod_would_have_failed():
    """Documents why the pipeline is integer-only: the naive float version breaks."""
    price, tick = 900.3, 0.1
    naive = int((price % (10 * tick)) / tick)
    exact = digit(round(price * 10), round(tick * 10))
    assert exact == 3
    assert naive != exact, "float mod happened to work here; the guard is still right"


# -------------------------------------------------------------------- sessions
def test_session_close_switches_on_2024_11_05():
    assert C.session_close_sec(dt.date(2024, 11, 1)) == 15 * 3600
    assert C.session_close_sec(dt.date(2024, 11, 5)) == 15 * 3600 + 1800
    assert C.session_close_sec(dt.date(2024, 11, 6)) == 15 * 3600 + 1800


def test_in_session_boundaries():
    before, after = dt.date(2024, 10, 1), dt.date(2024, 12, 2)
    assert not C.in_session(8 * 3600 + 59 * 60, before)      # pre-open
    assert C.in_session(9 * 3600, before)                     # open
    assert C.in_session(11 * 3600 + 30 * 60, before)          # morning close
    assert not C.in_session(12 * 3600, before)                # lunch
    assert C.in_session(12 * 3600 + 30 * 60, before)          # afternoon open
    assert C.in_session(15 * 3600, before)                    # old close
    assert not C.in_session(15 * 3600 + 60, before)           # past old close
    assert C.in_session(15 * 3600 + 29 * 60, after)           # inside extension
    assert not C.in_session(15 * 3600 + 31 * 60, after)


def test_buckets_cover_the_session_without_overlap():
    edges = C.BUCKET_EDGES
    assert len(edges) == 11
    for _, lo, hi in edges:
        assert hi > lo
    # Morning buckets are contiguous, and so are the afternoon ones.
    assert edges[0][1] == C.MORNING_OPEN
    assert edges[4][2] == C.MORNING_CLOSE
    assert edges[5][1] == C.AFTERNOON_OPEN
    assert edges[10][2] == 15 * 3600 + 1800


# --------------------------------------------------------------------- signing
def test_execution_type_signing_is_the_verified_mapping():
    """'At Sell Quote' means the ASK was consumed, i.e. the BUYER initiated."""
    assert C.EXEC_TYPE_MAP["At Sell Quote"] == C.SIGN_BUY
    assert C.EXEC_TYPE_MAP["At Buy Quote"] == C.SIGN_SELL


def test_stop_quote_variants_are_mapped():
    """Unmapped stop-quote codes would silently drop trades on volatile days."""
    for code, want in [("Unknown (148)", C.SIGN_BUY), ("Unknown (248)", C.SIGN_BUY),
                       ("Unknown (116)", C.SIGN_SELL), ("Unknown (216)", C.SIGN_SELL)]:
        assert C.EXEC_TYPE_MAP[code] == want


def test_auction_and_ambiguous_types_are_unsigned():
    for code in ("Opening", "Other", "Between Quotes", "Outside Quotes"):
        assert C.EXEC_TYPE_MAP[code] == C.SIGN_NONE
    assert "Opening" in C.AUCTION_EXEC_TYPES
    assert "Between Quotes" not in C.AUCTION_EXEC_TYPES  # zaraba, just unsigned


# ------------------------------------------------------------------ reference
def test_topix500_membership_loads():
    t500 = C.load_topix500()
    assert 400 < len(t500) < 700, f"implausible TOPIX500 size: {len(t500)}"
    for t in ("7203", "8306", "8604"):
        assert t in t500, f"{t} should be a TOPIX500 constituent"
    # Codes are four characters but not necessarily numeric: the exchange began
    # issuing alphanumeric codes such as "130A" in 2024. Anything that parses them
    # as integers will drop those stocks silently.
    assert all(len(t) == 4 and t.isalnum() for t in t500)


def test_write_guard_blocks_outside_paths():
    C.write_guard(os.path.join(C.PROJ, "results", "x.parquet"))
    C.write_guard(os.path.join(C.STORE, "individual_stock", "y.parquet"))
    for bad in (r"G:\needs\evil.txt", r"C:\Windows\evil.txt", r"E:\MTEC\evil.txt"):
        with pytest.raises(AssertionError):
            C.write_guard(bad)
