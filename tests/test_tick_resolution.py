"""The tick-resolution rule, and the bug it exists to prevent.

Assigning a stock a finer tick than it actually trades on is not a small error.
Every price then lands on a multiple of the assumed tick, so the last digit is
constant and the clustering measure reads 100% -- or 50% if the true grid is five
times the assumed one. Three real stock-days from 2024-01-04 did exactly that
before the tape was made authoritative in both directions.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import measures as M
import s0_common as C


def test_finer_tape_always_wins():
    """A stock cannot quote inside a grid it is not on, so a finer tape is proof."""
    tick, src = M.resolve_tick10(table_tick10=10, emp_tick10=5, n_prices=3)
    assert (tick, src) == (5, "inferred")
    tick, src = M.resolve_tick10(table_tick10=10, emp_tick10=1, n_prices=2)
    assert (tick, src) == (1, "inferred")


def test_coarser_tape_wins_when_there_is_enough_evidence():
    """4194 on 2024-01-04: table said 1 yen, the tape used 10 yen throughout.

    It reported M0 = 100% until the tape was allowed to win.
    """
    tick, src = M.resolve_tick10(table_tick10=10, emp_tick10=100, n_prices=28)
    assert (tick, src) == (100, "inferred")


def test_coarser_tape_ignored_when_the_stock_barely_traded():
    """An illiquid stock skipping grid points must not redefine the grid."""
    tick, src = M.resolve_tick10(table_tick10=10, emp_tick10=50, n_prices=3)
    assert (tick, src) == (10, "table")


def test_the_threshold_is_where_it_says_it_is():
    below = M.resolve_tick10(10, 50, M.MIN_PRICES_FOR_COARSE - 1)
    at = M.resolve_tick10(10, 50, M.MIN_PRICES_FOR_COARSE)
    assert below == (10, "table")
    assert at == (50, "inferred")


def test_agreement_needs_no_adjudication():
    assert M.resolve_tick10(10, 10, 50) == (10, "table")


def test_missing_inputs():
    assert M.resolve_tick10(None, 5, 20) == (5, "inferred")
    assert M.resolve_tick10(10, None, 0) == (10, "table")
    assert M.resolve_tick10(None, None, 0) == (None, "none")


# ------------------------------------------------------- filter (d), both halves
def test_a_day_spanning_a_band_boundary_is_not_tick_constant():
    """A TOPIX500 stock whose range crosses 3,000 yen ran on two grids that day.

    Below the boundary the grid is 0.5 yen and above it 1 yen, so the last digit
    means different things at different times of the day. Ohta's filter (d)
    requires the tick to be constant, and this is the case it is there to catch.
    """
    assert C.day_tick_constant10(29_900, 30_100, True) is None
    assert C.day_tick_constant10(30_100, 34_000, True) == 10


def test_contaminated_digits_are_what_the_bug_looked_like():
    """Documents the failure mode: a 10-yen grid read as a 1-yen grid.

    Every observed price is a multiple of ten ticks, so every last digit is zero.
    """
    tick_assumed, tick_real = 10, 100        # 1 yen assumed, 10 yen real
    prices10 = [86_600 + k * tick_real for k in range(20)]
    digits = {(p % (10 * tick_assumed)) // tick_assumed for p in prices10}
    assert digits == {0}, "the bug produced M0 = 100%"
    good = {(p % (10 * tick_real)) // tick_real for p in prices10}
    assert len(good) == 10, "with the right tick the digits spread over 0-9"
