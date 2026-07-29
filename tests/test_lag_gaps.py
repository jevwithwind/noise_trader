"""Lags must not bridge a gap in the calendar.

When only part of the year has been ingested the panel skips whole months. A
plain one-row shift then treats a gap of weeks as an overnight lag, which is
worse than having no lag: it produces a confident number from two unrelated
observations. These tests pin the guard.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s5_common as S5


def frame(dates: list[dt.date]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame({
        "ticker": ["7203"] * n,
        "date": dates,
        "close_px": [100.0 + i for i in range(n)],
        "open_px": [100.0 + i for i in range(n)],
        "effsprd_bps": [5.0] * n,
        "rv5": [0.0001] * n,
        "yenvol": [1e9] * n,
        "m_b_large0": [0.15 + 0.001 * i for i in range(n)],
        "tick10": [10] * n,
        "open_digit": [0] * n,
        "fine_tick_day": [False] * n,
    })


def test_consecutive_days_get_a_lag():
    d = S5.build_frame(frame([dt.date(2024, 1, 4), dt.date(2024, 1, 5)]),
                       winsorize=False)
    assert d["prev_ticker_ok"].to_list() == [False, True]
    assert d["m_b_large0_l1"][1] is not None


def test_a_weekend_still_counts():
    # Friday to Monday is a three-day calendar gap and a one-day trading gap.
    d = S5.build_frame(frame([dt.date(2024, 1, 5), dt.date(2024, 1, 8)]),
                       winsorize=False)
    assert d["prev_ticker_ok"][1]


def test_a_month_long_gap_does_not():
    """January to April is not an overnight lag."""
    d = S5.build_frame(frame([dt.date(2024, 1, 31), dt.date(2024, 4, 1)]),
                       winsorize=False)
    assert not d["prev_ticker_ok"][1]
    assert d["m_b_large0_l1"][1] is None
    assert d["ret_overnight"][1] is None


def test_the_boundary_is_where_it_says():
    ok = S5.build_frame(frame([dt.date(2024, 1, 4),
                               dt.date(2024, 1, 4) + dt.timedelta(days=S5.MAX_LAG_GAP_DAYS)]),
                        winsorize=False)
    bad = S5.build_frame(frame([dt.date(2024, 1, 4),
                                dt.date(2024, 1, 4) + dt.timedelta(days=S5.MAX_LAG_GAP_DAYS + 1)]),
                         winsorize=False)
    assert ok["prev_ticker_ok"][1]
    assert not bad["prev_ticker_ok"][1]


def test_lags_do_not_cross_stocks():
    df = pl.concat([frame([dt.date(2024, 1, 4), dt.date(2024, 1, 5)]),
                    frame([dt.date(2024, 1, 4), dt.date(2024, 1, 5)])
                    .with_columns(ticker=pl.lit("6758"))])
    d = S5.build_frame(df, winsorize=False).sort(["ticker", "date"])
    firsts = d.group_by("ticker").agg(pl.col("prev_ticker_ok").first())
    assert not any(firsts["prev_ticker_ok"].to_list())
