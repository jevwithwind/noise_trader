"""The two failure classes that shipped once, pinned so they cannot ship again.

1. The low-volatility indicator degenerated to all-False (a nested-window
   scoping accident), which zeroed every opening-digit contamination dummy;
   pyfixest then dropped the nine columns as collinear without changing any
   other number, so nothing downstream looked wrong.
2. The half-sample robustness split was hardcoded to a calendar date from an
   earlier study window; when the window moved, H1 became empty and H2 became
   the whole sample -- a check that no longer checked anything.
"""
from __future__ import annotations

import datetime as dt
import os
import random
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s5_common as S5


def synth_panel(n_tickers: int = 8, n_days: int = 30) -> pl.DataFrame:
    """A small panel with everything build_frame touches, nothing more."""
    dates = []
    d = dt.date(2025, 1, 6)
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    rng = random.Random(7)
    rows = []
    for i in range(n_tickers):
        for j, date in enumerate(dates):
            rows.append({
                "ticker": f"{7000 + i}",
                "date": date,
                "open_px": 1000.0 + i,
                "close_px": 1000.0 + i + (j % 5) - 2,
                "rv5": rng.uniform(0.5, 2.0) * (1 + (i % 3)),
                "open_digit": (i * 7 + j) % 10,
                "m_b_large0": 0.10 + 0.01 * ((i + j) % 5),
                "effsprd_bps": 5.0 + (j % 7),
                "yenvol": 1e8,
                "tick10": 10,
                "fine_tick_day": False,
            })
    return pl.DataFrame(rows)


def test_lowvol_marks_roughly_half_the_cross_section():
    df = S5.build_frame(synth_panel())
    for col in ("lowvol_now", "lowvol_l1"):
        share = float(df[col].fill_null(False).mean())
        # A below-the-daily-median indicator that is not near one half is the
        # degenerate-window bug (x < x is identically False) coming back.
        assert 0.2 < share < 0.8, f"{col} share {share:.3f} is degenerate"


def test_opening_digit_dummies_have_support():
    df = S5.build_frame(synth_panel())
    for lagged in (True, False):
        names, out = S5.opening_digit_dummies(df, lagged=lagged)
        assert len(names) == 9
        total = sum(float(out[n].sum()) for n in names)
        assert total > 0, "contamination dummies must not be identically zero"


def test_opening_digit_dummies_refuse_a_degenerate_set():
    df = S5.build_frame(synth_panel(n_tickers=8, n_days=180))
    df = df.with_columns(lowvol_l1=pl.lit(False))
    with pytest.raises(AssertionError, match="identically zero"):
        S5.opening_digit_dummies(df, lagged=True)


def test_sample_halves_partition_and_order():
    df = synth_panel()
    halves = S5.sample_halves(df)
    assert len(halves) == 2
    (lab1, h1), (lab2, h2) = halves
    assert lab1.startswith("H1") and lab2.startswith("H2")
    assert h1.height > 0 and h2.height > 0
    assert h1.height + h2.height == df.height
    assert h1["date"].cast(pl.Utf8).max() < h2["date"].cast(pl.Utf8).min()


def test_sample_halves_never_returns_an_empty_half():
    # Fewer than four distinct dates: refuse rather than fabricate a split.
    df = synth_panel(n_days=3)
    assert S5.sample_halves(df) == []
