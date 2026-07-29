"""Shared helpers for the analysis stages: the panel, weighting, and inference."""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

PANEL = os.path.join(C.RESULTS, "s3_panel", "panel_stockday.parquet")
PANEL_INTRADAY = os.path.join(C.RESULTS, "s3_panel", "panel_intraday.parquet")

#: Ohta's headline cells: the buy- and sell-initiated large-trade measures.
M_CELLS = ["m_b_large0", "m_s_large0", "m_b_small0", "m_s_small0"]


def load_panel(final_only: bool = True, drop_fine_tick: bool = False) -> pl.DataFrame:
    df = pl.read_parquet(PANEL)
    if final_only:
        df = df.filter(pl.col("in_sample_final"))
    if drop_fine_tick:
        # Ohta drops 0.1-yen-tick days from the regressions: clustering there is
        # elevated for a mechanical reason (the grid is so fine that orders
        # collapse onto whole-yen prices) that has nothing to do with the
        # hypothesis being tested.
        df = df.filter(~pl.col("fine_tick_day"))
    return df


def size_quintiles(df: pl.DataFrame, col: str = "mktcap") -> pl.DataFrame:
    """Assign each stock to a size quintile from its median value of `col`.

    Assigned once per stock rather than per stock-day, so a stock does not drift
    between quintiles as its price moves during the year.
    """
    use = col if col in df.columns and df[col].is_not_null().any() else "yenvol"
    per = (df.group_by("ticker").agg(sz=pl.col(use).median())
             .drop_nulls("sz").sort("sz"))
    if per.height == 0:
        return df.with_columns(size_q=pl.lit(None, dtype=pl.Int8))
    per = per.with_columns(
        size_q=(pl.col("sz").rank("ordinal") * 5 // (per.height + 1) + 1)
        .cast(pl.Int8))
    return df.join(per.select("ticker", "size_q"), on="ticker", how="left")


def twoway_cluster_mean(df: pl.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean of a stock-day variable, with a standard error clustered two ways.

    A panel mean whose standard error ignores clustering is badly overconfident:
    the same stock appears on 240 days and the same day covers a thousand stocks,
    so observations are dependent along both margins. This uses the
    Cameron-Gelbach-Miller combination -- cluster by stock, plus by day, minus by
    stock-day -- applied to the constant-only regression.
    """
    d = df.select("ticker", "date", pl.col(col).alias("y")).drop_nulls("y")
    n = d.height
    if n < 10:
        return (float("nan"), float("nan"), n)
    # A two-way clustered variance is only meaningful with enough clusters on
    # each margin; below that it is not conservative, it is undefined.
    if d["date"].n_unique() < 5 or d["ticker"].n_unique() < 5:
        return (float(d["y"].mean()), float("nan"), n)
    y = d["y"].to_numpy().astype(float)
    mu = float(y.mean())
    e = y - mu

    def meat(keys) -> float:
        s = 0.0
        order = np.argsort(keys, kind="stable")
        k, ee = np.asarray(keys)[order], e[order]
        bounds = np.flatnonzero(k[1:] != k[:-1]) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(k)]):
            s += ee[lo:hi].sum() ** 2
        return s

    st = d["ticker"].to_numpy()
    dt_ = d["date"].cast(pl.Utf8).to_numpy()
    both = np.char.add(np.char.add(st.astype(str), "|"), dt_.astype(str))
    v = (meat(st) + meat(dt_) - meat(both)) / (n * n)
    g = min(len(np.unique(st)), len(np.unique(dt_)))
    if g > 1:
        v *= g / (g - 1)
    return (mu, float(np.sqrt(max(v, 0.0))), n)


def stars(t: float) -> str:
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.96 else "*" if a > 1.645 else ""


def fmt(v, nd: int = 3) -> str:
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{nd}f}"


# ------------------------------------------------------------------- LaTeX
def latex_table(path: str, caption: str, label: str, header: list[str],
                rows: list[list[str]], notes: str = "", align: str | None = None,
                small: bool = True) -> None:
    """Write a booktabs table in the thesis's style: caption above, notes below."""
    al = align or ("l" + "r" * (len(header) - 1))
    out = ["\\begin{table}[htbp]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if small:
        out.append("\\small")
    if notes:
        out.append("\\begin{threeparttable}")
    out += [f"\\begin{{tabular}}{{@{{}}{al}@{{}}}}", "\\toprule",
            " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(" & ".join("" if x is None else str(x) for x in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        out += ["\\begin{tablenotes}[flushleft]\\footnotesize",
                f"\\item {notes}", "\\end{tablenotes}", "\\end{threeparttable}"]
    out.append("\\end{table}")
    from s0_common import write_guard
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(write_guard(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


TABLES = os.path.join(C.REPORT, "tables")
FIGURES = os.path.join(C.REPORT, "figures")


def setup_mpl():
    """Figure style copied from the thesis, so the report looks of a piece."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "-", "font.family": "serif",
        "figure.figsize": (6.4, 3.2),
    })
    return plt


BLUE, ORANGE = "#3b6ea5", "#b5651d"
