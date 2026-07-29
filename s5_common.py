"""Regression machinery: frame construction, fixed effects, clustered inference.

Design decisions worth stating, because they shape every number in the report.

*Raw measures, not residuals.* An earlier design used a two-step procedure --
residualise the clustering measure on the contamination controls, then regress on
the residual. That is redundant when the same controls appear in the second stage,
and it makes the standard errors wrong (the residual is generated, not observed).
The controls go directly into every specification instead. The residualised signal
survives in exactly one place, the strategy demonstration, where a tradable
quantity has to be formed from information available at the time.

*Lagged dependent variable.* The headline specification includes the outcome's own
lag. Clustering is highly persistent and so is liquidity, so a regression without
it mostly recovers the fact that liquid stocks are liquid. With it, the
coefficient answers a sharper question: does today's clustering tell us anything
about tomorrow's liquidity that yesterday's liquidity did not already say? With
240 time periods the resulting dynamic-panel bias is of order 1/T and negligible.

*Nothing here is causal.* One year, no experiment, no instrument. Every statement
in the report is descriptive or predictive.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4

OUT = os.path.join(C.RESULTS, "s5_reg")

#: Outcomes named in advance as primary. Everything else is exploratory and
#: labelled as such, so the reader can price the multiple testing.
PRIMARY_Y = ["ln_effsprd", "imp60_bps"]
PRIMARY_X = "m_b_large0"

CONTROLS = ["rel_tick", "ln_yenvol_l1", "ln_rv5_l1", "ln_effsprd_l1",
            "ret_overnight", "fine_tick_day"]


def build_frame(df: pl.DataFrame, winsorize: bool = True) -> pl.DataFrame:
    """Transform, lag and winsorise the panel for estimation."""
    df = df.sort(["ticker", "date"])

    pos = {"effsprd_bps": "ln_effsprd", "qspread_twa_bps": "ln_qspread",
           "rv5": "ln_rv5", "amihud": "ln_amihud", "yenvol": "ln_yenvol",
           "mktcap": "ln_mktcap"}
    for src, dst in pos.items():
        if src in df.columns:
            df = df.with_columns(
                **{dst: pl.when(pl.col(src) > 0).then(pl.col(src).log()).otherwise(None)})
    for src, dst in (("depth_best_ln", "ln_depth_best"), ("depth10_ln", "ln_depth10")):
        if src in df.columns:
            df = df.with_columns(**{dst: pl.col(src)})

    if "tick10" in df.columns and "open_px" in df.columns:
        df = df.with_columns(
            rel_tick=pl.when(pl.col("open_px") > 0)
            .then(pl.col("tick10") / 10.0 / pl.col("open_px") * 1e4).otherwise(None))

    # Overnight return: previous close to this open, within stock.
    df = df.with_columns(prev_close=pl.col("close_px").shift(1).over("ticker"),
                         prev_ticker_ok=pl.col("ticker") == pl.col("ticker").shift(1))
    df = df.with_columns(
        ret_overnight=pl.when(pl.col("prev_ticker_ok") & (pl.col("prev_close") > 0))
        .then(pl.col("open_px") / pl.col("prev_close") - 1.0).otherwise(None))

    # Low-volatility indicator, defined cross-sectionally within each day.
    if "rv5" in df.columns:
        df = df.with_columns(
            lowvol_l1=(pl.col("rv5") < pl.col("rv5").median().over("date"))
            .shift(1).over("ticker"))

    lag_me = ["m_b_large0", "m_s_large0", "m_b_small0", "m_s_small0", "m0_all",
              "ln_effsprd", "ln_qspread", "imp60_bps", "imp1_bps", "imp300_bps",
              "rs60_bps", "ln_depth_best", "ln_depth10", "ln_rv5", "vr_absdev",
              "ln_amihud", "ln_yenvol", "rdepth_ask0", "rdepth_bid0",
              "l_s0", "l_b0", "l_s0c", "l_b0c", "dimp60_b_large", "dimp60_s_large",
              "ofi_sum", "dslope"] + [f"m_b_large{d}" for d in range(1, 10)]
    for c in lag_me:
        if c in df.columns:
            df = df.with_columns(
                **{f"{c}_l1": pl.when(pl.col("prev_ticker_ok"))
                   .then(pl.col(c).shift(1).over("ticker")).otherwise(None)})

    if winsorize:
        # Within stock, at 2.5% and 97.5%, following the paper. Only the panel
        # used for estimation is trimmed; the stored panel stays raw.
        wins = [c for c in df.columns if c.startswith(("imp", "dimp", "rs", "ln_",
                                                       "l_s0", "l_b0", "vr_", "ofi"))]
        exprs = []
        for c in wins:
            if df.schema[c] not in (pl.Float64, pl.Float32):
                continue
            lo = pl.col(c).quantile(0.025).over("ticker")
            hi = pl.col(c).quantile(0.975).over("ticker")
            exprs.append(pl.col(c).clip(lo, hi).alias(c))
        if exprs:
            df = df.with_columns(exprs)
    return df


def opening_digit_dummies(df: pl.DataFrame) -> list[str]:
    """The interaction of the opening digit with low volatility (Ohta's control).

    Digit 0 is the omitted category, so coefficients read against it.
    """
    names = []
    for d in range(1, 10):
        n = f"d{d}_lowvol"
        df_col = ((pl.col("open_digit") == d) & pl.col("lowvol_l1").fill_null(False))
        df = df.with_columns(**{n: df_col.cast(pl.Float64)})
        names.append(n)
    return names, df


def fit(df: pl.DataFrame, y: str, xs: list[str], *, entity="ticker", time="date",
        backend: str = "pyfixest") -> dict:
    """Two-way fixed effects with standard errors clustered on both margins."""
    cols = [y] + xs + [entity, time]
    d = df.select([c for c in cols if c in df.columns]).drop_nulls()
    have = [x for x in xs if x in d.columns]
    if d.height < 200 or not have:
        return {"n": d.height, "coef": {}, "se": {}, "t": {}, "error": "insufficient data"}

    pdf = d.to_pandas()
    pdf[entity] = pdf[entity].astype(str)
    pdf[time] = pdf[time].astype(str)

    try:
        if backend == "pyfixest":
            import pyfixest as pf
            fml = f"{y} ~ " + " + ".join(have) + f" | {entity} + {time}"
            res = pf.feols(fml, data=pdf, vcov={"CRV1": f"{entity}+{time}"})
            coef, se = res.coef().to_dict(), res.se().to_dict()
            return {"n": int(res._N), "coef": coef, "se": se,
                    "t": {k: coef[k] / se[k] if se.get(k) else float("nan")
                          for k in coef},
                    "r2": float(getattr(res, "_r2_within", float("nan")) or float("nan")),
                    "backend": "pyfixest"}
        import linearmodels.panel as lmp
        pdf["_t"] = pd_to_dt(pdf[time])
        pdf = pdf.set_index([entity, "_t"])
        mod = lmp.PanelOLS(pdf[y], pdf[have], entity_effects=True, time_effects=True,
                           drop_absorbed=True)
        r = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        return {"n": int(r.nobs), "coef": r.params.to_dict(),
                "se": r.std_errors.to_dict(), "t": r.tstats.to_dict(),
                "r2": float(r.rsquared_within), "backend": "linearmodels"}
    except Exception as exc:
        return {"n": d.height, "coef": {}, "se": {}, "t": {}, "error": str(exc)}


def pd_to_dt(s):
    import pandas as pd
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")


def cell(res: dict, x: str, nd: int = 4) -> tuple[str, str]:
    if x not in res.get("coef", {}):
        return ("--", "")
    b, se = res["coef"][x], res["se"].get(x, float("nan"))
    t = res["t"].get(x, float("nan"))
    return (f"{b:.{nd}f}{S4.stars(t)}", f"({se:.{nd}f})")
