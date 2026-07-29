"""S6 -- a strategy demonstration, and what it is not.

This exists because the internship task is to build a *rebalancing strategy* on
this indicator, and the gap between "the indicator predicts liquidity" and "here
is a tradable rule" is exactly where such projects fail. So the machinery is
built and exercised end to end: form a tradable signal from information available
at the time, sort on it, rebalance, and charge realistic costs.

It is a demonstration of plumbing, not an alpha claim. One short sample, one market, no
out-of-sample period, and returns computed from raw closing prices that carry no
adjustment for splits or dividends. The honest output is the cost accounting and
the turnover, not the Sharpe ratio.

Two variants are run. The daily variant trades the raw daily signal, whose
first-order autocorrelation is low -- most of a day's reading is transitory --
so it churns. The smoothed variant trades a five-day rolling mean of the same
signal; if the churn is measurement noise rather than information, smoothing
should cut turnover far faster than it cuts the gross return. The comparison
between the two lines is the point.

The signal is deliberately size-neutralised. Clustering is strongest in small,
retail-heavy stocks, so an unneutralised sort is mostly a small-cap bet wearing a
microstructure costume.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4
import s5_common as S5

OUT = os.path.join(C.RESULTS, "s6_strategy")
MAX_ABS_RET = 0.25   # a daily move beyond this is a corporate action, not a return
ANN_DAYS = 245


def residual_signal(df: pl.DataFrame, dig_cols: list[str]) -> pl.DataFrame:
    """Cross-sectional residual of clustering on everything that mechanically drives it.

    Run separately each day, so nothing from the future enters. The controls
    are the ones the panel regressions use, in their same-day form: the
    signal is built from day t's cross-section, so the contamination controls
    are day t's opening digit interacted with day t's low-volatility flag --
    both known by the close, when the position would be formed.
    """
    feats = [c for c in ["rel_tick", "ln_yenvol_l1", "ln_rv5_l1", "ln_effsprd_l1",
                         "ln_mktcap", "ret_overnight"] if c in df.columns]
    feats += [c for c in dig_cols if c in df.columns]
    out = []
    for (d,), g in df.group_by(["date"], maintain_order=True):
        g = g.drop_nulls(["m_b_large0"] + feats)
        if g.height < 40:
            continue
        X = np.column_stack([np.ones(g.height)]
                            + [g[c].to_numpy().astype(float) for c in feats])
        y = g["m_b_large0"].to_numpy().astype(float)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
        except Exception:
            continue
        sd = r.std()
        out.append(g.select("date", "ticker", "close_px", "effsprd_bps", "imp60_bps")
                    .with_columns(signal=pl.Series("signal", r / sd if sd > 0 else r)))
    return pl.concat(out) if out else pl.DataFrame()


def add_forward_returns(sig: pl.DataFrame) -> pl.DataFrame:
    """Next genuine trading day's close-to-close return, within stock.

    If the sample skips a stretch of the calendar, the next row can be weeks
    away, and treating that as an overnight return would invent a position
    nobody could have held.
    """
    sig = sig.sort(["ticker", "date"]).with_columns(
        nxt=pl.col("close_px").shift(-1).over("ticker"),
        nxt_date=pl.col("date").shift(-1).over("ticker"),
        same=(pl.col("ticker") == pl.col("ticker").shift(-1)))
    sig = sig.with_columns(
        gap=(pl.col("nxt_date").cast(pl.Date) - pl.col("date").cast(pl.Date))
        .dt.total_days())
    return sig.with_columns(
        fwd=pl.when(pl.col("same") & (pl.col("close_px") > 0)
                    & pl.col("gap").is_not_null()
                    & (pl.col("gap") <= S5.MAX_LAG_GAP_DAYS))
        .then(pl.col("nxt") / pl.col("close_px") - 1.0).otherwise(None))


def evaluate(sig: pl.DataFrame, sigcol: str, label: str) -> dict | None:
    """Quintile-sort `sigcol`, rebalance daily, and account for costs.

    Costs charge the half-spread actually faced by the traded legs (Q1 and
    Q5), not the panel average: the extreme-signal quintiles are wider than
    the middle, and a cost model that averages over names it never trades
    understates the bill.
    """
    d = sig.drop_nulls([sigcol, "fwd"]).filter(pl.col("fwd").abs() <= MAX_ABS_RET)
    if d.is_empty():
        return None
    d = d.with_columns(
        q=(pl.col(sigcol).rank("ordinal").over("date") * 5
           // (pl.len().over("date") + 1) + 1).cast(pl.Int8))
    by_q = (d.group_by(["date", "q"]).agg(
        r=pl.col("fwd").mean(), n=pl.len(),
        hs=pl.col("effsprd_bps").mean()).sort(["date", "q"]))

    res: dict = {"label": label, "quintiles": [], "n_stockdays": d.height}
    print(f"\n[{label}] next-day return by signal quintile (Q5 = most clustered):")
    for q in range(1, 6):
        g = by_q.filter(pl.col("q") == q)
        r = g["r"].drop_nulls().to_numpy()
        if len(r) < 3:
            print(f"  Q{q}: only {len(r)} day(s); skipped")
            continue
        mu = float(r.mean()) * 1e4
        se = float(r.std(ddof=1)) / np.sqrt(len(r)) * 1e4
        t = mu / se if se > 0 else float("nan")
        hs = float(g["hs"].mean())
        print(f"  Q{q}: {mu:+7.2f} bp/day  (t={t:5.2f})  "
              f"avg names {g['n'].mean():.0f}  eff. half-spread {hs:.2f} bp")
        res["quintiles"].append({"q": q, "bp_day": mu, "se_bp": se, "t": t,
                                 "names": float(g["n"].mean()), "hs_bp": hs})

    wide = (by_q.pivot(values="r", index="date", on="q").sort("date"))
    if not {"1", "5"}.issubset(set(wide.columns)):
        return res
    ls_f = wide.select("date", ls=pl.col("5") - pl.col("1")).drop_nulls("ls")
    ls = ls_f["ls"].to_numpy().astype(float)
    if len(ls) < 5:
        return res
    mu, sd = float(ls.mean()), float(ls.std(ddof=1))
    se = sd / np.sqrt(len(ls))

    # Turnover: how much of each traded leg is replaced from one day to the next.
    memb = d.select("date", "ticker", "q").sort("date")
    days = memb["date"].unique().sort().to_list()
    turns, prev = [], {}
    for day in days:
        gg = memb.filter(pl.col("date") == day)
        cur = {qq: set(gg.filter(pl.col("q") == qq)["ticker"].to_list())
               for qq in (1, 5)}
        if prev:
            t_ = [len(cur[qq] - prev[qq]) / max(len(prev[qq]), 1)
                  for qq in (1, 5) if prev.get(qq)]
            if t_:
                turns.append(float(np.mean(t_)))
        prev = cur
    turnover = float(np.mean(turns)) if turns else float("nan")

    hs_legs = float(by_q.filter(pl.col("q").is_in([1, 5]))["hs"].mean())
    hs_panel = float(d["effsprd_bps"].mean())
    # Cost arithmetic, spelled out because it is the point of the section.
    # `turnover` is the fraction of one leg's names replaced per day. A
    # replacement is two trades -- sell the leaver, buy the entrant -- and
    # each crosses the half-spread, so one leg pays 2 * turnover * half-spread
    # per day. The long-short portfolio has two legs and its return is quoted
    # per unit of each, so the total is twice that again. The half-spread
    # charged is the average of the two legs actually traded.
    cost_bp = 4 * turnover * hs_legs
    net_bp = mu * 1e4 - cost_bp

    res.update({
        "gross_bp_day": mu * 1e4, "se_bp": se * 1e4,
        "t": mu / se if se > 0 else float("nan"),
        "annualised_pct": mu * ANN_DAYS * 100,
        "sharpe": mu / sd * np.sqrt(ANN_DAYS) if sd > 0 else float("nan"),
        "turnover": turnover, "half_spread_legs_bp": hs_legs,
        "half_spread_panel_bp": hs_panel, "cost_bp_day": cost_bp,
        "net_bp_day": net_bp, "n_days": len(ls),
        "ls_dates": [str(x) for x in ls_f["date"].to_list()],
        "ls_path": [float(x) for x in ls],
    })
    print(f"  long-short (Q5 minus Q1): gross {mu*1e4:+7.2f} bp/day "
          f"(t={res['t']:.2f})   turnover {100*turnover:.1f}%/day   "
          f"cost {cost_bp:.2f} bp/day = 4 x {100*turnover:.1f}% x "
          f"{hs_legs:.2f} bp legs' half-spread   net {net_bp:+7.2f} bp/day")
    return res


def main() -> int:
    tee = C.Tee("s6_step1_strategy")
    try:
        print("=== S6: strategy demonstration ===\n")
        raw = S4.load_panel(final_only=True, drop_fine_tick=True)
        df = S5.build_frame(raw, winsorize=False)
        # Same-day contamination dummies: the signal is a day-t cross-section.
        dig_now, df = S5.opening_digit_dummies(df, lagged=False)
        sig = residual_signal(df, dig_now)
        if sig.is_empty():
            print("no signal could be formed")
            return 1
        print(f"signal formed on {sig.height:,} stock-days, "
              f"{sig['date'].n_unique()} days")

        # The smoothed variant: a five-day rolling mean of the daily signal,
        # re-standardised within the day so the sort is comparable. If the
        # daily churn is measurement noise, this cuts turnover much faster
        # than it cuts whatever signal exists.
        sig = sig.sort(["ticker", "date"]).with_columns(
            signal5_raw=pl.col("signal")
            .rolling_mean(window_size=5, min_samples=3).over("ticker"))
        sig = sig.with_columns(
            signal5=((pl.col("signal5_raw") - pl.col("signal5_raw").mean().over("date"))
                     / pl.col("signal5_raw").std().over("date")))

        sig = add_forward_returns(sig)
        n_before = sig.height
        usable = sig.drop_nulls("fwd").filter(pl.col("fwd").abs() <= MAX_ABS_RET)
        print(f"usable stock-day returns: {usable.height:,} "
              f"(dropped {n_before - usable.height:,} for missing or extreme moves)")

        base = evaluate(sig, "signal", "daily signal")
        smooth = evaluate(sig, "signal5", "5-day smoothed signal")
        if base is None or "gross_bp_day" not in base:
            print("could not evaluate the daily variant")
            return 1

        rows = [[f"Q{q['q']}", f"{q['bp_day']:+.2f}", f"({q['se_bp']:.2f})",
                 f"{q['t']:.2f}", f"{q['names']:.0f}", f"{q['hs_bp']:.2f}"]
                for q in base["quintiles"]]
        rows.append(["Q5 -- Q1 gross, daily", f"{base['gross_bp_day']:+.2f}",
                     f"({base['se_bp']:.2f})", f"{base['t']:.2f}", "--",
                     f"{base['half_spread_legs_bp']:.2f}"])
        rows.append(["\\quad net of costs", f"{base['net_bp_day']:+.2f}",
                     "--", "--", "--", "--"])
        if smooth and "gross_bp_day" in smooth:
            rows.append(["Q5 -- Q1 gross, 5-day signal",
                         f"{smooth['gross_bp_day']:+.2f}",
                         f"({smooth['se_bp']:.2f})", f"{smooth['t']:.2f}", "--",
                         f"{smooth['half_spread_legs_bp']:.2f}"])
            rows.append(["\\quad net of costs", f"{smooth['net_bp_day']:+.2f}",
                         "--", "--", "--", "--"])

        C.ensure_dir(OUT)
        payload = {
            # Backward-compatible top-level fields describe the daily variant.
            "gross_bp_day": base["gross_bp_day"], "se_bp": base["se_bp"],
            "t": base["t"], "annualised_pct": base["annualised_pct"],
            "sharpe": base["sharpe"], "turnover": base["turnover"],
            "cost_bp_day": base["cost_bp_day"], "net_bp_day": base["net_bp_day"],
            "half_spread_bp": base["half_spread_legs_bp"],
            "half_spread_panel_bp": base["half_spread_panel_bp"],
            "n_days": base["n_days"], "max_abs_ret_filter": MAX_ABS_RET,
            "quintiles": base["quintiles"],
        }
        if smooth and "gross_bp_day" in smooth:
            payload["smoothed"] = {k: smooth[k] for k in
                                   ("gross_bp_day", "se_bp", "t", "turnover",
                                    "cost_bp_day", "net_bp_day",
                                    "half_spread_legs_bp", "n_days")}
        C.atomic_json(os.path.join(OUT, "strategy.json"), payload)

        # Cumulative gross and net paths, dates aligned with the series they plot.
        plt = S4.setup_mpl()
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        import datetime as _dt
        dts = [_dt.date.fromisoformat(s[:10]) for s in base["ls_dates"]]
        ls = np.asarray(base["ls_path"])
        ax.plot(dts, 100 * np.cumsum(ls), color=S4.BLUE, lw=1.2,
                label="daily signal, gross")
        ax.plot(dts, 100 * np.cumsum(ls - base["cost_bp_day"] / 1e4),
                color=S4.ORANGE, lw=1.2, ls="--",
                label="daily signal, net of costs")
        if smooth and "ls_path" in smooth:
            dts2 = [_dt.date.fromisoformat(s[:10]) for s in smooth["ls_dates"]]
            ls2 = np.asarray(smooth["ls_path"])
            ax.plot(dts2, 100 * np.cumsum(ls2 - smooth["cost_bp_day"] / 1e4),
                    color="0.45", lw=1.2, ls=":",
                    label="5-day signal, net of costs")
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(frameon=False, fontsize=9)
        fig.autofmt_xdate()
        fig.savefig(C.write_guard(os.path.join(S4.FIGURES, "f_strategy.pdf")))
        plt.close(fig)
        print("\nwrote f_strategy.pdf")

        S4.latex_table(
            os.path.join(S4.TABLES, "t_strategy.tex"),
            "Next-day returns to a size-neutralised clustering signal",
            "tab:strategy",
            ["Portfolio", "Return (bp/day)", "SE", "$t$", "Names", "Eff. half-spread (bp)"],
            rows,
            notes="The signal is the daily cross-sectional residual of "
                  "$M^{BLarge0}$ on the same-day opening-digit contamination "
                  "controls, relative tick size, log market capitalisation, "
                  "lagged log turnover, lagged volatility, lagged spread and "
                  "the overnight return, standardised within the day; the "
                  "5-day variant sorts on a five-day rolling mean of that "
                  "residual. Portfolios are equally weighted and rebalanced "
                  "daily. Costs charge the average measured effective "
                  "half-spread of the two traded legs on both sides of every "
                  "replacement in both legs, which is four times turnover "
                  "times the legs' half-spread. Stock-days with an absolute "
                  "move above 25\\% are dropped, since closing prices here "
                  "carry no adjustment for splits or dividends -- note the "
                  "March fiscal-year-end ex-dividend dates sit inside the "
                  "sample. This is a demonstration that the pipeline runs end "
                  "to end, not evidence of a profitable strategy: one short "
                  "sample, one market, and no out-of-sample period.")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
