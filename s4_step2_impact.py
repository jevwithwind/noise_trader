"""S4 step 2 -- do trades at round prices move the price more?

This is the paper's second headline result and the one that makes clustering
matter for execution: a trade at a round price is executing against an order
whose price has gone stale, so the midquote moves further afterwards. Ohta puts
the gap at roughly one basis point at the one-minute horizon.

Two estimates are reported. The stock-day estimate averages each day's difference
between round-price and other impact, which is what the paper's regressions use
but is noisy, since a day needs enough of both kinds of trade. The pooled
estimate takes the difference across all stock-days at once with stock and day
fixed effects, which is far more efficient and is treated here as primary.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4

OUT = os.path.join(C.RESULTS, "s4_rq1")
HORIZONS = (1, 60, 300)
BENCH = {"b": 1.259, "s": 1.121}   # Ohta's Table 3 Delta-Imp, one-minute horizon


def pooled_diff(df: pl.DataFrame, h: int, tag: str, size: str) -> tuple:
    """Volume-weighted round-price impact minus other impact, with stock and day
    effects removed and two-way clustered inference.

    Working from the per-cell impacts and their trade counts recovers the pooled
    difference without needing to carry every trade through the panel.
    """
    c0, c1 = f"imp{h}_{tag}_{size}0", f"imp{h}_{tag}_{size}1"
    n0, n1 = f"n_imp{h}_{tag}_{size}0", f"n_imp{h}_{tag}_{size}1"
    need = [c0, c1, n0, n1]
    if any(c not in df.columns for c in need):
        return (float("nan"),) * 3 + (0,)
    d = df.select(["ticker", "date"] + need).drop_nulls()
    d = d.filter((pl.col(n0) >= 5) & (pl.col(n1) >= 5))
    # Two-way clustering needs enough clusters on both margins. With only a
    # handful of dates the day-demeaning removes nearly all the variation and the
    # variance collapses to zero, which would print as an infinitely precise
    # estimate rather than as the undefined quantity it is.
    if d.height < 50 or d["date"].n_unique() < 20 or d["ticker"].n_unique() < 20:
        return (float("nan"),) * 3 + (d.height,)
    d = d.with_columns(gap=pl.col(c0) - pl.col(c1),
                       w=pl.min_horizontal(pl.col(n0), pl.col(n1)).cast(pl.Float64))
    # Absorb stock and day effects, then take the weighted mean of what is left.
    d = d.with_columns(
        g=pl.col("gap") - pl.col("gap").mean().over("ticker")
        - pl.col("gap").mean().over("date") + pl.col("gap").mean())
    y = d["g"].to_numpy().astype(float)
    w = d["w"].to_numpy().astype(float)
    raw = d["gap"].to_numpy().astype(float)
    mu = float(np.average(raw, weights=w))
    e = (y - np.average(y, weights=w)) * w / w.mean()

    def meat(keys):
        s = 0.0
        o = np.argsort(keys, kind="stable")
        k, ee = np.asarray(keys)[o], e[o]
        b = np.flatnonzero(k[1:] != k[:-1]) + 1
        for lo, hi in zip(np.r_[0, b], np.r_[b, len(k)]):
            s += ee[lo:hi].sum() ** 2
        return s

    st = d["ticker"].to_numpy().astype(str)
    dt_ = d["date"].cast(pl.Utf8).to_numpy().astype(str)
    both = np.char.add(np.char.add(st, "|"), dt_)
    n = len(y)
    v = (meat(st) + meat(dt_) - meat(both)) / (n * n)
    g = min(len(np.unique(st)), len(np.unique(dt_)))
    if g > 1:
        v *= g / (g - 1)
    se = float(np.sqrt(max(v, 0.0)))
    return mu, se, (mu / se if se else float("nan")), n


def main() -> int:
    tee = C.Tee("s4_step2_impact")
    try:
        print("=== S4 step 2: price impact at round prices ===\n")
        df = S4.load_panel(final_only=True)
        payload = {}

        print("liquidity levels (volume-weighted, basis points):")
        for c, lab in (("effsprd_bps", "effective half-spread"),
                       ("qspread_twa_bps", "quoted spread (time-weighted)"),
                       ("imp1_bps", "impact, 1 second"),
                       ("imp60_bps", "impact, 60 seconds"),
                       ("imp300_bps", "impact, 5 minutes"),
                       ("rs60_bps", "realized spread, 60 seconds")):
            if c not in df.columns:
                continue
            mu, se, n = S4.twoway_cluster_mean(df, c)
            print(f"  {lab:<32} {mu:>8.3f} ({se:.3f})  n={n:,}")
            payload[c] = {"mean": mu, "se": se, "n": n}

        # The decomposition should hold on average as well as per stock-day.
        if all(c in df.columns for c in ("effsprd_bps", "imp60_bps", "rs60_bps")):
            e = payload["effsprd_bps"]["mean"]
            i = payload["imp60_bps"]["mean"]
            r = payload["rs60_bps"]["mean"]
            print(f"\n  decomposition check: impact {i:.3f} + realized {r:.3f} "
                  f"= {i+r:.3f} vs effective spread {e:.3f}")

        print("\nround-price impact premium (Delta-Imp), in basis points:")
        print(f"{'horizon':>8} {'side':>6} {'pooled':>9} {'SE':>7} {'t':>7} "
              f"{'stock-day':>10} {'n':>8}   paper")
        rows = []
        for h in HORIZONS:
            for tag, side in (("b", "buy"), ("s", "sell")):
                mu, se, t, n = pooled_diff(df, h, tag, "large")
                sd_mu, sd_se, sd_n = S4.twoway_cluster_mean(df, f"dimp{h}_{tag}_large")
                b = BENCH[tag] if h == 60 else None
                print(f"{h:>7}s {side:>6} {mu:>9.3f} {se:>7.3f} {t:>7.2f}"
                      f"{S4.stars(t):<3} {sd_mu:>10.3f} {n:>8,}"
                      + (f"   {b:.2f}" if b else ""))
                rows.append([f"{h}s", side, f"{mu:.3f}{S4.stars(t)}", f"({se:.3f})",
                             f"{t:.2f}", f"{sd_mu:.3f}", f"{n:,}",
                             f"{b:.2f}" if b else "--"])
                payload[f"dimp{h}_{tag}"] = {"pooled": mu, "se": se, "t": t, "n": n,
                                             "stockday": sd_mu, "stockday_se": sd_se,
                                             "paper": b}
        S4.latex_table(
            os.path.join(S4.TABLES, "t_dimp.tex"),
            "The round-price price-impact premium in large trades",
            "tab:dimp",
            ["Horizon", "Side", "Pooled", "SE", "$t$", "Stock-day mean",
             "Stock-days", "Ohta"],
            rows,
            notes="$\\Delta\\mathit{Imp}$ is the volume-weighted midquote change "
                  "after trades at a last price digit of zero minus the same "
                  "quantity for other trades, within large trades on each side, in "
                  "basis points and signed so that a positive value means the price "
                  "moved the way the initiator pushed it. The pooled column removes "
                  "stock and day effects and weights each stock-day by the smaller "
                  "of its two cell counts; standard errors are clustered by stock "
                  "and by day. Cells with fewer than five trades are dropped. The "
                  "final column is Ohta's (2026) 2010--2022 mean at the one-minute "
                  "horizon. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.")

        # Small trades: the paper's mechanism does not predict a premium here, and
        # this is the contrast that makes the large-trade result meaningful.
        print("\nsmall trades (the mechanism predicts no premium):")
        for tag, side in (("b", "buy"), ("s", "sell")):
            mu, se, t, n = pooled_diff(df, 60, tag, "small")
            print(f"  {side:>5}, 60s: {mu:>7.3f} ({se:.3f})  t={t:.2f}"
                  f"{S4.stars(t)}  n={n:,}")
            payload[f"dimp60_{tag}_small"] = {"pooled": mu, "se": se, "t": t, "n": n}

        # Depth at the best quote before round-price trades versus other trades.
        if "dep_ask0" in df.columns:
            print("\nlog depth at the best quote immediately before a trade:")
            for a, b, lab in (("dep_ask0", "dep_ask1", "ask, buy-initiated"),
                              ("dep_bid0", "dep_bid1", "bid, sell-initiated")):
                d = df.drop_nulls([a, b]).with_columns(gap=pl.col(a) - pl.col(b))
                mu, se, n = S4.twoway_cluster_mean(d, "gap")
                t = mu / se if se else float("nan")
                print(f"  {lab:<22} round minus other: {mu:+.4f} ({se:.4f}) "
                      f"t={t:.2f}{S4.stars(t)}  n={n:,}")
                payload[f"depgap_{a}"] = {"gap": mu, "se": se, "t": t, "n": n}

        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "impact.json"), payload)
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
