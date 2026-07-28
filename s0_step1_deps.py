"""S0 step 1 -- panel-econometrics backends, verified against a known truth.

Discovering at the regression stage that the estimator is broken (or that its
two-way clustered SEs disagree between libraries) would be very expensive. So we
simulate a panel whose true coefficient we know, fit it with every backend we
plan to use, and require them to agree with each other and with the truth.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

OUT = os.path.join(C.RESULTS, "s0_inst")

N_STOCK, N_DAY, BETA_TRUE = 50, 100, 2.0
SEED = 20260729


def simulate() -> pd.DataFrame:
    """Panel with stock effects, day effects, and clustered errors on both margins."""
    rng = np.random.default_rng(SEED)
    stock = np.repeat(np.arange(N_STOCK), N_DAY)
    day = np.tile(np.arange(N_DAY), N_STOCK)
    a_stock = rng.normal(0, 1.0, N_STOCK)[stock]
    a_day = rng.normal(0, 0.7, N_DAY)[day]
    # x correlates with both effects, so an estimator that fails to absorb them
    # will be visibly biased rather than merely noisy.
    x = 0.5 * a_stock + 0.5 * a_day + rng.normal(0, 1.0, N_STOCK * N_DAY)
    err = (rng.normal(0, 0.5, N_STOCK)[stock]
           + rng.normal(0, 0.5, N_DAY)[day]
           + rng.normal(0, 1.0, N_STOCK * N_DAY))
    y = BETA_TRUE * x + a_stock + a_day + err
    return pd.DataFrame({"stock": stock, "day": day, "x": x, "y": y})


def fit_linearmodels(df: pd.DataFrame) -> tuple[float, float]:
    from linearmodels.panel import PanelOLS

    d = df.copy()
    d["stock_i"] = d["stock"].astype("category")
    d["day_i"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(d["day"], unit="D")
    d = d.set_index(["stock_i", "day_i"])
    res = PanelOLS.from_formula("y ~ 1 + x + EntityEffects + TimeEffects", data=d).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    return float(res.params["x"]), float(res.std_errors["x"])


def fit_pyfixest(df: pd.DataFrame) -> tuple[float, float]:
    import pyfixest as pf

    res = pf.feols("y ~ x | stock + day", data=df, vcov={"CRV1": "stock+day"})
    return float(res.coef()["x"]), float(res.se()["x"])


def fit_manual_cgm(df: pd.DataFrame) -> tuple[float, float]:
    """Within-transform on both margins + Cameron-Gelbach-Miller two-way SEs.

    V = V_stock + V_day - V_(stock,day). Kept as an independent third opinion:
    if the two libraries ever disagree, this arbitrates.
    """
    d = df.copy()
    for col in ("y", "x"):
        d[col + "_t"] = (d[col]
                         - d.groupby("stock")[col].transform("mean")
                         - d.groupby("day")[col].transform("mean")
                         + d[col].mean())
    x = d["x_t"].to_numpy()[:, None]
    y = d["y_t"].to_numpy()
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = float((xtx_inv @ x.T @ y)[0])
    resid = y - x[:, 0] * beta

    def meat(groups: np.ndarray) -> np.ndarray:
        m = np.zeros((1, 1))
        order = np.argsort(groups, kind="stable")
        g_sorted, xs, rs = groups[order], x[order], resid[order]
        bounds = np.flatnonzero(np.diff(g_sorted)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(g_sorted)]):
            s = (xs[lo:hi].T @ rs[lo:hi])[:, None]
            m += s @ s.T
        return m

    both = d["stock"].to_numpy() * (N_DAY + 1) + d["day"].to_numpy()
    v = xtx_inv @ (meat(d["stock"].to_numpy()) + meat(d["day"].to_numpy())
                   - meat(both)) @ xtx_inv
    # Degrees-of-freedom correction for the two absorbed fixed-effect dimensions.
    n, k = len(y), 1 + N_STOCK + N_DAY - 1
    g_min = min(N_STOCK, N_DAY)
    adj = (n - 1) / (n - k) * g_min / (g_min - 1)
    return beta, float(np.sqrt(v[0, 0] * adj))


def main() -> int:
    tee = C.Tee("s0_step1_deps")
    try:
        print("=== S0 step 1: panel backends vs known truth ===\n")
        print(f"simulated panel: {N_STOCK} stocks x {N_DAY} days, true beta = {BETA_TRUE}\n")
        df = simulate()

        results, fails = {}, []
        for name, fn in (("linearmodels", fit_linearmodels),
                         ("pyfixest", fit_pyfixest),
                         ("manual_cgm", fit_manual_cgm)):
            try:
                b, se = fn(df)
                results[name] = {"beta": b, "se": se}
                t_from_truth = abs(b - BETA_TRUE) / se
                ok = t_from_truth < 3.0
                print(f"[{'ok' if ok else 'FAIL'}] {name:13s} beta={b:.6f}  se={se:.6f}"
                      f"  |beta-truth|/se={t_from_truth:.2f}")
                if not ok:
                    fails.append(f"{name} beta {b:.4f} too far from {BETA_TRUE}")
            except Exception as exc:
                fails.append(f"{name} raised: {exc}")
                print(f"[FAIL] {name}: {exc}")

        # Point estimates must be numerically identical -- all three solve the same
        # least-squares problem, so any disagreement is a bug, not an opinion.
        betas = [v["beta"] for v in results.values()]
        if len(betas) > 1:
            spread = max(betas) - min(betas)
            ok = spread < 1e-8
            print(f"\n[{'ok' if ok else 'FAIL'}] point-estimate agreement: spread={spread:.2e}")
            if not ok:
                fails.append(f"backends disagree on beta by {spread:.2e}")

        ses = [v["se"] for v in results.values()]
        if len(ses) > 1:
            rel = (max(ses) - min(ses)) / np.mean(ses)
            ok = rel < 0.10
            print(f"[{'ok' if ok else 'FAIL'}] two-way clustered SE agreement: "
                  f"rel spread={rel:.4f} (tolerance 0.10, small-sample dof "
                  f"conventions differ between libraries)")
            if not ok:
                fails.append(f"clustered SEs disagree by {rel:.1%}")

        import linearmodels, pyfixest
        payload = {
            "beta_true": BETA_TRUE, "results": results, "fails": fails,
            "versions": {"linearmodels": linearmodels.__version__,
                         "pyfixest": pyfixest.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
            "backend_roles": {"daily_panel": "linearmodels",
                              "intraday_highdim_fe": "pyfixest",
                              "arbiter": "manual_cgm"},
        }
        C.ensure_dir(OUT)
        C.atomic_json(os.path.join(OUT, "backends.json"), payload)

        print()
        if fails:
            print(f"GATE FAILED ({len(fails)}):")
            for f in fails:
                print("  -", f)
            return 1
        print("GATE PASSED -- linearmodels (daily), pyfixest (intraday), "
              "manual CGM (arbiter)")
        return 0
    finally:
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
