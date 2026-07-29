"""Ohta (2026) clustering measures, liquidity, price formation, and book measures.

One module, used unchanged by the pilot (S1, reading raw zips) and by the full
panel build (S3, reading the Parquet store). Validating it on the anchor
stock-days is therefore validation of production code, not of a look-alike.

Everything price-related is computed in integer deci-yen. Float `mod` silently
corrupts the 0.1-yen grid, which is exactly where clustering is strongest, so the
conversion happens once in `normalize_day` and never again.

Conventions used throughout
---------------------------
* The tape is one row per event. A row is a *trade* iff `Execution Type` is
  non-null; every other row is a book update. Both kinds carry the full ten-level
  book snapshot *as of after the event*, so the state a trade executed against is
  the **previous row's** snapshot. That is what makes the vendor trade-direction
  field verifiable, and it is the basis of every "pre-trade" quantity here.
* Trade direction comes from `Execution Type`, never from Lee-Ready.
* Buy-side quantities are sign-flipped so a positive price impact always means
  "the price moved the way the initiator pushed it".
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

# --------------------------------------------------------------------- columns
ASK_PX = ["Sell Quote 1 Best"] + [f"Sell Quote {i}" for i in range(2, 11)]
ASK_VOL = [f"Sell Quote Vol {i}" for i in range(1, 11)]
ASK_FLAG = [f"Sell Quote Flag {i}" for i in range(1, 11)]
BID_PX = ["Buy Quote 1 Best"] + [f"Buy Quote {i}" for i in range(2, 11)]
BID_VOL = [f"Buy Quote Vol {i}" for i in range(1, 11)]
BID_FLAG = [f"Buy Quote Flag {i}" for i in range(1, 11)]

#: Columns needed for the clustering, spread, impact, volatility and OFI measures.
THIN_COLS = ["Update Time", "Execution Time", "Execution Price", "Execution Type",
             "Ayumi Flag", "Volume", "Session",
             "Sell Quote 1 Best", "Sell Quote Vol 1", "Sell Quote Flag 1",
             "Buy Quote 1 Best", "Buy Quote Vol 1", "Buy Quote Flag 1"]
#: Additional columns needed for the ten-level book measures and ladder inference.
WIDE_COLS = THIN_COLS + ASK_PX[1:] + ASK_VOL[1:] + BID_PX[1:] + BID_VOL[1:]

US = 1_000_000  # microseconds per second

# Minimum observations before a cell is reported rather than nulled. Stock-day
# cells built from a handful of trades are noise, and a difference of two such
# cells (which is what Delta-Imp is) is worse.
MIN_CELL_TRADES = 5
MIN_BUCKET_TRADES = 10


# ------------------------------------------------------------------- normalize
def _time_us(col: str) -> pl.Expr:
    """`HHMMSSffffff` (or `HHMMSS`) as microseconds from midnight."""
    s = pl.col(col)
    base = (s.str.slice(0, 2).cast(pl.Int64) * 3600
            + s.str.slice(2, 2).cast(pl.Int64) * 60
            + s.str.slice(4, 2).cast(pl.Int64)) * US
    frac = pl.when(s.str.len_chars() >= 12).then(
        s.str.slice(6, 6).cast(pl.Int64)).otherwise(0)
    return base + frac


def normalize_day(df: pl.DataFrame, trade_date: _dt.date, unit: int = 100,
                  wide: bool = False) -> pl.DataFrame:
    """Put one stock-day into canonical form.

    Adds the integer time axis, the trade flag and signed direction, deci-yen
    prices, the previous row's book state, and the session/bucket labels.
    """
    sign_map = C.EXEC_TYPE_MAP

    # The tape arrives in shard order, which is *almost* time order: a handful of
    # stray pre-open rows can appear at the end of the file. A stable sort on the
    # microsecond clock fixes those without disturbing genuine same-timestamp
    # sequences, whose file order is the real sequence.
    df = (df.with_row_index("_file_rn")
            .with_columns(t_us=_time_us("Update Time"))
            .sort("t_us", maintain_order=True)
            .with_row_index("rn"))

    close_us = C.session_close_sec(trade_date) * US
    is_trade = pl.col("Execution Type").is_not_null()

    df = df.with_columns(
        is_trade=is_trade,
        sign=pl.col("Execution Type").replace_strict(
            sign_map, default=None, return_dtype=pl.Int8),
        p10=(pl.col("Execution Price") * 10).round(0).cast(pl.Int64),
        t_sec=(pl.col("t_us") // US).cast(pl.Int32),
        in_session=(
            ((pl.col("t_us") >= C.MORNING_OPEN * US) & (pl.col("t_us") <= C.MORNING_CLOSE * US))
            | ((pl.col("t_us") >= C.AFTERNOON_OPEN * US) & (pl.col("t_us") <= close_us))
        ),
        is_morning=pl.col("t_us") <= C.MORNING_CLOSE * US,
    )

    # Previous row's book state = the state the current event met.
    shift_cols = {
        "prev_ask": pl.col("Sell Quote 1 Best").shift(1),
        "prev_bid": pl.col("Buy Quote 1 Best").shift(1),
        "prev_ask_vol": pl.col("Sell Quote Vol 1").shift(1),
        "prev_bid_vol": pl.col("Buy Quote Vol 1").shift(1),
        "prev_ask_flag": pl.col("Sell Quote Flag 1").shift(1),
        "prev_bid_flag": pl.col("Buy Quote Flag 1").shift(1),
    }
    df = df.with_columns(**shift_cols)

    ordinary = list(C.ORDINARY_QUOTE_FLAGS)
    df = df.with_columns(
        # A two-sided ordinary quote: both sides quoted at a real price, and both
        # flagged as ordinary rather than special / pre-opening / market-order.
        prev_two_sided=(pl.col("prev_ask") > 0) & (pl.col("prev_bid") > 0),
        prev_ordinary=(pl.col("prev_ask") > 0) & (pl.col("prev_bid") > 0)
        & pl.col("prev_ask_flag").is_in(ordinary) & pl.col("prev_bid_flag").is_in(ordinary),
        prev_mid=pl.when((pl.col("prev_ask") > 0) & (pl.col("prev_bid") > 0))
        .then((pl.col("prev_ask") + pl.col("prev_bid")) / 2.0).otherwise(None),
    )

    # The book state carried by this row (used for the sampled book measures).
    if wide:
        df = df.with_columns(
            cur_ordinary=(pl.col("Sell Quote 1 Best") > 0) & (pl.col("Buy Quote 1 Best") > 0)
            & pl.col("Sell Quote Flag 1").is_in(ordinary)
            & pl.col("Buy Quote Flag 1").is_in(ordinary),
            cur_mid=pl.when((pl.col("Sell Quote 1 Best") > 0) & (pl.col("Buy Quote 1 Best") > 0))
            .then((pl.col("Sell Quote 1 Best") + pl.col("Buy Quote 1 Best")) / 2.0)
            .otherwise(None),
        )

    # 30-minute bucket label.
    bucket = pl.lit(None, dtype=pl.Int8)
    for bid, lo, hi in reversed(C.BUCKET_EDGES):
        bucket = pl.when((pl.col("t_us") >= lo * US) & (pl.col("t_us") < hi * US)) \
                   .then(pl.lit(bid, dtype=pl.Int8)).otherwise(bucket)
    df = df.with_columns(bucket=bucket)

    # Zaraba trades: continuous-session, signed, against a two-sided ordinary book.
    df = df.with_columns(
        is_auction=pl.col("Execution Type").is_in(list(C.AUCTION_EXEC_TYPES)),
        ayumi_ok=pl.col("Ayumi Flag").is_in(list(C.ZARABA_AYUMI_FLAGS)),
    )
    df = df.with_columns(
        is_zaraba=(pl.col("is_trade") & pl.col("in_session") & ~pl.col("is_auction")
                   & pl.col("ayumi_ok") & (pl.col("Volume") > 0)),
    )
    df = df.with_columns(
        # Two variants, because they answer different questions. The *ordinary*
        # variant is Ohta's stated filter and is what the panel reports; the
        # *two-sided* variant reproduces the looser specification used in the
        # preparation drill, and exists so the anchor check is like-for-like.
        zaraba_ord=pl.col("is_zaraba") & pl.col("prev_ordinary"),
        zaraba_2s=pl.col("is_zaraba") & pl.col("prev_two_sided"),
        size_class=pl.when(pl.col("Volume") == unit).then(pl.lit("small"))
        .when(pl.col("Volume") > unit).then(pl.lit("large"))
        .otherwise(pl.lit("odd")),
    )
    return df


# ------------------------------------------------------------------ tick size
def infer_tick10(df: pl.DataFrame) -> int | None:
    """Smallest positive gap on the observed price grid, in deci-yen.

    Uses quotes and trades together. For an active stock this recovers the true
    tick exactly; for a quiet one it can overstate it, because the stock simply
    never used adjacent grid points. `resolve_tick10` handles that asymmetry.
    """
    px = pl.concat([
        df.select(p=pl.col("Sell Quote 1 Best")),
        df.select(p=pl.col("Buy Quote 1 Best")),
        df.filter(pl.col("is_trade")).select(p=pl.col("Execution Price")),
    ]).filter(pl.col("p") > 0)
    if px.height == 0:
        return None
    grid = (px.with_columns(p10=(pl.col("p") * 10).round(0).cast(pl.Int64))
              .get_column("p10").unique().sort())
    if grid.len() < 2:
        return None
    d = grid.diff().drop_nulls()
    d = d.filter(d > 0)
    return int(d.min()) if d.len() else None


def resolve_tick10(table_tick10: int | None, emp_tick10: int | None) -> tuple[int | None, str]:
    """Reconcile the coded JPX grid with the grid the tape actually used.

    The asymmetry matters. A tape *finer* than the table means the stock is on a
    grid the table did not predict -- index membership drifted -- and the tape
    wins. A tape *coarser* than the table means nothing: an illiquid stock simply
    skipped grid points, and the table remains correct.
    """
    if table_tick10 is None:
        return (emp_tick10, "inferred") if emp_tick10 else (None, "none")
    if emp_tick10 is None:
        return table_tick10, "table"
    if emp_tick10 < table_tick10:
        return emp_tick10, "inferred"
    return table_tick10, "table"


def digit_expr(p10: pl.Expr, tick10: int) -> pl.Expr:
    """Ohta's last price digit, in exact integer arithmetic."""
    return (p10 % (10 * tick10)) // tick10


# --------------------------------------------------------------- sample filters
def sample_filters(df: pl.DataFrame, tick10: int | None, is_t500: bool) -> dict:
    """Ohta's stock-day admission criteria, plus the inputs they need."""
    trades = df.filter(pl.col("is_trade") & (pl.col("Volume") > 0))
    out: dict = {"n_rows": df.height, "n_trade_rows": int(trades.height)}

    if trades.height == 0:
        out.update(first_trade_sec=None, open_px=None, close_px=None,
                   n_zaraba=0, n_zaraba_2s=0, pass_open910=False, pass_open200=False,
                   pass_n20=False, pass_tick=False, in_sample=False, open_digit=None,
                   pmin=None, pmax=None)
        return out

    first_sec = int(trades["t_sec"].min())
    # The opening price is the auction print when there is one, else the day's
    # first trade -- the same object Ohta's filter (b) and his digit-of-open
    # contamination control refer to.
    auction = trades.filter(pl.col("is_auction"))
    open_px = float(auction["Execution Price"][0]) if auction.height else \
        float(trades["Execution Price"][0])
    close_px = float(trades["Execution Price"][-1])

    n_zar = int(df["zaraba_ord"].sum())
    n_zar_2s = int(df["zaraba_2s"].sum())

    pmin = float(trades["Execution Price"].min())
    pmax = float(trades["Execution Price"].max())

    out.update(
        first_trade_sec=first_sec, open_px=open_px, close_px=close_px,
        n_zaraba=n_zar, n_zaraba_2s=n_zar_2s, pmin=pmin, pmax=pmax,
        pass_open910=first_sec <= 9 * 3600 + 10 * 60,        # filter (a)
        pass_open200=open_px > 200.0,                         # filter (b)
        pass_n20=n_zar > 20,                                  # filter (c)
        pass_tick=tick10 is not None and tick10 in C.POWER_OF_TEN_TICKS10,  # filter (d)
        open_digit=None if tick10 is None else
        int((round(open_px * 10) % (10 * tick10)) // tick10),
        tick10=tick10, topix500=is_t500,
    )
    out["in_sample"] = bool(out["pass_open910"] and out["pass_open200"]
                            and out["pass_n20"] and out["pass_tick"])
    return out


# ------------------------------------------------------------- the M measures
def m_measures(df: pl.DataFrame, tick10: int, gate: str = "zaraba_ord",
               prefix: str = "") -> dict:
    """Volume shares executing at each last price digit, split by side and size.

    `M0` pools everything; the four cells split buyer/seller-initiated by
    large/small. Large means "more than one trading unit" -- Ohta's definition,
    and the reason the measure is a large-trade phenomenon.
    """
    z = df.filter(pl.col(gate)).with_columns(digit=digit_expr(pl.col("p10"), tick10))
    out: dict = {}
    if z.height == 0:
        return {f"{prefix}m0_all": None}

    tot = int(z["Volume"].sum())
    zero = int(z.filter(pl.col("digit") == 0)["Volume"].sum())
    out[f"{prefix}m0_all"] = zero / tot if tot else None
    out[f"{prefix}n_zaraba_used"] = z.height

    # Full digit distribution: the placebo tests in S5 need digits 1-9, and a
    # distribution that fails to sum to one is a loud bug signal.
    dist = (z.group_by("digit").agg(v=pl.col("Volume").sum())
             .with_columns(share=pl.col("v") / tot))
    for r in dist.iter_rows(named=True):
        out[f"{prefix}m{int(r['digit'])}_all"] = r["share"]
    for d in range(10):
        out.setdefault(f"{prefix}m{d}_all", 0.0)

    for side, tag in ((C.SIGN_BUY, "b"), (C.SIGN_SELL, "s")):
        for size in ("large", "small"):
            cell = z.filter((pl.col("sign") == side) & (pl.col("size_class") == size))
            n = cell.height
            v = int(cell["Volume"].sum()) if n else 0
            v0 = int(cell.filter(pl.col("digit") == 0)["Volume"].sum()) if n else 0
            out[f"{prefix}m_{tag}_{size}0"] = (v0 / v) if v else None
            out[f"{prefix}n_{tag}_{size}"] = n
            out[f"{prefix}vol_{tag}_{size}"] = v
            if size == "large" and n:
                # Digits 1-9 within the headline cell, for the placebo panel.
                for d in range(1, 10):
                    vd = int(cell.filter(pl.col("digit") == d)["Volume"].sum())
                    out[f"{prefix}m_{tag}_large{d}"] = vd / v if v else None

    # Pooled size cells, and the small-trade share Ohta reports as `Small`.
    for size in ("large", "small"):
        cell = z.filter(pl.col("size_class") == size)
        v = int(cell["Volume"].sum()) if cell.height else 0
        v0 = int(cell.filter(pl.col("digit") == 0)["Volume"].sum()) if cell.height else 0
        out[f"{prefix}m_{size}0"] = (v0 / v) if v else None
    small_v = int(z.filter(pl.col("size_class") == "small")["Volume"].sum())
    out[f"{prefix}small_share"] = small_v / tot if tot else None
    out[f"{prefix}n_odd_lot"] = int((z["size_class"] == "odd").sum())
    return out


# ----------------------------------------------------- spreads and price impact
def _mid_series(df: pl.DataFrame) -> pl.DataFrame:
    """Time-indexed midquote series from rows showing a two-sided ordinary book."""
    return (df.filter(pl.col("prev_ordinary"))
              .select(q_us=pl.col("t_us"), mid=pl.col("prev_mid"))
              .unique(subset="q_us", keep="last")
              .sort("q_us"))


def spread_impact(df: pl.DataFrame, trade_date: _dt.date, tick10: int,
                  horizons=(1, 60, 300), gate: str = "zaraba_ord") -> dict:
    """Effective spread, price impact at several horizons, realized spread, depth.

    These are Ohta's liquidity measures. Impact is the signed midquote change a
    fixed interval after the trade; the effective spread is the signed distance
    from the pre-trade midquote to the trade price; the realized spread is the
    difference, so `ES = Imp + RS` holds by construction on each horizon's own
    trade set.

    Trades whose horizon would run past the end of *their own* session are
    dropped for that horizon. Carrying the last morning midquote across the lunch
    break as if it were a price one minute later would manufacture zero impacts.
    """
    z = (df.filter(pl.col(gate) & pl.col("prev_mid").is_not_null())
           .with_columns(digit=digit_expr(pl.col("p10"), tick10)))
    out: dict = {}
    if z.height == 0:
        return out

    quotes = _mid_series(df)
    close_us = C.session_close_sec(trade_date) * US
    morning_close_us = C.MORNING_CLOSE * US

    # Signed effective half-spread in basis points.
    z = z.with_columns(
        es_bps=pl.col("sign") * (pl.col("Execution Price") - pl.col("prev_mid"))
        / pl.col("prev_mid") * 1e4)
    w = pl.col("Volume")
    out["effsprd_bps"] = float(
        (z["es_bps"] * z["Volume"]).sum() / z["Volume"].sum())
    out["es_sign_violations"] = int((z["es_bps"] < 0).sum())

    for h in horizons:
        tgt = pl.col("t_us") + h * US
        sess_end = pl.when(pl.col("is_morning")).then(morning_close_us).otherwise(close_us)
        zh = z.with_columns(target_us=tgt, sess_end=sess_end).filter(
            pl.col("target_us") <= pl.col("sess_end"))
        if zh.height == 0:
            continue
        zh = (zh.sort("target_us")
                .join_asof(quotes, left_on="target_us", right_on="q_us", strategy="backward")
                .filter(pl.col("mid").is_not_null()))
        if zh.height == 0:
            continue
        zh = zh.with_columns(
            imp_bps=pl.col("sign") * (pl.col("mid") - pl.col("prev_mid"))
            / pl.col("prev_mid") * 1e4)
        zh = zh.with_columns(rs_bps=pl.col("es_bps") - pl.col("imp_bps"))

        tw = zh["Volume"].sum()
        out[f"imp{h}_bps"] = float((zh["imp_bps"] * zh["Volume"]).sum() / tw)
        out[f"rs{h}_bps"] = float((zh["rs_bps"] * zh["Volume"]).sum() / tw)
        out[f"n_imp{h}"] = zh.height

        # Per-cell impact and the digit-0 minus non-digit-0 difference.
        for side, tag in ((C.SIGN_BUY, "b"), (C.SIGN_SELL, "s")):
            for size in ("large", "small"):
                cell = zh.filter((pl.col("sign") == side) & (pl.col("size_class") == size))
                c0 = cell.filter(pl.col("digit") == 0)
                c1 = cell.filter(pl.col("digit") != 0)
                v0, v1 = c0["Volume"].sum(), c1["Volume"].sum()
                i0 = float((c0["imp_bps"] * c0["Volume"]).sum() / v0) \
                    if c0.height >= MIN_CELL_TRADES and v0 else None
                i1 = float((c1["imp_bps"] * c1["Volume"]).sum() / v1) \
                    if c1.height >= MIN_CELL_TRADES and v1 else None
                out[f"imp{h}_{tag}_{size}0"] = i0
                out[f"imp{h}_{tag}_{size}1"] = i1
                out[f"dimp{h}_{tag}_{size}"] = (i0 - i1) if (i0 is not None and i1 is not None) else None
                out[f"n_imp{h}_{tag}_{size}0"] = c0.height
                out[f"n_imp{h}_{tag}_{size}1"] = c1.height

    # Depth at the best quote immediately before a trade -- Ohta's Dep, split by
    # whether the trade printed at a round price.
    for side, tag, volcol in ((C.SIGN_BUY, "ask", "prev_ask_vol"),
                              (C.SIGN_SELL, "bid", "prev_bid_vol")):
        cell = z.filter(pl.col("sign") == side)
        for dtag, sub in (("0", cell.filter(pl.col("digit") == 0)),
                          ("1", cell.filter(pl.col("digit") != 0))):
            v = sub[volcol].mean() if sub.height >= MIN_CELL_TRADES else None
            out[f"dep_{tag}{dtag}"] = float(math.log(v)) if v and v > 0 else None
    return out


# ------------------------------------------------------ volatility and efficiency
def _grid_mids(quotes: pl.DataFrame, trade_date: _dt.date, step_sec: int) -> list[pl.Series]:
    """Midquotes sampled on a fixed grid, one series per session.

    Returned per session so that no return ever spans the lunch break or the
    overnight gap.
    """
    out = []
    for lo, hi in C.session_ranges(trade_date):
        pts = list(range(lo * US, hi * US + 1, step_sec * US))
        if len(pts) < 3:
            continue
        grid = pl.DataFrame({"q_us": pts}, schema={"q_us": pl.Int64}).sort("q_us")
        j = grid.join_asof(quotes, on="q_us", strategy="backward")
        s = j["mid"].drop_nulls()
        if s.len() >= 3:
            out.append(s)
    return out


def rv_vr(df: pl.DataFrame, trade_date: _dt.date) -> dict:
    """Realized variance from 5-minute midquotes, and a variance ratio.

    The variance ratio |VR - 1| is a price-formation measure rather than a
    liquidity one: it says how far the midquote departs from a random walk at the
    five-minute horizon.
    """
    quotes = _mid_series(df)
    out: dict = {"rv5": None, "vr5": None, "vr_absdev": None, "n_rv_obs": 0}
    if quotes.height < 3:
        return out

    m5 = _grid_mids(quotes, trade_date, 300)
    r5 = []
    for s in m5:
        lr = (s.log().diff().drop_nulls())
        r5.append(lr)
    n5 = sum(x.len() for x in r5)
    if n5 >= 10:
        out["rv5"] = float(sum(float((x ** 2).sum()) for x in r5))
        out["n_rv_obs"] = n5

    m1 = _grid_mids(quotes, trade_date, 60)
    r1 = [s.log().diff().drop_nulls() for s in m1]
    n1 = sum(x.len() for x in r1)
    if n1 >= 100 and n5 >= 10:
        import numpy as np
        v1 = float(np.var(np.concatenate([x.to_numpy() for x in r1]), ddof=1))
        v5 = float(np.var(np.concatenate([x.to_numpy() for x in r5]), ddof=1))
        if v1 > 0:
            out["vr5"] = v5 / (5.0 * v1)
            out["vr_absdev"] = abs(out["vr5"] - 1.0)
    return out


def ofi_measures(df: pl.DataFrame) -> dict:
    """Order-flow imbalance at the best quote (Cont, Kukanov and Stoikov, 2014).

    Reimplemented rather than taken from `tse_tick.features`, whose version
    assigns the execution *price* to its volume variable.
    """
    q = df.filter(pl.col("in_session") & (pl.col("Sell Quote 1 Best") > 0)
                  & (pl.col("Buy Quote 1 Best") > 0)).select(
        "bucket", ap=pl.col("Sell Quote 1 Best"), av=pl.col("Sell Quote Vol 1"),
        bp=pl.col("Buy Quote 1 Best"), bv=pl.col("Buy Quote Vol 1"))
    if q.height < 2:
        return {"ofi_sum": None, "ofi_abs": None}
    q = q.with_columns(ap_p=pl.col("ap").shift(1), av_p=pl.col("av").shift(1),
                       bp_p=pl.col("bp").shift(1), bv_p=pl.col("bv").shift(1)).drop_nulls()
    if q.height == 0:
        return {"ofi_sum": None, "ofi_abs": None}
    e_bid = (pl.when(pl.col("bp") > pl.col("bp_p")).then(pl.col("bv"))
               .when(pl.col("bp") < pl.col("bp_p")).then(-pl.col("bv_p"))
               .otherwise(pl.col("bv") - pl.col("bv_p")))
    e_ask = (pl.when(pl.col("ap") < pl.col("ap_p")).then(pl.col("av"))
               .when(pl.col("ap") > pl.col("ap_p")).then(-pl.col("av_p"))
               .otherwise(pl.col("av") - pl.col("av_p")))
    q = q.with_columns(ofi=(e_bid - e_ask).cast(pl.Float64))
    return {"ofi_sum": float(q["ofi"].sum()), "ofi_abs": float(q["ofi"].abs().sum())}


# ------------------------------------------------------------- ten-level book
def book_grid(df: pl.DataFrame, trade_date: _dt.date, tick10: int,
              step_sec: int = 60) -> dict:
    """Time-averaged shape of the visible book, sampled on a fixed grid.

    `RDepth` is the measure Ohta's data could not support and this project adds:
    the share of *visible resting depth* sitting at round prices. Where the M
    measures see round-price orders only once they have been executed against,
    RDepth sees the stale round-price inventory while it is still standing.
    """
    snap = df.filter(pl.col("in_session") & pl.col("cur_ordinary"))
    out: dict = {"rdepth_ask0": None, "rdepth_bid0": None, "qspread_twa_bps": None,
                 "depth_best_ln": None, "depth10_ln": None, "dslope": None,
                 "n_book_snaps": 0}
    if snap.height == 0:
        return out

    keep = ["t_us", "Sell Quote 1 Best", "Buy Quote 1 Best", "cur_mid",
            "Sell Quote Vol 1", "Buy Quote Vol 1"] + ASK_PX[1:] + ASK_VOL[1:] \
        + BID_PX[1:] + BID_VOL[1:]
    snap = snap.select([c for c in keep if c in snap.columns]).sort("t_us")

    # Sample on the grid so that a stock quoting ten thousand times a second does
    # not dominate its own daily average.
    pts: list[int] = []
    for lo, hi in C.session_ranges(trade_date):
        pts.extend(range(lo * US, hi * US + 1, step_sec * US))
    grid = pl.DataFrame({"t_us": pts}, schema={"t_us": pl.Int64}).sort("t_us")
    s = grid.join_asof(snap, on="t_us", strategy="backward").drop_nulls(subset="cur_mid")
    if s.height == 0:
        return out

    ask_v = [pl.col(c).fill_null(0).cast(pl.Int64) for c in ASK_VOL]
    bid_v = [pl.col(c).fill_null(0).cast(pl.Int64) for c in BID_VOL]
    ask_round, bid_round = [], []
    for px, vol in zip(ASK_PX, ASK_VOL):
        p10 = (pl.col(px) * 10).round(0).cast(pl.Int64)
        ask_round.append(pl.when((pl.col(px) > 0) & (digit_expr(p10, tick10) == 0))
                         .then(pl.col(vol).fill_null(0)).otherwise(0).cast(pl.Int64))
    for px, vol in zip(BID_PX, BID_VOL):
        p10 = (pl.col(px) * 10).round(0).cast(pl.Int64)
        bid_round.append(pl.when((pl.col(px) > 0) & (digit_expr(p10, tick10) == 0))
                         .then(pl.col(vol).fill_null(0)).otherwise(0).cast(pl.Int64))

    s = s.with_columns(
        ask_tot=sum(ask_v[1:], ask_v[0]), bid_tot=sum(bid_v[1:], bid_v[0]),
        ask_r=sum(ask_round[1:], ask_round[0]), bid_r=sum(bid_round[1:], bid_round[0]),
        qspread_bps=(pl.col("Sell Quote 1 Best") - pl.col("Buy Quote 1 Best"))
        / pl.col("cur_mid") * 1e4,
        best_depth=(pl.col("Sell Quote Vol 1").fill_null(0)
                    + pl.col("Buy Quote Vol 1").fill_null(0)).cast(pl.Float64),
    )
    at, bt = float(s["ask_tot"].sum()), float(s["bid_tot"].sum())
    out["rdepth_ask0"] = float(s["ask_r"].sum()) / at if at > 0 else None
    out["rdepth_bid0"] = float(s["bid_r"].sum()) / bt if bt > 0 else None
    out["qspread_twa_bps"] = float(s["qspread_bps"].mean())
    bd = float(s["best_depth"].mean())
    out["depth_best_ln"] = math.log(bd) if bd > 0 else None
    d10 = float((s["ask_tot"] + s["bid_tot"]).mean())
    out["depth10_ln"] = math.log(d10) if d10 > 0 else None
    out["n_book_snaps"] = s.height

    # Depth slope: how fast cumulative size grows as you walk away from the best
    # quote, averaged over the two sides (Naes and Skjeltorp, 2006).
    import numpy as np
    lv = np.arange(1, 11, dtype=float)
    x = lv - lv.mean()
    denom = float((x * x).sum())
    a = np.nan_to_num(s.select(ASK_VOL).to_numpy().astype(float))
    b = np.nan_to_num(s.select(BID_VOL).to_numpy().astype(float))
    slope = ((a * x).sum(axis=1) / denom + (b * x).sum(axis=1) / denom) / 2.0
    out["dslope"] = float(np.mean(slope))
    return out


# ------------------------------------------------- limit-order flow from the ladder
def ladder_lc(df: pl.DataFrame, trade_date: _dt.date, tick10: int) -> dict:
    """Infer limit-order submission and cancellation volume from book deltas.

    Between two consecutive snapshots, resting volume at a price can only change
    by execution, cancellation or submission:

        V_n(p) = V_{n-1}(p) - executed(p) - cancelled(p) + submitted(p)

    so with `d = V_n - V_{n-1}` and `X` the volume executed at `p` in the
    interval, `submitted = max(d + X, 0)` and `cancelled = max(-(d + X), 0)`.

    Two things make or break this. First, prices are matched **by price, never by
    level index** -- when the book shifts a level, index matching invents a
    submission and a cancellation that never happened. Second, a price is only
    counted when it is observable in *both* snapshots: on the ask side that means
    at or below the deepest visible ask in each, since anything inside that window
    either shows its volume or is genuinely empty, while anything beyond it may
    carry depth we cannot see.
    """
    out: dict = {"n_ladder_intervals": 0}
    cols = ["rn", "t_us", "is_trade", "sign", "Execution Price", "Volume",
            "is_morning", "cur_ordinary"] + ASK_PX + ASK_VOL + BID_PX + BID_VOL
    d = (df.filter(pl.col("in_session"))
           .select([c for c in cols if c in df.columns])
           .sort("rn").with_row_index("seq"))
    if d.height < 2:
        return out

    # An interval is usable only when both of its endpoints show an ordinary
    # two-sided book, the two rows are genuinely adjacent on the tape, and they
    # sit in the same session. Comparing across a dropped stretch would silently
    # attribute a whole halt's worth of book movement to one interval.
    valid = d.select(
        "seq",
        ok=(pl.col("cur_ordinary") & pl.col("cur_ordinary").shift(1)
            & (pl.col("rn") - pl.col("rn").shift(1) == 1)
            & (pl.col("is_morning") == pl.col("is_morning").shift(1))).fill_null(False),
    ).filter(pl.col("ok")).select("seq")
    if valid.height == 0:
        return out

    # Long form: one row per (snapshot, side, price). Unpopulated levels carry a
    # zero price and are dropped.
    parts = []
    for side, pxs, vols in ((1, ASK_PX, ASK_VOL), (-1, BID_PX, BID_VOL)):
        for px, vol in zip(pxs, vols):
            parts.append(d.select(
                seq=pl.col("seq"),
                side=pl.lit(side, dtype=pl.Int8),
                p10=(pl.col(px) * 10).round(0).cast(pl.Int64),
                vol=pl.col(vol).fill_null(0).cast(pl.Int64),
            ).filter(pl.col("p10") > 0))
    long = pl.concat(parts)

    # Observability frontier per snapshot and side: the deepest visible price.
    # Anything at or inside it is observable -- listed levels show their volume,
    # and unlisted prices inside the window are genuinely empty. Anything beyond
    # it may carry depth we cannot see.
    bounds = long.group_by(["seq", "side"]).agg(
        f_hi=pl.col("p10").max(), f_lo=pl.col("p10").min())
    bounds = bounds.with_columns(
        frontier=pl.when(pl.col("side") == 1).then(pl.col("f_hi")).otherwise(pl.col("f_lo"))
    ).select("seq", "side", "frontier")

    cur = long.rename({"vol": "v_cur"})
    prv = long.with_columns(seq=pl.col("seq") + 1).rename({"vol": "v_prev"})
    j = cur.join(prv, on=["seq", "side", "p10"], how="full", coalesce=True)
    j = j.with_columns(v_cur=pl.col("v_cur").fill_null(0),
                       v_prev=pl.col("v_prev").fill_null(0))
    j = j.join(valid, on="seq", how="semi")

    fr_cur = bounds.select("seq", "side", f_cur="frontier")
    fr_prv = bounds.select(seq=pl.col("seq") + 1, side="side", f_prev="frontier")
    j = j.join(fr_cur, on=["seq", "side"]).join(fr_prv, on=["seq", "side"])
    j = j.filter(
        ((pl.col("side") == 1) & (pl.col("p10") <= pl.min_horizontal("f_cur", "f_prev")))
        | ((pl.col("side") == -1) & (pl.col("p10") >= pl.max_horizontal("f_cur", "f_prev")))
    )
    if j.height == 0:
        return out

    # Executions in this interval: the trade printed on the current row, charged
    # to the side of the book it consumed.
    ex = (d.filter(pl.col("is_trade") & (pl.col("sign") != 0)
                   & pl.col("sign").is_not_null())
            .select(seq="seq",
                    side=pl.when(pl.col("sign") == C.SIGN_BUY)
                    .then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(-1, dtype=pl.Int8)),
                    p10=(pl.col("Execution Price") * 10).round(0).cast(pl.Int64),
                    x=pl.col("Volume").cast(pl.Int64))
            .group_by(["seq", "side", "p10"]).agg(x=pl.col("x").sum()))
    j = j.join(ex, on=["seq", "side", "p10"], how="left").with_columns(
        x=pl.col("x").fill_null(0))

    j = j.with_columns(net=pl.col("v_cur") - pl.col("v_prev") + pl.col("x"))
    j = j.with_columns(
        sub=pl.when(pl.col("net") > 0).then(pl.col("net")).otherwise(0),
        can=pl.when(pl.col("net") < 0).then(-pl.col("net")).otherwise(0),
        digit=digit_expr(pl.col("p10"), tick10),
    )

    # Distance class relative to the previous snapshot's best quote on that side.
    best = d.select(
        seq=pl.col("seq") + 1,
        best_ask=(pl.col(ASK_PX[0]) * 10).round(0).cast(pl.Int64),
        best_bid=(pl.col(BID_PX[0]) * 10).round(0).cast(pl.Int64))
    j = j.join(best, on="seq", how="left")
    ref = pl.when(pl.col("side") == 1).then(pl.col("best_ask")).otherwise(pl.col("best_bid"))
    away = pl.when(pl.col("side") == 1).then(pl.col("p10") - ref).otherwise(ref - pl.col("p10"))
    j = j.with_columns(dist=pl.when(away < 0).then(pl.lit("inside"))
                       .when(away == 0).then(pl.lit("atbest"))
                       .when(away.cast(pl.Float64) / ref.cast(pl.Float64) < 0.005)
                       .then(pl.lit("near")).otherwise(pl.lit("far")))

    out["n_ladder_intervals"] = int(valid.height)
    for side, tag in ((1, "s"), (-1, "b")):
        g = j.filter(pl.col("side") == side)
        sub_t, can_t = int(g["sub"].sum()), int(g["can"].sum())
        g0 = g.filter(pl.col("digit") == 0)
        g1 = g.filter(pl.col("digit") != 0)
        sub0, can0, x0 = int(g0["sub"].sum()), int(g0["can"].sum()), int(g0["x"].sum())
        sub1, can1, x1 = int(g1["sub"].sum()), int(g1["can"].sum()), int(g1["x"].sum())
        # L: round-price share of submitted limit volume.
        out[f"l_{tag}0"] = sub0 / sub_t if sub_t else None
        # C: round-price share of cancelled limit volume.
        out[f"c_{tag}0"] = can0 / can_t if can_t else None
        # Cancellation and execution ratios, at round prices and elsewhere.
        out[f"l_{tag}0c"] = can0 / sub0 if sub0 else None
        out[f"l_{tag}1c"] = can1 / sub1 if sub1 else None
        out[f"l_{tag}0e"] = x0 / sub0 if sub0 else None
        out[f"l_{tag}1e"] = x1 / sub1 if sub1 else None
        out[f"sub_vol_{tag}"] = sub_t
        out[f"canc_vol_{tag}"] = can_t
        for dist in ("inside", "atbest", "near", "far"):
            gd = g.filter(pl.col("dist") == dist)
            s_t = int(gd["sub"].sum())
            s_0 = int(gd.filter(pl.col("digit") == 0)["sub"].sum())
            out[f"l_{tag}0_{dist}"] = s_0 / s_t if s_t else None

    # Diagnostics that tell the reader how much of the flow we could actually see.
    over = df.filter(pl.col("in_session")).select(
        o=pl.col("Sell Quote Vol OVER").fill_null(0).cast(pl.Float64),
        u=pl.col("Buy Quote Vol UNDER").fill_null(0).cast(pl.Float64),
        a=pl.sum_horizontal([pl.col(c).fill_null(0) for c in ASK_VOL]).cast(pl.Float64),
        b=pl.sum_horizontal([pl.col(c).fill_null(0) for c in BID_VOL]).cast(pl.Float64),
    ) if "Sell Quote Vol OVER" in df.columns else None
    if over is not None and over.height:
        tot = float((over["o"] + over["u"] + over["a"] + over["b"]).sum())
        out["over_vol_share"] = float((over["o"] + over["u"]).sum()) / tot if tot else None

    # How often the observability frontier moved between snapshots: the exposure
    # of the far-distance class to level-10 censoring.
    fr = bounds.sort(["side", "seq"]).with_columns(
        moved=pl.col("frontier") != pl.col("frontier").shift(1).over("side"))
    out["frontier_move_share"] = float(fr["moved"].fill_null(False).mean())
    return out


# ------------------------------------------------------------------ intraday
def bucket_rows(df: pl.DataFrame, trade_date: _dt.date, tick10: int,
                ticker: str, gate: str = "zaraba_ord") -> list[dict]:
    """One row per 30-minute bucket, for the intraday panel."""
    z = df.filter(pl.col(gate)).with_columns(digit=digit_expr(pl.col("p10"), tick10))
    if z.height == 0:
        return []
    quotes = _mid_series(df)
    rows = []
    for bid, lo, hi in C.BUCKET_EDGES:
        if hi * US > (C.session_close_sec(trade_date) + 1) * US:
            continue
        b = z.filter(pl.col("bucket") == bid)
        if b.height < MIN_BUCKET_TRADES:
            continue
        tot = int(b["Volume"].sum())
        v0 = int(b.filter(pl.col("digit") == 0)["Volume"].sum())
        row = {"date": trade_date, "ticker": ticker, "bucket": bid,
               "n_trades": b.height, "yenvol": float((b["Execution Price"] * b["Volume"]).sum()),
               "m0": v0 / tot if tot else None}
        for side, tag in ((C.SIGN_BUY, "b"), (C.SIGN_SELL, "s")):
            cell = b.filter((pl.col("sign") == side) & (pl.col("size_class") == "large"))
            v = int(cell["Volume"].sum()) if cell.height else 0
            vz = int(cell.filter(pl.col("digit") == 0)["Volume"].sum()) if cell.height else 0
            row[f"m_{tag}_large0"] = (vz / v) if (v and cell.height >= MIN_BUCKET_TRADES) else None
        bb = b.with_columns(
            es_bps=pl.col("sign") * (pl.col("Execution Price") - pl.col("prev_mid"))
            / pl.col("prev_mid") * 1e4).drop_nulls("es_bps")
        if bb.height:
            row["effsprd_bps"] = float((bb["es_bps"] * bb["Volume"]).sum() / bb["Volume"].sum())
        # 60-second impact within the bucket.
        close_us = C.session_close_sec(trade_date) * US
        sess_end = pl.when(pl.col("is_morning")).then(C.MORNING_CLOSE * US).otherwise(close_us)
        bh = (b.with_columns(target_us=pl.col("t_us") + 60 * US, sess_end=sess_end)
                .filter(pl.col("target_us") <= pl.col("sess_end"))
                .drop_nulls("prev_mid").sort("target_us"))
        if bh.height:
            bh = bh.join_asof(quotes, left_on="target_us", right_on="q_us",
                              strategy="backward").drop_nulls("mid")
            if bh.height:
                bh = bh.with_columns(
                    imp=pl.col("sign") * (pl.col("mid") - pl.col("prev_mid"))
                    / pl.col("prev_mid") * 1e4)
                row["imp60_bps"] = float((bh["imp"] * bh["Volume"]).sum() / bh["Volume"].sum())
        # Midquote move across the bucket, and OFI inside it.
        qb = df.filter((pl.col("bucket") == bid) & pl.col("prev_ordinary"))
        if qb.height >= 2:
            m0_, m1_ = float(qb["prev_mid"][0]), float(qb["prev_mid"][-1])
            row["ret_mid_bps"] = (m1_ / m0_ - 1.0) * 1e4 if m0_ > 0 else None
            row["ofi"] = ofi_measures(df.filter(pl.col("bucket") == bid)).get("ofi_sum")
        rows.append(row)
    return rows


# ------------------------------------------------------------------ orchestrator
def stock_day(df: pl.DataFrame, ticker: str, trade_date: _dt.date, *,
              is_t500: bool, unit: int = 100, wide: bool = False,
              do_ladder: bool = False, gate: str = "zaraba_ord",
              horizons=(1, 60, 300)) -> tuple[dict, list[dict]]:
    """Compute every measure for one stock-day.

    Returns the daily row and the list of 30-minute bucket rows. A stock-day that
    fails Ohta's admission filters still returns a row -- with the filter flags
    set and the measures null -- so the sample-construction waterfall can be
    reported honestly rather than inferred from what is missing.
    """
    n = normalize_day(df, trade_date, unit=unit, wide=wide)
    emp = infer_tick10(n)

    px = n.filter(pl.col("is_trade") & (pl.col("Volume") > 0))
    if px.height:
        pmin10 = int(round(float(px["Execution Price"].min()) * 10))
        pmax10 = int(round(float(px["Execution Price"].max()) * 10))
        table_tick = C.day_tick_constant10(pmin10, pmax10, is_t500)
    else:
        table_tick = None
    tick10, source = resolve_tick10(table_tick, emp)

    row: dict = {"date": trade_date, "ticker": ticker}
    row.update(sample_filters(n, tick10, is_t500))
    row.update(tick_source=source, tick_table10=table_tick, tick_emp10=emp,
               tick_mismatch=bool(table_tick is not None and emp is not None
                                  and table_tick != emp),
               fine_tick_day=bool(tick10 == 1), unit=unit)

    unknown = n.filter(pl.col("is_trade") & pl.col("sign").is_null())
    row["n_unmapped_exec_type"] = unknown.height
    row["unmapped_exec_types"] = ";".join(
        sorted(set(unknown["Execution Type"].to_list()))) if unknown.height else ""

    if not row["in_sample"] or tick10 is None:
        return row, []

    row.update(m_measures(n, tick10, gate=gate))
    row.update(spread_impact(n, trade_date, tick10, horizons=horizons, gate=gate))
    row.update(rv_vr(n, trade_date))
    row.update(ofi_measures(n))

    z = n.filter(pl.col(gate))
    yenvol = float((z["Execution Price"] * z["Volume"]).sum()) if z.height else 0.0
    row["yenvol"] = yenvol
    row["sh_vol"] = int(z["Volume"].sum()) if z.height else 0
    if row.get("open_px") and row.get("close_px"):
        row["ret_oc"] = row["close_px"] / row["open_px"] - 1.0
    # Amihud illiquidity, kept as an auxiliary yardstick against the paper's own
    # liquidity measures rather than as a substitute for them.
    if yenvol > 0 and row.get("ret_oc") is not None:
        row["amihud"] = abs(row["ret_oc"]) / (yenvol / 1e6)

    if wide:
        row.update(book_grid(n, trade_date, tick10))
        if do_ladder:
            row.update(ladder_lc(n, trade_date, tick10))

    return row, bucket_rows(n, trade_date, tick10, ticker, gate=gate)
