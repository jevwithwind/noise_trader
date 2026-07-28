# Verification + drill answer-key generator (kept OUT of E:\MTEC on purpose).
# 1) Confirms 約定種別 (Execution Type) semantics against prevailing quotes.
# 2) Infers the empirical tick size per stock-day (deci-yen integer arithmetic).
# 3) Computes Ohta (2026) M-measures for the drill stock-days.
import polars as pl
from tse_tick import read_ticks

ROOT = r"G:\NEEDS\個別株式{y}\TICST120"

CASES = [
    ("4666", "20230515", "Park24? pre 2023-06-05 change"),
    ("4666", "20230718", "same stock, post change"),
    ("8604", "20240201", "Nomura, expect 0.1 tick"),
    ("8306", "20240201", "MUFG, expect 0.5 tick"),
    ("7203", "20240401", "Toyota, expect 1 tick (fine table 3000-5000)"),
]

def deci(x):  # yen -> integer deci-yen
    return (x * 10).round(0).cast(pl.Int64)

def analyze(ticker, date, note):
    df = read_ticks(ROOT.format(y=date[:4]), ticker_filter={ticker}, date=date, language="en")
    # book state BEFORE each row's event = previous row's best quotes
    df = df.with_columns(
        prev_ask=pl.col("Sell Quote 1 Best").shift(1),
        prev_bid=pl.col("Buy Quote 1 Best").shift(1),
        prev_ask_flag=pl.col("Sell Quote Flag 1").shift(1),
        prev_bid_flag=pl.col("Buy Quote Flag 1").shift(1),
    )
    trades = df.filter(pl.col("Execution Type").is_not_null())
    print(f"\n=== {ticker} {date}  ({note}) ===")
    print("rows:", df.height, " trade rows:", trades.height)
    print(trades["Execution Type"].value_counts().sort("count", descending=True))
    print(trades["Ayumi Flag"].value_counts().sort("count", descending=True))

    # ---- tick inference from the quote grid (deci-yen)
    prices = pl.concat([
        df.select(p=pl.col("Sell Quote 1 Best")).drop_nulls(),
        df.select(p=pl.col("Buy Quote 1 Best")).drop_nulls(),
        trades.select(p=pl.col("Execution Price")),
    ])
    grid = prices.with_columns(p10=deci(pl.col("p"))).get_column("p10").unique().sort()
    diffs = grid.diff().drop_nulls().filter(grid.diff().drop_nulls() > 0)
    tick10 = int(diffs.min())
    pmin, pmax = float(prices["p"].min()), float(prices["p"].max())
    print(f"price range: {pmin:.1f} - {pmax:.1f} yen   inferred tick: {tick10/10:.1f} yen")

    # ---- Execution Type semantics cross-tab vs previous best quotes
    zaraba = trades.filter(~pl.col("Execution Type").str.contains("Opening"))
    ct = (
        zaraba.with_columns(
            at_prev_ask=(pl.col("Execution Price") == pl.col("prev_ask")),
            at_prev_bid=(pl.col("Execution Price") == pl.col("prev_bid")),
        )
        .group_by("Execution Type")
        .agg(
            n=pl.len(),
            pct_at_prev_ask=(pl.col("at_prev_ask").mean() * 100).round(1),
            pct_at_prev_bid=(pl.col("at_prev_bid").mean() * 100).round(1),
        )
        .sort("n", descending=True)
    )
    print(ct)

    # ---- Ohta digit + M measures (drill definition: zaraba trades only —
    #      no opening/closing itayose, no 'Other' — with a two-sided book
    #      just before the trade; digit via integer deci-yen)
    z = zaraba.filter(
        ~pl.col("Execution Type").is_in(["Opening", "Other"])
        & ~pl.col("Ayumi Flag").str.contains("Closing")
        & (pl.col("prev_ask") > 0)
        & (pl.col("prev_bid") > 0)
    ).with_columns(p10=deci(pl.col("Execution Price")))
    z = z.with_columns(digit=(pl.col("p10") % (10 * tick10)) // tick10)

    dist = (
        z.group_by("digit")
        .agg(vol=pl.col("Volume").sum())
        .with_columns(pct=(pl.col("vol") / pl.col("vol").sum() * 100).round(2))
        .sort("digit")
    )
    print("volume-weighted last-digit distribution (%):")
    print(dist)

    def m0(frame):
        if frame.height == 0:
            return float("nan")
        v = frame.group_by(pl.col("digit") == 0).agg(vol=pl.col("Volume").sum())
        tot = v["vol"].sum()
        z0 = v.filter(pl.col("digit"))["vol"].sum() if v.filter(pl.col("digit")).height else 0
        return round(100 * z0 / tot, 2)

    buy_init = z.filter(pl.col("Execution Type").str.contains("At Sell Quote"))
    sell_init = z.filter(pl.col("Execution Type").str.contains("At Buy Quote"))
    res = {
        "M0_all": m0(z),
        "M_BLarge0": m0(buy_init.filter(pl.col("Volume") > 100)),
        "M_SLarge0": m0(sell_init.filter(pl.col("Volume") > 100)),
        "M_BSmall0": m0(buy_init.filter(pl.col("Volume") == 100)),
        "M_SSmall0": m0(sell_init.filter(pl.col("Volume") == 100)),
        "n_zaraba_trades": z.height,
        "buy_init_share_%": round(100 * buy_init.height / max(z.height, 1), 1),
    }
    print(res)
    return res

for t, d, note in CASES:
    try:
        analyze(t, d, note)
    except Exception as e:
        print(f"\n=== {t} {d} FAILED: {e}")
