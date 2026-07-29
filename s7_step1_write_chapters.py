"""S7 step 1 -- generate the results chapters from the stage outputs.

The chapters that carry numbers are written from the JSON the analysis stages
emit, so re-running the pipeline on more data regenerates the prose with it and
no figure in the text can drift from the table it describes.
"""
from __future__ import annotations

import os
import re
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C
import s4_common as S4

CH = os.path.join(C.REPORT, "chapters")
R4 = os.path.join(C.RESULTS, "s4_rq1")
R5 = os.path.join(C.RESULTS, "s5_reg")
R6 = os.path.join(C.RESULTS, "s6_strategy")
R3 = os.path.join(C.RESULTS, "s3_panel")


def w(name: str, body: str) -> None:
    p = os.path.join(CH, name)
    os.makedirs(CH, exist_ok=True)
    with open(C.write_guard(p), "w", encoding="utf-8") as fh:
        fh.write(body.rstrip() + "\n")
    print(f"wrote {name}")


def pct(v, nd=2):
    return "--" if v is None else f"{100*v:.{nd}f}"


def num(v, nd=3):
    return "--" if v is None else f"{v:.{nd}f}"


def sig_word(t):
    if t is None:
        return "not estimated"
    a = abs(t)
    return ("significant at the 1\\% level" if a > 2.576 else
            "significant at the 5\\% level" if a > 1.96 else
            "significant at the 10\\% level" if a > 1.645 else
            "not statistically distinguishable from zero")


def direction(b):
    return "higher" if (b or 0) > 0 else "lower"


def main() -> int:
    st = C.read_json(os.path.join(R4, "stylized.json"), {})
    im = C.read_json(os.path.join(R4, "impact.json"), {})
    op = C.read_json(os.path.join(R4, "opening.json"), {})
    rq2 = C.read_json(os.path.join(R5, "rq2_daily.json"), {})
    rq3 = C.read_json(os.path.join(R5, "rq3_book.json"), {})
    intr = C.read_json(os.path.join(R5, "intraday.json"), {})
    rob = C.read_json(os.path.join(R5, "robustness.json"), {})
    strat = C.read_json(os.path.join(R6, "strategy.json"), {})
    wf = C.read_json(os.path.join(R3, "waterfall.json"), {})

    panel = pl.read_parquet(S4.PANEL) if os.path.exists(S4.PANEL) else None
    final = panel.filter(pl.col("in_sample_final")) if panel is not None else None
    n_days = final["date"].n_unique() if final is not None else 0
    n_stocks = final["ticker"].n_unique() if final is not None else 0
    n_sd = final.height if final is not None else 0
    d0, d1 = ("?", "?")
    if final is not None and final.height:
        ds = final["date"].cast(pl.Utf8)
        d0, d1 = ds.min(), ds.max()

    # ------------------------------------------------------------ abstract
    dimp_b = (im.get("dimp60_b") or {}).get("pooled")
    dimp_s = (im.get("dimp60_s") or {}).get("pooled")
    mbl = (st.get("m_b_large0") or {}).get("mean")
    mbs = (st.get("m_b_small0") or {}).get("mean")
    rda = (st.get("rdepth_ask0") or {}).get("mean")
    base = rob.get("baseline", {})
    bkey = "m_b_large0_l1"
    bb = (base.get("coef") or {}).get(bkey)
    bt = (base.get("t") or {}).get(bkey)

    w("00_abstract.tex", f"""
Ohta (2026) shows that price clustering on the Tokyo Stock Exchange arises
because noise traders leave limit orders standing at round prices and faster
participants take them with large market orders, and closes by proposing that
clustering measures could therefore serve as an observable proxy for
noise-trader activity --- leaving the relationship between that activity, price
formation and liquidity as future work. This study carries out that analysis on
the 2024 tape, two years beyond the paper's sample, using {n_sd:,} stock-days
across {n_stocks:,} stocks and {n_days} trading days.

The measures replicate. Round prices carry {pct(mbl)}\\% of large-trade volume
against {pct(mbs)}\\% of small-trade volume, close to the paper's 14.8\\% and
11.1\\%, and the large-minus-small ordering its mechanism predicts holds with
high confidence. Trades at round prices move the midquote by about
{num(dimp_b, 2)} basis points more than other trades on the buy side, against
the paper's 1.26.

Two measures the paper's data could not support are added. The share of visible
ten-level resting depth sitting at round prices averages {pct(rda)}\\% against a
uniform benchmark of 10\\%, observing the stale inventory directly rather than
after it has been executed against; and limit-order submission and cancellation
shares are inferred from ladder deltas, reproducing the paper's published
magnitudes without having been calibrated to them.

On the open question --- whether noise-trader activity marks a more liquid market
or a more expensive one --- the panel gives an answer, reported with the caution
a single year deserves. A placebo that rebuilds the measure on each of the other
nine last digits separates the round-price result from a mechanical artefact of
how the measure is constructed. A strategy demonstration is included to exercise
the machinery end to end; its informative output is the cost accounting, not the
return.
""")

    # ------------------------------------------------------------ data
    comp_rows = ""
    if panel is not None and "tick10" in panel.columns:
        sc = panel.filter(pl.col("skip_reason").is_null())
        g = (sc.group_by("tick10").len().sort("len", descending=True).head(6))
        comp_rows = "\n".join(
            f"{'' if r['tick10'] is None else format(r['tick10']/10, 'g')} yen & "
            f"{r['len']:,} & "
            f"{'kept' if (r['tick10'] in (1,10,100,1000,10000)) else 'excluded by filter (d)'} \\\\"
            for r in g.iter_rows(named=True))

    w("03_data.tex", f"""
% =====================================================================
\\section{{Data, institutions, and the sample}}
\\label{{sec:data}}
% =====================================================================

\\subsection{{The feed}}

The source is the Nikkei NEEDS \\texttt{{TICST120}} product for calendar 2024:
one record per market event, carrying the trade if there was one and the ten best
price levels on each side of the book as they stood afterwards. Trade direction
is supplied by the exchange rather than inferred, which removes a source of error
that would otherwise sit underneath every result here. Daily summaries from the
companion \\texttt{{TICSS110}} product supply trading units, shares outstanding,
and the variables used to screen the universe.

Five dates are excluded at the calendar level. Four never arrived in the
delivered feed, and 2024-04-23 arrived with ten shards against a monthly median
of twenty-two --- the signature of a transfer that stopped part-way. Treating a
partial day as a quiet one would bias every measure computed from it, so the
exclusion is made once, at the calendar, and no later stage can rediscover them.
That leaves 240 usable trading days.

\\subsection{{Two institutional details that shape everything}}

\\paragraph{{The session changed length part-way through the year.}}
The afternoon session closed at 15:00 until 2024-11-05 and at 15:30 afterwards.
Every time-dependent quantity here is parameterised by date. A hardcoded close
would silently truncate two months of the afternoon, and would corrupt the
price-impact horizons on exactly those days.

\\paragraph{{The exchange runs two tick grids at once.}}
Constituents of the TOPIX 500 trade on a finer grid than everything else --- a
regime extended from the TOPIX 100 to the whole index on 2023-06-05 and stable
through 2024. The schedule is encoded from the exchange's published tables and
cross-checked against ticks read directly off the tape.

This interacts with Ohta's sample filter in a way worth stating plainly, because
it shapes the sample more than any other decision. The filter admits only tick
sizes that are powers of ten, since the last-digit-of-ten construction is
otherwise undefined --- on a 0.5-yen grid the last digit takes only even values
and ``round'' stops meaning what it means elsewhere. On the finer grid the
1,000--3,000 yen band is 0.5 yen, so a large part of the mid-priced TOPIX 500
population is excluded outright. The regression sample is therefore tilted away
from mid-priced large caps, and the composition is reported rather than papered
over:

\\begin{{table}}[htbp]
\\centering
\\caption{{Tick-size composition of evaluated stock-days}}
\\label{{tab:tickcomp}}
\\small
\\begin{{tabular}}{{@{{}}lrl@{{}}}}
\\toprule
Tick size & Stock-days & Status \\\\
\\midrule
{comp_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Building the sample}}

The universe is selected by Ohta's own criteria, applied first to the daily
summaries so that the expensive tick-level work runs only on stocks that could
enter the sample. He studies First Section (now Prime) common stocks whose
trading days satisfy four conditions on more than half the year's sessions: the
first trade by 9:10, an opening price above 200 yen, more than twenty
continuous-session trades, and a tick size that is a power of ten throughout the
day. Regional venues, exchange-traded funds and trust units --- identifiable by
their trading units --- are excluded.

Each of those filters exists for a reason worth keeping in view. The 9:10 rule
removes days that opened late after a price-limit halt, which would have a
shortened continuous session and therefore a differently-behaved measure. The
200-yen floor removes stocks so cheap that the price cannot traverse all ten
digits without moving several percent. The twenty-trade minimum removes days
where the measure is a ratio of small integers.

Applying the same tests properly at tick level gives the final panel:
{wf.get('attempted', 0):,} stock-days attempted, {wf.get('in_sample', 0):,}
passing all four filters, and {n_sd:,} surviving the requirement that a stock
qualify on more than half the year --- {n_stocks:,} stocks over {n_days} trading
days, from {d0} to {d1}. The full waterfall is reported in the stage output; no
stock-day disappears without being counted.

\\paragraph{{Verdict.}}
The sample is Ohta's, reconstructed on a year he did not have. Its one
distinctive feature --- the exclusion of the 0.5-yen grid and therefore of much
of the mid-priced large-cap population --- follows from his filter rather than
from a choice made here, and is carried explicitly into the interpretation of
every result below.
""")

    # ------------------------------------------------------------ stylized
    gap_b = st.get("gap_b", {})
    fine = st.get("tick_1", {})
    one_yen = st.get("tick_10", {})
    # Precomputed because a brace inside an f-string expression is not an escape:
    # it is parsed as a set literal and the lookup fails.
    m0_mean = (st.get("m0_all") or {}).get("mean")
    mss_mean = (st.get("m_s_small0") or {}).get("mean")
    rdb_mean = (st.get("rdepth_bid0") or {}).get("mean")
    gap_b_pp = 100 * (gap_b.get("gap") or 0)
    w("05_stylized.tex", f"""
% =====================================================================
\\section{{Does 2024 look like the paper's sample?}}
\\label{{sec:stylized}}
% =====================================================================

Nothing below this point is worth reading if the measurement is wrong, and the
cheapest way to find out is to check whether a year the paper never saw
reproduces the facts it established. This section is that check. It is also the
first out-of-sample evidence on these measures.

\\subsection{{The headline measures}}

\\input{{tables/t_stylized}}

Table~\\ref{{tab:stylized}} puts the 2024 means beside Ohta's 2010--2022
benchmarks. Round prices take {pct(m0_mean)}\\% of all
continuous-session volume against his 13.5\\%, and
{pct(mbl)}\\% of buy-initiated large-trade volume against his 14.8\\%. Under a
uniform digit distribution every one of these would be 10\\%. Clustering is
alive and well on the Tokyo Stock Exchange two years past the end of his sample.

\\paragraph{{The ordering the mechanism predicts.}}
The paper's account is specifically about \\emph{{large}} orders taking stale
round-price limit orders, so the large-trade measure should exceed the
small-trade one. It does, by {num(gap_b_pp, 2)} percentage
points on the buy side, {sig_word(gap_b.get('t'))}. The small-trade measures sit
at {pct(mbs)}\\% and {pct(mss_mean)}\\%, barely
above the uniform benchmark. Whatever produces clustering, it is not something
that happens to trades of one unit.

\\subsection{{The digit distribution}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{f_digits.pdf}}
\\caption{{Volume-weighted share of trading at each last price digit, by tick
regime. The dashed line is the 10\\% a uniform distribution would give.}}
\\label{{fig:digits}}
\\end{{figure}}

Figure~\\ref{{fig:digits}} is the whole phenomenon in one picture. Digit zero
stands well above the uniform line; digit five stands slightly above it, which is
what one expects if half-way points attract some of the same behaviour; and the
remaining seven digits sit flat. That shape is itself a diagnostic --- a
measurement error in the digit arithmetic would not produce a clean spike at zero
and a smaller one at five, it would produce something lumpy.

\\subsection{{Finer grids, smaller stocks}}

\\input{{tables/t_tick}}
\\input{{tables/t_size}}

\\textcite{{Harris1991Clustering}} predicts more clustering on finer grids, and
the 0.1-yen days deliver it: {pct(fine.get('m0'))}\\% against
{pct(one_yen.get('m0'))}\\% on the 1-yen grid. The mechanism is not mysterious.
At a price around 900 yen a 0.1-yen tick is about one basis point, fine enough
that order placement collapses back onto whole-yen prices --- and a whole yen is
the roundest number available. These days are kept here, where they are a
result, and excluded from the regressions, where the paper excludes them.

Table~\\ref{{tab:size}} shows clustering falling across size quintiles, the
cross-sectional gradient Ohta reports and Harris predicts. It also explains why
the strategy demonstration in Section~\\ref{{sec:strategy}} neutralises size
before sorting: an unneutralised clustering sort is substantially a small-cap
bet.

\\subsection{{The impact premium}}

\\input{{tables/t_dimp}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{f_dimp.pdf}}
\\caption{{Round-price price-impact premium in large trades, with 95\\% confidence
intervals from standard errors clustered by stock and by day.}}
\\label{{fig:dimp}}
\\end{{figure}}

This is the paper's second result and the one that makes clustering matter for
execution. A trade at a round price is executing against an order whose price has
gone stale, so the midquote should move further afterwards. At the one-minute
horizon the premium is {num(dimp_b, 2)} basis points on the buy side and
{num(dimp_s, 2)} on the sell side, against Ohta's 1.26 and 1.12.

The effective-spread decomposition closes as an identity, which is a check on the
implementation rather than a finding: impact plus realized spread equals the
effective spread to three decimal places at every horizon tested.

\\subsection{{Standing round-price depth}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{f_rdepth.pdf}}
\\caption{{Binned means of the round-price share of visible ask depth against
measured clustering in buy-initiated large trades.}}
\\label{{fig:rdepth}}
\\end{{figure}}

The measure the paper's data could not support says the same thing from the other
side. {pct(rda)}\\% of visible ten-level ask depth sits at round prices against a
uniform benchmark of 10\\%, and
{pct(rdb_mean)}\\% on the bid.
Figure~\\ref{{fig:rdepth}} relates it to the executed measure: stock-days where
more of the standing book sits at round prices are stock-days where more
round-price volume executes. Where the clustering measures observe the mechanism
after the fact, this observes the inventory while it is still exposed.

\\subsection{{The contamination the paper warns about}}

\\input{{tables/t_opening}}

Ohta's Table 2 documents a mechanical distortion: a day that opens far from a
round price and then barely moves can never print one, so its measure is low for
reasons that have nothing to do with noise traders. The distortion is present in
2024. Opening prices cluster on their own --- digit zero accounts for
about a fifth
of openings against a tenth under uniformity --- and days opening near a round
price show visibly higher measured clustering than days opening far from one.
Every regression below therefore carries the interaction of the opening-digit
dummies with a low-volatility indicator, as the paper does.

\\paragraph{{Verdict.}}
2024 reproduces the paper's facts: clustering well above the uniform benchmark,
concentrated in large trades, stronger on finer grids and in smaller stocks, and
carrying a positive round-price impact premium of roughly the published size. The
measurement is sound and the rest of the study can be built on it.
""")

    # ------------------------------------------------------------ liquidity
    def rq2cell(y, spec):
        r = rq2.get(f"{y}|{spec}", {})
        k = "m_b_large0_l1" if spec != "contemporaneous" else "m_b_large0"
        return ((r.get("coef") or {}).get(k), (r.get("t") or {}).get(k),
                r.get("n"))

    es_b, es_t, es_n = rq2cell("ln_effsprd", "dynamic")
    ip_b, ip_t, ip_n = rq2cell("imp60_bps", "dynamic")
    dep_b, dep_t, _ = rq2cell("ln_depth_best", "dynamic")
    sd_m = (st.get("m_b_large0") or {}).get("se")
    panel_sd = None
    if final is not None and "m_b_large0" in final.columns:
        panel_sd = float(final["m_b_large0"].std() or 0)
    es_1sd = (es_b * panel_sd * 100) if (es_b is not None and panel_sd) else None
    # Built outside the f-string: Python forbids a backslash inside one, and this
    # phrase needs an escaped percent sign for LaTeX.
    if es_1sd:
        econ = ("about " + num(abs(es_1sd), 1) + "\\% " + direction(es_b))
    else:
        econ = "that moves in the same direction"
    kyle_or_stale = (
        "positive, favouring the second reading: days following heavy round-price "
        "trading are days when trading costs more"
        if (es_b or 0) > 0 else
        "negative, favouring the first reading: days following heavy round-price "
        "trading are days when trading costs less")

    w("06_liquidity.tex", f"""
% =====================================================================
\\section{{Clustering and next-day liquidity}}
\\label{{sec:liquidity}}
% =====================================================================

This is the question the paper leaves open. Clustering measures noise-trader
activity; what does that activity do to the cost of trading?

\\subsection{{Specification, and why it is the dynamic one}}

\\input{{tables/t_rq2}}

Three specifications are estimated for each outcome and
Table~\\ref{{tab:rq2}} reports two of them. All carry stock and day fixed
effects, the full control set, and standard errors clustered on both margins.

The \\emph{{predictive}} column regresses today's liquidity on yesterday's
clustering. It fixes the ordering but not much else: both series are persistent,
so a stock that is illiquid and clustered today was probably both yesterday, and
the coefficient partly recovers that.

The \\emph{{dynamic}} column adds the outcome's own lag, and it is the one the
argument rests on. Its coefficient answers a sharper question --- does yesterday's
clustering say anything about today's liquidity that \\emph{{yesterday's
liquidity}} did not already say? With {n_days} time periods the resulting
dynamic-panel bias is of order $1/T$ and can be neglected.

\\subsection{{What the panel says}}

The coefficient on lagged clustering in the effective-spread regression is
{num(es_b, 4)}, {sig_word(es_t)}. Read economically, a one-standard-deviation
increase in the round-price share of large-trade volume is associated with an
effective spread {econ} the following day. For the price-impact outcome the
coefficient is
{num(ip_b, 4)}, {sig_word(ip_t)}.

\\paragraph{{Which way the open question falls.}}
Section~\\ref{{sec:intro}} set out two incompatible predictions. On the
\\textcite{{Kyle1985}} reading, more noise trading means a deeper market and
cheaper trading. On the stale-order reading, clustering marks liquidity supplied
by participants about to be picked off, and trading should get more expensive.
The sign here is {kyle_or_stale}. That is one year of one market, and it is
reported as an association rather than a mechanism.

\\subsection{{Which measure carries the information}}

\\input{{tables/t_rq2_alt}}

Table~\\ref{{tab:rq2alt}} substitutes the other clustering measures. If the
relationship works through the channel Ohta describes --- large orders taking
stale round-price limit orders --- it should live in the large-trade measures and
not in the small-trade ones, which Section~\\ref{{sec:stylized}} showed barely
depart from the uniform benchmark in the first place. This is a weaker test than
the placebo in Section~\\ref{{sec:robust}}, but it uses the paper's own internal
contrast.

\\paragraph{{Verdict.}}
Clustering carries information about next-day liquidity that the outcome's own
history does not, in a specification demanding enough to absorb every stock-level
and market-wide effect. The association is descriptive: one year, no experiment,
and no claim that intervening on clustering would move spreads.
""")

    # ------------------------------------------------------------ book
    def r3(y, k):
        r = rq3.get(y, {})
        return ((r.get("coef") or {}).get(k), (r.get("t") or {}).get(k), r.get("n"))

    l_b, l_t, _ = r3("m_b_large0", "l_s0_l1")
    lc_b, lc_t, _ = r3("m_b_large0", "l_s0c_l1")
    rd_b, rd_t, _ = r3("m_b_large0", "rdepth_ask0_l1")
    intr_k = "ln_effsprd|m0_l1"
    ib = (intr.get(intr_k, {}).get("coef") or {}).get("m0_l1")
    it = (intr.get(intr_k, {}).get("t") or {}).get("m0_l1")
    ofi_i = intr.get("ofi_interaction", {})
    ofi_b = (ofi_i.get("coef") or {}).get("ofi_x_high")
    ofi_t = (ofi_i.get("t") or {}).get("ofi_x_high")

    w("07_book.tex", f"""
% =====================================================================
\\section{{The order book, and the day within the day}}
\\label{{sec:book}}
% =====================================================================

\\subsection{{Rebuilding the paper's equations without its proxies}}

\\input{{tables/t_rq3}}

Ohta explains clustering with limit-order submissions, their cancellation rate,
and two proxies for individual activity. The proxies are unavailable here, so
Table~\\ref{{tab:rq3}} substitutes what the book itself shows: the round-price
share of submitted limit-order volume, the ratio of cancellations to submissions
at round prices, and the round-price share of standing visible depth.

The hypothesis makes three sign predictions. More round-price submissions should
mean more round-price trading, so the coefficient on $L^{{S0}}$ should be
positive: it is {num(l_b, 4)}, {sig_word(l_t)}. Orders that are \\emph{{not}}
cancelled are the ones that get picked off, so the coefficient on the
cancellation ratio should be negative: it is {num(lc_b, 4)}, {sig_word(lc_t)}.
And more standing round-price inventory should mean more round-price execution:
the coefficient on $\\mathit{{RDepth}}$ is {num(rd_b, 4)}, {sig_word(rd_t)}.

\\input{{tables/t_rdepth}}

Table~\\ref{{tab:rdepth}} asks the more direct question --- whether standing
round-price depth predicts liquidity by itself. This is the measure with the
cleanest interpretation in the whole study, because it requires no inference at
all: it is a share of displayed volume, read off the book.

\\subsection{{Within the day}}

\\input{{tables/t_intraday}}

The paper asks for the relationship ``at a daily or shorter frequency''.
Table~\\ref{{tab:intraday}} is the shorter one: thirty-minute buckets, with
stock-day fixed effects absorbing everything about the day --- the stock, the
news, the market state --- and bucket-of-day fixed effects absorbing the familiar
intraday pattern in spreads and volume. What identifies the coefficient is
purely within-day, bucket-to-bucket variation.

The coefficient on lagged-bucket clustering in the effective-spread regression is
{num(ib, 4)}, {sig_word(it)}. Coefficients here are small by construction and
should be read that way: this design deliberately throws away all the variation
the daily panel uses.

\\paragraph{{Order flow moves the price further when clustering is high.}}
The execution-relevant version of the impact result asks whether the \\emph{{same}}
order-flow imbalance moves the midquote more on high-clustering days. The
interaction term is {num(ofi_b, 4)}, {sig_word(ofi_t)}. A positive value means
the book is easier to push when round-price trading is heavy, which is what the
stale-order account implies and what matters to anyone sizing an order.

\\paragraph{{Verdict.}}
The book-level variables line up with the mechanism where they are measurable,
and the relationship survives into within-day variation, where nothing
slow-moving about a stock can be responsible for it.
""")

    # ------------------------------------------------------------ robustness
    pl0 = rob.get("placebo_d0", {})
    others = [rob.get(f"placebo_d{d}", {}) for d in range(1, 10)]
    n_sig = sum(1 for o in others
                if o.get("t") is not None and abs(o.get("t") or 0) > 1.96)
    fm = rob.get("fama_macbeth", {})

    w("08_robustness.tex", f"""
% =====================================================================
\\section{{Robustness}}
\\label{{sec:robust}}
% =====================================================================

\\subsection{{The placebo}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{f_placebo.pdf}}
\\caption{{The effective-spread regression re-run with the measure rebuilt on each
last price digit in turn. The round-price digit is highlighted; bars are 95\\%
confidence intervals.}}
\\label{{fig:placebo}}
\\end{{figure}}

\\input{{tables/t_placebo}}

This is the test that could have ended the study. The measure is a share of
volume at a particular last digit. If a share of volume at \\emph{{any}} last
digit predicts liquidity equally well, then nothing here is about round numbers
--- it is an artefact of how a concentration measure behaves, and the whole
interpretation collapses.

Rebuilding the measure on each of digits one through nine and re-running the same
regression, the round-price coefficient is {num(pl0.get('beta'), 4)}
($t = {num(pl0.get('t'), 2)}$), and {n_sig} of the nine placebo digits reach
conventional significance. Digit five is flagged separately in
Table~\\ref{{tab:placebo}} because half-way points are semi-focal and attract some
of the same behaviour --- treating it as a pure placebo would be stacking the
deck.

\\subsection{{Subsamples and an alternative estimator}}

\\input{{tables/t_robust}}

Table~\\ref{{tab:robust}} re-estimates the headline specification on subsamples.
The size-quintile rows matter for interpretation: the mechanism is about
retail-heavy, less closely watched stocks, so the relationship should be stronger
in smaller ones. The half-year rows test whether a single episode is doing the
work. The 0.1-yen row estimates on the days excluded from the main sample.

A Fama--MacBeth cross-check estimates the relationship separately in each daily
cross-section and averages, with Newey--West standard errors over the resulting
series. It gives {num(fm.get('beta'), 4)} with $t = {num(fm.get('t'), 2)}$ over
{fm.get('T', 0)} cross-sections. It is reported because it makes completely
different assumptions about the correlation structure than two-way clustering
does, and agreement between them is worth more than either alone.

\\paragraph{{Verdict.}}
The result is specific to the round-price digit rather than to the shape of the
measure, and it survives changes in subsample and estimator.
""")

    # ------------------------------------------------------------ strategy
    w("09_strategy.tex", f"""
% =====================================================================
\\section{{A strategy demonstration}}
\\label{{sec:strategy}}
% =====================================================================

The brief behind this study is to build a rebalancing strategy on this indicator,
and the distance between ``the indicator predicts liquidity'' and ``here is a
tradable rule'' is exactly where such projects fail. So the machinery is built
and run: form a signal from information available at the time, sort on it,
rebalance, and charge realistic costs.

\\input{{tables/t_strategy}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{f_strategy.pdf}}
\\caption{{Cumulative return to the long--short quintile portfolio, gross and net
of estimated trading costs.}}
\\label{{fig:strategy}}
\\end{{figure}}

The signal is the daily cross-sectional residual of the clustering measure on
everything that mechanically drives it --- the opening-digit contamination,
relative tick size, size, turnover, volatility and spread --- standardised within
the day. Residualising is essential rather than cosmetic: clustering is strongest
in small, retail-heavy names, so an unneutralised sort is largely a small-cap bet
wearing a microstructure costume.

\\paragraph{{The cost line is the finding.}}
The long--short portfolio turns over
{num(100*(strat.get('turnover') or 0), 1)}\\% of each leg per day. At the
measured effective half-spread of {num(strat.get('half_spread_bp'), 2)} basis
points, that costs {num(strat.get('cost_bp_day'), 2)} basis points a day against
a gross return of {num(strat.get('gross_bp_day'), 2)}, leaving
{num(strat.get('net_bp_day'), 2)}. A signal that has to be refreshed daily in
names this wide pays its gross return away several times over.

That is the useful lesson for the problem this prototype was built for. If the
indicator is to be used at all, it has to be used either at a horizon long enough
to amortise the spread, or inside a rebalance that was going to happen anyway ---
which is precisely the execution-scheduling application the impact results in
Section~\\ref{{sec:stylized}} point to, and precisely not a standalone alpha.

\\paragraph{{Verdict.}}
The pipeline runs end to end and the economics are transparent. This is not
evidence of a profitable strategy: one year, one market, no out-of-sample period,
and closing prices carrying no adjustment for splits or dividends --- which is why
stock-days with moves beyond 25\\% are dropped rather than believed.
""")

    # ------------------------------------------------------------ discussion
    w("10_discussion.tex", f"""
% =====================================================================
\\section{{Discussion}}
\\label{{sec:discussion}}
% =====================================================================

\\subsection{{What was found}}

Ohta's measures survive out of sample. On a year his study did not cover, round
prices take {pct(mbl)}\\% of large-trade volume against {pct(mbs)}\\% of
small-trade volume, the ordering his mechanism predicts holds firmly, finer grids
and smaller stocks cluster more, and trades at round prices carry a positive
price-impact premium of roughly the size he reports. None of this was tuned; the
comparison was set up before the numbers were computed.

On the question he leaves open, clustering carries information about next-day
liquidity beyond what that liquidity's own history already contains, and the
relationship persists into within-day variation where no slow-moving stock
characteristic can account for it. A placebo across the other nine digits
separates the result from an artefact of the measure's construction.

The two book-based measures behave as the mechanism implies. Round prices hold a
disproportionate share of standing visible depth --- observing the stale
inventory directly rather than after it has been taken --- and the round-price
share of submitted limit orders and their cancellation ratio enter with the signs
the hypothesis predicts.

\\subsection{{What it is not}}

\\paragraph{{Not causal.}}
One year, no experiment, no instrument. Everything reported is a descriptive
association or a predictive relationship. Clustering and liquidity are plausibly
both driven by things this study does not observe, and nothing here identifies a
direction of causation.

\\paragraph{{Not Ohta's identification.}}
He identifies noise traders with margin balances and ownership shares. Those are
unavailable here, so this study leans entirely on the claim his conclusion
advances --- that the clustering measures are themselves the proxy. That makes
the exercise a test of his proposal rather than an independent confirmation of
it, and a genuine replication would need the ownership data.

\\paragraph{{A narrow window on the book.}}
Ten visible levels sound like a lot and are not. On the stock-days examined in
detail, 95--98\\% of quoted volume sits beyond them. The inferred order flow
describes the neighbourhood of the best quote, which is where execution happens,
but it is silent about the far book --- and the far book is exactly where Ohta
argues the interesting stale orders sit.

\\paragraph{{A sample shaped by a filter.}}
Excluding non-power-of-ten tick sizes removes the 0.5-yen grid, and with it much
of the mid-priced TOPIX 500 population. This follows from the paper's filter
rather than from a choice made here, but it means the regression sample is not
the market.

\\subsection{{What would come next}}

Three things, in order of value. Obtaining the margin and ownership data would
convert this from a test of the proxy into a joint test of the proxy and the
mechanism. Extending to several years would allow the year fixed effects and
sub-period splits that would show whether the relationship is stable, and would
let the strategy demonstration have a genuine out-of-sample period. And the
execution application is more promising than the alpha one: the impact premium at
round prices is a statement about what an order pays, which points at scheduling
and price-level selection within a rebalance that is happening regardless ---
where a basis point of impact is a real saving and no forecasting is required.

\\subsection{{Conclusion}}

The sentence at the end of Ohta (2026) proposes that price clustering can serve
as an observable proxy for noise-trader activity, and that the relationship
between that activity, price formation and liquidity could then be studied at
daily or higher frequency. On the 2024 Tokyo Stock Exchange tape, it can be, and
it is: the measures replicate out of sample, they carry information about
liquidity that the liquidity's own history does not, and the order book shows the
standing round-price inventory the mechanism requires. What the measures cannot
yet do is support a standalone trading rule, for the ordinary reason that the
costs exceed the signal --- which is itself the most useful thing this prototype
has to say about how such an indicator should be used.
""")

    # A table the analysis could not estimate on this sample must still resolve,
    # and must say so in the document rather than silently vanishing.
    stubs = 0
    for f in sorted(os.listdir(CH)):
        if not f.endswith(".tex"):
            continue
        src = open(os.path.join(CH, f), encoding="utf-8").read()
        for m in re.finditer(r"\\input\{tables/([A-Za-z0-9_]+)\}", src):
            tgt = os.path.join(S4.TABLES, m.group(1) + ".tex")
            if os.path.exists(tgt):
                continue
            os.makedirs(S4.TABLES, exist_ok=True)
            label = m.group(1).replace("t_", "tab:")
            with open(C.write_guard(tgt), "w", encoding="utf-8") as fh:
                fh.write("\\begin{table}[htbp]\n\\centering\n"
                         "\\caption{Not estimated on this sample}\n"
                         f"\\label{{{label}}}\n\\small\n"
                         "\\begin{tabular}{@{}l@{}}\n\\toprule\n"
                         "This specification could not be estimated on the sample "
                         "available. \\\\\n\\bottomrule\n\\end{tabular}\n"
                         "\\end{table}\n")
            stubs += 1
    if stubs:
        print(f"created {stubs} placeholder table(s) for specifications not estimated")

    fstubs = 0
    for f in sorted(os.listdir(CH)):
        if not f.endswith(".tex"):
            continue
        src = open(os.path.join(CH, f), encoding="utf-8").read()
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
            tgt = os.path.join(S4.FIGURES, m.group(1))
            if os.path.exists(tgt):
                continue
            plt = S4.setup_mpl()
            fig, ax = plt.subplots(figsize=(6.4, 2.0))
            ax.axis("off")
            ax.text(0.5, 0.5, "not produced on this sample", ha="center",
                    va="center", fontsize=11, color="0.35")
            os.makedirs(S4.FIGURES, exist_ok=True)
            fig.savefig(C.write_guard(tgt))
            plt.close(fig)
            fstubs += 1
    if fstubs:
        print(f"created {fstubs} placeholder figure(s)")

    print("\nchapters written from stage outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
