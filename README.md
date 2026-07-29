# Noise-trader activity, price formation and liquidity on the TSE

A prototype implementation of the **future work proposed by Ohta (2026)**, *"Noise Traders and
Price Clustering"* (現代ファイナンス / *Gendai Finance*, doi:
[10.24487/gendaifinance.480001](https://doi.org/10.24487/gendaifinance.480001)).

Ohta shows that price clustering — the disproportionate share of trades executing at round
prices — arises because noise traders leave stale limit orders at round prices outside the best
quote, which fast participants pick off with large market orders. The paper's conclusion argues
that clustering measures can therefore serve as an *observable proxy for noise-trader activity*,
and that relating that activity to price formation and liquidity "at a daily or shorter
frequency" remains a task for future work.

**This repository is that analysis**, on the Tokyo Stock Exchange tape for January–April 2025 —
three years beyond the paper's 1997–2022 sample.

## What it does

1. Computes the Ohta clustering measures (`M0`, and `M` split by trade initiator and by size)
   for every stock-day on the tape, under the paper's own sample filters.
2. Validates the pipeline against the paper's published magnitudes and two hand-verified anchor
   stock-days before trusting any of it.
3. Relates clustering to **price formation** and to **liquidity as the paper defines it** —
   price impact, effective spread, realized spread, quoted depth — at daily and 30-minute
   frequency, adding two measures the paper's data could not support: the share of visible
   ten-level resting depth sitting at round prices, and limit-order submission/cancellation
   shares inferred from ladder deltas.
4. Demonstrates a small cost-aware strategy on the residualized signal. This is plumbing, not an
   alpha claim, and is labelled as such throughout.
5. Produces a LaTeX report documenting the method, the results and the limitations.

## Layout

| Path | Contents |
|---|---|
| `s0_*.py` … `s7_*.py` | Flat numbered stage scripts. `step0` is always a gate; the last step of each stage writes a markdown report. |
| `measures.py` | The measure library. Used unchanged by the pilot and the full pass, so validating it on the anchor days validates production code. |
| `tests/` | pytest suite: tick tables, digit arithmetic, sessions, trade signing, the ladder algorithm on a hand-built tape. |
| `results/s<N>_*/` | Per-stage outputs and reports (not in git). |
| `report/` | LaTeX source, generated tables and figures, and the compiled PDF. |
| `reference/` | Read-only reference implementation from the preparation drill. |

The Parquet tick store lives at `D:\MTEC_tick_store`, outside the repository.

## Running it

```powershell
conda activate localdb
cd E:\MTEC\prototype

python s0_step0_gate.py           # environment gate
python s0_step1_deps.py           # estimator backends vs a known-truth panel
python s0_step2_raw_inventory.py  # freeze the trading calendar
python s0_step3_yobine_tables.py  # tick-size reference
python s0_step4_report.py

pytest tests -q                   # unit tests
```

Later stages follow the same pattern; each refuses to run until the previous stage's gate has
passed. Stages 2 and 3 are long jobs and are checkpointed, so an interrupted run resumes.

## Data

Source is the Nikkei NEEDS `TICST120` feed (tick data with the ten best quotes per side) for
2025, read strictly read-only from `G:\needs`, plus `TICSS110` daily summaries for trading units
and shares outstanding. **No data — raw or derived — is committed to this repository or leaves
the machine.** The licence belongs to the university, not to this project.

## If you are picking this up for the real task

This was built as preparation, so the parts worth reusing are not evenly distributed. In rough
order of what would survive contact with a different specification:

| Asset | Where | Why it transfers |
|---|---|---|
| Tick-size schedule | `s0_common.py` (`YOBINE_*`, `tick_for10`, `day_tick_constant10`) | The exchange's two grids in integer deci-yen, cross-checked against ticks read off the tape. Nothing in `tse_tick` has this, and the digit measure is undefined without it. |
| Trade signing | `s0_common.py` (`EXEC_TYPE_MAP`) | Includes the stop-quote variants that decode to an unrecognised label and would otherwise be dropped silently — exactly on the most volatile days. |
| Digit arithmetic | `measures.py` (`digit_expr`, `observed_grid`, `resolve_tick10`) | Integer-only, and the resolution rule handles index-membership drift. Getting this wrong reads as 50% or 100% clustering, not as noise. |
| Liquidity measures | `measures.py` (`spread_impact`) | Effective spread, impact at several horizons, realized spread, quote depth — Ohta's definitions, with session-end truncation handled per horizon. |
| Ladder inference | `measures.py` (`ladder_lc`) + `report/appendices/B_ladder.tex` | Submission and cancellation volume from book deltas. Validated against the paper's published magnitudes. |
| Ingest configuration | `s2_step1_ingest.py` | The default full-frame ingest runs out of memory at current message rates. The header comment explains what works and why the obvious fix makes it worse. |
| Panel inference | `s4_common.py`, `s5_common.py` | Two-way clustered means and fixed-effects fits, with guards that refuse degenerate panels instead of reporting a zero standard error. |

**Where the real task will differ.** The rebalancing brief could mean execution-cost reduction,
a signal tilt, or trigger design. This prototype builds the spine all three need — the indicator,
the panel, and a costed evaluation — and deliberately stops short of committing to one. The
strategy demonstration in `s6_step1_strategy.py` is the signal-tilt reading, and its main finding
is that the costs exceed the signal at daily frequency, which points at the execution reading
instead.

**What is missing and would have to be added.** Margin-trading and ownership data (Ohta's direct
noise-trader proxies) are not available here, so the study leans on his claim that clustering is
itself the proxy. With those series the same panel becomes a joint test of proxy and mechanism.

## Caveats worth reading before the results

- Four months of one market. Every result is descriptive or predictive; none is causal.
- Margin-trading and ownership data (Ohta's direct noise-trader proxies) are unavailable here.
  Their absence is deliberate: the paper's own conclusion is that clustering *is* the proxy, so
  this prototype uses book-derived measures only.
- Ohta's tick filter excludes the 0.5-yen grid, which covers the 1,000–3,000 yen band
  for TOPIX500 constituents. The regression sample is tilted accordingly, and the report
  reports the composition rather than pooling incompatible grids.
- The study covers four months, not a full year; the report states its own coverage.
