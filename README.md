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

**This repository is that analysis**, on the full Tokyo Stock Exchange tape for calendar 2024 —
two years beyond the paper's 1997–2022 sample.

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
python s0_step2_raw_inventory.py  # freeze the 2024 trading calendar
python s0_step3_yobine_tables.py  # tick-size reference
python s0_step4_report.py

pytest tests -q                   # unit tests
```

Later stages follow the same pattern; each refuses to run until the previous stage's gate has
passed. Stages 2 and 3 are long jobs and are checkpointed, so an interrupted run resumes.

## Data

Source is the Nikkei NEEDS `TICST120` feed (tick data with the ten best quotes per side) for
2024, read strictly read-only from `G:\needs`, plus `TICSS110` daily summaries for trading units
and shares outstanding. **No data — raw or derived — is committed to this repository or leaves
the machine.** The licence belongs to the university, not to this project.

## Caveats worth reading before the results

- One calendar year. Every result is descriptive or predictive; none is causal.
- Margin-trading and ownership data (Ohta's direct noise-trader proxies) are unavailable here.
  Their absence is deliberate: the paper's own conclusion is that clustering *is* the proxy, so
  this prototype uses book-derived measures only.
- Ohta's tick filter excludes the 0.5-yen grid, which in 2024 covers the 1,000–3,000 yen band
  for TOPIX500 constituents. The regression sample is tilted accordingly, and the report
  reports the composition rather than pooling incompatible grids.
- 2024-04-23 through 2024-04-30 are partly missing from the delivered feed and are excluded.
