## Project
Prototype of the **stated future work** of Ohta, W. (2026), "Noise Traders and Price Clustering,"
*Gendai Finance*, doi:10.24487/gendaifinance.480001 (J-STAGE advance publication 2026-04-25).
The paper's conclusion says price clustering measures can serve as an observable proxy for
noise-trader activity (NTA), and that "it may be possible to analyze the relationship between
noise trader activity and price formation and liquidity at a daily or shorter frequency. Such
analysis, however, remains a task for future work." **This repository is that analysis**, on TSE
2024 tick / 10-level limit-order-book data. Built as internship preparation (MTEC 三菱UFJ信託
投資工学研究所, starts 2026-07-31, task = a rebalancing strategy on this indicator).

## Research goal (the spine, in order)
1. **COMPUTE** the Ohta (2026) clustering measures (M0, M^{B/S,Large/Small,0}) for the full TSE
   tape, calendar 2024 -- two years beyond the paper's 1997-2022 sample.
2. **VALIDATE** against the paper's published magnitudes and against two hand-verified anchor
   stock-days. A pipeline that does not reproduce the literature's magnitudes is not trusted.
3. **RELATE** NTA to *price formation* and *liquidity* at daily and 30-minute frequency, using
   the paper's own liquidity definitions (price impact, effective spread, realized spread,
   best-quote depth) -- never plain volume -- plus multi-level-book measures the paper could not
   use: RDepth (share of visible 10-level resting depth sitting at round prices) and
   ladder-inferred limit-order submission / cancellation shares (L, L^C).
4. **DEMONSTRATE** a light, cost-aware strategy on the residualized signal -- explicitly a
   plumbing demonstration, not an alpha claim.
5. **DOCUMENT** in a LaTeX report styled on `G:\flash_crash\thesis`.

## Scope boundary
IN scope: everything derived from the Nikkei NEEDS TICST120 feed for calendar 2024, plus
TICSS110 daily summaries (trading unit, shares outstanding) and JPX/TOPIX reference tables.
OUT of scope: margin-trading (日証金) and ownership (有報) data -- unavailable locally. Their
absence is *by design*: the paper's conclusion is that clustering itself measures NTA, so this
prototype uses book-derived NTA measures only. Also OUT: causal claims. One year, no exogenous
variation, no identification strategy. Every result is descriptive or predictive, and the report
says so.

## Stack
Python 3.11 (conda env `localdb`), Windows / PowerShell. polars (primary), pandas, pyarrow,
duckdb; statsmodels, linearmodels (PanelOLS two-way FE), pyfixest (high-dimensional FE);
matplotlib. Domain library `tse_tick` 0.15.1 (installed in `localdb`; source `G:\tse_tick`),
which reads NEEDS TICST120 zips into 95-column frames and runs the two-stage
(INGEST -> Parquet -> QUERY) workflow. LaTeX: TeX Live 2026, `latexmk -pdf` (pdflatex + biber).

## Architecture decisions (resolved -- do not relitigate)
- **Two-shot pipeline** -- tse_tick Stage 1 ingests the raw tape once into a Parquet store; every
  later stage reads the store. Chosen because the raw source is a seek-bound HDD and the panel
  build needs many passes over the same days.
- **Integer deci-yen digit arithmetic** -- `p10 = round(price*10)`, `digit = (p10 % (10*tick10))
  // tick10`. Chosen because float `mod` silently corrupts the 0.1-yen grid, which is where
  clustering is strongest.
- **Vendor trade signing** -- `Execution Type` (約定種別) is authoritative: "At Sell Quote" =
  buyer-initiated, "At Buy Quote" = seller-initiated (verified 100/0 on ~76k trades). Never
  Lee-Ready. Stop-quote variants decode as `Unknown (116/148/216/248)` and are mapped explicitly.
- **Tick authority** -- the coded JPX yobine tables are primary; per-stock-day empirical
  inference from the tape is the cross-check. A tape *finer* than the table wins (membership
  drift); a tape *coarser* than the table does not (illiquid names simply skip grid points).
  Override rate must stay under 1% of stock-days or the run halts.
- **Zaraba-only measures** -- opening and closing auctions excluded from every measure. Auction
  prices are not set by individual limit orders, and the closing itayose is one giant
  volume-weighted print that moves a daily measure by percentage points.
- **Session close parameterized by date** -- 15:00 before 2024-11-05, 15:30 on/after. Never
  hardcoded.
- **OFI reimplemented** -- `tse_tick.features.compute_flow_imbalance` has a real bug (it assigns
  `Execution Price` to its volume variable). Never call the shipped feature functions.
- **Single-pass panel build** -- one read per (date, ticker) file produces the stock-day row, the
  30-minute bucket rows, and the ladder L/C aggregates together. The store is read once.
- **L/C ladder inference is core, with a logged downscope** -- if the wall-clock gate trips, fall
  back to a 300-stock size-stratified subsample. RDepth stays universal in every mode.
- **Winsorization happens in the regression stage only.** The panel on disk is raw.
- **0.1-yen-tick days are excluded from regressions** (the paper's practice) but kept in the
  stylized-facts stage, where their elevated clustering is a result.
- **April 2024 gap days** (2024-04-24/25/26/30 absent; 2024-04-23 truncated) are excluded at the
  calendar level and disclosed in the report.

## Repository
ONE local git repo rooted at `E:\MTEC\prototype`. Code + docs + report only; results, logs and
all Parquet are gitignored. The Parquet store lives OUTSIDE the repo at `D:\MTEC_tick_store`.
Published as private GitHub repo `jevwithwind/noise_trader` -- code and report only, never data.

## Conventions
- Raw data (READ-ONLY source): `G:\needs\個別株式2024\TICST120\YYYYMM\HTICST120.YYYYMMDD.N.zip`
  (shard numbers are unpadded -- sort numerically). Sibling `TICSS110` = daily summary.
- Parquet store: `D:\MTEC_tick_store\individual_stock\date=YYYYMMDD\ticker=NNNN.parquet` and
  `...\stock_summary\`. ~1M small files: discovery goes through `store_manifest.parquet`,
  never a recursive glob.
- Stage scripts are flat and numbered: `s<N>_common.py`, `s<N>_step<M>_<what>.py`. `step0` is
  always a gate; the last step of every stage writes a markdown report.
- Outputs: `results\s<N>_<stage>\`. Versioned, never overwritten. Logs: `logs\<script>_<ts>.log`.
- Checkpoints: atomic JSON (write `.tmp`, then `os.replace`).
- Every writer calls `write_guard(path)` before touching disk.
- The measure library is ONE module (`measures.py`) used unchanged by the pilot and the full
  pass, so validation on anchors is validation of production code.

## Hard rules
- NEVER write outside `E:\MTEC\prototype\` and `D:\MTEC_tick_store\`.
- `G:\needs\` is READ-ONLY, always. `G:\flash_crash\` and `G:\tse_tick\` are read-only references.
- NEVER let anything else read G: while an ingest is running (the HDD is seek-bound).
- ALWAYS validate on the anchor stock-days before launching a full-range run.
- ALWAYS report counts and coverage at the end of any full run.
- NEVER add a dependency, or change a resolved architecture decision, without flagging it.
- NEVER copy NEEDS raw or derived data off this machine, or into git. The data licence is the
  university's/lab's. Figures and aggregates in the report are fine; files are not.
- NEVER report a validation as passing without the numbers that show it.

## Definition of done (per stage)
Gate passed; full run completes with checkpoints intact; outputs match the documented schema;
counts, coverage and every judgment call reported back; CLAUDE.md updated if a decision changed.

## Autonomy contract
Work autonomously through any plan I have approved.
DECIDE AND PROCEED, without asking, on anything mechanical or reversible:
file layout, boilerplate, naming, refactors, writing/running tests, reads,
and any change that's trivially undoable.
STOP AND ASK ME, every time, only at a genuine decision point:
- an architecture fork (two real designs, not just two syntaxes),
- an irreversible or destructive action (deletes, migrations, force-push, anything outside the repo),
- an ambiguous or underspecified requirement,
- a material tradeoff (performance vs. simplicity, a new dependency, a public API shape).
When you stop: batch all open questions into one message, and for each give
2-3 options, your recommendation, and a sensible default I can approve in one word.
Do NOT ask permission for routine work. Do NOT make silent decisions on consequential work.
When in doubt about which bucket something falls in, treat it as consequential and ask.
