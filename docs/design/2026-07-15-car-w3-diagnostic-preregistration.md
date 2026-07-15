# CAR-W3 Fresh Diagnostic Preregistration

Date: 2026-07-15
Executor: Codex
Status: frozen before the 3M-FE run

## Purpose

This is the single final diagnostic for the writeback channel. It tests whether
CAR-W3's first-pair futility stop reduces unavoidable probe exposure while
preserving the unchanged CAR-W2 utility and safety gate. It is a pilot-level
diagnostic, not a full-24 performance evaluation.

## Fixed protocol

- Problems: `E1 E2 S3 R4 A5 E6`.
- Seeds: `26 27 28` (fresh and disjoint from the earlier W1 `9-11`, W2
  `22-24`, the focused 5k seed-25 smoke, and v33/v36 replay seed sets).
- Budget: strict `3,000,000` FE per lane/problem/seed.
- Lanes: `v33_fallback`, `car_w3`, `car_w3_shuffled`,
  `car_w3_paired_fallback`, `no_action_negative_control`.
- Parallelism: `18` worker jobs, pinned HCC environment, `phase_i_mmes`.
- Output: `results/car_w3_diagnostic_6case_seed26_28_3m_20260715`.
- Runtime inputs: current-run `DispatchEvidence` only. Case identity, function
  family, paper-best, historical outcomes, final error, and win/catastrophic
  labels are audit-only and cannot reach dispatch.

Exact command (the wrapper Python and thread variables are part of the
protocol):

```powershell
$env:PYTHONPATH='src'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m experiments.pilots.exp_003_hcc_runtime_consumer_smoke.run `
  --output-dir E:/ARAC/results/car_w3_diagnostic_6case_seed26_28_3m_20260715 `
  --seeds 26 27 28 `
  --problems E1 E2 S3 R4 A5 E6 `
  --jobs 18 `
  --max-fes 3000000 `
  --budget-accounting strict `
  --lane-profile car_w3_diagnostic `
  --hcc-root E:/ARAC/vendor/hcc `
  --hcc-runner E:/ARAC/scripts/hcc_smoke_runner.py
```

## Pre-registered gates

Integrity is mandatory: `90/90` fresh executions, zero same-budget violations,
all AOB hashes unchanged, all anti-leakage rows pass, branch manifests are
complete, and no missing or non-finite probe rows.

The W release gate is unchanged from CAR-W2:

1. at least six graph-candidate commits covering at least three problems and two
   observed overlap strata;
2. probe-to-3M sign agreement at least 60% where a candidate was committed;
3. paired log-error mean `< 0` and median `<= 0` for CAR-W3 versus v33;
4. zero catastrophic losses under the frozen `<= -20%` offline threshold; and
5. total probe overhead `<= 3%` of the strict FE budget (the fixed CAR lease
   fraction; W3 may spend less through first-pair abort).

First-pair abort count, candidate coverage, commit rate, pair count per run,
and per-seed/worst-seed outcomes are descriptive mandatory reports, not
post-hoc gates. `paper-best` is joined only after runtime completion and is a
secondary offline comparison.

## Stop rule

If any integrity gate fails, or any W release gate fails, stop CAR iteration and
do not start resource/search-state channels, held-out expansion, or a CAR full-24
run. The paper then reports either safe-but-ineffective abstention or candidate
long-horizon mismatch, with the raw artifacts retained.
