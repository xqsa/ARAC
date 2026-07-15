# CAR-W3 Diagnostic Result: 2026-07-15

Status: integrity-valid, utility gate failed; stop before R/S, held-out, or
CAR full-24 expansion.

## Frozen run

- Entry: exp003 `car_w3_diagnostic`
- Cases: `E1/E2/S3/R4/A5/E6`
- Seeds: `26/27/28`
- Lanes: v33 fallback, W3 graph, shuffled W3, paired-fallback W3, no-action
- Budget: strict `3,000,000` FE per lane/case
- Fresh trajectories: `90/90`
- Frozen code: `e2a65da`
- Raw artifacts: `results/car_w3_diagnostic_6case_seed26_28_3m_20260715`

## Integrity

| Check | Result |
|---|---:|
| Fresh trajectories | PASS, 90/90 |
| Same-budget violations | PASS, 0/90 |
| AOB unchanged | PASS, 885/885 |
| Anti-leakage | PASS, 16/16 |
| CAR dispatch boundary | PASS, 28/28 |
| Branch requested/actual FE | PASS, 68/68 |
| Branch checkpoint fingerprints | PASS, 34/34 paired slots |
| No-plan zero-cost parity | PASS, 10/10 W3 graph runs |

The no-plan runs consumed zero probe FE and matched the v33 final error and FE:
`E1` seeds 26-28, `R4` seeds 27-28, `A5` seeds 27-28, and `E6` seed 26.

## Utility

All paired log deltas below are `log(W3_error / v33_error)`; negative is better.
The offline catastrophic threshold is the frozen relative gain `<= -20%`.

| Metric | Observed |
|---|---:|
| Stable graph plans / graph runs | 8/18 |
| Candidate commits / stable plans | 1/8 |
| Commit coverage / all graph runs | 1/18 |
| Commit cases / overlap strata | 1 case / 1 stratum |
| Probe rows | 14 |
| First-pair futility aborts | 5 |
| Full three-pair probes | 3 |
| Mean normalized probe delta | -0.009892754 |
| Positive probe rows | 7/14 |
| W3-v33 mean log delta, all 18 | +0.019356626 |
| W3-v33 median log delta, all 18 | 0 |
| W3-v33 mean log delta, overlap 15 | +0.023227951 |
| W3-v33 median log delta, overlap 15 | 0 |
| Numeric final-error wins | 2/18 |
| Meaningful wins (>=5% relative) | 0/18 |
| Catastrophic losses (<=-20% relative) | 0/18 |
| Total probe FE | 419,772 |
| Mean probe overhead, all 18 | 0.777356% |
| Mean probe overhead, probed 8 | 1.749050% |
| Maximum probe overhead | 2.998600% |

The only commit was `A5 seed 26`. Its final paired log delta was
`-0.000008147`, effectively a tie; the one-commit probe-to-3M sign agreement is
therefore descriptive only and cannot satisfy the six-commit release gate.

Per-seed paired summaries (numeric wins, not meaningful wins):

| Seed | Mean log delta | Wins | Catastrophic |
|---:|---:|---:|---:|
| 26 | +0.029218818 | 1/6 | 0/6 |
| 27 | -0.000670750 | 1/6 | 0/6 |
| 28 | +0.029521810 | 0/6 | 0/6 |

Seed 28 is the worst seed by mean paired log delta and has no numeric win.
The shuffled-control comparison also failed on `E2`, where the shuffled lane
stably outperformed the graph candidate across the three seeds. The no-action
control had 9/18 catastrophic losses, so it is not a replacement baseline; it
is retained only as a negative control.

## Gate decision

The CAR-W3 release gate fails for three independent reasons:

1. `1/8` commits is below the required six commits across at least three cases
   and two topology strata.
2. The all-run and overlap mean log deltas are positive, so W3 is worse on
   average despite the lower-tail safety gate.
3. The worst-seed and shuffled-control evidence do not support stable utility.

The first-pair futility rule did reduce wasted probe exposure: five candidates
stopped after one pair and mean overhead fell below 1% over all graph runs. It
did not create candidate utility or increase identifiable coverage. This is a
cost-control result, not a performance improvement.

## Decision for July

Do not relax the LCB, lower-tail, endpoint, or catastrophic thresholds. Do not
implement W4+, resource allocation (`R`), or search-start (`S`) channels, and do
not spend the remaining July budget on a CAR held-out or full-24 run.

The defensible paper claim is limited to the protocol:

> ARAC-CAR treats an overlap-derived intervention as a runtime hypothesis and
> calibrates it against the native fallback from the same optimizer checkpoint
> under an equal-FE ledger; W3 can abstain early when the first paired horizon
> is already futile. In this diagnostic, integrity and auditability passed, but
> candidate coverage and long-horizon utility were insufficient for a stable
> performance claim.

This result supports a risk-aware, reference-blind calibration and failure
analysis paper. It does not support a claim of stable improvement, SOTA, or
general superiority over v33.
