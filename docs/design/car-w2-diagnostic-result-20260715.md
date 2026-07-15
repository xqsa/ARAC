# CAR-W2 Diagnostic Result: 2026-07-15

Status: integrity-valid, utility gate failed; no R/S or full-24 expansion.

## Frozen run

- Entry: exp003 `car_w2_diagnostic`
- Cases: `E1/E2/S3/R4/A5/E6`
- Seeds: `22/23/24`
- Lanes: v33 fallback, W2 graph, shuffled W2, paired-fallback W2, no-action
- Budget: strict 3,000,000 FE per lane/case
- Fresh trajectories: 90/90
- Frozen commit: `554f7fd`

## Integrity

| Check | Result |
|---|---:|
| Fresh trajectories | PASS, 90/90 |
| Same-budget violations | PASS, 0/90 |
| AOB unchanged | PASS, 885/885 |
| Anti-leakage | PASS, 16/16 |
| CAR dispatch boundary | PASS, 28/28 |
| Branch requested/actual FE | PASS, 198/198 |
| Branch isolation | PASS, 99 paired branch slots |
| No-plan zero-cost parity | PASS, 7/7 W2 graph runs |

The seven W2 graph runs without a stable plan (`E1` seeds 22-24, `R4` seeds
22-24, `A5` seed 24) used zero probe FE and matched v33 final error and FE.
All three W2 lanes were numerically identical within every case/seed triplet,
so the control branches did not introduce a hidden second trajectory.

## Utility

| Metric | Observed |
|---|---:|
| Stable W2 graph plans | 11/18 graph runs |
| Candidate commits | 0/11 |
| Paired probe observations | 33 |
| Mean normalized probe delta | -0.0215582 |
| Positive probe observations | 6/33 |
| W2-v33 mean log-error delta, all 18 | +0.0842704 |
| W2-v33 median log-error delta, all 18 | 0 |
| W2-v33 mean log-error delta, overlap 15 | +0.101125 |
| Meaningful wins (>=5% relative) | 0/15 |
| Catastrophic losses (<=-20% relative) | 2/15 |
| Max probe overhead | 2.9986% |
| Mean probe overhead | 1.8323% |

The two catastrophic losses are S3 seeds 22 and 23. They are probe-path
opportunity costs: no W2 candidate was committed. This is not evidence that
the LCB or lower-tail gate should be relaxed.

## Root cause

The lazy lease fixes the W1 failure for missing plans. It does not fix the
second failure mode: once a stable coordinate plan appears, W2 spends the full
three-pair lease before learning that the candidate will abstain. In this run,
all 11 graph plans were `allow_beneficial_coordination`, all 11 abstained, and
all 11 consumed roughly 3% probe capacity.

## Decision

CAR-W2 is not released. Do not start R/S, held-out expansion, or full-24 from
this candidate. Do not retune the LCB, tail, endpoint, or catastrophic gates.

The next redesign is CAR-W3: use the first equal-FE paired component horizon as
a preregistered futility stage. If its candidate contrast is non-positive or
the candidate endpoint is worse than the checkpoint, stop after that pair and
adopt the fallback; only a positive first pair may spend the remaining two
pairs. The W2 lazy native prefix and zero-cost no-plan invariant remain fixed.
