# CAR Actionability Audit Result: 2026-07-15

Status: integrity-valid diagnostic; long-horizon utility gate failed. No
runtime-performance or SOTA claim is made.

Executor: Codex. Frozen commit: `aa1e3f2`.

## Frozen run

- Entry: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke`
- Profile: `car_actionability_audit`
- Cases: `E1/E2/S3/R4/A5/E6`
- Seeds: `1/2/3`
- Lanes: `oracle_fallback` and `oracle_candidate`
- Budget: strict `3,000,000` FE per lane/case
- Raw artifacts: `results/car_actionability_audit_6case_seed123_3m_20260715_v2`
- Protocol: `car-actionability-v2`

The terminal estimand is the common absolute-FE prefix
`max(checkpoint_fe + intervention_fe, max_fes - terminal_tolerance_fe)`. A
terminal row is usable only after a strict post-intervention continuation and
an endpoint within the recorded tolerance. Case identity, function family,
paper-best values, historical outcomes, and final outcomes were not runtime
dispatch inputs.

## Integrity

| Check | Result |
|---|---:|
| Fresh optimizer lanes | PASS, 36/36 provenance artifacts complete and fresh |
| Freeze/provenance | PASS, all requests bind commit `aa1e3f2` and pinned Python 3.12.13 environment |
| Same-budget FE | PASS, 0 violations; observed lane FE `2,999,984..3,000,000` |
| AOB inputs | PASS, 354/354 unchanged |
| Anti-leakage | PASS, 16/16 rows pass |
| Horizon coverage | PASS, 47/47 paired summary rows integrity-clean |
| Actionability gate | PASS, no raw-artifact or pairing blocker |

The 47 paired rows are the reachable horizons under the frozen protocol. Eight
of 18 case-seed pairs had no applicable plan and therefore produce a terminal
control with zero intervention FE; the remaining 10 pairs applied the one-shot
candidate action. The raw trace contains 94 lane-horizon rows.

## Utility diagnosis

The reported contrast is
`Y_h = log(error_fallback,h) - log(error_candidate,h)`. Positive is better for
the candidate. The table below is descriptive only: there are three seeds per
case and 10 applied pairs, so no confirmatory significance claim is made.

| Horizon (applied pairs) | n | Mean log advantage | Median | Numeric wins | Meaningful wins (>=5%) | Catastrophic losses |
|---|---:|---:|---:|---:|---:|---:|
| `closure_1` | 10 | -0.003291 | -0.000011 | 0 | 0 | 0 |
| `budget_3x` | 10 | -0.003302 | -0.000044 | 1 | 0 | 0 |
| `budget_9x` | 9 | -0.008195 | -0.000105 | 0 | 0 | 0 |
| `terminal` | 10 | -0.066855 | 0.000000 | 5 | 0 | 1 |

The applied terminal median is effectively zero, while the mean is dominated
by one S3 seed: relative gain `-1.079679` (candidate error about 2.08x the
fallback error), log advantage `-0.732213`. The best terminal relative gain is
only `+3.5746%`, below the preregistered meaningful-win threshold.

### Per-case terminal view

| Case | Applied n | Mean log advantage | Seed wins | Worst applied log | Catastrophic |
|---|---:|---:|---:|---:|---:|
| A5 | 3 | +0.000597 | 2/3 | -0.000060 | 0 |
| E1 | 0 | not applicable (3 no-plan ties) | 0 | 0 | 0 |
| E2 | 3 | -0.000264 | 1/3 | -0.005964 | 0 |
| E6 | 2 | +0.018200 | 1/2 | 0.000000 | 0 |
| R4 | 0 | not applicable (3 no-plan ties) | 0 | 0 | 0 |
| S3 | 2 | -0.352976 | 1/2 | -0.732213 | 1 |

Across all 18 terminal rows (including the eight zero-intervention no-plan
controls), the mean log advantage is `-0.037142`; the controls are exact ties
and must not be counted as candidate utility.

## Does early evidence predict terminal utility?

Among the 10 applied case-seed pairs with both closure and terminal rows:

- strict non-zero closure/terminal sign agreement: `4/10` (`4/8` among
  pairs that were non-zero at both endpoints);
- non-zero closure/terminal sign reversals: `4/10`;
- Spearman rank correlation: `0.158`;
- Pearson correlation: `-0.101`.

The single `0 -> 0` pair would raise a descriptive equality count to `5/10`,
but it is not counted by the frozen non-zero sign-agreement metric. One
additional pair moved from zero at closure to positive at terminal.

Closure has zero numeric wins (two ties), while terminal has five numeric wins
but one catastrophic loss. This is the direct evidence that the current
overlap/grouping signal can trigger an action but cannot safely predict its
long-horizon payoff. The failure is not an integrity problem and is not a
reason to relax the lower-tail or catastrophic gates.

## Release decision

Do not start the CAR full-24 matrix, R/S channels, or threshold tuning from this
candidate. The preregistered hard stop is met for two independent reasons:

1. applied terminal mean utility is negative and has no meaningful wins;
2. a catastrophic terminal loss remains, with unstable short-to-terminal
   ordering.

The defensible paper claim is narrower:

> ARAC can convert overlap evidence into an auditable, reference-blind,
> same-budget one-shot actionability contrast. In the frozen six-case audit,
> the contrast was integrity-clean but did not provide stable long-horizon
> utility; short-horizon evidence was not a reliable terminal predictor.

The v33.8 full-24 result remains the main performance evidence. This CAR audit
is a mechanism/limitation result, not a replacement performance table.

## Next work

1. Preserve this output directory as immutable raw evidence and update the
   manuscript discussion with the negative result and catastrophic case.
2. Use the already completed v33.8 24-case artifacts for the paper's
   best/mean/worst/seed-win reporting; do not mix CAR oracle labels into the
   runtime claim.
3. If a future method revision is attempted after the paper deadline, require
   a held-out calibration design with at least five seeds, an explicit
   delayed-credit model, and a pre-registered abstention/coverage tradeoff
   before any new full matrix.
