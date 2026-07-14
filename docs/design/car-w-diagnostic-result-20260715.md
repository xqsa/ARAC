# CAR-W Diagnostic Result: 2026-07-15

Status: integrity-valid diagnostic, W utility gate failed; no performance or SOTA claim.

Executor: Codex. Frozen commit: `bab6df7`.

## Protocol

The run used `car_w_diagnostic_6case_seed9_11_3m_parityfix_20260715` with
E1/E2/S3/R4/A5/E6, seeds 9/10/11, five lanes (`v33_fallback`, `car_w`,
`car_w_shuffled`, `car_w_paired_fallback`, and `no_action_negative_control`),
strict 3,000,000 FE, and 18 parallel jobs. The pre-registered runtime inputs
were limited to Phase-I/runtime evidence. Case identity, function family,
paper-best values, historical results, and final outcomes were not runtime
dispatch inputs.

## Integrity gates

| Check | Result | Evidence |
|---|---:|---|
| Fresh trajectories | PASS | 90/90 `fresh_execution=1` |
| Same-budget | PASS | 0/90 violations; every branch requested FE equaled actual FE |
| AOB inputs | PASS | 885/885 rows unchanged |
| Anti-leakage | PASS | 16/16 audit rows pass |
| CAR type boundary | PASS | 28/28 audit rows pass |
| Pair branch isolation | PASS | 162 branch rows; 81 complete fallback/candidate pairs; requested=actual FE for every arm |
| No-overlap parity | PASS | E1 CAR lanes emit `no_overlap_component_candidate`, probe FE 0, and match v33 on the same seed |

## W gate

The frozen release gate requires at least six candidate commits over at least
three cases and two topology strata, probe-to-3M sign agreement at least 60%,
negative mean and median paired log-error delta, zero catastrophic losses, and
probe overhead at most 6%.

| Criterion | Observed | Decision |
|---|---:|---|
| Candidate commits | 0 | FAIL (required >=6) |
| Probe coverage | 9 CAR-W runs in A5/E2/S3; 0 commits | FAIL as a release signal |
| Probe-to-3M sign agreement | Not identifiable because commits=0 | FAIL/undefined |
| Mean paired log delta vs v33 | +0.342483 | FAIL (required <0) |
| Median paired log delta vs v33 | +0.270987 | FAIL (required <=0) |
| Meaningful wins | 2/15 overlap pairs | Informational only |
| Catastrophic losses | 9/15 under the repository's -20% relative-gain rule | FAIL (required 0) |
| Probe overhead | 2.999% maximum; 2.998% mean | PASS in isolation, insufficient to offset utility loss |

All 27 CAR-W probe observations abstained. The observed reasons were
`lcb_not_positive`, `lower_tail_negative`, and, for shuffled controls,
`candidate_endpoint_worse_than_start`. The candidate was never deployed.

## Root cause

This is not evidence that the LCB threshold is too strict. The coordinate
candidate itself was not promising in the paired probes: mean normalized probe
delta was negative for A5 (-9.4e-5), E2 (-1.6e-5), and S3 (-2.38e-2).
The safety gate correctly abstained.

The net loss came from paying the probe opportunity cost before knowing that a
candidate could be committed. CAR-W reserves the probe channel and reshapes
the early sweep schedule whenever any overlap edge exists. When the candidate
is rejected, the run still pays that cost. R4 and E6 often never reached a
stable non-fallback action (`mixed_non_fallback_action_family` or
`no_stable_non_fallback_action`), but their CAR lanes still entered the
CAR-shaped route. The three CAR lanes were numerically identical in all 18
per-seed triplets, confirming that no graph candidate was causally deployed in
this diagnostic; the observed final differences are probe-path cost, not a
candidate win.

## Release decision

The result is integrity-valid but utility-invalid. Do not tune thresholds
against these final outcomes. Do not implement R/S channels or start the
held-out/full-24 protocol from this candidate.

## Required redesign before another run

1. Register a lazy probe lease: do not reserve or reshape FE for an overlap
   case until a runtime-evidence candidate plan is available. If no stable plan
   is available, return the untouched v33 state with zero probe FE.
2. Separate opportunity discovery from deployment. The discovery stage must be
   budget-neutral with respect to the v33 prefix; only a preregistered paired
   probe may consume a bounded lease after discovery.
3. Replace the current coordinate-only candidate contract with a preregistered
   candidate set whose effect is tested by a futility stage before the full
   component horizon. Keep the positive-LCB and lower-tail safety gates; do not
   relax them to rescue this result.
4. Re-run CLI/5k parity and a new held-out diagnostic before any R/S or full-24
   work. A new method version needs a new spec, frozen parameters, and a new
   output directory; this result remains immutable evidence for CAR-W v1.
