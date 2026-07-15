---
session_no: S22
suggested_title: "[ARAC] S23 close runtime component-lease route and finish paper"
parent_session: S21
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

The final falsifiable scheduler-state correction is complete. The deterministic
revisit cap and runtime component mutex are technically correct, but the v41
paired performance gate failed. This controller route is permanently stopped;
v41 is dispatch-blocked and no full-24 extension is authorized.

## Completed

- S22 action-time scheduler-cap replay used fresh A4/S2/E2 seeds 34/35/36.
  It selected 111 actions in six runs and three cases; 111/111 resolved, with
  zero overlap, cap underprediction, and cap-contract drift.
- Implemented v41 as the existing v38 precision action gated only by the
  shared pure `component mutex + deterministic scheduler cap` eligibility.
  No case, seed, family, paper-best, historical/final outcome, current action
  resolution, gain, overwrite, or survival enters runtime dispatch.
- v38/v41 E2 seed1 5k parity passed: final error 6.101156e+12, FE 4996,
  20 trace rows and 101 common fields with zero difference, AOB 10/10
  unchanged, and anti-leakage 16/16 per arm.
- Fresh v41 and v38 A4/S2/E2 seeds 37/38/39 completed at strict 3M FE. Both
  arms were 9/9 fresh, with zero FE overspend, AOB 90/90 unchanged, and
  anti-leakage 16/16.
- v41 runtime state was exact: 116 selected, 116 resolved, 2,142 mutex
  abstentions, 134 scheduler-horizon abstentions, zero overlap, zero cap
  underprediction, zero contract failure, and zero decision mismatch.
- Verification before FE: 391 focused tests passed with one skip; tracked full
  suite passed 812 tests with one skip. The 37 separately discovered failures
  belong to user-untracked exp008 tests that require a missing exp005 module.

## Frozen failure

The v41 gate failed five fixed conditions:

- selected case coverage was 2, not 3; E2 selected zero leases;
- changed paired runs were 1 win and 5 losses;
- mean paired log advantage was -0.0931242;
- median paired log advantage was -0.000264807;
- S2 seeds 37 and 38 were catastrophic losses.

Per seed, A4 was one win and two tiny losses; E2 was unchanged in all three
seeds; S2 lost all three seeds, including relative gains -38.32% and -56.80%
on seeds 37/38. Do not relax or replace these gates.

## Root cause

The missing state was real but addressed only credit observability. The
deterministic cap answers whether a selected action can reach its semantic
resolution window. The mutex ensures that the observed credit belongs to one
action. Neither answers whether suppressing the other repeated precision
actions improves terminal search. S2 provides the counterexample: the mutex
made attribution clean while removing useful repeated precision exposure and
causing two catastrophic terminal regressions.

This means the remaining instability is not another scheduler-state bug. It
is action-utility non-identifiability: the available pre-action structural and
scheduler state does not rank the long-horizon benefit of action frequency.

## Next step

1. Do not run v41, tune this controller, or start a full-24 matrix.
2. Keep v33.8 as the July paper's full-24 performance result: best-of-three
   13/24, with mean/worst/catastrophic limitations reported honestly.
3. Use v40/S22/v41 as a mechanism result: component scope, credit horizon, and
   action utility are three distinct layers; solving the first two did not
   solve the third.
4. Finish reproducibility packaging, figures, result tables, limitations, and
   manuscript rather than opening another optimizer-controller branch.

## Required reading

1. `.light/handoff/S22-arac-runtime-component-lease-failed.md`
2. `docs/superpowers/specs/2026-07-15-runtime-component-lease-controller-design.md`
3. `results/controller_v41_runtime_lease_audit_20260715/runtime_component_lease_gate.csv`
4. `results/controller_v41_runtime_lease_audit_20260715/runtime_component_lease_paired_performance.csv`
5. `docs/design/state-evidence-sufficiency-audit-20260715.md`
6. `docs/design/core-method.md`

## Prohibited

- No v41 threshold/margin/fallback retuning, seed substitution, case
  substitution, R/S expansion, or full-24 run.
- No case, function, seed, paper-best, history, final error, gain, overwrite,
  resolution, win, or catastrophic label in runtime dispatch.
- No claim that 116/116 lease closure is performance evidence.
