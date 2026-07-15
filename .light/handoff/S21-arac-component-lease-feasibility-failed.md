---
session_no: S21
suggested_title: "[ARAC] S22 deterministic scheduler revisit-cap audit"
parent_session: S20
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

The exploratory offline component-lease replay is complete. The component
mutex produced clean, fully closed selected leases, but the prior-cycle
horizon estimator failed the frozen coverage gate and was not a reliable upper
bound. v40 remains trace-only and no runtime controller is authorized.

## Completed

- Added `scripts/replay_component_lease_feasibility.py` and focused tests for
  prior-only cycle projection, component-level history, late-horizon
  abstention, mutex hold/release, and current-action outcome blindness.
- Replayed the frozen S20 9-run v40 matrix without optimizer execution or new
  FE. S20 integrity, v38 parity, AOB, and anti-leakage evidence was required as
  input and passed.
- Original actions: 1,803. Selected: 71. Abstained: 1,732. Every selected lease
  resolved and selected overlap violations were zero.
- Selected coverage was five runs and two cases: A4 seeds 32/33 and S2 seeds
  31/32/33. E2 seed32 and seed33 selected no leases; A4/E2 seed31 had no
  precision actions in the original traces.
- Selected diagnostics: one neighbour-harm row; overwrite/survival in 10 rows,
  including four full overwrites and four full survivals; projection
  underestimation in 13/71 rows, maximum 76,595 FE.
- The gate failed only the frozen coverage minima: at least six selected runs
  and three selected cases.
- Verification: 10 focused tests passed; the complete tracked suite including
  the new tests passed `792 passed, 1 skipped`; compile and diff checks passed.

## Root cause

The mutex is necessary but not sufficient. The horizon estimator assumes that
the latest completed component cycle predicts the next revisit. End-budget
group work is strongly nonstationary. For E2 seed32's first precision action,
remaining FE was 12,433, the prior-cycle component maximum was 342,185, and the
observed revisit delay was only 11,503. The same estimator also underpredicted
13 selected delays, so it is neither a useful late-budget estimate nor a hard
upper cap.

The source check matches the trace diagnosis. Upstream
`E:\HCC-main\2025_HCC_GECCO-main\HCC_SRC\HCC-ES.py` and ARAC
`scripts/hcc_smoke_runner.py` both recompute a uniform group budget from
`ceil(remaining_fes / group_count)` at each sweep. A next-sweep cap should use
that monotone shrinking schedule directly, including all per-group evaluation
overhead and strict-budget break conditions.

This is not evidence for relaxing the horizon threshold after seeing the
outcome. It is evidence that historical duration is the wrong state variable.

## Next step

1. Inspect the canonical v38 scheduler and FE-ledger path to determine whether
   the next semantic revisit has a deterministic action-time upper cap from
   already committed per-group/block allocations.
2. Specify `projected_next_revisit_cap_fe` as a runtime-state quantity. It may
   use only the current ledger, current scheduler position, component topology,
   and already committed block caps; it may not use observed future duration.
3. If the cap is not present in existing traces, add trace-only instrumentation
   and prove v38/v40 parity at CLI/5k before any held-out run.
4. Freeze a new offline coverage gate before replaying or collecting evidence.
   Do not reuse S21 outcomes to fit a margin.

## Required reading

1. `.light/handoff/S21-arac-component-lease-feasibility-failed.md`
2. `scripts/replay_component_lease_feasibility.py`
3. `results/controller_v40_component_lease_feasibility_replay_20260715/component_lease_feasibility_gate.csv`
4. `docs/design/state-evidence-sufficiency-audit-20260715.md`
5. `docs/design/core-method.md`

## Prohibited

- No runtime activation, empirical margin tuning, new 3M matrix, or full-24 run.
- No current-action resolution, gain, neighbour outcome, overwrite, case,
  function, seed, paper-best, historical best, or final outcome in eligibility.
- No claim that 71/71 observed closure proves counterfactual performance or
  future closure.
