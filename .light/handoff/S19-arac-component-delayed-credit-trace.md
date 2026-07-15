---
session_no: S19
suggested_title: "[ARAC] S20 held-out delayed-credit coverage"
parent_session: S18
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

S19 implemented the v40 component delayed-credit tracker as trace-only
instrumentation. It inherits v38 behavior and does not change dispatch, FE,
RNG, candidates, optimizer state, or AOB inputs. No runtime controller or
threshold is authorized yet.

## Completed

- Added `src/arac/policy/component_delayed_credit.py` with overlap-connected
  component topology, action-specific IDs, proposal disagreement, pending and
  lock-conflict state, revisit resolution, gain/spillover, and shared-variable
  overwrite/survival fields.
- Registered `arac_evidence_action_controller_v40` with v38 behavior and an
  explicit `trace_affects_dispatch: false` boundary. v40 has no v39 sigma
  continuation capability.
- Added schema isolation, CLI parse, action execution plan, and anti-leakage
  tests. Focused result: `386 passed, 1 skipped`; v40/component subset passed.
- Full Git-tracked suite: `782 passed, 1 skipped`. A raw all-files pytest run
  additionally discovered 37 failures confined to user-owned untracked exp008
  tests whose untracked exp005 dependency is absent; those files were not
  changed or included in S19.
- Ran pinned E2/seed1/5k parity:
  `results/controller_v40_component_credit_5k_20260715_final_v38` and
  `results/controller_v40_component_credit_5k_20260715_final_v40`.
- Parity evidence: final error `6.101156e+12` and FE `4996` in both lanes;
  101 common trace fields matched across 20 rows; AOB `10/10` unchanged and
  anti-leakage `16/16` pass per lane; v40 added 19 relation observations.

## Decision

The instrumentation gate is a WARN, not a performance pass. The smoke did not
trigger `post_retirement_precision_reanchor`, so no row reached `resolved` or
`unresolved_run_end`. The tracker is therefore verified for wiring and parity,
not for long-horizon credit validity.

## Next steps

1. Run a small, preregistered held-out coverage probe where v40 actually emits
   search-start precision actions; verify resolved and run-end-unresolved rows,
   FE monotonicity, candidate immutability, and overwrite/survival semantics.
2. Keep v40 trace-only while checking coverage across several cases/seeds. Do
   not fit thresholds, enable credit-gated leases, or run CAR R/S/full-24.
3. If coverage is adequate, freeze one search-start exposure rule with a
   same-budget paired reference and report action coverage, abstention, mean,
   worst seed, and catastrophic loss. Otherwise report the evidence gap and
   keep the July performance result on v33.8.

## Prohibited

- No case/function/seed/paper-best/history/final-outcome runtime dispatch.
- No treating relation/group rows as independent utility samples.
- No shared-variable writeback or resource dispatch changes from this trace.
- No full-24 or 3M matrix until coverage and a preregistered utility gate pass.
