---
session_no: S20
suggested_title: "[ARAC] S21 offline component lease feasibility replay"
parent_session: S19
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

The preregistered v40 held-out trace-coverage experiment completed on commit
`fedee07`. Raw integrity and v38 parity passed, but the frozen coverage gate
failed because one precision-bearing run had no resolved credit. v40 remains
trace-only and no controller change is authorized.

## Completed

- Added and froze `scripts/audit_component_credit_coverage.py` plus tests before
  new FE. The gate uses run-level coverage, not row volume or outcomes.
- Ran v40 on A4/S2/E2 seeds 31/32/33 and v38 parity anchors on seed31, all at
  strict 3M FE in the pinned environment.
- Integrity: 12/12 fresh, zero FE overspend, v40 AOB `90/90` and v38 AOB
  `30/30` unchanged, anti-leakage `16/16` pass in both arms.
- Parity: 3/3 exact final error and FE, 101 common trace fields with zero
  differences, equal row counts, and equal AOB hashes.
- Coverage: precision in 7/9 runs and cross-seed coverage in all three cases;
  resolved `1668/1803` (`92.51%`), unresolved 135, overwrite coverage in six
  runs/three cases, and 1,796 lock-conflict rows.
- The first audit invocation exposed a parser assumption: exp003 repeats an
  identical action-plan row per seed. The parser now collapses only identical
  repetitions and fails on inconsistent plans; no threshold or raw artifact
  changed.
- Verification: 17 focused tests passed; the full Git-tracked suite passed
  `787 passed, 1 skipped`; compileall, diff check, passport parsing, and the
  offline forbidden-input scan passed.

## Decision

The gate result is FAIL because E2 seed33 emitted 15 precision actions and all
15 ended before their next canonical group revisit. Its first precision action
started at FE 2,995,118 with only 0.1627% budget remaining.

The deeper issue is credit interference: `1796/1803 = 99.61%` precision actions
started while another action in the same overlap component was pending, with a
maximum pending depth of 19. The current per-group action stream therefore does
not provide clean action-level long-horizon attribution.

## Next step

Run an offline eligibility replay only, using the frozen v40 traces:

1. Allow a hypothetical lease only when `component_pending_count == 0`.
2. Require `remaining_fe >= projected_next_revisit_fe`, where the projection is
   estimated from prior completed current-run cycles only.
3. Report retained action/run/case coverage, resolved rate, neighbour harm,
   overwrite/survival, and abstention. Do not use terminal outcomes to tune the
   replay rule.
4. Only if that offline replay preserves useful coverage and closes every
   credit horizon may a new runtime controller be separately preregistered.

## Required reading

1. `.light/handoff/S20-arac-component-credit-coverage-failed.md`
2. `docs/design/state-evidence-sufficiency-audit-20260715.md`
3. `docs/design/core-method.md`
4. `results/controller_v40_component_credit_heldout_audit_20260715/component_credit_gate.csv`
5. `.light/passport.yaml`

## Prohibited

- No case/function/seed/paper-best/history/final-outcome runtime dispatch.
- No controller activation, threshold fitting, additional 3M matrix, or
  full-24 run from the S20 traces.
- No treating 1,803 action rows as independent samples.
- No changing the preregistered S20 gate after observing the result.
