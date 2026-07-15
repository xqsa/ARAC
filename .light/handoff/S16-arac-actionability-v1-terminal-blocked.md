---
session_no: S16
suggested_title: "[ARAC] S17 CAR actionability v2 3M validation"
parent_session: S15
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

The first E2 seed1 3M actionability pilot reached a real applied checkpoint,
but protocol v1 correctly blocked terminal pairing because the two natural HCC
population endpoints differed by one FE. Protocol v2 defines a common terminal
absolute-FE prefix and needs fresh validation.

## Completed

- Freeze commit `001d188` is on `origin/main`.
- `results/car_actionability_pilot_E2_seed1_3m_20260715` has 2/2 fresh lanes,
  zero FE overspend, matching prefix/action/CRN/context/AOB evidence, and a real
  one-shot action at checkpoint FE 2,578,640 with 14,993 intervention FE.
- `closure_1`, `budget_3x`, and `budget_9x` are integrity-clean. Candidate log
  advantages are -0.000870, -0.000870, and -0.011043 respectively; these are
  valid pilot diagnostics, not a release result.
- v1 terminal is invalid and blanked: fallback ended at 2,999,986 FE and
  candidate at 2,999,985 FE. No terminal win/loss claim is allowed.
- Protocol v2 sets terminal target to `max(intervention closure, max_fes -
  terminal_completion_tolerance_fe)`, keeps each natural endpoint and shortfall
  in raw metadata, and requires a strict post-closure continuation plus both
  lanes reaching the common target.
- v2 tests: runner/CLI 222 passed and 1 skipped; exp003/adapter 140 passed;
  tracked tests 765 passed and 1 skipped; compileall and diff check passed.
- The v2 semantic coverage validator independently recomputes terminal and
  nested-horizon metadata, enforces zero-cost non-applied controls, and is
  reused by resume acceptance. Any actionability integrity failure redacts
  summary outcome/headroom fields.

## Workspace state

Protocol v2 changes are prepared for commit after this card. User-owned
untracked FlyKI, exp006-008, manuscript, and historical files remain untouched.
Refresh Git status/log before running anything.

## Next steps

1. Commit and push protocol v2, then run a fresh 5k E1/E2 smoke in a new output
   directory and verify the new protocol/commit provenance.
2. Rerun only E2 seed1 fallback/candidate at 3M and require all four horizons,
   including common terminal, to pass integrity.
3. Only after step 2 passes, start the six-case, three-seed matrix; otherwise
   stop and report the repeated runtime-evidence blocker.

## Risks

- v1 pilot suggests the E2 candidate is already slightly worse through 9x;
  this is one seed and must not be generalized, but it lowers expected headroom.
- Common-terminal comparison is a best-so-far prefix at an exact shared FE, not
  either lane's natural population boundary. The natural endpoints remain raw
  audit metadata to prevent semantic overstatement.

## Required reading

1. `.light/handoff/S16-arac-actionability-v1-terminal-blocked.md`
2. `.light/passport.yaml`
3. `docs/design/car-actionability-audit-protocol.md`
4. `results/car_actionability_pilot_E2_seed1_3m_20260715/car_actionability_gate.json`

## Prohibited

- Do not reuse or relabel the v1 terminal row; it is integrity-invalid.
- Do not use offline errors, oracle labels, case identity, paper-best, or final
  outcomes in runtime dispatch.
- Do not start the full six-case matrix before the v2 E2 pair passes.
