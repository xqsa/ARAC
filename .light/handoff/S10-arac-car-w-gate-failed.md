---
session_no: S10
suggested_title: "[ARAC] S11 Zero-regret CAR redesign preregistration"
parent_session: S09
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

CAR-W parity-fixed 3M diagnostic is complete. Integrity passes, but the
pre-registered W utility gate fails, so the pipeline is stopped before R/S and
the full 24-case protocol.

## Completed

- `results/car_w_diagnostic_6case_seed9_11_3m_parityfix_20260715` — 90/90
  fresh trajectories; same-budget, AOB, anti-leakage, type boundary, and pair
  isolation audits pass.
- `docs/design/car-w-diagnostic-result-20260715.md` — W gate table, root cause,
  and the required zero-regret redesign.
- `bab6df7` — parity fix committed and pushed to `origin/main`.

## Workspace state

- `main` is pushed through `bab6df7`.
- User-owned untracked FlyKI, exp006-exp008, manuscript, and historical files
  remain untouched.
- The diagnostic results are raw local evidence and are not added to Git.

## Next step

1. Treat CAR-W v1 as a failed but integrity-valid candidate; do not tune its
   thresholds or start R/S/full-24.
2. If development continues, write and freeze a new spec for a lazy, budget-
   neutral probe lease and a preregistered futility stage.
3. Re-run CLI/5k parity before any new 3M diagnostic.

## Blockers and risks

- Candidate commits: 0/required 6.
- Mean/median paired log delta: +0.342483/+0.270987.
- Catastrophic losses: 9/15 under the repository's -20% relative-gain rule.
- All 27 CAR-W probe observations abstained; probe-to-3M agreement is not
  identifiable.

## Must read

1. This card
2. `.light/passport.yaml`
3. `docs/design/core-method.md`
4. `docs/design/car-w-diagnostic-result-20260715.md`
5. `results/car_w_diagnostic_6case_seed9_11_3m_parityfix_20260715/run_manifest.md`

## Forbidden

- Do not treat CAR-W v1 final differences as candidate utility; no candidate
  was committed.
- Do not relax LCB, tail, or catastrophic gates to make this run pass.
- Do not implement R/S channels or launch full-24 from this failed gate.
- Do not use case labels, function families, paper-best, historical results, or
  final outcomes as runtime dispatch inputs.
- Do not overwrite or commit the user's unrelated untracked files.
