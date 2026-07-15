---
session_no: S17
suggested_title: "[ARAC] S18 paper completion after CAR hard stop"
parent_session: S16
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

CAR actionability protocol v2 is integrity-valid, but the frozen six-case
utility gate failed. CAR full-24, R/S expansion, and threshold tuning are
stopped. The project should now return to the July paper and reproducibility
package, using v33.8 full-24 as the main performance evidence.

## Completed

- Commit `aa1e3f2` is on `origin/main` and binds the v2 common-terminal,
  provenance, resume, and fail-closed audit semantics.
- `results/car_actionability_smoke_20260715_v7` passed 2/2 fresh 5k lanes,
  with zero FE overspend, AOB unchanged, and anti-leakage pass.
- `results/car_actionability_pilot_E2_seed1_3m_20260715_v2` passed all four
  paired horizons at the common absolute-FE terminal target. Its terminal log
  advantage was `-0.005588`.
- `results/car_actionability_audit_6case_seed123_3m_20260715_v2` completed
  E1/E2/S3/R4/A5/E6, seeds 1/2/3: 36/36 fresh provenance, zero FE overspend,
  AOB 354/354 unchanged, anti-leakage 16/16, and 47/47 paired horizons clean.
- Ten case-seed pairs applied the one-shot candidate; eight had no applicable
  action and zero intervention FE. Applied terminal mean log advantage was
  `-0.066855`, with
  5/10 numeric wins, 0/10 meaningful wins, and one catastrophic loss
  (S3 seed1, log advantage `-0.732213`, relative gain `-1.079679`).
- Strict non-zero closure/terminal sign agreement was 4/10 and Spearman rank
  correlation was 0.158. The current short evidence does not reliably order
  terminal action utility.
- `docs/design/car-actionability-audit-result-20260715.md` records the complete
  audit, root cause, narrow defensible claim, and hard-stop decision.

## Workspace state

The report, passport, and this card are the only files in this handoff scope.
Concurrent commit `bfbb7ed` added the separate exp009 convergence comparison
and must be preserved. User-owned untracked FlyKI, exp006-exp008, manuscript,
historical reports, and reference files remain untouched and must not be
staged without a new audit. Refresh Git status and log before relying on this
card.

## Next steps

1. Freeze the CAR result as a mechanism/limitation result and do not launch a
   CAR full-24 matrix under the failed preregistered gate.
2. Update the manuscript result/discussion around the already completed v33.8
   full-24 evidence: best 13/24, mean 4/24, worst 2/24, seed wins 21/72, and
   catastrophic losses 31/72, with three-seed limitations explicit.
3. Build the paper reproducibility table and artifact index from the frozen
   v33.8 and CAR directories without copying runtime oracle labels into a
   deployable-performance claim.

## Risks

- Three seeds are descriptive pilot evidence, not confirmatory significance.
- Dropping S3 seed1 would make the mean look positive, but that exclusion is
  not allowed; the catastrophic tail is part of the frozen result.
- The failure is utility and delayed-credit mismatch, not implementation
  integrity. Relaxing thresholds or adding case-specific dispatch would hide
  the blocker and violate the runtime boundary.

## Required reading

1. `.light/handoff/S17-arac-actionability-sixcase-hard-stop.md`
2. `.light/passport.yaml`
3. `docs/design/car-actionability-audit-result-20260715.md`
4. `docs/design/car-actionability-audit-protocol.md`
5. `results/car_actionability_audit_6case_seed123_3m_20260715_v2/car_actionability_gate.json`

## Prohibited

- Do not start CAR full-24, R/S expansion, threshold tuning, or critic training
  from this failed six-case gate.
- Do not use case identity, function family, paper-best, historical/final
  outcomes, oracle-selected arms, or offline errors in runtime dispatch.
- Do not relabel actionability integrity pass as a utility or performance pass.
- Do not count no-plan ties as candidate wins or remove catastrophic seeds.
