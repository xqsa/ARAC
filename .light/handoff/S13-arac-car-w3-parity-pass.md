---
session_no: S13
suggested_title: "[ARAC] S14 CAR-W3 diagnostic preregistration"
parent_session: S12
project: arac
date: 2026-07-15
author: Codex
---

## Current stage

CAR-W3 first-pair futility abort is implemented and passed focused tests plus
real E2 seed9 5k no-plan parity. W1/W2 behavior remains opt-in and unchanged.
No W3 3M diagnostic is registered or running.

Workspace state: local `f22d093` contains the W3 implementation; `origin/main`
is still at `554f7fd` because the latest two push attempts failed to connect to
GitHub port 443. No user-owned untracked files were staged.

## Completed

- Registered `arac_counterfactual_action_racing_w3`.
- Added first-pair fail-closed gate and one-pair ledger accounting.
- Verification: 49 focused tests and 734 tracked tests passed; 1 tracked skip.
- Negative pair 0 aborts before pairs 1-2; positive pair 0 continues to the
  unchanged K=3 LCB/lower-tail gate.
- W3-v33 E2 seed9 5k: final error `6.772894435189659e+12`, FE 4996 for both,
  identical action/AOB hashes, W3 probe FE zero.

## Evidence boundary

W3 was motivated by the W2 observation that 9/11 graph candidates had a
non-positive first pair. That final diagnostic is offline design evidence only.
Runtime W3 sees only its own pair-0 branch contrast and identity-free CAR
evidence. Case labels, paper-best, historical outcomes, final error, and
win/catastrophic labels remain forbidden.

## Next step

1. Freeze a fresh W3 diagnostic with new seeds and an immutable output path.
2. Include v33, W3, shuffled W3, paired-fallback W3, and no-action controls.
3. Gate on integrity, no-plan parity, first-pair abort coverage, commit coverage,
   paired utility, catastrophic losses, and actual lease overhead.
4. Stop before R/S/full24 if the W gate fails.

## Risks

- Early futility reduces probe cost but does not create a stronger candidate.
- Positive pair 0 can still fail later LCB/lower-tail gates and spend 3%.
- Candidate coverage and long-horizon utility remain unproven.
