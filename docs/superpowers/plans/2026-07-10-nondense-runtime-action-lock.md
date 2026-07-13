# Non-Dense Runtime Action Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the canonical controller's fixed non-dense v26 route with a canonical-input-validated runtime action route while preserving the dense E6/S6 behavior.

**Architecture:** Keep `canonical_evidence_controller_v1` as the only final entry. First verify that the existing single-trajectory repair/phase-rescue behavior is reachable under the explicit ARAC AOB data root and strict FE accounting; only then update the v31 run state. Dense overlap continues to use the existing three-relation v24/v26 prefix lock.

**Tech Stack:** Python 3.12, pytest, existing HCC MMES/CMAES runner, AOB, exp005.

---

### Task 1: Verify the non-dense candidate under canonical inputs

**Files:**
- Read: `results/nondense_composite_4case_seed123_strict_20260710/our_result_by_case.csv`
- Read: `results/nondense_composite_4case_seed123_strict_20260710/run_manifest.md`

- [x] Run A4/R2/R3/S3 at seeds 1-3, 3M FE, strict accounting, explicit ARAC AOB data root, and the existing `historical_13_runtime_composite` single lane.
- [x] Require 12 fresh executions, no FE violations, unchanged AOB hashes, and anti-leakage pass.
- [x] Continue only if best-of-3 improves the current canonical result on at least three of the four cases and S3 has no catastrophic regression.

### Task 2: Specify the non-dense route with a failing test

Before changing runtime code, run the existing `focused_compare` profile for R3
at seeds 1-3 under the same explicit AOB root and strict accounting. Use this
offline ablation only to establish whether coordinate continuation remains a
reachable candidate; never expose its final errors to runtime dispatch.

Fixed coordinate remained unreachable. The single-trajectory
`historical_13_runtime_composite_v2` CC-harm/NDA candidate reached `3.214242e5`
on R3 seed 3, but a follow-up A4/R2/S3 preservation ablation showed that early
full-budget refresh regresses R2 and S3. Therefore CC-harm takeover is not part
of this runtime-state change. The selected route is the validated non-dense
v26 relation policy with the repair/refine sigma multiplier and phase rescue.

**Files:**
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] Rename the non-dense v31 test to describe the v26 precision route.
- [x] Assert that degree of overlap below `0.18` keeps v26 relation decisions, applies the `0.5` CC sigma multiplier, and keeps phase rescue enabled.
- [x] Run the exact test and verify it fails because the current sigma dispatch does not accept the v31 run state.

### Task 3: Implement the minimal runtime-state change

**Files:**
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py`

- [x] Pass `EvidenceActionControllerV31RunState` into CC sigma refinement and apply the validated multiplier only for non-dense v31 runs.
- [x] Keep the dense prefix lock, relation policy, rescue gate, action names, FE accounting, and schemas unchanged.
- [x] Run the exact unit test and the runner test module.

### Task 4: Verify protocol and regression gates

**Files:**
- Test: `tests/test_relation_policy.py`
- Test: `tests/test_hcc_smoke_runner_cli.py`
- Test: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`
- Test: `tests/test_exp_005_hcc_final_protocol_pilot.py`

- [x] Run the focused canonical suite.
- [x] Run the full pytest suite.
- [x] Run `git diff --check`.

### Task 5: Run dense/non-dense 3M-FE anchors

**Files:**
- Output only: `E:/ARAC/results/nondense_lock_anchor_s3_e6_20260710/`

- [x] Run S3 seed 3 and E6 seed 3 at 3M FE with `canonical_evidence_controller_v1` in the pinned HCC environment.
- [x] Record the failed gate honestly: E6 reached `2.143181e7` and passed, while S3 reached `1.092374e4` and failed the `9.72e3` threshold.

The S3 fixed-action ablation showed that fixed repair (`6.940804e3`) and
repair/refine (`5.222168e3`) are reachable in the same pinned environment.
The canonical S3 prefix is three shared-variable relations with guarded
actions `fallback/fallback/fallback`; the last two are rejected by
`action_value_delta_guard_exceeded`. R3 seed 2 instead ends the same
three-fallback prefix with `high_fallback_margin_keeps_native_overlap_blend`,
while E3 has an early coordinate action. This identifies the missing runtime
selector evidence without using case or function labels.

### Task 5A: Add the non-dense prefix repair lock with TDD

**Files:**
- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py`

- [x] Add a failing test for the exact label-free prefix: three relations with
  three shared variables, all guarded to fallback, with the final two rejected
  by the action-value delta guard.
- [x] Lock subsequent shared-variable writeback to repair only after that
  prefix and bypass the guard that produced the observed rejection loop.
- [x] Record the lock in both the action-decision reason and action-trace policy
  source; do not change dense v31 behavior.
- [x] Run focused and full unit regressions (`222 passed, 1 skipped`; full
  `319 passed, 1 skipped`).

### Task 5B: Verify pinned preservation anchors

**Files:**
- Output only: `E:/ARAC/results/`

- [x] Run S3 seed 3, R3 seed 2, E3 seed 3, and E6 seed 3 at 3M FE.
- [x] Require S3 below `9.72e3`, no prefix-lock activation on R3/E3/E6, and
  no regression of their established controller routes.
- [x] If anchors pass, run pinned A4/R2/R3/S3 seeds 1-3 before the 13-case run.

The first post-prefix implementation failed because it acted only after three
fallback writes had already changed the CC trajectory. Runtime trace comparison
showed that S3's first fallback combined a low contribution delta ratio with a
large writeback norm. The corrected controller locks repair immediately on
that joint evidence and disables phase rescue once repair owns the trajectory.
S3 seed 3 then reproduced `repair_protect_refine` exactly at `5.222168e3`.
E3, E6, and R3 remained bit-identical with zero false locks. The 12-run
preservation batch was fresh and audit-clean; S3 seeds 1-3 reached
`2.447260e3`, `8.651703e3`, and `5.222168e3`.

### Task 5C: Pin the final HCC environment

**Files:**
- Modify: `pyproject.toml`
- Modify: existing `experiments/exp_005_*` final protocol entry and tests

- [x] Pin the validated Python package versions instead of broad lower bounds.
- [x] Add a fail-fast environment audit to the existing final protocol entry.
- [x] Verify that incompatible NumPy/SciPy/Torch versions cannot silently run a
  result-comparable final protocol.

### Task 6: Run the 13-case three-seed regression

**Files:**
- Output only: `E:/ARAC/results/canonical_13target_nondense_lock_seed123_20260710/`

- [ ] Launch 13 cases at seeds 1-3 only after both anchors pass.
- [ ] Report seed-level wins, best-of-3 wins, and 3-seed pilot means separately.
- [ ] Do not claim a 25-run mean or use paper/history values in runtime dispatch.
