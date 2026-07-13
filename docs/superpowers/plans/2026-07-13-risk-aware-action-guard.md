# Risk-Aware Action Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in v33 runtime trust guard that makes shared-variable actions progressively cautious and preserves the canonical v32 behavior.

**Architecture:** Keep relation scoring and v31/v32 execution intact. Add a small pure `action_trust_policy` module for state transitions and bounded writeback, then adapt the runner's v33 path to feed it downstream runtime signals and write audit fields. Add one explicit exp003 v33 lane for the pilot.

**Tech Stack:** Python, NumPy, pytest, existing HCC/AOB runner and CMA-ES backend.

---

### Task 1: Add the pure trust-policy contract

**Files:**
- Create: `src/arac/policy/action_trust_policy.py`
- Test: `tests/test_action_trust_policy.py`

- [x] Write tests for probation damping, trusted promotion, quarantine, cooldown recovery, exposure cap, and bounded writeback.
- [x] Run `pytest tests/test_action_trust_policy.py -q` and verify it fails because the module does not exist.
- [x] Implement `ActionTrustConfig`, `ActionTrustState`, `ActionTrustDecision`, `ActionTrustPolicy`, `make_action_key`, and `robust_damped_writeback` with finite-value validation and no objective calls.
- [x] Run the focused test file and verify it passes.

### Task 2: Add the explicit v33 runner contract

**Files:**
- Modify: `scripts/hcc_smoke_runner.py:ACTION_TRACE_FIELDS`, controller constants, predicates, and state construction
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] Add `arac_evidence_action_controller_v33` as an explicit action and make v33 state opt-in only.
- [x] Add trace columns for trust key, phase, reason, score, exposure, cooldown, credit, and instability.
- [x] Test CLI parsing, action classification, and v32 default preservation.

### Task 3: Integrate relation-level trust and robust writeback

**Files:**
- Modify: `scripts/hcc_smoke_runner.py:EvidenceActionControllerV31RunState`, relation execution helper, and runtime relation loop
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] Add a v33-only trust policy instance to the existing run state.
- [x] Credit a pending relation/action from the next already-evaluated downstream group signal; do not add FE.
- [x] Apply phase-dependent damping and norm clipping after the existing v31 value-delta guard.
- [x] Force protected fallback during quarantine/cooldown/exposure-cap and log the reason.
- [x] Verify v31/v32 paths never instantiate or consult the trust policy.

### Task 4: Add the pilot experiment lane

**Files:**
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- Modify: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/expected_outputs.md`

- [x] Add one `evidence_action_controller_v33` lane using one runtime trajectory and `controller_v31` relation policy.
- [x] Keep the lane out of canonical v32 profiles and document the pilot output fields.
- [x] Run the exp003 focused tests.

### Task 5: Verify and run the first pilot

**Files:**
- Generate only: `results/` outputs, excluded from Git

- [x] Run focused tests and `python -m compileall src scripts experiments`; full pytest and final diff review remain for the commit gate.
- [x] Run real-HCC 5k v33 smokes and verify paired objective credit, exact FE bound, and no leakage.
- [x] Run E2/E4/E6/S6/R1/R2/A4/A5 for seeds 1/2/3 at 3M FE with the existing parallel experiment runner.
- [x] Compare against the frozen paper-best table offline. v33.7 completed 24/24 fresh runs with zero same-budget violations and passed the anti-leakage audit, but reached only 7/8: S6 recovered while R2 regressed.

### Task 6: Scope protected fallback by runtime topology

**Files:**
- Modify: `scripts/hcc_smoke_runner.py:apply_relation_action_with_controller_v33`
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [x] **Step 1: Write the dense-overlap preservation test**

Change `test_controller_v33_preserves_v31_protected_fallback_writeback` to
construct v33 state with degree `0.20`. Keep the expected fallback proposal and
norm unchanged:

```python
state = runner.build_evidence_action_controller_v31_run_state(
    0.20,
    action_name=runner.EVIDENCE_ACTION_CONTROLLER_V33,
)
assert state.dense_overlap is True
assert delta_norm == pytest.approx(10.0)
np.testing.assert_allclose(adjusted, np.array([10.0]))
```

- [x] **Step 2: Write the non-dense bounded-fallback test**

Add a separate test using degree `0.10` and the same fallback-producing
relation. The existing `0.5` guard must be restored only for this topology:

```python
assert state.dense_overlap is False
assert executed.relation_action_name == "fallback"
assert trust is None
assert delta_norm == pytest.approx(0.5)
np.testing.assert_allclose(adjusted, np.array([0.5]))
```

- [x] **Step 3: Run both tests and verify RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "preserves_v31_protected_fallback or bounds_non_dense_protected_fallback" -q
```

Expected: the dense test passes under v33.7 and the new non-dense test fails
with a `10.0` norm, proving the missing topology branch.

- [x] **Step 4: Implement the minimal topology branch**

Keep active coordinate/repair/isolate actions unchanged. For every other v31
result, preserve dense fallback exactly and apply the previous bounded route
only when `controller_run_state.dense_overlap` is false:

```python
if canonical_action_name not in {
    "allow_beneficial_coordination",
    "repair_shared_variable_binding",
    "isolate_conflicting_relation",
}:
    if controller_run_state is not None and controller_run_state.dense_overlap:
        return executed_action, adjusted_values, action_value_delta_norm, None
    if action_value_delta_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        return executed_action, np.asarray(current_values, dtype=float).copy(), 0.0, None
    adjusted_values = robust_damped_writeback(
        current_values=np.asarray(current_values, dtype=float),
        proposed_values=np.asarray(adjusted_values, dtype=float),
        blend_strength=1.0,
        max_delta_norm=ACTION_VALUE_DELTA_GUARD_THRESHOLD,
    )
    return executed_action, adjusted_values, float(
        np.linalg.norm(adjusted_values - np.asarray(current_values, dtype=float))
    ), None
```

- [x] **Step 5: Run focused and full verification**

Run the two topology tests, then the runner, trust-policy, exp003, and full
pytest suites. Run `compileall` and `git diff --check`. Expected: all tests pass
and legacy v32 trace/schema tests remain unchanged.

- [x] **Step 6: Run the real-HCC 5k topology smoke**

Run R2/S6, seeds 1/2/3, strict 5k FE, jobs 6. Verify `6/6` fresh runs,
same-budget violations `0/6`, anti-leakage `16/16`, unchanged AOB hashes, R2
non-dense fallback norm at most `0.5`, and S6 dense fallback norms above `0.5`.

- [x] **Step 7: Run the protected 3M gate**

Run E2/E4/E6/S6/R1/R2/A4/A5, seeds 1/2/3, strict 3M FE, jobs 24. Compare
best-of-three offline with `references/paper_reported_table2_best_by_case.csv`.
Acceptance requires 8/8 wins, 24/24 fresh runs, no FE overrun, unchanged AOB
inputs, and a clean anti-leakage audit.

### Task 7: Review and commit

**Files:**
- Stage only source, tests, and design/plan documentation

- [x] Inspect `git diff` for hidden fallback, duplicate route, forbidden runtime input, and unintended v32 changes.
- [x] Commit with `feat: add risk-aware v33 action guard`.
- [x] Report test and pilot evidence; do not push without explicit user request.
