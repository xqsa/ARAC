# Downstream Recovery Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in v34 controller that commits a relation writeback trajectory only when the next CC group recovers past the pre-writeback checkpoint, then verify that it improves full-24 mean and worst-seed wins without losing the v33.8 best-of-three gates.

**Architecture:** A small pure policy module owns immutable recovery checkpoints and commit/restore decisions. The HCC runner owns one pending checkpoint per run, resolves it after the downstream group and before the next relation decision, and emits v34-only trace fields. Existing v33 trust and topology behavior is reused unchanged; exp003 gains a single v34 lane and a derived per-run recovery summary.

**Tech Stack:** Python 3.12, NumPy, pytest, ARAC exp003 HCC adapter, strict same-budget HCC runtime.

---

## File Map

- Create `src/arac/policy/trajectory_guard.py`: immutable checkpoint and pure commit/restore policy.
- Create `tests/test_trajectory_guard.py`: pure policy validation and forbidden-field audit.
- Modify `scripts/hcc_smoke_runner.py`: v34 constant, pending checkpoint lifecycle, CC-loop integration, versioned trace schema, CLI registration.
- Modify `src/arac/actions/contracts.py`: register v34 as trajectory/core intervention.
- Modify `src/arac/backends/hcc_plan.py`: declare v34 optimizer-consumed semantics.
- Modify `src/arac/backends/hcc_shared_writeback.py`: map v34 to the existing changed HCC semantic surfaces.
- Modify `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`: add the single v34 lane, schema propagation, and recovery summary artifact.
- Modify `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/expected_outputs.md`: document v34 trace and summary outputs.
- Modify `tests/test_hcc_smoke_runner_cli.py`: runner lifecycle, v33 compatibility, CLI, schema, and FE tests.
- Modify `tests/test_hcc_action_execution_plan.py`: v34 backend contract.
- Modify `tests/test_exp_003_hcc_runtime_consumer_smoke.py`: v34 profile and summary tests.

### Task 1: Pure Recovery Policy

**Files:**
- Create: `tests/test_trajectory_guard.py`
- Create: `src/arac/policy/trajectory_guard.py`

- [ ] **Step 1: Write failing checkpoint tests**

Add tests that express strict commit, equality/degradation restore, preemption,
copy isolation, validation, and forbidden fields:

```python
from dataclasses import fields

import numpy as np
import pytest

from arac.policy.trajectory_guard import (
    make_recovery_checkpoint,
    preempt_recovery_checkpoint,
    resolve_recovery_checkpoint,
)


def test_strict_downstream_improvement_commits() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0, 2.0]), 10.0)

    resolved = resolve_recovery_checkpoint(
        checkpoint,
        downstream_candidate=np.array([3.0, 4.0]),
        downstream_fitness=8.0,
    )

    assert resolved.status == "committed"
    assert resolved.restored is False
    assert resolved.fitness == 8.0
    assert resolved.effective_delta == pytest.approx(2.0)
    assert resolved.recovery_credit == pytest.approx(0.2)
    np.testing.assert_allclose(resolved.candidate, np.array([3.0, 4.0]))


@pytest.mark.parametrize("downstream_fitness", [10.0, 12.0])
def test_non_improving_downstream_candidate_restores(downstream_fitness: float) -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0, 2.0]), 10.0)

    resolved = resolve_recovery_checkpoint(
        checkpoint,
        downstream_candidate=np.array([3.0, 4.0]),
        downstream_fitness=downstream_fitness,
    )

    assert resolved.status == "restored"
    assert resolved.restored is True
    assert resolved.fitness == 10.0
    assert resolved.effective_delta == 0.0
    np.testing.assert_allclose(resolved.candidate, np.array([1.0, 2.0]))


def test_preemption_restores_without_claiming_recovery() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0]), 10.0)
    resolved = preempt_recovery_checkpoint(checkpoint)

    assert resolved.status == "preempted_restored"
    assert resolved.restored is True
    assert resolved.recovery_credit is None
    np.testing.assert_allclose(resolved.candidate, np.array([1.0]))


def test_checkpoint_isolated_from_source_mutation() -> None:
    source = np.array([1.0, 2.0])
    checkpoint = make_recovery_checkpoint(source, 10.0)
    source[:] = -1.0
    np.testing.assert_allclose(checkpoint.candidate, np.array([1.0, 2.0]))


def test_recovery_policy_rejects_non_finite_and_shape_mismatch() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0]), 10.0)
    with pytest.raises(ValueError, match="shape"):
        resolve_recovery_checkpoint(
            checkpoint,
            downstream_candidate=np.array([1.0, 2.0]),
            downstream_fitness=9.0,
        )
    with pytest.raises(ValueError, match="finite"):
        make_recovery_checkpoint(np.array([np.nan]), 10.0)
    with pytest.raises(ValueError, match="finite"):
        resolve_recovery_checkpoint(
            checkpoint,
            downstream_candidate=np.array([1.0]),
            downstream_fitness=np.nan,
        )


def test_recovery_dataclasses_exclude_forbidden_runtime_fields() -> None:
    from arac.policy.trajectory_guard import RecoveryCheckpoint, RecoveryResolution

    forbidden = {
        "case_id", "problem_id", "function_family", "paper_best",
        "historical_outcome", "final_error", "relative_gain",
    }
    for kind in (RecoveryCheckpoint, RecoveryResolution):
        assert not forbidden.intersection(field.name for field in fields(kind))
```

- [ ] **Step 2: Run the tests and observe RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_trajectory_guard.py -q
```

Expected: collection fails with `ModuleNotFoundError: arac.policy.trajectory_guard`.

- [ ] **Step 3: Implement the pure policy**

Create `src/arac/policy/trajectory_guard.py` with these public types and
functions. Candidate arrays must be copied and made read-only in
`__post_init__`; recovery credit reuses `normalized_objective_credit`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from arac.policy.action_trust_policy import normalized_objective_credit

RecoveryStatus = Literal["committed", "restored", "preempted_restored"]


def _frozen_candidate(value: np.ndarray) -> np.ndarray:
    candidate = np.asarray(value, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(candidate)):
        raise ValueError("recovery candidate must be finite")
    candidate.setflags(write=False)
    return candidate


@dataclass(frozen=True)
class RecoveryCheckpoint:
    candidate: np.ndarray
    fitness: float

    def __post_init__(self) -> None:
        fitness = float(self.fitness)
        if not math.isfinite(fitness):
            raise ValueError("checkpoint fitness must be finite")
        object.__setattr__(self, "candidate", _frozen_candidate(self.candidate))
        object.__setattr__(self, "fitness", fitness)


@dataclass(frozen=True)
class RecoveryResolution:
    candidate: np.ndarray
    fitness: float
    effective_delta: float
    status: RecoveryStatus
    restored: bool
    recovery_credit: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate", _frozen_candidate(self.candidate))


def make_recovery_checkpoint(candidate: np.ndarray, fitness: float) -> RecoveryCheckpoint:
    return RecoveryCheckpoint(candidate=candidate, fitness=fitness)


def resolve_recovery_checkpoint(
    checkpoint: RecoveryCheckpoint,
    *,
    downstream_candidate: np.ndarray,
    downstream_fitness: float,
) -> RecoveryResolution:
    candidate = _frozen_candidate(downstream_candidate)
    if candidate.shape != checkpoint.candidate.shape:
        raise ValueError("downstream candidate shape must match checkpoint shape")
    fitness = float(downstream_fitness)
    if not math.isfinite(fitness):
        raise ValueError("downstream fitness must be finite")
    credit = normalized_objective_credit(checkpoint.fitness, fitness)
    if fitness < checkpoint.fitness:
        return RecoveryResolution(
            candidate=candidate,
            fitness=fitness,
            effective_delta=checkpoint.fitness - fitness,
            status="committed",
            restored=False,
            recovery_credit=credit,
        )
    return RecoveryResolution(
        candidate=checkpoint.candidate,
        fitness=checkpoint.fitness,
        effective_delta=0.0,
        status="restored",
        restored=True,
        recovery_credit=credit,
    )


def preempt_recovery_checkpoint(checkpoint: RecoveryCheckpoint) -> RecoveryResolution:
    return RecoveryResolution(
        candidate=checkpoint.candidate,
        fitness=checkpoint.fitness,
        effective_delta=0.0,
        status="preempted_restored",
        restored=True,
        recovery_credit=None,
    )
```

- [ ] **Step 4: Run the pure policy tests GREEN**

Run the Task 1 command again. Expected: all tests in
`tests/test_trajectory_guard.py` pass.

- [ ] **Step 5: Commit the isolated policy**

```powershell
git add -- src/arac/policy/trajectory_guard.py tests/test_trajectory_guard.py
git diff --cached --check
git commit -m "feat: add downstream recovery checkpoint policy"
```

### Task 2: Explicit v34 Contracts And Trace Schema

**Files:**
- Modify: `src/arac/actions/contracts.py`
- Modify: `src/arac/backends/hcc_plan.py`
- Modify: `src/arac/backends/hcc_shared_writeback.py`
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `tests/test_hcc_action_execution_plan.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Add failing v34 contract tests**

Add tests for CLI parsing, action state opt-in, v34 backend parameters, and
versioned trace fields. The expected v34 HCC plan is:

```python
assert plan.optimizer_consumed_parameters == {
    "relation_runtime_hook": "controller_v33_risk_aware_action_guard",
    "mode_selector": "early_runtime_overlap_relation_evidence_with_relation_first_lock",
    "candidate_relation_policies": ["adaptive_v24", "adaptive_v26"],
    "search_state_runtime_hooks": ["phase_rescue_multistart"],
    "guard": "probation_trust_quarantine_and_exposure_cap",
    "writeback": "topology_scoped_fallback_and_bounded_active_damping",
    "trajectory_guard": "downstream_recovery_checkpoint",
    "dispatch_boundary": "runtime_evidence_only",
}
```

The CLI test must assert all of:

```python
assert args.arac_action == "arac_evidence_action_controller_v34"
assert runner.is_evidence_action_controller_v34(args.arac_action)
assert runner.is_evidence_action_controller(args.arac_action)
```

The schema test must verify that v34 output contains the existing trust fields
plus exactly these new fields, while v33 output excludes them:

```python
V34_RECOVERY_TRACE_FIELDS = {
    "trajectory_guard_status",
    "trajectory_guard_pre_fitness",
    "trajectory_guard_post_writeback_fitness",
    "trajectory_guard_downstream_fitness",
    "trajectory_guard_recovery_credit",
    "trajectory_guard_restored",
}
```

- [ ] **Step 2: Run focused tests and observe RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_action_execution_plan.py tests/test_hcc_smoke_runner_cli.py -q
```

Expected: the new v34 action, predicate, plan, CLI choice, and trace constants
are missing.

- [ ] **Step 3: Register v34 without changing v33**

Make these bounded changes:

```python
# src/arac/actions/contracts.py
ActionSpec("arac_evidence_action_controller_v34", ActionFamily.TRAJECTORY, "core_intervention")
```

```python
# scripts/hcc_smoke_runner.py
EVIDENCE_ACTION_CONTROLLER_V34 = "arac_evidence_action_controller_v34"

def is_evidence_action_controller_v34(action_name: str) -> bool:
    return action_name == EVIDENCE_ACTION_CONTROLLER_V34
```

Include v34 wherever v33 currently enables trust, phase rescue, guarded
relation application, and action-controller classification. Do not add v34 to
`uses_resumable_phase_i_state_during_run`.

Define `V34_RECOVERY_TRACE_FIELDS`, `V33_ACTION_TRACE_FIELDS`, and
`V34_ACTION_TRACE_FIELDS`. Extend `_write_action_trace` with
`include_recovery_fields: bool = False`; select legacy, v33, or v34 fields
without changing the v33 header.

Add v34 to `HCC_ACTION_EXECUTION_BINDINGS` and to the trajectory semantic set
in `hcc_shared_writeback.py`, using the exact contract asserted above.

- [ ] **Step 4: Run the focused tests GREEN**

Run the Task 2 command again. Expected: all existing and new tests pass; the
existing v33 schema compatibility assertion remains green.

- [ ] **Step 5: Commit v34 contracts**

```powershell
git add -- src/arac/actions/contracts.py src/arac/backends/hcc_plan.py src/arac/backends/hcc_shared_writeback.py scripts/hcc_smoke_runner.py tests/test_hcc_action_execution_plan.py tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "feat: register v34 recovery controller contracts"
```

### Task 3: Runner Checkpoint Lifecycle

**Files:**
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Write failing run-state lifecycle tests**

Add tests that construct a v34 run state and prove the lifecycle:

```python
state.register_pending_trajectory_guard(
    candidate=np.array([1.0, 2.0]),
    pre_writeback_fitness=10.0,
    trace_row=trace_row,
)
state.observe_pending_trajectory_guard(post_writeback_fitness=12.0)
resolved = state.resolve_pending_trajectory_guard(
    downstream_candidate=np.array([3.0, 4.0]),
    downstream_fitness=9.0,
)
assert resolved is not None
assert resolved.status == "committed"
assert trace_row["trajectory_guard_status"] == "committed"
assert trace_row["trajectory_guard_restored"] == "0"
```

Add a degraded case that restores exactly, a duplicate-registration error, a
resolve-before-observe error, and preemption that writes
`preempted_restored`. Assert a v33 run state has recovery disabled and never
opens a checkpoint.

- [ ] **Step 2: Run the lifecycle tests and observe RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "trajectory_guard or recovery_checkpoint" -q
```

Expected: missing pending recovery state and methods.

- [ ] **Step 3: Add pending recovery state**

In the runner, add:

```python
@dataclass
class PendingTrajectoryRecovery:
    checkpoint: RecoveryCheckpoint
    trace_row: dict[str, str]
    post_writeback_fitness: float | None = None
```

Extend `EvidenceActionControllerV31RunState` with
`trajectory_guard_enabled: bool` and
`pending_trajectory_recovery: PendingTrajectoryRecovery | None`. Implement
register, observe, resolve, and preempt methods by delegating candidate
semantics to `trajectory_guard.py` and formatting trace floats with `.17e`.

`build_evidence_action_controller_v31_run_state` enables both trust and recovery
for v34, trust only for v33, and neither for v32.

- [ ] **Step 4: Integrate the lifecycle into the CC loop**

Make the order explicit:

1. Before a v34 relation writeback, copy the full `best_individual` and compute
   `pre_writeback_fitness = original_fitness - current_delta`.
2. Register only when `action_value_delta_norm > ACTION_TRUST_MIN_WRITEBACK_NORM`.
3. At the next group start, after its existing objective evaluation, observe
   both immediate trust credit and post-writeback checkpoint fitness.
4. After primary optimization and phase rescue, but before
   `fitness_delta_list.append(current_delta)`, resolve the checkpoint.
5. Set `best_individual`, `original_best`, `original_fitness`, and
   `current_delta` from the resolution so discarded work cannot enter the next
   relation evidence.
6. Before a scheduled search-state action, preempt and restore any unresolved
   checkpoint.
7. At run exit, preempt unresolved state to make the trace complete.

Do not call the objective from checkpoint code. The existing next-group call is
the only post-writeback evaluation.

- [ ] **Step 5: Add a no-extra-FE integration assertion**

Extend the existing fake-HCC v33 credit integration test with a v34 route. For
the same fake optimizer sequence, assert the objective call count and
`current_fitness_evaluations(fun)` are identical between v33 and v34; assert
the v34 trace contains a terminal commit or restore status before the next
relation row is emitted.

- [ ] **Step 6: Run runner and policy tests GREEN**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_trajectory_guard.py tests/test_action_trust_policy.py tests/test_hcc_smoke_runner_cli.py -q
```

Expected: all tests pass, including unchanged v33 tests.

- [ ] **Step 7: Commit runner integration**

```powershell
git add -- scripts/hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "feat: guard v34 downstream trajectory recovery"
```

### Task 4: exp003 Lane And Recovery Summary

**Files:**
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/expected_outputs.md`
- Modify: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`

- [ ] **Step 1: Write failing profile and summary tests**

Assert `lanes_for_profile("evidence_action_controller_v34")` returns exactly one
lane using v34 for lane, selected, runner, and plan action names, with
`controller_v31` relation mode.

Add a pure summary test with committed, restored, preempted, and blank trace
rows. The expected row is:

```python
{
    "run_id": RUN_ID,
    "lane_id": "arac_evidence_action_controller_v34",
    "problem_id": "E2",
    "seed": "1",
    "pending_count": 0,
    "committed_count": 1,
    "restored_count": 1,
    "preempted_restored_count": 1,
    "total_resolved_count": 3,
    "restore_rate": pytest.approx(2.0 / 3.0),
}
```

- [ ] **Step 2: Run exp003 tests and observe RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_003_hcc_runtime_consumer_smoke.py -q
```

Expected: v34 profile, schema, and summary function are absent.

- [ ] **Step 3: Add the v34 lane and derived artifact**

Add `EVIDENCE_ACTION_CONTROLLER_V34_LANES`, profile/CLI choices, and return trust
plus recovery fields from `action_trace_fields_for_lanes` only for v34.

Implement `_trajectory_guard_summary_rows(action_trace_rows)` using a dictionary
keyed by `(run_id, lane_id, problem_id, seed)`. Count only nonblank
`trajectory_guard_status`; calculate restore rate as
`(restored + preempted_restored) / total_resolved` or an empty string when no
rows resolve. Write `trajectory_guard_summary.csv` beside `action_trace.csv` and
list it in the manifest artifacts.

- [ ] **Step 4: Run exp003 and contract tests GREEN**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_hcc_action_execution_plan.py -q
```

Expected: all tests pass and v33 profile/schema assertions remain unchanged.

- [ ] **Step 5: Commit the experiment surface**

```powershell
git add -- experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py experiments/pilots/exp_003_hcc_runtime_consumer_smoke/expected_outputs.md tests/test_exp_003_hcc_runtime_consumer_smoke.py
git diff --cached --check
git commit -m "feat: expose v34 recovery experiment lane"
```

### Task 5: Static Verification And Real-HCC 5k Smoke

**Files:**
- Modify only if a verified implementation defect is found.
- Generate under `results/controller_v34_recovery_5k_smoke_20260714/`.

- [ ] **Step 1: Run focused and full static verification**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_trajectory_guard.py tests/test_action_trust_policy.py tests/test_hcc_action_execution_plan.py tests/test_hcc_smoke_runner_cli.py tests/test_exp_003_hcc_runtime_consumer_smoke.py -q
E:\ARAC\.venv\Scripts\python.exe -m pytest -q
E:\ARAC\.venv\Scripts\python.exe -m compileall -q src scripts experiments
git diff --check
```

Expected: focused and full tests pass, compileall exits 0, and diff check is
clean.

- [ ] **Step 2: Run a real-HCC 5k smoke**

Use one dense and one non-dense case across all three seeds:

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v34_recovery_5k_smoke_20260714 --seeds 1 2 3 --problems R2 S6 --jobs 6 --max-fes 5000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v34
```

Expected: 6/6 fresh executions, zero FE violations, unchanged AOB hashes,
anti-leakage pass, v34 trace fields present, no unresolved `pending` rows at run
exit, and at least one committed or restored recovery row when a nonzero
writeback occurs.

- [ ] **Step 3: Inspect only the relevant diff and status**

```powershell
git status --short --branch
git diff --stat HEAD~4..HEAD
git diff --check HEAD~4..HEAD
```

Confirm no `results/`, cache, log, user paper, FlyKi, exp006-exp008, or external
source material is tracked.

### Task 6: Protected 8-Case 3M Gate

**Files:**
- Generate under `results/controller_v34_recovery_8case_seed123_3m_20260714/`.

- [ ] **Step 1: Run the protected protocol**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v34_recovery_8case_seed123_3m_20260714 --seeds 1 2 3 --problems E2 E4 E6 S6 R1 R2 A4 A5 --jobs 24 --max-fes 3000000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v34
```

Expected protocol gates: 24/24 fresh, zero FE violations, unchanged AOB input
hashes, and anti-leakage pass.

- [ ] **Step 2: Join paper-best offline and enforce 8/8**

Read `our_result_by_case.csv` and join
`references/paper_reported_table2_best_by_case.csv` by case only after the run.
Report every seed, best, mean, worst, seed wins, and catastrophic count. Continue
to Task 7 only when all eight cases retain a strict best-of-three win.

- [ ] **Step 3: Record checkpoint evidence**

Verify `trajectory_guard_summary.csv` covers all 24 case/seed runs, has no
pending rows, and reports commit/restore/preempt counts. Cross-check a sample of
summary counts against `action_trace.csv`.

### Task 6A: Preserve Local CC Evidence On Recovery Commit

**Files:**
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `docs/superpowers/specs/2026-07-14-downstream-recovery-checkpoint-design.md`

- [ ] **Step 1: Write the failing context reconciliation test**

Add a focused test that resolves one committed and one restored checkpoint,
then calls this wished-for runner helper:

```python
best, original, fitness, delta = runner.reconcile_trajectory_recovery_context(
    resolution=committed,
    checkpoint_candidate=np.array([1.0, 2.0]),
    original_best=np.array([2.0, 2.0]),
    original_fitness=12.0,
    current_delta=4.0,
)
np.testing.assert_allclose(best, np.array([3.0, 4.0]))
np.testing.assert_allclose(original, np.array([2.0, 2.0]))
assert fitness == 12.0
assert delta == 4.0
```

For the restored resolution, assert both candidates equal the checkpoint,
fitness equals the checkpoint fitness, and delta is zero. Mutating returned
arrays must not mutate the inputs.

- [ ] **Step 2: Run the test and observe RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "recovery_context" -q
```

Expected: `AttributeError` because
`reconcile_trajectory_recovery_context` does not exist.

- [ ] **Step 3: Implement the pure runner reconciliation helper**

Add the following helper beside the v34 run-state lifecycle:

```python
def reconcile_trajectory_recovery_context(
    *,
    resolution: RecoveryResolution,
    checkpoint_candidate: np.ndarray,
    original_best: np.ndarray,
    original_fitness: float,
    current_delta: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    best = resolution.candidate.copy()
    if not resolution.restored:
        return (
            best,
            np.asarray(original_best, dtype=float).copy(),
            float(original_fitness),
            float(current_delta),
        )
    return (
        best,
        np.asarray(checkpoint_candidate, dtype=float).copy(),
        float(resolution.fitness),
        0.0,
    )
```

Replace the unconditional checkpoint-baseline rewrite in `run_problem` with
this helper. Do not change checkpoint resolution, trust credit, FE accounting,
or trace fields.

- [ ] **Step 4: Run focused tests GREEN**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_trajectory_guard.py tests/test_action_trust_policy.py tests/test_hcc_smoke_runner_cli.py tests/test_exp_003_hcc_runtime_consumer_smoke.py -q
```

Expected: all focused tests pass, including unchanged v33 and no-extra-FE
assertions.

- [ ] **Step 5: Commit the isolated correction**

```powershell
git add -- scripts/hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py docs/superpowers/specs/2026-07-14-downstream-recovery-checkpoint-design.md docs/superpowers/plans/2026-07-14-downstream-recovery-checkpoint.md
git diff --cached --check
git commit -m "fix: preserve local CC evidence on recovery commit"
```

- [ ] **Step 6: Run real-HCC 5k verification**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v34_recovery_local_evidence_5k_20260714 --seeds 1 2 3 --problems R2 S6 --jobs 6 --max-fes 5000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v34
```

Require 6/6 fresh, zero FE violations, unchanged AOB inputs, anti-leakage pass,
and no pending recovery rows.

- [ ] **Step 7: Rerun the protected gate in a new directory**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v34_recovery_local_evidence_8case_seed123_3m_20260714 --seeds 1 2 3 --problems E2 E4 E6 S6 R1 R2 A4 A5 --jobs 24 --max-fes 3000000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v34
```

Preserve the first failed v34 directory. Continue to Task 7 only if the new
run is 24/24 fresh, protocol-clean, and strict best-of-three `8/8`.

### Task 7: Full-24 3M Stability Gate

**Files:**
- Generate under `results/controller_v34_recovery_full24_seed123_3m_20260714/`.
- Modify: `docs/superpowers/specs/2026-07-14-downstream-recovery-checkpoint-design.md` with the final evidence section.
- Modify: `.light/passport.yaml` and create the next `.light/handoff/S03-*.md` at the stage boundary.

- [ ] **Step 1: Run the full configured protocol**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v34_recovery_full24_seed123_3m_20260714 --seeds 1 2 3 --problems E1 E2 E3 E4 E5 E6 S1 S2 S3 S4 S5 S6 R1 R2 R3 R4 R5 R6 A1 A2 A3 A4 A5 A6 --jobs 24 --max-fes 3000000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v34
```

Expected protocol gates: 72/72 fresh, zero FE violations, 708/708 unchanged AOB
rows, anti-leakage pass, backend semantics changed 72/72, and no forbidden
runtime dispatch fields.

- [ ] **Step 2: Produce the offline stability audit**

Using the frozen paper-best CSV only after runtime completion, produce per-case
and aggregate best/mean/worst wins, all 72 seed wins, catastrophic flags, and
trajectory guard counts. The adoption gate is:

```text
protected best-of-three = 8/8
full-24 best-of-three >= 13/24
full-24 mean wins >= 6/24
full-24 worst-seed wins >= 4/24
catastrophic seeds <= 27/72
FE/AOB/anti-leakage/case-dispatch gates all pass
```

- [ ] **Step 3: Update tracked evidence honestly**

Append the exact fresh results, artifact paths, and claim boundary to the v34
design. Update `.light/passport.yaml` and the S03 handoff. If any performance
gate fails, label v34 a failed candidate and preserve the raw artifacts; do not
adopt it as the canonical route or hide losing seeds.

- [ ] **Step 4: Final verification and evidence commit**

```powershell
git status --short --branch
E:\ARAC\.venv\Scripts\python.exe -m pytest -q
git diff --check
git add -- docs/superpowers/specs/2026-07-14-downstream-recovery-checkpoint-design.md .light/passport.yaml .light/handoff/S03-*.md
git diff --cached --check
git commit -m "docs: record v34 stability evidence"
```

Do not stage result payloads or any pre-existing untracked user files. Do not
push until the repository's explicit push confirmation boundary is satisfied.
