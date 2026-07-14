# First-Sweep Evidence Maturity Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in v36 controller that transparently executes existing repair-lock actions and first-sweep-mature sparse coordinate actions while preserving v33.8 for every other relation.

**Architecture:** Extend the existing per-run v31 state with one first-sweep evidence accumulator. The v36 relation executor runs v31 once, chooses either the transparent v31 proposal or the extracted v33 trust/fallback guard, and emits reconstructable maturity fields. The route is isolated behind a new runner action and the existing exp003 lane profile.

**Tech Stack:** Python 3.12, NumPy, dataclasses, pytest, the existing HCC subprocess runner, CSV trace/audit artifacts.

---

## File Map

- Modify `scripts/hcc_smoke_runner.py`: v36 state, pure maturity decision, relation executor, trace fields, CLI route, and runtime integration.
- Modify `src/arac/actions/contracts.py`: register v36 as one trajectory action.
- Modify `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`: expose the single v36 experiment lane and trace schema.
- Modify `tests/test_hcc_smoke_runner_cli.py`: pure boundary, execution, CLI, isolation, and matched-FE tests.
- Modify `tests/test_exp_003_hcc_runtime_consumer_smoke.py`: lane/profile/schema tests.
- Modify `tests/test_package_boundaries.py`: action taxonomy test.
- Modify `docs/superpowers/specs/2026-07-14-first-sweep-evidence-maturity-guard-design.md`: append verified implementation and experiment results only after they exist.
- Modify `.light/passport.yaml` and create `.light/handoff/S05-arac-v36-sweep-maturity-result.md`: record the audited stage outcome without committing generated `results/`.

## Task 1: First-Sweep Maturity State

**Files:**
- Modify: `scripts/hcc_smoke_runner.py`
- Test: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Write failing pure-state tests**

Add tests covering the inclusive boundaries, family consensus, invalid numeric
evidence, and one-time finalization. Use synthetic relations rather than case
labels:

```python
def test_v36_first_sweep_latches_sparse_coordinate_maturity() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(
        0.10,
        action_name=runner.EVIDENCE_ACTION_CONTROLLER_V36,
    )
    for index in range(19):
        relation = _v36_relation(index=index, outer_iter=0, rank_signal=0.55)
        action = _v36_action("coordinate" if index < 5 else "fallback", 0.98)
        state.prepare_v36_outer_iter(0)
        state.observe_v36_relation(relation, action)

    state.prepare_v36_outer_iter(1)

    assert state.coordinate_maturity_latched is True
    assert state.sweep_evidence_reason == "first_sweep_sparse_coordinate_mature"
    assert state.sweep_evidence_relation_count == 19
    assert state.sweep_evidence_active_count == 5
    assert state.sweep_evidence_active_fraction == pytest.approx(5 / 19)
    assert state.sweep_evidence_support == pytest.approx(0.55 * 0.98)
```

Also add parameterized failures for `active_count=3`, fractions below `0.20`
or above `0.30`, mixed coordinate/repair families, support below `0.50`, repair
lock active, non-finite confidence, non-finite rank, and an incomplete outer-0
prefix that has not advanced to outer 1.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "v36_first_sweep" -q
```

Expected: failures because `EVIDENCE_ACTION_CONTROLLER_V36` and the v36 state
methods do not exist.

- [ ] **Step 3: Implement constants and the minimal accumulator**

Add normalized constants and fields to `EvidenceActionControllerV31RunState`:

```python
V36_FIRST_SWEEP_OUTER_ITER = 0
V36_MIN_ACTIVE_COUNT = 4
V36_MIN_ACTIVE_FRACTION = 0.20
V36_MAX_ACTIVE_FRACTION = 0.30
V36_MIN_CONFIDENCE_RANK_SUPPORT = 0.50

v36_enabled: bool = False
sweep_evidence_outer_iter: int | None = None
sweep_evidence_relation_count: int = 0
sweep_evidence_active_count: int = 0
sweep_evidence_active_families: set[str] = field(default_factory=set)
sweep_evidence_support_sum: float = 0.0
sweep_evidence_valid: bool = True
sweep_evidence_finalized: bool = False
coordinate_maturity_latched: bool = False
sweep_evidence_reason: str = ""
```

Implement `prepare_v36_outer_iter()`, `observe_v36_relation()`, and a private
`_finalize_v36_first_sweep()` with these fail-closed semantics:

```python
active_fraction = active_count / relation_count
support = support_sum / active_count
coordinate_maturity_latched = (
    evidence_valid
    and active_count >= V36_MIN_ACTIVE_COUNT
    and V36_MIN_ACTIVE_FRACTION <= active_fraction <= V36_MAX_ACTIVE_FRACTION
    and active_families == {"coordinate"}
    and support >= V36_MIN_CONFIDENCE_RANK_SUPPORT
    and not non_dense_repair_locked
)
```

`observe_v36_relation()` counts every relation and adds `confidence *
relation.rank_signal` only for non-fallback actions. A non-finite or out-of-
range confidence/rank marks the sweep invalid without raising or latching.

Update `build_evidence_action_controller_v31_run_state()` so v36 receives an
`ActionTrustPolicy`, has recovery disabled, and sets `v36_enabled=True`.

- [ ] **Step 4: Run pure-state tests and existing v33-v35 state tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "v36_first_sweep or trust_state_is_opt_in or v34_enables_trust or v35_enables_topology" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the state boundary**

```powershell
git add -- scripts/hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "feat: add v36 first-sweep maturity state"
```

## Task 2: v36 Relation Execution

**Files:**
- Modify: `scripts/hcc_smoke_runner.py`
- Test: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Write failing execution tests**

Add four focused tests:

```python
def test_controller_v36_transparently_executes_repair_lock() -> None:
    runner = _load_runner_module()
    state = _v36_state_with_repair_lock(runner)
    relation, repair = _v36_repair_fixture(runner)

    actual = runner.apply_relation_action_with_controller_v36(
        relation=relation,
        action=repair,
        previous_values=np.array([10.0]),
        current_values=np.array([0.0]),
        previous_delta=1.0,
        current_delta=1.0,
        controller_run_state=state,
    )

    executed, adjusted, norm, trust, fallback, maturity = actual
    assert executed.relation_action_name == "reassign_repair"
    np.testing.assert_allclose(adjusted, np.array([10.0]))
    assert norm == pytest.approx(10.0)
    assert trust is None
    assert fallback == ""
    assert maturity == "repair_lock_transparent"
```

The other tests prove: a matured coordinate row equals v35/v31; an unmatured
coordinate row equals v33 including trust damping; dense and non-dense fallback
rows exactly retain `dense_preserve_v31` and `non_dense_bounded_0_5`.

- [ ] **Step 2: Run execution tests and confirm RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "controller_v36" -q
```

Expected: failures because `apply_relation_action_with_controller_v36()` is
undefined.

- [ ] **Step 3: Extract the post-v31 v33 guard**

Refactor without behavior change:

```python
def apply_v33_guard_to_executed_relation(
    *,
    relation: OverlapRelation,
    executed_action: RelationActionDecision,
    adjusted_values: np.ndarray | None,
    action_value_delta_norm: float,
    current_values: np.ndarray | None,
    controller_run_state: EvidenceActionControllerV31RunState | None,
) -> tuple[RelationActionDecision, np.ndarray | None, float,
           ActionTrustDecision | None, str]:
    policy = (
        None
        if controller_run_state is None
        else controller_run_state.action_trust_policy
    )
    if policy is None or current_values is None or adjusted_values is None:
        return (
            executed_action,
            adjusted_values,
            action_value_delta_norm,
            None,
            "",
        )

    canonical_action_name = _canonical_relation_action_name(executed_action)
    adjusted_values, action_value_delta_norm, fallback_route = (
        apply_topology_scoped_fallback_guard(
            executed_action=executed_action,
            adjusted_values=adjusted_values,
            action_value_delta_norm=action_value_delta_norm,
            current_values=current_values,
            controller_run_state=controller_run_state,
        )
    )
    if fallback_route:
        return (
            executed_action,
            adjusted_values,
            action_value_delta_norm,
            None,
            fallback_route,
        )
    if action_value_delta_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        return (
            executed_action,
            np.asarray(current_values, dtype=float).copy(),
            0.0,
            None,
            "",
        )

    trust_key = make_action_key(
        group_left=relation.group_left,
        group_right=relation.group_right,
        shared_vars=relation.shared_vars,
        canonical_action_name=canonical_action_name,
    )
    trust_decision = policy.decide(trust_key)
    if not trust_decision.allow_intervention:
        fallback_action = RelationActionDecision(
            relation_id=relation.relation_id,
            action_name="fallback",
            action_family="fallback",
            confidence=0.0,
            trigger_reason=f"controller_v33_{trust_decision.reason}",
        )
        return (
            fallback_action,
            np.asarray(current_values, dtype=float).copy(),
            0.0,
            trust_decision,
            "",
        )

    guard_threshold = (
        COORDINATE_ACTION_VALUE_DELTA_GUARD_THRESHOLD
        if canonical_action_name == "allow_beneficial_coordination"
        else ACTION_VALUE_DELTA_GUARD_THRESHOLD
    )
    guarded_values = robust_damped_writeback(
        current_values=np.asarray(current_values, dtype=float),
        proposed_values=np.asarray(adjusted_values, dtype=float),
        blend_strength=trust_decision.blend_strength,
        max_delta_norm=guard_threshold,
    )
    guarded_norm = float(
        np.linalg.norm(
            guarded_values - np.asarray(current_values, dtype=float)
        )
    )
    if guarded_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        policy.rollback_decision(trust_decision)
        return (
            executed_action,
            np.asarray(current_values, dtype=float).copy(),
            0.0,
            None,
            "",
        )
    return executed_action, guarded_values, guarded_norm, trust_decision, ""
```

Make `apply_relation_action_with_controller_v33()` call v31 once and then call
this helper. Run the existing v33 trust/fallback tests before adding v36.

- [ ] **Step 4: Implement the v36 executor**

The new executor must call v31 exactly once:

```python
def apply_relation_action_with_controller_v36(
    relation: OverlapRelation,
    action: RelationActionDecision,
    previous_values: np.ndarray | None = None,
    current_values: np.ndarray | None = None,
    previous_delta: float = 0.0,
    current_delta: float = 0.0,
    controller_run_state: EvidenceActionControllerV31RunState | None = None,
) -> tuple[
    RelationActionDecision,
    np.ndarray | None,
    float,
    ActionTrustDecision | None,
    str,
    str,
]:
    if controller_run_state is not None:
        controller_run_state.prepare_v36_outer_iter(relation.outer_iter)
    executed, adjusted, norm = apply_relation_action_with_controller_v31(
        relation=relation,
        action=action,
        previous_values=previous_values,
        current_values=current_values,
        previous_delta=previous_delta,
        current_delta=current_delta,
        controller_v31_run_state=controller_run_state,
    )
    if controller_run_state is not None:
        controller_run_state.observe_v36_relation(relation, executed)

    canonical = _canonical_relation_action_name(executed)
    if (
        controller_run_state is not None
        and controller_run_state.non_dense_repair_locked
        and canonical == "repair_shared_variable_binding"
    ):
        return executed, adjusted, norm, None, "", "repair_lock_transparent"
    if (
        controller_run_state is not None
        and controller_run_state.coordinate_maturity_latched
        and canonical == "allow_beneficial_coordination"
    ):
        return (
            executed,
            adjusted,
            norm,
            None,
            "",
            "first_sweep_sparse_coordinate_mature",
        )

    protected_action, protected_values, protected_norm, trust, fallback = (
        apply_v33_guard_to_executed_relation(
            relation=relation,
            executed_action=executed,
            adjusted_values=adjusted,
            action_value_delta_norm=norm,
            current_values=current_values,
            controller_run_state=controller_run_state,
        )
    )
    return (
        protected_action,
        protected_values,
        protected_norm,
        trust,
        fallback,
        "",
    )
```

Do not create trust state for transparent rows. A no-op remains a no-op.

- [ ] **Step 5: Run v33-v36 relation tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "controller_v33 or controller_v35 or controller_v36" -q
```

Expected: all selected tests pass with unchanged v33/v35 assertions.

- [ ] **Step 6: Commit relation execution**

```powershell
git add -- scripts/hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "feat: execute v36 maturity-scoped writeback"
```

## Task 3: Trace And Runtime Integration

**Files:**
- Modify: `scripts/hcc_smoke_runner.py`
- Test: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Write failing trace and runtime-routing tests**

Assert all six fields exist and are formatted by `build_action_trace_row()`:

```python
V36_MATURITY_TRACE_FIELDS = [
    "active_maturity_route",
    "sweep_evidence_relation_count",
    "sweep_evidence_active_count",
    "sweep_evidence_active_fraction",
    "sweep_evidence_support",
    "sweep_evidence_reason",
]
```

Add a route test that patches `apply_relation_action_with_controller_v36()` and
proves a v36 `SmokeConfig` reaches it. Add assertions that transparent v36 rows
have empty trust fields and protected rows retain trust fields.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "v36_trace or v36_runtime_route" -q
```

Expected: missing fields and missing route failures.

- [ ] **Step 3: Add trace fields and row arguments**

Add the six fields to `ACTION_TRACE_FIELDS`, define
`V36_MATURITY_TRACE_FIELDS`, and extend `build_action_trace_row()` with explicit
arguments. Empty evidence writes empty strings; finite evidence uses six-decimal
scientific formatting, matching existing trace conventions.

- [ ] **Step 4: Route v36 inside the HCC loop**

Add a v36 branch next to v33/v35, capture `active_maturity_route`, and pass the
state snapshot to `build_action_trace_row()`. Include v36 in:

- guarded-controller membership;
- v33 trust trace schema membership;
- scheduled search-state and phase-rescue membership;
- CLI choices and parser predicates;
- action trace writing with trust plus maturity fields.

Do not add problem-id or family checks.

- [ ] **Step 5: Run routing, schema, and trace tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "v36 or v35 or v34 or v33" -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit runtime integration**

```powershell
git add -- scripts/hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "feat: connect v36 runtime maturity trace"
```

## Task 4: Stable Action And Experiment Interfaces

**Files:**
- Modify: `src/arac/actions/contracts.py`
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- Test: `tests/test_package_boundaries.py`
- Test: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`

- [ ] **Step 1: Write failing contract and lane tests**

Add:

```python
def test_v36_is_registered_as_one_core_trajectory_action() -> None:
    action = action_by_name("arac_evidence_action_controller_v36")
    assert action.family == ActionFamily.TRAJECTORY
    assert action.backend_role == "core_intervention"


def test_exp_003_v36_is_one_maturity_lane() -> None:
    lanes = lanes_for_profile("evidence_action_controller_v36")
    assert len(lanes) == 1
    assert lanes[0].runner_action_name == "arac_evidence_action_controller_v36"
    assert lanes[0].relation_policy_mode == "controller_v31"
```

Also test parser acceptance and that v36 includes trust and maturity trace fields
but excludes recovery fields.

- [ ] **Step 2: Run interface tests and confirm RED**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_package_boundaries.py tests\test_exp_003_hcc_runtime_consumer_smoke.py -k "v36" -q
```

Expected: unknown action/profile failures.

- [ ] **Step 3: Register v36 and expose one lane**

Add exactly one `ActionSpec` and one `LaneConfig`:

```python
ActionSpec(
    "arac_evidence_action_controller_v36",
    ActionFamily.TRAJECTORY,
    "core_intervention",
)
```

```python
EVIDENCE_ACTION_CONTROLLER_V36_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v36",
        ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v36",
        "arac_evidence_action_controller_v36",
        "single_run_first_sweep_evidence_maturity_controller_v36",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v36",
        relation_policy_mode="controller_v31",
    ),
)
```

Wire `lanes_for_profile()`, parser choices, and
`action_trace_fields_for_lanes()` without changing any prior profile.

- [ ] **Step 4: Run interface tests**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_package_boundaries.py tests\test_exp_003_hcc_runtime_consumer_smoke.py -k "v36 or v35" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit stable interfaces**

```powershell
git add -- src/arac/actions/contracts.py experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py tests/test_package_boundaries.py tests/test_exp_003_hcc_runtime_consumer_smoke.py
git diff --cached --check
git commit -m "feat: expose v36 experiment lane"
```

## Task 5: Matched-FE Integration And Regression Suite

**Files:**
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Extend the fake-HCC matched-FE test**

Run v36 after the existing v33/v34/v35 lanes with the same fake objective and
optimizer. Assert:

```python
assert v36_function.objective_calls == v33_objective_calls
assert runner.current_fitness_evaluations(v36_function) == v33_fes
assert len(v36_record) == len(v33_record)
assert all(row.get("trajectory_guard_status", "") == "" for row in v36_rows)
assert any("active_maturity_route" in row for row in v36_rows)
```

Add a static source check that no v36 branch compares `problem_id`, function
name, paper values, relative gain, or final error.

- [ ] **Step 2: Run the matched-FE test and confirm RED if assertions are not wired**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_hcc_smoke_runner_cli.py -k "matched_fe or v36" -q
```

Expected: pass only after runtime integration is complete.

- [ ] **Step 3: Run focused controller verification**

Run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests\test_action_trust_policy.py tests\test_relation_policy.py tests\test_search_state_policy.py tests\test_hcc_smoke_runner_cli.py tests\test_exp_003_hcc_runtime_consumer_smoke.py tests\test_package_boundaries.py -q
```

Expected: all focused tests pass, with only the repository's existing skip.

- [ ] **Step 4: Run all Git-tracked tests and compile check**

Run:

```powershell
$tracked = git ls-files 'tests/test_*.py'
E:\ARAC\.venv\Scripts\python.exe -m pytest $tracked -q
E:\ARAC\.venv\Scripts\python.exe -m compileall -q src scripts experiments\pilots\exp_003_hcc_runtime_consumer_smoke
git diff --check
```

Expected: all tracked tests pass, compileall exits 0, and diff check is clean.

- [ ] **Step 5: Commit integration coverage**

```powershell
git add -- tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "test: cover v36 matched FE runtime path"
```

## Task 6: Real-HCC 5k Smoke

**Files:**
- Generate only: `results/controller_v36_sweep_maturity_5k_20260714/`
- Modify after verification: `docs/superpowers/specs/2026-07-14-first-sweep-evidence-maturity-guard-design.md`

- [ ] **Step 1: Run the registered smoke**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v36_sweep_maturity_5k_20260714 --seeds 1 2 3 --problems S2 S3 A4 R2 --jobs 12 --max-fes 5000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v36
```

- [ ] **Step 2: Audit smoke artifacts**

Verify `12/12` fresh runs, zero `same_budget_violation`, no FE overspend,
`120/120` unchanged AOB hashes, `16/16` anti-leakage pass, non-empty repair
maturity on S3, no maturity route on A4/R2, and both topology fallback routes.
At 5k, S2 may not complete a first sweep; report that as expected rather than
fabricating a latch.

- [ ] **Step 3: Stop on gate failure**

If any provenance, FE, AOB, leakage, or route assertion fails, diagnose and
repair it before starting 3M. Do not weaken the smoke gate.

## Task 7: Current-Winning-13 3M Gate

**Files:**
- Generate only: `results/controller_v36_sweep_maturity_13win_seed123_3m_20260714/`

- [ ] **Step 1: Run 39 fresh trajectories**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v36_sweep_maturity_13win_seed123_3m_20260714 --seeds 1 2 3 --problems A4 A5 E1 E2 E3 E4 E6 R1 R2 S2 S3 S5 S6 --jobs 24 --max-fes 3000000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v36
```

- [ ] **Step 2: Verify runtime artifacts before offline comparison**

Require `39/39` fresh completion, zero FE violation/overspend, unchanged AOB
hashes, anti-leakage pass, one raw trace/decision/mismatch/relations/budget/
manifest/evaluation set per run, and no forbidden runtime dispatch fields.

- [ ] **Step 3: Join frozen paper-best offline**

Reuse the existing offline comparison helper/entry point; do not hand-code a
second paper table. Report per case best, mean, worst, seed-win count, and
catastrophic count.

- [ ] **Step 4: Apply the registered stage gate**

Require best `13/13`, mean at least `5/13`, worst at least `4/13`, seed wins at
least `24/39`, catastrophic at most `9/39`, and route evidence showing only
legal maturity inputs. A failure leaves v33.8 canonical.

- [ ] **Step 5: Decide next experiment without weakening the final target**

If v36 reaches only expected mean `5` and worst `4`, do not run full-24. Record
v36 as a useful but incomplete candidate and design a separately attributable
second stability mechanism. Run full-24 only when the target-13 stage reaches
evidence sufficient for project-level mean `>=6`, worst `>=4`, and catastrophic
`<=27/72`.

## Task 8: Evidence, Documentation, And Handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-07-14-first-sweep-evidence-maturity-guard-design.md`
- Modify: `.light/passport.yaml`
- Create: `.light/handoff/S05-arac-v36-sweep-maturity-result.md`

- [ ] **Step 1: Append only observed results**

Record exact test counts, artifact paths, FE/AOB/leakage audits, route counts,
and performance metrics. Explicitly retain failures and three-seed limits.

- [ ] **Step 2: Update the project truth source**

Add one passport stage for implementation/smoke and one for the 13-case result.
Do not create `.autonomous/` or another experiment ledger; `.light/passport.yaml`
remains the project stage truth source.

- [ ] **Step 3: Verify final tracked changes**

```powershell
git status --short --branch
git diff --check
$tracked = git ls-files 'tests/test_*.py'
E:\ARAC\.venv\Scripts\python.exe -m pytest $tracked -q
```

- [ ] **Step 4: Commit the evidence record**

```powershell
git add -- docs/superpowers/specs/2026-07-14-first-sweep-evidence-maturity-guard-design.md .light/passport.yaml .light/handoff/S05-arac-v36-sweep-maturity-result.md
git diff --cached --check
git commit -m "docs: record v36 sweep maturity result"
```

Do not add `results/`, user-owned untracked files, caches, or logs. Do not push
unless the user's separate no-push boundary is changed explicitly.
