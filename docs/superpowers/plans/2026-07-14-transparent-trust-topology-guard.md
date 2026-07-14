# Transparent-Trust Topology Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user pre-authorized inline autonomous execution; do not dispatch subagents or create a second worktree.

**Goal:** Add an opt-in v35 controller that preserves v33.8 topology-scoped fallback protection while making active relation actions transparent to the unsupported exact-key trust policy, then verify higher full-24 mean and worst-seed wins.

**Architecture:** Keep v1-v34 frozen. Add one v35 relation executor that reuses the v31 decision/repair/value guard, applies the existing dense/non-dense fallback guard only to executed fallback actions, and otherwise returns the v31 active writeback unchanged. Register one exp003 lane, reuse v33 trace fields for fallback-route audit, and gate full-24 behind a 10-case protocol.

**Tech Stack:** Python 3.11, NumPy, pytest, ARAC exp003, vendored real HCC/AOB, strict FE ledger, PowerShell.

---

### Task 1: Pure v35 Relation Semantics

**Files:**
- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `scripts/hcc_smoke_runner.py`

- [ ] **Step 1: Write failing v35 state and active-transparency tests**

Add tests beside the existing v33/v34 contracts:

```python
def test_v35_enables_topology_guard_without_trust_or_recovery() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(
        0.10, action_name=runner.EVIDENCE_ACTION_CONTROLLER_V35
    )
    assert state.action_trust_policy is None
    assert state.trajectory_guard_enabled is False


def test_controller_v35_active_action_is_v31_transparent() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(
        0.10, action_name=runner.EVIDENCE_ACTION_CONTROLLER_V35
    )
    relation = runner.OverlapRelation(
        relation_id="O0_0_1", problem_id="runtime_case", outer_iter=0,
        group_left=0, group_right=1, shared_vars=(10,), overlap_strength=1.0,
        delta_signal=0.0, rank_signal=1.0, budget_remaining_ratio=0.8,
        shared_var_count=1,
    )
    action = runner.RelationActionDecision(
        relation_id=relation.relation_id, action_name="coordinate",
        action_family="coordinate", confidence=0.8,
        trigger_reason="runtime_coordinate_candidate",
    )
    expected = runner.apply_relation_action_with_controller_v31(
        relation=relation, action=action, previous_values=np.array([0.0]),
        current_values=np.array([1.0]), previous_delta=1.0,
        current_delta=1.0, controller_run_state=state,
    )
    actual = runner.apply_relation_action_with_controller_v35(
        relation=relation, action=action, previous_values=np.array([0.0]),
        current_values=np.array([1.0]), previous_delta=1.0,
        current_delta=1.0, controller_run_state=state,
    )
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1], expected[1])
    assert actual[2] == pytest.approx(expected[2])
    assert actual[3:] == (None, "")
```

- [ ] **Step 2: Run RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "v35" -q
```

Expected: fail because the v35 constant and executor do not exist.

- [ ] **Step 3: Write failing fallback tests**

Use the same relation/action fixture, with `previous_values=np.array([10.0])`,
`current_values=np.array([0.0])`, `previous_delta=1.0`, and
`current_delta=0.0`. Parameterize:

```python
@pytest.mark.parametrize(
    ("overlap", "expected", "norm", "route"),
    [
        (0.20, np.array([10.0]), 10.0, "dense_preserve_v31"),
        (0.10, np.array([0.5]), 0.5, "non_dense_bounded_0_5"),
    ],
)
def test_controller_v35_keeps_topology_scoped_fallback(
    overlap, expected, norm, route
) -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(
        overlap, action_name=runner.EVIDENCE_ACTION_CONTROLLER_V35
    )
    relation = runner.OverlapRelation(
        relation_id="O0_0_1", problem_id="runtime_case", outer_iter=0,
        group_left=0, group_right=1, shared_vars=(10,), overlap_strength=1.0,
        delta_signal=0.0, rank_signal=1.0, budget_remaining_ratio=0.8,
        shared_var_count=1,
    )
    action = runner.RelationActionDecision(
        relation_id=relation.relation_id, action_name="coordinate",
        action_family="coordinate", confidence=0.8,
        trigger_reason="runtime_coordinate_candidate",
    )
    executed, adjusted, actual_norm, trust, actual_route = (
        runner.apply_relation_action_with_controller_v35(
            relation=relation, action=action,
            previous_values=np.array([10.0]),
            current_values=np.array([0.0]),
            previous_delta=1.0, current_delta=0.0,
            controller_run_state=state,
        )
    )
    assert executed.relation_action_name == "fallback"
    np.testing.assert_allclose(adjusted, expected)
    assert actual_norm == pytest.approx(norm)
    assert trust is None
    assert actual_route == route
```

- [ ] **Step 4: Implement a shared topology fallback helper**

Add `EVIDENCE_ACTION_CONTROLLER_V35` and this pure helper:

```python
def apply_topology_scoped_fallback_guard(
    *, executed_action, adjusted_values, action_value_delta_norm,
    current_values, controller_run_state,
) -> tuple[np.ndarray | None, float, str]:
    route = controller_v33_fallback_route(
        canonical_action_name=_canonical_relation_action_name(executed_action),
        controller_run_state=controller_run_state,
    )
    if not route or current_values is None or adjusted_values is None:
        return adjusted_values, float(action_value_delta_norm), ""
    current = np.asarray(current_values, dtype=float)
    if action_value_delta_norm <= ACTION_TRUST_MIN_WRITEBACK_NORM:
        return current.copy(), 0.0, ""
    if route == "dense_preserve_v31":
        return adjusted_values, float(action_value_delta_norm), route
    bounded = robust_damped_writeback(
        current_values=current,
        proposed_values=np.asarray(adjusted_values, dtype=float),
        blend_strength=1.0,
        max_delta_norm=ACTION_VALUE_DELTA_GUARD_THRESHOLD,
    )
    return bounded, float(np.linalg.norm(bounded - current)), route
```

Refactor v33 to call the helper without changing trust behavior. Add a v35
executor that calls v31, then the helper, and always returns `trust=None`.

- [ ] **Step 5: Run GREEN and commit**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "v33 or v34 or v35 or topology_scoped" -q
git add -- scripts/hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git diff --cached --check
git commit -m "feat: add v35 transparent topology guard"
```

### Task 2: Runner And Action Registration

**Files:**
- Modify: `scripts/hcc_smoke_runner.py`
- Modify: `src/arac/actions/contracts.py`
- Modify: `tests/test_hcc_smoke_runner_cli.py`
- Modify: `tests/test_package_boundaries.py`

- [ ] **Step 1: Write failing CLI and registry tests**

```python
def test_hcc_smoke_runner_parses_explicit_evidence_action_controller_v35() -> None:
    runner = _load_runner_module()
    args = runner.parse_args([
        "--functions", "rastrigin", "--ids", "2", "--seed", "1",
        "--max-fes", "5000", "--output-root", "results/test",
        "--arac-action", "arac_evidence_action_controller_v35",
    ])
    assert args.arac_action == runner.EVIDENCE_ACTION_CONTROLLER_V35
    assert runner.is_evidence_action_controller_v35(args.arac_action)
    assert not runner.is_risk_aware_evidence_action_controller(args.arac_action)
```

Assert in `tests/test_package_boundaries.py` that v35 is a trajectory/core
action and that the legacy re-export still shares the same registry object.

- [ ] **Step 2: Run RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py tests/test_package_boundaries.py -k "v35" -q
```

- [ ] **Step 3: Wire the explicit version**

Add `is_evidence_action_controller_v35` and include v35 in trajectory,
guarded-controller, evidence-controller, phase-rescue, scheduled-search-state,
and CLI choice sets. Keep `is_risk_aware_evidence_action_controller` v33/v34
only. Route v35 to its executor in `run_problem`.

Add the trace-only predicate:

```python
def uses_v33_trust_trace_schema(action_name: str) -> bool:
    return (
        is_risk_aware_evidence_action_controller(action_name)
        or is_evidence_action_controller_v35(action_name)
    )
```

Use it for CSV fields only; it must not create trust state.

- [ ] **Step 4: Register and verify**

Add `ActionSpec("arac_evidence_action_controller_v35",
ActionFamily.TRAJECTORY, "core_intervention")`, then run:

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py tests/test_package_boundaries.py -k "v33 or v34 or v35" -q
git add -- scripts/hcc_smoke_runner.py src/arac/actions/contracts.py tests/test_hcc_smoke_runner_cli.py tests/test_package_boundaries.py
git diff --cached --check
git commit -m "feat: register v35 runtime route"
```

### Task 3: Exp003 Profile And Schema

**Files:**
- Modify: `experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py`
- Modify: `tests/test_exp_003_hcc_runtime_consumer_smoke.py`

- [ ] **Step 1: Write failing profile tests**

```python
def test_exp_003_v35_is_one_topology_lane() -> None:
    lanes = lanes_for_profile("evidence_action_controller_v35")
    assert len(lanes) == 1
    assert lanes[0].runner_action_name == "arac_evidence_action_controller_v35"
    assert lanes[0].relation_policy_mode == "controller_v31"


def test_exp_003_v35_reuses_v33_fields_without_recovery() -> None:
    fields = set(action_trace_fields_for_lanes(
        lanes_for_profile("evidence_action_controller_v35")
    ))
    assert set(V33_TRUST_TRACE_FIELDS).issubset(fields)
    assert not set(V34_RECOVERY_TRACE_FIELDS).intersection(fields)
```

Also assert the CLI accepts `evidence_action_controller_v35`.

- [ ] **Step 2: Run RED**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_003_hcc_runtime_consumer_smoke.py -k "v35" -q
```

- [ ] **Step 3: Add one lane**

```python
EVIDENCE_ACTION_CONTROLLER_V35_LANES = (
    LaneConfig(
        "arac_evidence_action_controller_v35", ActionFamily.TRAJECTORY,
        "arac_evidence_action_controller_v35",
        "arac_evidence_action_controller_v35",
        "single_run_transparent_trust_topology_guard_controller_v35",
        relation_dispatch_enabled=True,
        plan_action_name="arac_evidence_action_controller_v35",
        relation_policy_mode="controller_v31",
    ),
)
```

Wire it into profile selection, CLI choices, and v33 trace fields. Keep recovery
fields v34-only.

- [ ] **Step 4: Run GREEN and commit**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_exp_003_hcc_runtime_consumer_smoke.py -k "v33 or v34 or v35" -q
git add -- experiments/pilots/exp_003_hcc_runtime_consumer_smoke/run.py tests/test_exp_003_hcc_runtime_consumer_smoke.py
git diff --cached --check
git commit -m "feat: expose v35 experiment lane"
```

### Task 4: Integration And Regression Verification

**Files:**
- Modify: `tests/test_hcc_smoke_runner_cli.py`

- [ ] **Step 1: Add matched-FE integration coverage**

Extend the existing synthetic v33/v34 matched-FE fixture with v35. Assert the
same optimizer-call and objective-evaluation counts as v32/v33, no trust key,
no recovery status, and at least one auditable fallback route.

- [ ] **Step 2: Run focused and tracked tests**

```powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_trajectory_guard.py tests/test_action_trust_policy.py tests/test_hcc_smoke_runner_cli.py tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_package_boundaries.py -q
$tracked = @(git ls-files 'tests/test_*.py')
E:\ARAC\.venv\Scripts\python.exe -m pytest -q $tracked
git diff --check
```

Expected: zero failures; untracked exp008 tests remain outside the canonical
tracked set.

### Task 5: Real-HCC 5k Smoke

**Files:**
- Generate: `results/controller_v35_transparent_trust_5k_20260714/`

- [ ] **Step 1: Run 9 trajectories**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v35_transparent_trust_5k_20260714 --seeds 1 2 3 --problems S3 R2 S6 --jobs 9 --max-fes 5000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v35
```

- [ ] **Step 2: Enforce smoke audit**

Require 9/9 fresh, zero FE violations/overspends, unchanged AOB hashes,
anti-leakage 16/16, empty trust/recovery values, and both
`non_dense_bounded_0_5` and `dense_preserve_v31` rows.

### Task 6: Protected 10-Case Gate

**Files:**
- Generate: `results/controller_v35_transparent_trust_10case_seed123_3m_20260714/`

- [ ] **Step 1: Run the protected protocol**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v35_transparent_trust_10case_seed123_3m_20260714 --seeds 1 2 3 --problems E2 E4 E6 S6 R1 R2 A4 A5 S2 S3 --jobs 24 --max-fes 3000000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v35
```

- [ ] **Step 2: Audit before offline comparison**

Require 30/30 fresh, zero FE violations/overspends, all five raw artifact
classes, unchanged AOB hashes, and anti-leakage pass.

- [ ] **Step 3: Join paper-best offline and enforce the gate**

Generate `offline_paper_best_comparison.csv/.md` only after completion. Continue
only if the original eight are best `8/8`, S2/S3 seed wins are `6/6`, aggregate
best `10/10`, mean `>=2/10`, worst `>=2/10`, seed wins `>=15/30`, catastrophic
`<=8/30`, and `runtime_dispatch_used=0` for the offline join.

### Task 7: Full-24 Stability Gate

**Files:**
- Generate conditionally: `results/controller_v35_transparent_trust_full24_seed123_3m_20260714/`
- Modify: `docs/superpowers/specs/2026-07-14-transparent-trust-topology-guard-design.md`
- Modify: `.light/passport.yaml`
- Create: `.light/handoff/S04-arac-v35-*.md`

- [ ] **Step 1: Run full-24 only after Task 6 passes**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\pilots\exp_003_hcc_runtime_consumer_smoke\run.py --output-dir results\controller_v35_transparent_trust_full24_seed123_3m_20260714 --seeds 1 2 3 --problems E1 E2 E3 E4 E5 E6 S1 S2 S3 S4 S5 S6 R1 R2 R3 R4 R5 R6 A1 A2 A3 A4 A5 A6 --jobs 24 --max-fes 3000000 --budget-accounting strict --hcc-root E:\ARAC\vendor\hcc --hcc-runner E:\ARAC\scripts\hcc_smoke_runner.py --lane-profile evidence_action_controller_v35
```

- [ ] **Step 2: Audit and evaluate offline**

Require 72/72 fresh and protocol-clean before paper join. Adoption requires
best `>=13/24`, mean `>=6/24`, worst `>=4/24`, seed wins above `21/72`, and
catastrophic `<=27/72`.

- [ ] **Step 3: Record exact evidence and verify**

Append the result to the spec, passport, and S04 handoff. If a gate fails,
label v35 failed without weakening the goal. Run all tracked tests and
`git diff --check`, then commit only the tracked evidence files.
