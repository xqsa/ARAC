# Bounded Late NDA Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one runtime-evidence-driven, budget-bounded NDA refresh to canonical v31, resume CC afterward, and test whether it closes the pinned R3 gap without losing the current 12/13 best-of-three wins.

**Architecture:** Keep `canonical_evidence_controller_v1` and its single optimizer trajectory as the only final entry. Add a pure eligibility/budget planner to the existing v31 run state, reuse the guarded MMES continuation with an explicit bounded budget, then resume the existing CC loop from the protected incumbent. Extend the current trace and FE ledger instead of creating a new runner or result schema.

**Tech Stack:** Python 3.12, NumPy, existing HCC CMAES/MMES backend, pytest, exp_003 artifact aggregation, exp_005 pinned 3M-FE protocol.

---

## File Map

- Modify `HCC_SRC/arac_hcc_smoke_runner.py`: constants, v31 state, pure trigger and budget plan, bounded MMES invocation, CC-loop integration, trace fields, and FE accounting.
- Modify `tests/test_hcc_smoke_runner_cli.py`: pure trigger tests, budget tests, optimizer-result validation, resumed-CC integration, trace assertions, and regression of the existing full-takeover exclusion.
- Modify `experiments/exp_003_hcc_runtime_consumer_smoke/run.py`: preserve the new trace fields when per-run traces are aggregated.
- Modify `tests/test_exp_003_hcc_runtime_consumer_smoke.py`: verify aggregated bounded-refresh audit fields.
- Test `tests/test_exp_005_hcc_final_protocol_pilot.py`: canonical gate and same-budget behavior remain unchanged; no new final entry is added.
- Output only under `E:/ARAC/results/`: pinned pilot and preservation artifacts; these remain ignored by Git.

### Task 1: Add the pure runtime trigger and budget plan

**Files:**
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py:229-292,347-466,885-922`
- Test: `tests/test_hcc_smoke_runner_cli.py:1343-1610`

- [x] **Step 1: Write failing tests for eligible and rejected runtime states**

Add the following helper and tests near the existing controller-v31 tests:

```python
def _bounded_refresh_relation(
    runner,
    *,
    index: int,
    shared_var_count: int = 3,
    budget_remaining_ratio: float = 0.20,
):
    return runner.OverlapRelation(
        relation_id=f"O4_{index}_{index + 1}",
        problem_id="runtime_case",
        outer_iter=4,
        group_left=index,
        group_right=index + 1,
        shared_vars=tuple(range(shared_var_count)),
        overlap_strength=float(shared_var_count),
        delta_signal=0.0,
        rank_signal=0.5,
        budget_remaining_ratio=budget_remaining_ratio,
        previous_delta=0.0,
        current_delta=0.0,
        both_positive=False,
        one_side_zero=False,
        delta_ratio_gap=0.0,
        rank_stability=1.0,
        shared_var_count=shared_var_count,
        shared_var_support_ratio=0.1,
        feature_coverage=1.0,
        fallback_margin_proxy=1.0,
    )


def test_controller_v31_plans_bounded_late_refresh_from_runtime_evidence() -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(0.10)
    relations = [_bounded_refresh_relation(runner, index=index) for index in range(2)]

    plan = runner.plan_bounded_late_nda_refresh(
        controller_v31_run_state=state,
        current_outer_relations=relations,
        fitness_deltas=[0.0, 0.0, 0.0],
        overlap_writeback_norms=[0.0, 0.0],
        reference_fitness=1_000_000.0,
        remaining_fes=600_000,
        max_fes=3_000_000,
        population_size=40,
    )

    assert plan is not None
    assert plan.refresh_budget == 450_000
    assert plan.continuation_reserve == 150_000
    assert plan.remaining_budget_ratio == pytest.approx(0.20)
    assert plan.shared_var_count == 3
    assert plan.trigger_reason == "low_cc_gain+severe_group_stagnation"


@pytest.mark.parametrize(
    ("state_overlap", "shared_count", "remaining_fes", "repair_locked", "deltas"),
    [
        (0.18, 3, 600_000, False, [0.0, 0.0, 0.0]),
        (0.10, 5, 600_000, False, [0.0, 0.0, 0.0]),
        (0.10, 1, 600_000, False, [0.0, 0.0, 0.0]),
        (0.10, 3, 1_050_000, False, [0.0, 0.0, 0.0]),
        (0.10, 3, 150_000, False, [0.0, 0.0, 0.0]),
        (0.10, 3, 600_000, True, [0.0, 0.0, 0.0]),
        (0.10, 3, 600_000, False, [100.0, 100.0, 100.0]),
    ],
)
def test_controller_v31_rejects_nonmatching_bounded_refresh_evidence(
    state_overlap: float,
    shared_count: int,
    remaining_fes: int,
    repair_locked: bool,
    deltas: list[float],
) -> None:
    runner = _load_runner_module()
    state = runner.build_evidence_action_controller_v31_run_state(state_overlap)
    state.non_dense_repair_locked = repair_locked
    relations = [
        _bounded_refresh_relation(runner, index=index, shared_var_count=shared_count)
        for index in range(2)
    ]

    plan = runner.plan_bounded_late_nda_refresh(
        controller_v31_run_state=state,
        current_outer_relations=relations,
        fitness_deltas=deltas,
        overlap_writeback_norms=[0.0, 0.0],
        reference_fitness=1_000_000.0,
        remaining_fes=remaining_fes,
        max_fes=3_000_000,
        population_size=40,
    )

    assert plan is None
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "bounded_late_refresh or bounded_refresh_evidence" -q
```

Expected: failures because `plan_bounded_late_nda_refresh` and its plan type do not exist.

- [x] **Step 3: Implement constants, plan type, state flag, and pure planner**

Add beside the current CC-harm constants and v31 state:

```python
BOUNDED_LATE_NDA_REFRESH_ACTION = "bounded_late_nda_refresh"
BOUNDED_REFRESH_REMAINING_RATIO_MIN = 0.08
BOUNDED_REFRESH_REMAINING_RATIO_MAX = 0.30
BOUNDED_REFRESH_BUDGET_FRACTION = 0.15
BOUNDED_REFRESH_CONTINUATION_FRACTION = 0.05


@dataclass(frozen=True)
class BoundedLateNdaRefreshPlan:
    refresh_budget: int
    continuation_reserve: int
    remaining_budget_ratio: float
    shared_var_count: int
    trigger_reason: str
```

Add `bounded_late_nda_refresh_consumed: bool = False` to
`EvidenceActionControllerV31RunState` and implement:

```python
def plan_bounded_late_nda_refresh(
    *,
    controller_v31_run_state: EvidenceActionControllerV31RunState | None,
    current_outer_relations: list[OverlapRelation],
    fitness_deltas: list[float],
    overlap_writeback_norms: list[float],
    reference_fitness: float,
    remaining_fes: int,
    max_fes: int,
    population_size: int,
) -> BoundedLateNdaRefreshPlan | None:
    state = controller_v31_run_state
    if (
        state is None
        or state.dense_overlap
        or state.non_dense_repair_locked
        or state.bounded_late_nda_refresh_consumed
        or not state.phase_rescue_enabled
        or max_fes <= 0
        or len(fitness_deltas) < CC_HARM_MIN_GROUP_UPDATES
        or len(current_outer_relations) < CC_HARM_MIN_GROUP_UPDATES - 1
    ):
        return None

    shared_counts = {len(relation.shared_vars) for relation in current_outer_relations}
    if shared_counts != {V31_NON_DENSE_PREFIX_SHARED_VAR_COUNT}:
        return None

    remaining_ratio = max(0.0, remaining_fes / max_fes)
    if not (
        BOUNDED_REFRESH_REMAINING_RATIO_MIN
        <= remaining_ratio
        <= BOUNDED_REFRESH_REMAINING_RATIO_MAX
    ):
        return None

    triggered, reason = should_trigger_cc_harm_guard(
        fitness_deltas=fitness_deltas,
        overlap_writeback_norms=overlap_writeback_norms,
        reference_fitness=reference_fitness,
        remaining_fes=remaining_fes,
        minimum_refresh_budget=population_size,
    )
    if not triggered or not (
        "high_relation_conflict" in reason
        or "severe_group_stagnation" in reason
    ):
        return None

    continuation_reserve = math.ceil(
        max_fes * BOUNDED_REFRESH_CONTINUATION_FRACTION
    )
    available_refresh_fes = remaining_fes - continuation_reserve
    if available_refresh_fes < population_size:
        return None
    refresh_cap = math.floor(max_fes * BOUNDED_REFRESH_BUDGET_FRACTION)
    refresh_budget = bounded_population_budget(
        requested_fes=min(refresh_cap, available_refresh_fes),
        remaining_fes=available_refresh_fes,
        population_size=population_size,
    )
    if refresh_budget <= 0:
        return None
    return BoundedLateNdaRefreshPlan(
        refresh_budget=refresh_budget,
        continuation_reserve=continuation_reserve,
        remaining_budget_ratio=remaining_ratio,
        shared_var_count=next(iter(shared_counts)),
        trigger_reason=reason,
    )
```

- [x] **Step 4: Run focused and state regression tests**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "controller_v31 or bounded_refresh" -q
```

Expected: all selected tests pass, including
`test_controller_v31_never_enables_cc_harm_full_budget_takeover`.

- [x] **Step 5: Commit the pure planner**

```powershell
git add HCC_SRC/arac_hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git commit -m "Add bounded late refresh planner"
```

### Task 2: Bound and validate the guarded NDA invocation

**Files:**
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py:925-987`
- Test: `tests/test_hcc_smoke_runner_cli.py:3189-3250`

- [x] **Step 1: Write failing tests for explicit budget, action metadata, and invalid output**

Extend the existing guarded-continuation test with:

```python
def test_guarded_nda_continuation_honors_bounded_budget_and_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    options_seen: list[dict] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if np.asarray(vector).ndim == 1 else len(vector)
            self.fitness_record.extend([700.0] * batch_size)
            return [700.0] * batch_size

    class FakeMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            options_seen.append(dict(self.options))
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 700.0,
                "best_so_far_x": np.zeros(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda _dimension: 4)

    accepted, _candidate, best, used, candidate_best = (
        runner.run_guarded_nda_continuation(
            fun=FakeFunction(),
            info={"dimension": 3, "lower": -5.0, "upper": 5.0},
            config=runner.SmokeConfig(max_fes=100, seed=7, verbose=0),
            fun_name="rastrigin",
            fun_id=3,
            outer_iter=4,
            guard_individual=np.ones(3),
            guard_fitness=800.0,
            remaining_fes=40,
            requested_fes=20,
            search_state_action=runner.BOUNDED_LATE_NDA_REFRESH_ACTION,
        )
    )

    assert accepted is True
    assert best == 700.0
    assert candidate_best == 700.0
    assert used == 20
    assert options_seen[0]["max_function_evaluations"] == 20
    assert options_seen[0]["arac_search_state_action"] == "bounded_late_nda_refresh"
    assert options_seen[0]["seed_rng"] == runner.derive_optimizer_seed(
        7, "rastrigin", 3, 5, 23011
    )


def test_guarded_nda_continuation_rejects_nonfinite_optimizer_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()

    class InvalidMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            return {
                "n_function_evaluations": 4,
                "best_so_far_y": float("nan"),
                "best_so_far_x": np.zeros(3),
            }

    monkeypatch.setattr(runner, "MMES", InvalidMMES)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda _dimension: 4)

    with pytest.raises(RuntimeError, match="guarded NDA returned non-finite fitness"):
        runner.run_guarded_nda_continuation(
            fun=lambda _vector: [1.0],
            info={"dimension": 3, "lower": -5.0, "upper": 5.0},
            config=runner.SmokeConfig(max_fes=100, seed=7, verbose=0),
            fun_name="rastrigin",
            fun_id=3,
            outer_iter=4,
            guard_individual=np.ones(3),
            guard_fitness=800.0,
            remaining_fes=40,
            requested_fes=20,
            search_state_action=runner.BOUNDED_LATE_NDA_REFRESH_ACTION,
        )
```

- [x] **Step 2: Run the two tests and verify RED**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "honors_bounded_budget or rejects_nonfinite_optimizer_output" -q
```

Expected: failures because the new keyword arguments and validation do not exist.

- [x] **Step 3: Extend the existing guarded continuation without changing old defaults**

Change the signature and budget/action construction:

```python
def run_guarded_nda_continuation(
    *,
    fun,
    info: dict,
    config: SmokeConfig,
    fun_name: str,
    fun_id: int,
    outer_iter: int,
    guard_individual: np.ndarray,
    guard_fitness: float,
    remaining_fes: int,
    requested_fes: int | None = None,
    search_state_action: str = CC_HARM_GUARDED_SEP_REFRESH_ACTION,
) -> tuple[bool, np.ndarray, float, int, float]:
    population_size = calculate_cmaes_population_size(int(info["dimension"]))
    requested_budget = remaining_fes if requested_fes is None else min(
        remaining_fes,
        max(0, int(requested_fes)),
    )
    refresh_budget = bounded_population_budget(
        requested_fes=requested_budget,
        remaining_fes=remaining_fes,
        population_size=population_size,
    )
```

Set `options["arac_search_state_action"] = search_state_action`. After optimize,
validate the result before acceptance:

```python
candidate_best = float(results["best_so_far_y"])
candidate = np.asarray(results["best_so_far_x"], dtype=float).reshape(-1)
used_fes = int(results["n_function_evaluations"])
if used_fes < 0 or used_fes > refresh_budget:
    raise RuntimeError("guarded NDA reported invalid FE usage")
if not math.isfinite(candidate_best):
    raise RuntimeError("guarded NDA returned non-finite fitness")
if candidate.shape != np.asarray(guard_individual).reshape(-1).shape:
    raise RuntimeError("guarded NDA returned invalid candidate shape")
if not np.all(np.isfinite(candidate)):
    raise RuntimeError("guarded NDA returned non-finite candidate")
```

Use `candidate.copy()` in the accepted return and retain the current incumbent
for a finite but non-improving candidate.

- [x] **Step 4: Run new and existing guarded-continuation tests**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "guarded_nda_continuation or cc_harm_guarded_sep_refresh" -q
```

Expected: all selected tests pass; the existing full-budget action still uses all
remaining FE because its new arguments use defaults.

- [x] **Step 5: Commit the bounded optimizer call**

```powershell
git add HCC_SRC/arac_hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git commit -m "Bound guarded NDA continuation budget"
```

### Task 3: Integrate the bounded action and resume canonical CC

**Files:**
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py:251-267,2177-2962`
- Test: `tests/test_hcc_smoke_runner_cli.py:3070-3250`

- [x] **Step 1: Write a failing integration test for one refresh followed by CC**

Add a compact fake-run test that forces the pure planner once and records the
optimizer call order:

```python
def test_controller_v31_runs_one_bounded_refresh_then_resumes_cc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    call_order: list[str] = []

    class FakeFunction:
        def __init__(self) -> None:
            self.fitness_record: list[float] = []

        def __call__(self, vector):
            batch_size = 1 if np.asarray(vector).ndim == 1 else len(vector)
            self.fitness_record.extend([1000.0] * batch_size)
            return [1000.0] * batch_size

    class FakeBenchmark:
        def __init__(self, output_dir: str, data_dir=None) -> None:
            self.output_dir = output_dir

        def get_function(self, _fun_name: str, _fun_id: int):
            return FakeFunction()

        def get_info(self, _fun_name: str, _fun_id: int):
            return {"dimension": 5, "lower": -5.0, "upper": 5.0}

    class FakeMMES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            call_order.append("refresh")
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 700.0,
                "best_so_far_x": np.ones(self.problem["ndim_problem"]),
            }

    class FakeCMAES:
        def __init__(self, problem, options) -> None:
            self.problem = problem
            self.options = options

        def optimize(self):
            call_order.append("cc")
            budget = self.options["max_function_evaluations"]
            self.problem["fitness_function"](
                np.zeros((budget, self.problem["ndim_problem"]))
            )
            return {
                "n_function_evaluations": budget,
                "best_so_far_y": 650.0,
                "best_so_far_x": np.zeros(self.problem["ndim_problem"]),
                "mean": np.zeros(self.problem["ndim_problem"]),
            }

    monkeypatch.setattr(runner, "Benchmark", FakeBenchmark)
    monkeypatch.setattr(runner, "MMES", FakeMMES)
    monkeypatch.setattr(runner, "CMAES", FakeCMAES)
    monkeypatch.setattr(
        runner,
        "decompose_problem",
        lambda _fun_id, data_root=None: [[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    monkeypatch.setattr(
        runner,
        "remove_overlapping_groups",
        lambda grouping: (grouping, [[1], [2], [3]], [[1], [2], [3]]),
    )
    monkeypatch.setattr(
        runner,
        "load_aob_metadata",
        lambda _fun_id, data_root=None: {
            "dimension": 5,
            "overlap_degree": 1,
            "subgroups": [2, 2, 2, 2],
        },
    )
    monkeypatch.setattr(runner, "calculate_global_fes", lambda _max_fes, _degree: 0)
    monkeypatch.setattr(runner, "calculate_cmaes_population_size", lambda _dimension: 4)
    planner_calls = 0

    def fake_plan(**kwargs):
        nonlocal planner_calls
        planner_calls += 1
        state = kwargs["controller_v31_run_state"]
        if state.bounded_late_nda_refresh_consumed:
            return None
        return runner.BoundedLateNdaRefreshPlan(
            refresh_budget=20,
            continuation_reserve=8,
            remaining_budget_ratio=0.20,
            shared_var_count=3,
            trigger_reason="low_cc_gain+high_relation_conflict",
        )

    monkeypatch.setattr(runner, "plan_bounded_late_nda_refresh", fake_plan)

    _record, _elapsed, trace_rows = runner.run_problem(
        "rastrigin",
        3,
        tmp_path,
        runner.SmokeConfig(
            max_fes=160,
            seed=3,
            verbose=0,
            arac_action=runner.EVIDENCE_ACTION_CONTROLLER_V31,
            enable_relation_dispatch=True,
            relation_policy_mode="controller_v31",
        ),
    )

    refresh_index = call_order.index("refresh")
    assert "cc" in call_order[refresh_index + 1 :]
    bounded_rows = [
        row for row in trace_rows
        if row["selected_action_name"] == runner.BOUNDED_LATE_NDA_REFRESH_ACTION
    ]
    assert len(bounded_rows) == 2
    assert bounded_rows[0]["bipop_restart_mode"].startswith(
        "bounded_late_nda_refresh:start"
    )
    assert bounded_rows[1]["bipop_restart_mode"] == (
        "bounded_late_nda_refresh:completion"
    )
    assert bounded_rows[0]["restart_accepted"] == "1"
    assert planner_calls >= 1
```

- [x] **Step 2: Run the integration test and verify RED**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py::test_controller_v31_runs_one_bounded_refresh_then_resumes_cc -q
```

Expected: failure because the action is not connected to the run loop.

- [x] **Step 3: Connect the planner without enabling full CC-harm takeover**

Add `BOUNDED_LATE_NDA_REFRESH_ACTION` to `TRAJECTORY_ACTION_NAMES`. Keep
`uses_cc_harm_guard_during_run(EVIDENCE_ACTION_CONTROLLER_V31, ...)` returning
`False` so the old all-remaining-budget branch stays disabled.

Add a narrow predicate and dedicated audit semantics so this internal action is
not mislabeled as generic trajectory mean blending:

```python
def is_bounded_late_nda_refresh_action(action_name: str) -> bool:
    return action_name == BOUNDED_LATE_NDA_REFRESH_ACTION
```

Return `"bounded_guarded_incumbent_refresh"` from `_owner_selected` and
`"bounded_late_nda_refresh_and_cc_continuation"` from `_semantic_surface` before
their generic trajectory branches when this predicate is true.

Inside `run_problem`, initialize:

```python
bounded_refresh_completion: dict[str, object] | None = None
```

After a relation row has been added and before the existing full CC-harm block,
select the protected incumbent exactly as the existing guard does, then call the
pure planner for canonical v31. When a plan exists:

```python
refresh_seed = derive_optimizer_seed(
    config.seed,
    fun_name,
    fun_id,
    outer_iter + 1,
    23011,
) if config.seed is not None else None
accepted, refreshed_individual, refreshed_best, used_fes, candidate_best = (
    run_guarded_nda_continuation(
        fun=fun,
        info=info,
        config=config,
        fun_name=fun_name,
        fun_id=fun_id,
        outer_iter=outer_iter,
        guard_individual=guard_individual,
        guard_fitness=guard_fitness,
        remaining_fes=remaining_fes,
        requested_fes=plan.refresh_budget,
        search_state_action=BOUNDED_LATE_NDA_REFRESH_ACTION,
    )
)
sum_fes += used_fes
refresh_fe += used_fes
best_individual = refreshed_individual.copy()
guarded_incumbent = best_individual.copy()
guarded_incumbent_fitness = refreshed_best
controller_v31_run_state.bounded_late_nda_refresh_consumed = True
```

Append the start trace row with a `bipop_restart_mode` prefix of
`bounded_late_nda_refresh:start`, save the values needed for the completion row,
and `break` only the current group loop. Do not set `cc_harm_guard_consumed`; the
outer `while` must resume CC with the reserved FE.

After the `while` loop, append a completion row using
`min(float(value) for value in fun.fitness_record)` as the post-continuation best.
Set its existing `bipop_restart_mode` field to
`bounded_late_nda_refresh:completion`. This reads existing fitness evidence and
must not call the objective again. Task 4 enriches both rows with dedicated audit
fields after the integration behavior is green.

- [x] **Step 4: Run integration, v31, and old CC-harm tests**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "bounded_refresh or controller_v31 or cc_harm_guarded" -q
```

Expected: all selected tests pass; the integration call order contains CC after
refresh; the full-takeover exclusion test remains green.

- [ ] **Step 5: Commit runtime integration**

```powershell
git add HCC_SRC/arac_hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git commit -m "Integrate bounded refresh into canonical v31"
```

### Task 4: Extend trace aggregation and FE audit

**Files:**
- Modify: `HCC_SRC/arac_hcc_smoke_runner.py:103-142,1475-1566`
- Modify: `experiments/exp_003_hcc_runtime_consumer_smoke/run.py:3647-3692`
- Test: `tests/test_hcc_smoke_runner_cli.py:1612-1645`
- Test: `tests/test_exp_003_hcc_runtime_consumer_smoke.py:390-420`

- [x] **Step 1: Write failing schema and aggregation assertions**

Add these fields to the expected trace subset in both runner and exp_003 tests:

```python
{
    "trace_event",
    "remaining_budget_ratio",
    "shared_var_count",
    "repair_lock_active",
    "refresh_budget",
    "continuation_reserve",
    "optimizer_seed",
}
```

For a constructed bounded start row, assert:

```python
assert row["trace_event"] == "start"
assert row["remaining_budget_ratio"] == "2.000000e-01"
assert row["shared_var_count"] == "3"
assert row["repair_lock_active"] == "0"
assert row["refresh_budget"] == "450000"
assert row["continuation_reserve"] == "150000"
assert row["optimizer_seed"] == "12345"
```

- [x] **Step 2: Run the two schema tests and verify RED**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -k "action_trace" tests/test_exp_003_hcc_runtime_consumer_smoke.py::test_exp_003_writes_runtime_consumer_smoke_artifacts -q
```

Expected: failures because the new fields are absent or dropped during aggregation.

- [x] **Step 3: Extend the existing trace row and aggregator**

Append the seven field names to `ACTION_TRACE_FIELDS`, add keyword parameters to
`build_action_trace_row`, and serialize them as follows:

```python
"trace_event": trace_event,
"remaining_budget_ratio": "" if remaining_budget_ratio is None else f"{remaining_budget_ratio:.6e}",
"shared_var_count": "" if shared_var_count is None else str(shared_var_count),
"repair_lock_active": "" if repair_lock_active is None else str(int(repair_lock_active)),
"refresh_budget": "" if refresh_budget is None else str(refresh_budget),
"continuation_reserve": "" if continuation_reserve is None else str(continuation_reserve),
"optimizer_seed": "" if optimizer_seed is None else str(optimizer_seed),
```

Replace the hardcoded downstream scope with a backward-compatible keyword:

```python
downstream_consumption_scope: str = "same_outer_iteration"
```

Serialize the supplied value. Bounded-refresh start rows pass
`"subsequent_outer_iterations"`; completion rows pass `"run_completion"`.
Existing relation rows retain the default.

Append the same names to the exp_003 `_write_csv(... action_trace.csv ...)`
field list. Do not create another trace artifact.

- [x] **Step 4: Verify budget reconciliation and canonical gate tests**

Run:

```powershell
D:\python\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_exp_005_hcc_final_protocol_pilot.py -q
```

Expected: all selected tests pass. `refresh_fe` includes bounded refresh FE,
`cc_phase_fe` includes resumed CC, and `same_budget_violation` remains `0`.

- [ ] **Step 5: Commit trace and audit integration**

```powershell
git add HCC_SRC/arac_hcc_smoke_runner.py experiments/exp_003_hcc_runtime_consumer_smoke/run.py tests/test_hcc_smoke_runner_cli.py tests/test_exp_003_hcc_runtime_consumer_smoke.py
git commit -m "Audit bounded refresh runtime state"
```

### Task 5: Run code-level regression gates

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run the focused canonical suite**

```powershell
D:\python\python.exe -m pytest tests/test_relation_policy.py tests/test_hcc_smoke_runner_cli.py tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_exp_005_hcc_final_protocol_pilot.py tests/test_exp_005_hcc_final_protocol_pilot_cli.py -q
```

Expected: zero failures; the existing skipped integration marker may remain skipped.

- [ ] **Step 2: Run the full test suite**

```powershell
D:\python\python.exe -m pytest -q
```

Expected: zero failures.

- [ ] **Step 3: Run diff and repository hygiene checks**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and no staged/unstaged result artifacts, caches,
or temporary logs.

- [ ] **Step 4: Push implementation commits after local verification**

```powershell
git push origin HEAD:main
```

Expected: `origin/main` advances to the verified implementation commit. Stop
after three identical network failures and report the local commit if push is
temporarily unavailable.

### Task 6: Run the pinned R3 seed-3 gate

**Files:**
- Output only: `E:/ARAC/results/r3_bounded_late_refresh_seed3_pinned_20260711/`

- [ ] **Step 1: Run one canonical 3M-FE trajectory**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\exp_005_hcc_final_protocol_pilot\run.py `
  --output-dir E:\ARAC\results\r3_bounded_late_refresh_seed3_pinned_20260711 `
  --hcc-root E:\HCC-main `
  --aob-data-root C:\Users\83718\.config\superpowers\worktrees\ARAC\codex-nondense-v24\HCC_SRC\AOB\AOBG\datafile `
  --python-executable E:\ARAC\.venv\Scripts\python.exe `
  --seeds 3 --problems R3 --jobs 1 --max-fes 3000000 `
  --budget-accounting strict --lane-profile canonical_evidence_controller_v1
```

- [ ] **Step 2: Audit the runtime action and protocol**

Require all of the following before continuing:

```text
completed=1/1
fresh=1/1
same_budget_violations=0/1
anti_leakage_failures=0
aob_input_changed=0
bounded_late_nda_refresh start rows=1
bounded_late_nda_refresh completion rows=1
refresh_fe > 0
cc_phase_fe after refresh > 0
```

- [ ] **Step 3: Apply the directional gate**

Compare offline against the pinned canonical R3 seed-3 value `4.556898e5`.
Continue only if the new value is lower. Do not tune using the paper value inside
the runtime. If the action does not trigger or fails to improve, stop and diagnose
the observed trace before changing a threshold.

### Task 7: Run R3 three-seed and focused preservation gates

**Files:**
- Output only: `E:/ARAC/results/r3_bounded_late_refresh_seed123_pinned_20260711/`
- Output only: `E:/ARAC/results/bounded_late_refresh_preservation_seed123_pinned_20260711/`

- [ ] **Step 1: Run R3 seeds 1-3**

Use the Task 6 command with `--seeds 1 2 3 --jobs 3` and the R3 three-seed output
directory. Report all three values, best-of-three, and three-seed mean separately.

- [ ] **Step 2: Run E3/S3/R2/A4 preservation controls**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\exp_005_hcc_final_protocol_pilot\run.py `
  --output-dir E:\ARAC\results\bounded_late_refresh_preservation_seed123_pinned_20260711 `
  --hcc-root E:\HCC-main `
  --aob-data-root C:\Users\83718\.config\superpowers\worktrees\ARAC\codex-nondense-v24\HCC_SRC\AOB\AOBG\datafile `
  --python-executable E:\ARAC\.venv\Scripts\python.exe `
  --seeds 1 2 3 --problems E3 S3 R2 A4 --jobs 12 --max-fes 3000000 `
  --budget-accounting strict --lane-profile canonical_evidence_controller_v1
```

- [ ] **Step 3: Apply preservation and anti-leakage gates**

Require 12/12 completed and fresh, no FE/input/anti-leakage violation, and retain
at least one seed below each offline paper-best threshold:

```text
E3 < 1.60e7
S3 < 9.72e3
R2 < 2.48e5
A4 < 7.83e4
```

Also require no bounded-refresh trigger on the one- and five-shared-variable
controls. E3 may trigger only if its allowed runtime evidence satisfies every
generic gate; any loss of its existing best-of-three win blocks the full run.

### Task 8: Run the final 13-case, three-seed regression

**Files:**
- Output only: `E:/ARAC/results/canonical_13target_bounded_late_refresh_seed123_pinned_20260711/`

- [ ] **Step 1: Launch only after Tasks 6 and 7 pass**

```powershell
E:\ARAC\.venv\Scripts\python.exe experiments\exp_005_hcc_final_protocol_pilot\run.py `
  --output-dir E:\ARAC\results\canonical_13target_bounded_late_refresh_seed123_pinned_20260711 `
  --hcc-root E:\HCC-main `
  --aob-data-root C:\Users\83718\.config\superpowers\worktrees\ARAC\codex-nondense-v24\HCC_SRC\AOB\AOBG\datafile `
  --python-executable E:\ARAC\.venv\Scripts\python.exe `
  --seeds 1 2 3 `
  --problems E1 E2 E3 E4 E6 S2 S3 S6 R1 R2 R3 A4 A5 `
  --jobs 24 --max-fes 3000000 --budget-accounting strict `
  --lane-profile canonical_evidence_controller_v1
```

- [ ] **Step 2: Audit before performance comparison**

Require `39/39` completed and fresh, `0/39` FE violations, zero anti-leakage
failures, unchanged AOB inputs, pinned environment pass, canonical protocol pass,
and runner/policy/optimizer hashes recorded.

- [ ] **Step 3: Report the three distinct performance summaries**

Report without conflating them:

```text
seed-level wins / 39
best-of-three wins / 13
three-seed-mean wins / 13
```

The target is best-of-three `13/13` while retaining every previous win. Do not
present best-of-three as a mean and do not make a 25-run or SOTA claim.
