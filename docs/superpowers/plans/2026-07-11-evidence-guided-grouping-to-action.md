# Evidence-Guided Grouping-to-Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a real, single-track grouping-to-action controller that resumes Phase-I MMES state and reallocates a bounded search budget from runtime overlap evidence while preserving the current 12/13 three-seed best-of-three wins.

**Architecture:** Keep canonical_evidence_controller_v1 as the only final entry. Put the pure search-state evidence policy in src/arac/policy, add a checkpointable state/block API to the existing MMES backend, and let the HCC runner sequentially interleave canonical CC sweeps with bounded probes and continuation blocks. Extend the existing trace and FE schemas; do not create an oracle lane or select among complete final lanes.

**Tech Stack:** Python 3.12, NumPy, SciPy, existing HCC MMES/CMAES backend, pytest, CSV audit artifacts, exp_005_hcc_final_protocol_pilot.

---

## File Map

- Create src/arac/policy/search_state_policy.py: reference-blind evidence types, budget rounding, utility calculation, and pure probe/confirmation/expansion transitions.
- Create HCC_SRC/HCC/NDAs/MMES/state.py: MMES state and block-result dataclasses, validation, cloning, and fingerprinting.
- Modify HCC_SRC/HCC/NDAs/MMES/mmes.py: expose stateful execution while keeping optimize() compatible.
- Modify src/arac/action_space.py: register the stateful resume action.
- Modify HCC_SRC/arac_hcc_smoke_runner.py: capture Phase-I state, collect complete-sweep evidence, execute the scheduler, and extend audit fields.
- Modify src/arac/backends/hcc.py: expose state-action FE and trace metadata.
- Modify experiments/exp_003_hcc_runtime_consumer_smoke/run.py: aggregate the new trace and ledger fields.
- Modify experiments/exp_005_hcc_final_protocol_pilot/run.py: hash the new runtime modules while retaining the canonical profile.
- Create tests/test_search_state_policy.py and tests/test_mmes_stateful_execution.py.
- Modify existing runner, backend, exp003, and exp005 tests for integration and regression.
- Keep references/aob_paper_best_win_replay_matrix.csv as an offline-only threshold source.

## Locked Runtime Names And Policy Values

~~~python
CONTINUE_CANONICAL_CC = "continue_canonical_cc"
RESUME_PHASE_I_SEARCH_STATE = "resume_phase_i_search_state"
SEARCH_STATE_PROBE = "probe"
SEARCH_STATE_AWAITING_CONFIRMATION_CC = "awaiting_confirmation_cc"
SEARCH_STATE_CONFIRMATION = "confirmation"
SEARCH_STATE_EXPANSION = "expansion"
SEARCH_STATE_BLOCKED = "blocked"
~~~

~~~python
first_probe_fraction = 0.01
cumulative_intervention_fraction = 0.15
cc_reserve_fraction = 0.10
base_utility_ratio = 1.50
accelerating_cc_utility_ratio = 2.00
minimum_conflict_fraction = 0.50
~~~

These values are generic runtime policy configuration, not case-specific rules.

## Task 1: Add The Pure Search-State Policy

**Files:**
- Create: src/arac/policy/search_state_policy.py
- Modify: src/arac/action_space.py
- Test: tests/test_search_state_policy.py

- [ ] **Step 1: Write the failing policy tests**

Use this fixture:

~~~python
def eligible_evidence(policy):
    return policy.SearchStateEvidence(
        complete_sweep=True,
        overlap_degree=0.10,
        phase_rescue_enabled=True,
        repair_lock_active=False,
        phase_i_tail_utility=2.0e-6,
        non_coordinate_fraction=0.60,
        conflict_fraction=0.50,
        writeback_unstable=False,
        recent_cc_utilities=(1.0e-7,),
        remaining_fes=900_000,
        max_fes=3_000_000,
        population_size=24,
    )
~~~

Add focused tests for the initial probe, abstention, acceleration, transitions, and caps:

~~~python
def test_initial_probe_is_rounded_and_reserves_cc(policy):
    plan = policy.plan_search_state_action(
        eligible_evidence(policy),
        policy.SearchStateSchedulerState(),
    )
    assert plan.action_name == policy.RESUME_PHASE_I_SEARCH_STATE
    assert plan.stage == policy.SEARCH_STATE_PROBE
    assert plan.requested_fes == 30_000
    assert plan.requested_fes % 24 == 0
    assert plan.cc_reserve_fes == 300_000
    assert plan.required_utility_ratio == 1.50


def test_incomplete_or_repair_locked_evidence_abstains(policy):
    base = eligible_evidence(policy)
    for evidence in (
        replace(base, complete_sweep=False),
        replace(base, repair_lock_active=True),
        replace(base, overlap_degree=0.0),
        replace(base, phase_i_tail_utility=0.0),
    ):
        plan = policy.plan_search_state_action(
            evidence,
            policy.SearchStateSchedulerState(),
        )
        assert plan.action_name == policy.CONTINUE_CANONICAL_CC


def test_accelerating_cc_uses_two_x_gate(policy):
    evidence = replace(
        eligible_evidence(policy),
        recent_cc_utilities=(1.0e-7, 2.0e-7),
    )
    plan = policy.plan_search_state_action(
        evidence,
        policy.SearchStateSchedulerState(),
    )
    assert plan.required_utility_ratio == 2.0


def test_failed_probe_blocks_future_state_actions(policy):
    state = policy.record_search_state_outcome(
        policy.SearchStateSchedulerState(),
        stage=policy.SEARCH_STATE_PROBE,
        accepted=False,
        utility=0.0,
        required_utility_ratio=1.5,
        cc_utility=1.0e-7,
        used_fes=30_000,
    )
    assert state.phase == policy.SEARCH_STATE_BLOCKED
    plan = policy.plan_search_state_action(eligible_evidence(policy), state)
    assert plan.action_name == policy.CONTINUE_CANONICAL_CC
~~~

Also test normalized utility, non-improving candidates, population rounding, forbidden
field exclusion, the exact state transitions, and the cumulative 15% cap. The required
transition order is:

~~~text
successful probe -> awaiting_confirmation_cc
next complete canonical CC sweep -> confirmation
successful confirmation -> expansion
any failed acceptance or utility gate -> permanently blocked
~~~

The confirmation action must not be planned from the same sweep that triggered the
initial probe. Two qualified state-action blocks are therefore required before any
expansion block.

- [ ] **Step 2: Run and verify the missing-module failure**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_search_state_policy.py -q
~~~

Expected: collection fails because search_state_policy.py does not exist.

- [ ] **Step 3: Implement the policy types and transitions**

Define:

~~~python
@dataclass(frozen=True)
class SearchStateEvidence:
    complete_sweep: bool
    overlap_degree: float
    phase_rescue_enabled: bool
    repair_lock_active: bool
    phase_i_tail_utility: float
    non_coordinate_fraction: float
    conflict_fraction: float
    writeback_unstable: bool
    recent_cc_utilities: tuple[float, ...]
    remaining_fes: int
    max_fes: int
    population_size: int


@dataclass(frozen=True)
class SearchStateSchedulerState:
    phase: str = "initial_probe"
    probe_utilities: tuple[float, ...] = ()
    intervention_fe: int = 0


@dataclass(frozen=True)
class SearchStateActionPlan:
    action_name: str
    stage: str
    requested_fes: int
    cc_reserve_fes: int
    required_utility_ratio: float
    trigger_reason: str
~~~

Implement normalized_gain_utility and population_rounded_budget as pure functions.
plan_search_state_action must abstain for blocked, incomplete, non-overlap,
repair-locked, non-positive-tail, structurally unsupported, or underfunded evidence.
Structural support means non_coordinate_fraction >= 0.50, or conflict_fraction >=
0.50, or writeback_unstable. Reserve ceil(max_fes * 0.10), cap each block at
floor(max_fes * 0.01), cap cumulative state-action FE at floor(max_fes * 0.15),
and round every block to a complete population. record_search_state_outcome must
require strict acceptance and utility at least the required ratio times the latest
CC utility. A successful probe moves to awaiting_confirmation_cc; the next call at
a new complete CC-sweep boundary plans confirmation; a successful confirmation moves
to expansion. A failed acceptance or utility gate at probe, confirmation, or expansion
moves permanently to blocked. The runner is responsible for calling the planner only
after complete sweep boundaries, so awaiting_confirmation_cc cannot consume another
state-action block before one full canonical CC sweep has completed.

Register resume_phase_i_search_state in DEFAULT_ACTION_SPACE as a trajectory
core_intervention.

- [ ] **Step 4: Run policy tests and commit**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_search_state_policy.py -q
git add src/arac/policy/search_state_policy.py src/arac/action_space.py tests/test_search_state_policy.py
git commit -m "feat: add reference-blind search-state policy"
~~~

Expected: all policy tests pass.

## Task 2: Add The MMES State Container

**Files:**
- Create: HCC_SRC/HCC/NDAs/MMES/state.py
- Test: tests/test_mmes_stateful_execution.py

- [ ] **Step 1: Write failing state tests**

Load HCC_SRC onto sys.path and test wrong shapes, deep cloning, RNG fingerprinting,
and an exact capture/restore round trip. The round trip must preserve both RNG states,
sigma, population/parent/mirror counts, restart counters and histories, incumbent,
FE count, termination state, fitness list, and the last three `(FE, best)` checkpoints:

~~~python
def test_state_validation_rejects_wrong_shapes():
    state = make_state(ndim=4, population=6)
    state.x = np.zeros((5, 4))
    with pytest.raises(ValueError, match="x shape"):
        state.validate()


def test_state_clone_is_deep_and_fingerprint_changes():
    state = make_state(ndim=4, population=6)
    clone = state.clone()
    clone.mean[0, 0] += 1.0
    assert not np.shares_memory(state.mean, clone.mean)
    assert state.fingerprint() != clone.fingerprint()


def test_rng_state_is_part_of_fingerprint():
    left = make_state(ndim=4, population=6)
    right = left.clone()
    right.rng_optimization_state["state"]["state"] += 1
    assert left.fingerprint() != right.fingerprint()
~~~

- [ ] **Step 2: Run and verify import failure**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_mmes_stateful_execution.py -q
~~~

Expected: import failure for HCC.NDAs.MMES.state.

- [ ] **Step 3: Implement MMESState and MMESBlockResult**

MMESState must contain:

~~~python
@dataclass
class MMESState:
    x: np.ndarray
    mean: np.ndarray
    p: np.ndarray
    w: float
    q: np.ndarray
    t: np.ndarray
    v: np.ndarray
    y: np.ndarray
    sigma: float
    n_individuals: int
    n_parents: int
    n_mirror_sampling: int
    n_generations: int
    n_restart: int
    list_generations: list[int]
    list_fitness: list[float]
    list_initial_mean: list[np.ndarray]
    best_so_far_x: np.ndarray
    best_so_far_y: float
    n_function_evaluations: int
    termination_signal: int
    fitness: list[float]
    recent_best: list[tuple[int, float]]
    rng_initialization_state: dict[str, object]
    rng_optimization_state: dict[str, object]
~~~

validate() checks finite arrays, exact dimensions, positive population and parent
counts, n_parents <= n_individuals, and non-negative FE. clone() deep-copies
arrays, lists, nested initial means, and RNG dictionaries. fingerprint() hashes
dtype, shape, bytes, scalar fields, and stable JSON for both RNG states. Capture and
restore must round-trip every listed field, including sigma, current population
configuration, restart counters/history, incumbent, FE and termination state, fitness,
both RNG states, and exactly the retained last three recent-best checkpoints. A
missing or altered continuation field is an explicit validation/test failure rather
than a cold restart fallback.

Add:

~~~python
@dataclass(frozen=True)
class MMESBlockResult:
    state: MMESState
    best_before: float
    best_after: float
    actual_fes: int
    requested_fes: int
    unused_fes: int
    normalized_utility: float
    termination_reason: str
    state_fingerprint_before: str
    state_fingerprint_after: str
~~~

- [ ] **Step 4: Run state tests and commit**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_mmes_stateful_execution.py -q
git add HCC_SRC/HCC/NDAs/MMES/state.py tests/test_mmes_stateful_execution.py
git commit -m "feat: add checkpointable MMES state model"
~~~

Expected: all state tests pass.

## Task 3: Expose Deterministic MMES Blocks

**Files:**
- Modify: HCC_SRC/HCC/NDAs/MMES/mmes.py
- Modify: tests/test_mmes_stateful_execution.py

- [ ] **Step 1: Add failing lifecycle tests**

Use a deterministic batched sphere objective. Test that optimize() and
optimize_with_state() return identical best value, vector, FE, and termination
signal; run_block never exceeds its requested FE; blocks evaluate only complete
populations; sequential blocks preserve continuation; malformed state fails before
objective evaluation.

- [ ] **Step 2: Run and verify missing-method failures**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_mmes_stateful_execution.py -q
~~~

Expected: AttributeError for optimize_with_state and run_block.

- [ ] **Step 3: Refactor MMES around captured state**

Add these methods by extracting the current optimize loop without changing its
mathematical operations:

~~~python
def initialize_state(self, args=None) -> MMESState:
    fitness = ES.optimize(self)
    x, mean, p, w, q, t, v, y = self.initialize(args)
    self._print_verbose_info(fitness, y[0])
    return self._capture_state(x, mean, p, w, q, t, v, y, fitness)


def optimize_with_state(
    self,
    fitness_function=None,
    args=None,
) -> tuple[dict[str, object], MMESState]:
    fitness = ES.optimize(self, fitness_function)
    x, mean, p, w, q, t, v, y = self.initialize(args)
    self._print_verbose_info(fitness, y[0])
    while not self.termination_signal:
        y_bak = np.copy(y)
        x, y = self.iterate(x, mean, q, v, args)
        if self._check_terminations():
            break
        mean, p, w, q, t, v = self._update_distribution(
            x, mean, p, w, q, t, v, y, y_bak
        )
        self._n_generations += 1
        self._print_verbose_info(fitness, y)
        x, mean, p, w, q, t, v, y = self.restart_reinitialize(
            args, x, mean, p, w, q, t, v, y, fitness
        )
    state = self._capture_state(x, mean, p, w, q, t, v, y, fitness)
    results = self._collect(fitness, y, mean)
    results["p"] = p
    results["w"] = w
    return results, state


def run_block(
    self,
    state: MMESState,
    additional_function_evaluations: int,
    args=None,
) -> MMESBlockResult:
    state.validate()
    before = float(state.best_so_far_y)
    fingerprint_before = state.fingerprint()
    previous_limit = self.max_function_evaluations
    block_start = int(state.n_function_evaluations)
    block_limit = block_start + max(0, int(additional_function_evaluations))
    x, mean, p, w, q, t, v, y, fitness = self._restore_state(state)
    self.max_function_evaluations = block_limit
    self.termination_signal = self.Terminations.NO_TERMINATION
    try:
        while self.n_function_evaluations < block_limit:
            restart_reserve = 1 if self.is_restart else 0
            required = self.n_individuals + restart_reserve
            if self.n_function_evaluations + required > block_limit:
                break
            y_bak = np.copy(y)
            x, y = self.iterate(x, mean, q, v, args)
            if self._check_terminations():
                break
            mean, p, w, q, t, v = self._update_distribution(
                x, mean, p, w, q, t, v, y, y_bak
            )
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                args, x, mean, p, w, q, t, v, y, fitness
            )
        next_state = self._capture_state(x, mean, p, w, q, t, v, y, fitness)
    finally:
        self.max_function_evaluations = previous_limit
    actual = next_state.n_function_evaluations - block_start
    after = float(next_state.best_so_far_y)
    utility = max(0.0, before - after) / (max(abs(before), 1.0) * max(actual, 1))
    reason = "block_complete" if actual > 0 else "insufficient_population_budget"
    return MMESBlockResult(
        state=next_state,
        best_before=before,
        best_after=after,
        actual_fes=actual,
        requested_fes=additional_function_evaluations,
        unused_fes=max(0, additional_function_evaluations - actual),
        normalized_utility=utility,
        termination_reason=reason,
        state_fingerprint_before=fingerprint_before,
        state_fingerprint_after=next_state.fingerprint(),
    )
~~~

Replace the bodies with the existing MMES initialization, update, restart, and
collect logic. Keep optimize() as a compatibility wrapper that calls
optimize_with_state and returns only its result dictionary. Move local loop
variables into capture/restore helpers. Preserve initialization order, seed
derivation, restart settings, result keys, and collect behavior. Do not modify
ES._evaluate_fitness.

run_block validates and restores state, uses a temporary absolute FE limit, runs
only complete populations, reserves one FE for a possible restart initialization
before a population, restores the prior absolute limit, and returns objective-
observed FE plus before/after fingerprints. If a complete population cannot fit,
return zero FE with termination_reason=insufficient_population_budget. Never
silently reinitialize malformed state.

- [ ] **Step 4: Run lifecycle tests and commit**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_mmes_stateful_execution.py -q
git add HCC_SRC/HCC/NDAs/MMES/mmes.py tests/test_mmes_stateful_execution.py
git commit -m "feat: expose resumable MMES execution blocks"
~~~

Expected: all lifecycle tests pass.

## Task 4: Capture Phase-I State And Complete-Sweep Evidence

**Files:**
- Modify: HCC_SRC/arac_hcc_smoke_runner.py
- Modify: tests/test_hcc_smoke_runner_cli.py

- [ ] **Step 1: Add failing runner tests**

Extend fake MMES with optimize_with_state. Assert canonical Phase-I stores the
optimizer and state, zero-global-budget stores neither, partial group loops never
invoke the search-state policy, evidence has no forbidden fields, and accepted CC
group results update the protected global incumbent.

- [ ] **Step 2: Run and verify failure**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -q -k "phase_i_state or complete_sweep"
~~~

Expected: failure because the runner still calls only optimize().

- [ ] **Step 3: Add state capture and evidence helpers**

Add:

~~~python
def phase_i_tail_utility(state: MMESState) -> float:
    window = state.recent_best[-3:]
    if len(window) < 2:
        return 0.0
    start_fe, start_best = window[0]
    end_fe, end_best = window[-1]
    return normalized_gain_utility(start_best, end_best, end_fe - start_fe)


def build_search_state_evidence(
    *,
    complete_sweep: bool,
    overlap_degree: float,
    phase_rescue_enabled: bool,
    repair_lock_active: bool,
    phase_i_tail_utility_value: float,
    relations: list[OverlapRelation],
    decisions: list[RelationActionDecision],
    writeback_norms: list[float],
    fitness_deltas: list[float],
    reference_fitness: float,
    cc_utility_history: list[float],
    remaining_fes: int,
    max_fes: int,
    population_size: int,
) -> SearchStateEvidence:
    canonical_actions = [
        _canonical_relation_action_name(decision) for decision in decisions
    ]
    non_coordinate = sum(
        action != "allow_beneficial_coordination" for action in canonical_actions
    ) / max(1, len(canonical_actions))
    conflict = cc_harm_conflict_fraction(fitness_deltas, reference_fitness)
    unstable = any(norm > CC_HARM_WRITEBACK_NORM for norm in writeback_norms)
    return SearchStateEvidence(
        complete_sweep=complete_sweep and bool(relations),
        overlap_degree=overlap_degree,
        phase_rescue_enabled=phase_rescue_enabled,
        repair_lock_active=repair_lock_active,
        phase_i_tail_utility=phase_i_tail_utility_value,
        non_coordinate_fraction=non_coordinate,
        conflict_fraction=conflict,
        writeback_unstable=unstable,
        recent_cc_utilities=tuple(cc_utility_history[-2:]),
        remaining_fes=remaining_fes,
        max_fes=max_fes,
        population_size=population_size,
    )
~~~

Replace the canonical Phase-I call with MMES.optimize_with_state. Keep result
extraction and seed options unchanged. Build non-coordinate fraction from current
relation decisions, conflict fraction from current relation evidence, and
writeback instability from existing writeback norms. Track complete_sweep by
requiring every group to finish and at least one overlap relation to exist. Use
`complete_sweep=complete_sweep and bool(relations)` in the evidence constructor.
Do not make a decision from a partial prefix or an empty relation set.

- [ ] **Step 4: Run focused tests and commit**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -q -k "phase_i_state or complete_sweep"
git add HCC_SRC/arac_hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git commit -m "feat: capture Phase-I MMES state and sweep evidence"
~~~

Expected: all focused tests pass.

## Task 5: Integrate The Single-Track Scheduler

**Files:**
- Modify: HCC_SRC/arac_hcc_smoke_runner.py
- Modify: tests/test_hcc_smoke_runner_cli.py

- [ ] **Step 1: Add failing transition-order and no-harm tests**

Script fake utilities to require this order:

~~~text
canonical CC sweep -> probe -> canonical CC sweep -> confirmation
~~~

Assert no probe before a complete sweep; a failed first probe blocks all later
state actions; a worse candidate leaves both incumbents unchanged; a successful
probe cannot jump directly to expansion; state-action FE never exceeds 15% and
leaves the 10% CC reserve; canonical v1 never calls the old bounded planner.

- [ ] **Step 2: Run and verify failure**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -q -k "search_state or probe or confirmation"
~~~

Expected: failure because canonical v1 has no stateful scheduler path.

- [ ] **Step 3: Wire decisions at complete-sweep boundaries**

Add SearchStateSchedulerState to EvidenceActionControllerV31RunState. At each
complete CC sweep append normalized CC utility, mark a pending confirmation sweep
complete, build SearchStateEvidence, call plan_search_state_action, and either
continue CC or execute one resumed MMES block.

Add:

~~~python
def run_resumed_phase_i_state_block(
    *,
    optimizer,
    state: MMESState,
    requested_fes: int,
    guard_individual: np.ndarray,
    guard_fitness: float,
    fun,
) -> tuple[MMESState, bool, np.ndarray, float, MMESBlockResult]:
    evaluations_before = current_fitness_evaluations(fun)
    block = optimizer.run_block(state, requested_fes)
    observed = current_fitness_evaluations(fun) - evaluations_before
    if observed != block.actual_fes:
        raise RuntimeError("stateful MMES FE mismatch")
    candidate = np.asarray(block.state.best_so_far_x, dtype=float).reshape(-1)
    candidate_fitness = float(block.state.best_so_far_y)
    guard = np.asarray(guard_individual, dtype=float).reshape(-1)
    if candidate.shape != guard.shape or not np.all(np.isfinite(candidate)):
        raise RuntimeError("stateful MMES returned invalid candidate")
    if not math.isfinite(candidate_fitness):
        raise RuntimeError("stateful MMES returned non-finite fitness")
    accepted = candidate_fitness < float(guard_fitness)
    if accepted:
        return block.state, True, candidate.copy(), candidate_fitness, block
    return block.state, False, guard.copy(), float(guard_fitness), block
~~~

Replace the body with optimizer.run_block, objective-observed FE validation,
finite shape checks, and strict candidate acceptance. Do not create a new MMES or
derive a new seed; the saved RNG state is the continuation. After each block call
record_search_state_outcome and add actual FE to search_state_fe. Keep refresh_fe
exclusively for the legacy explicit cold-refresh profile. Canonical v1 must not
call plan_bounded_late_nda_refresh or run_guarded_nda_continuation.

- [ ] **Step 4: Run focused tests and commit**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_smoke_runner_cli.py -q -k "search_state or probe or confirmation"
git add HCC_SRC/arac_hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py
git commit -m "feat: add single-track stateful grouping-to-action scheduler"
~~~

Expected: transition, budget, and no-harm tests pass.

## Task 6: Extend Trace, FE Ledger, And Aggregation

**Files:**
- Modify: HCC_SRC/arac_hcc_smoke_runner.py
- Modify: src/arac/backends/hcc.py
- Modify: experiments/exp_003_hcc_runtime_consumer_smoke/run.py
- Modify: experiments/exp_005_hcc_final_protocol_pilot/run.py
- Test: tests/test_hcc_backbone_adapter.py
- Test: tests/test_hcc_execution_adapter.py
- Test: tests/test_exp_003_hcc_runtime_consumer_smoke.py
- Test: tests/test_exp_005_hcc_final_protocol_pilot.py

- [ ] **Step 1: Add failing schema tests**

Require these action-trace fields:

~~~text
scheduler_phase
decision_point
cc_block_fe
cc_utility
search_state_block_fe
search_state_utility
required_utility_ratio
state_action_fe
cc_reserve_fe
state_fingerprint_before
state_fingerprint_after
abstain_reason
~~~

Require search_state_fe in budget_summary.csv and same_budget_ledger.csv. Stage
totals must equal global_phase_fe + cc_phase_fe + rescue_fe + refresh_fe +
search_state_fe + separable_continuation_fe + overhead_fe. A legacy summary
without search_state_fe must parse it as zero.

- [ ] **Step 2: Run and verify schema failure**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_backbone_adapter.py tests/test_hcc_execution_adapter.py tests/test_exp_003_hcc_runtime_consumer_smoke.py -q -k "trace or budget or summary or manifest"
~~~

Expected: missing-field assertions fail.

- [ ] **Step 3: Extend schemas without a second result format**

Add the trace fields to ACTION_TRACE_FIELDS, build_action_trace_row, and exp003
aggregate columns. Add search_state_fe to BUDGET_SUMMARY_FIELDS,
_write_budget_summary, HccAobExecutionResult, _parse_hcc_budget_summary,
_ledger_for_result, and the exp003 ledger writer. Preserve refresh_fe as a
separate legacy stage and parse a missing search_state_fe as zero.

Add resume_phase_i_search_state to the backend execution plan with trajectory
family, resumable_mmes_state_block backend effect, optimizer_consumed=true,
execution_mode=hcc_stateful_search_action, and runtime_dispatch_allowed=true.

Add hashes for src/arac/policy/search_state_policy.py and
HCC_SRC/HCC/NDAs/MMES/state.py to the exp005 manifest. Keep the lane profile name
canonical_evidence_controller_v1 unchanged.

The exp005 execution request remains reference-blind. Add a regression assertion
that paper-best values and the offline comparison matrix never enter
`HccAobExecutionRequest`, its serialized payload, runner CLI arguments, or runtime
environment. After all optimizer processes finish and protocol audits pass, exp005
may read the offline threshold matrix to write `best_of_three_vs_paper_best.csv` with
exactly these columns:

~~~text
problem_id,seed_count,best_error,paper_best,best_of_three_win
~~~

This file is an offline report artifact only and must not influence dispatch,
acceptance, stopping, FE allocation, or action traces.

- [ ] **Step 4: Run schema tests and commit**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_hcc_backbone_adapter.py tests/test_hcc_execution_adapter.py tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_exp_005_hcc_final_protocol_pilot.py -q
git add HCC_SRC/arac_hcc_smoke_runner.py src/arac/backends/hcc.py experiments/exp_003_hcc_runtime_consumer_smoke/run.py experiments/exp_005_hcc_final_protocol_pilot/run.py tests/test_hcc_backbone_adapter.py tests/test_hcc_execution_adapter.py tests/test_exp_003_hcc_runtime_consumer_smoke.py tests/test_exp_005_hcc_final_protocol_pilot.py
git commit -m "feat: audit stateful search actions and FE accounting"
~~~

Expected: schema, parser, aggregate, and manifest tests pass.

## Task 7: Lock Canonical Regression And Anti-Leakage

**Files:**
- Modify: HCC_SRC/arac_hcc_smoke_runner.py
- Modify: tests/test_hcc_smoke_runner_cli.py
- Modify: tests/test_exp_005_hcc_final_protocol_pilot_cli.py

- [ ] **Step 1: Replace obsolete canonical cold-refresh assertions**

Retain tests for explicit legacy cc_harm_guarded_sep_refresh. Replace canonical
assertions about bounded_late_nda_refresh with assertions that canonical v1 emits
no bounded cold-refresh action and uses the new stateful action only after legal
evidence.

- [ ] **Step 2: Audit forbidden dispatch inputs**

~~~powershell
rg -n "paper|historical|final_error|problem_family|function_family|R3|rastrigin" src/arac/policy/search_state_policy.py HCC_SRC/arac_hcc_smoke_runner.py
~~~

Expected: no forbidden value is read by plan_search_state_action or
build_search_state_evidence. Existing offline labels and test fixtures may remain
outside those functions.

Also assert in backend/exp005 tests that neither the paper-best matrix nor any
`paper_best` value is present in an `HccAobExecutionRequest` or optimizer subprocess
command. The matrix may be opened only after execution results and protocol audits
have been collected.

- [ ] **Step 3: Run full local regression**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short
~~~

Expected: all tests pass, diff check is clean, and no result or cache file is
staged.

- [ ] **Step 4: Commit the regression lock**

~~~powershell
git add HCC_SRC/arac_hcc_smoke_runner.py tests/test_hcc_smoke_runner_cli.py tests/test_exp_005_hcc_final_protocol_pilot_cli.py
git commit -m "test: lock canonical grouping-to-action regression gates"
~~~

Do not start a 3M-FE run if Task 7 fails.

## Task 8: Run Fresh Runtime Gates In Order

**Files:**
- Output only: E:\ARAC\results (ignored, never commit)
- Offline threshold source: E:\ARAC\references\aob_paper_best_win_replay_matrix.csv
- Offline comparison artifact: `<output-dir>\best_of_three_vs_paper_best.csv`

- [ ] **Step 1: Re-run deterministic state and policy verification**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe -m pytest tests/test_mmes_stateful_execution.py tests/test_search_state_policy.py -q
~~~

Gate: exact optimize()/optimize_with_state() equivalence and hard block FE bounds
pass.

- [ ] **Step 2: Run the R3 seed-3 3M-FE diagnostic**

~~~powershell
E:\ARAC\.venv\Scripts\python.exe experiments\exp_005_hcc_final_protocol_pilot\run.py --output-dir E:\ARAC\results\exp_006_stateful_grouping_to_action_R3_seed3_3m --hcc-root E:\HCC-main --aob-data-root E:\ARAC\HCC_SRC\AOB\AOBG\datafile --python-executable E:\ARAC\.venv\Scripts\python.exe --seeds 3 --problems R3 --jobs 1 --max-fes 3000000 --budget-accounting strict --lane-profile canonical_evidence_controller_v1
~~~

Audit before reading performance: fresh execution, unchanged AOB inputs, zero FE
violation, zero anti-leakage failures, state trace present, and exact stage-ledger
reconciliation. Only after those checks may the report layer read paper-best values
and create `best_of_three_vs_paper_best.csv`; verify the required five-column schema.

- [ ] **Step 3: Run preservation controls**

Run E6, S6, R2, and A4 at seeds 1-3 with the same command and jobs=12. Stop if
any control has a catastrophic-loss label or loses its existing best-of-three
paper-best win.

- [ ] **Step 4: Run the 12-case preservation gate**

Run these cases at seeds 1-3:

~~~text
E1 E2 E3 E4 E6 S2 S3 S6 R1 R2 A4 A5
~~~

Use jobs=24 when the pinned environment remains stable. Compare only after the
protocol audit against the offline paper_best column in
references/aob_paper_best_win_replay_matrix.csv. Require 12/12 best-of-three wins
and no per-seed catastrophic_loss under the existing 20% relative-loss audit.

- [ ] **Step 5: Run the 13-case target only after 12/12 passes**

Add R3 to the same three-seed command. Report separately:

~~~text
seed-level wins / 39
best-of-three wins / 13
three-seed-mean wins / 13
~~~

Do not call best-of-three a mean and do not claim a 25-run result.

## Plan Self-Review Checklist

- Spec coverage: Tasks 2-3 cover state checkpointing; Task 1 covers legal evidence
  and pure action selection; Tasks 4-5 cover single-track execution and incumbent
  protection; Task 6 covers trace, FE, parser, and manifest audit; Task 7 covers
  regression and leakage; Task 8 covers preservation and target gates.
- Runtime leakage: the policy module has no benchmark, paper, historical, or
  final-outcome dependency. The threshold CSV appears only in offline Task 8.
- Budget consistency: each probe is 1%, cumulative state action is 15%, canonical
  reserve is 10%, and every block is population-rounded.
- Type consistency: SearchStateEvidence, SearchStateSchedulerState,
  SearchStateActionPlan, MMESState, and MMESBlockResult are used consistently.
- Fallback consistency: blocked or ambiguous evidence maps to
  continue_canonical_cc; legacy cold refresh remains outside canonical v1.
