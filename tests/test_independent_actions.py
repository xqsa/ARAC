from __future__ import annotations

import copy
from dataclasses import replace
import math

import numpy as np
import pytest

import arac.actions._execution as execution
import arac.actions.smp as smp
from arac.actions.aor import _optimizer_route
from arac.actions.registry import ActionRegistry
from arac.actions._execution import (
    BLOCK_POPULATION_SIZE,
    _PersistentBlockSession,
    _allocate_block_budgets,
    _block_population_size,
    _rank_blocks_by_directional_slope,
    derived_seed,
)
from arac.actions.ctp import _relation_cover
from arac.actions.gcb import GcbExecutor
from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.phase1 import run_phase1
from arac.runtime.contracts import (
    ACTION_NAMES,
    ActionContext,
    PhaseCheckpoint,
    RelationEvidence,
)
from arac.runtime.ledger import EvaluationLedger


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )


@pytest.mark.parametrize("action_name", ACTION_NAMES)
def test_every_action_uses_one_contract_and_exact_terminal_budget(action_name: str) -> None:
    problem = _problem()
    phase1_ledger = EvaluationLedger(problem, 500)
    checkpoint = run_phase1(problem, phase1_ledger, run_seed=11).checkpoint
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    context = ActionContext(
        action_name=action_name,
        checkpoint=checkpoint,
        problem=problem,
        ledger=ledger,
        action_seed=91,
    )

    result = ActionRegistry().execute(context)

    assert result.action_name == action_name
    assert result.checkpoint_hash == checkpoint.checkpoint_hash
    assert result.consumed_fes == 500 - 240
    assert result.terminal_fes == 500
    assert result.final_error <= checkpoint.incumbent_error
    assert len(result.result_hash) == 64


def test_gcb_internal_route_uses_relation_count_not_case_identity() -> None:
    problem = _problem()
    checkpoint = run_phase1(problem, EvaluationLedger(problem, 500), run_seed=17).checkpoint
    zero = replace(checkpoint, relations=())
    positive = replace(
        checkpoint,
        relations=(RelationEvidence(0, 1, strength=0.4, disagreement=0.2),),
    )

    def execute(active):
        ledger = EvaluationLedger.from_checkpoint(
            problem,
            total_budget=active.total_budget_fes,
            phase1_fes=active.phase1_fes,
            incumbent=active.incumbent,
            incumbent_error=active.incumbent_error,
        )
        return ActionRegistry().execute(
            ActionContext("gcb", active, problem, ledger, action_seed=3)
        )

    zero_result = execute(zero)
    positive_result = execute(positive)

    assert zero_result.route.startswith("zero_relation_compact_coordination_")
    assert positive_result.route.startswith("positive_relation_graph_")
    coordination_fes = int(
        positive_result.route.split("_coordination_", maxsplit=1)[1].split(
            "_", maxsplit=1
        )[0]
    )
    assert coordination_fes > 0


def test_gcb_full_schedule_uses_three_source_sweeps_and_three_fresh_windows() -> None:
    problem = _problem()
    base = run_phase1(problem, EvaluationLedger(problem, 500), run_seed=17).checkpoint
    checkpoint = replace(
        base,
        total_budget_fes=20_000,
        relations=(RelationEvidence(0, 1, strength=0.4, disagreement=0.2),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    context = ActionContext("gcb", checkpoint, problem, ledger, action_seed=3)
    events: list[dict[str, object]] = []

    result = GcbExecutor().execute_schedule(context, event_trace=events)

    group_events = [event for event in events if event["event"] == "cold_group_visit"]
    source_events = [
        event for event in group_events if event["namespace"] == "gcb-source"
    ]
    native_events = [
        event for event in group_events if event["namespace"] == "gcb-native"
    ]
    coordination = [
        event for event in events if event["event"] == "full_space_coordination"
    ]
    block_count = len(checkpoint.blocks)
    assert len(group_events) >= block_count * 6
    assert [
        sum(event["sweep_index"] == index for event in source_events)
        for index in range(3)
    ] == [block_count] * 3
    assert [
        sum(event["sweep_index"] == index for event in native_events)
        for index in range(3, 6)
    ] == [block_count] * 3
    assert all(event["cold_start"] is True for event in group_events)
    assert all(event["state_restored"] is False for event in group_events)
    assert len({event["stage_index"] for event in group_events}) == len(group_events)
    assert len(coordination) == 1
    assert coordination[0]["trigger"] == "relation_dispatch"
    source_fes = sum(
        int(event["actual_fes"])
        for event in source_events
        if event["sweep_index"] == 2
    )
    assert coordination[0]["actual_fes"] == source_fes
    assert result.terminal_fes == 20_000


def test_aor_route_matches_the_frozen_legacy_rule() -> None:
    base = {
        "line_high_frequency_fraction_median": 0.4,
        "log10_center_error": 1.0,
    }

    low_scale = _optimizer_route(base)
    high_scale = _optimizer_route(base | {"log10_center_error": 100.0})

    assert low_scale == "sepcmaes"
    assert high_scale == "mmes"
    assert _optimizer_route(base | {"line_high_frequency_fraction_median": 0.2}) == "sepcmaes"


def test_ctp_relation_cover_preserves_every_frozen_relation_block() -> None:
    problem = _problem()
    checkpoint = run_phase1(problem, EvaluationLedger(problem, 500), run_seed=17).checkpoint
    relations = tuple(
        RelationEvidence(
            left,
            right,
            strength=float(100 - 10 * left - right),
            disagreement=0.0,
        )
        for left in range(7)
        for right in range(left + 1, 7)
    )
    active = replace(checkpoint, relations=relations)
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=active.total_budget_fes,
        phase1_fes=active.phase1_fes,
        incumbent=active.incumbent,
        incumbent_error=active.incumbent_error,
    )

    cover = _relation_cover(ActionContext("ctp", active, problem, ledger, action_seed=3))

    assert cover[: len(active.blocks)] == active.blocks
    assert len(cover) == len(active.blocks) + len(relations)


def test_action_seed_does_not_change_with_checkpoint_metadata() -> None:
    problem = _problem()
    checkpoint = run_phase1(problem, EvaluationLedger(problem, 500), run_seed=17).checkpoint
    changed = replace(
        checkpoint,
        protocol="changed-protocol",
        feature_values=tuple(value + 1.0 for value in checkpoint.feature_values),
    )

    def context(active):
        ledger = EvaluationLedger.from_checkpoint(
            problem,
            total_budget=active.total_budget_fes,
            phase1_fes=active.phase1_fes,
            incumbent=active.incumbent,
            incumbent_error=active.incumbent_error,
        )
        return ActionContext("ctp", active, problem, ledger, action_seed=91)

    assert checkpoint.checkpoint_hash != changed.checkpoint_hash
    assert derived_seed(context(checkpoint), "block", 3) == derived_seed(
        context(changed), "block", 3
    )


def test_ctp_relation_route_uses_an_overlapping_evidence_cover() -> None:
    problem = _problem()
    checkpoint = run_phase1(problem, EvaluationLedger(problem, 500), run_seed=17).checkpoint
    positive = replace(
        checkpoint,
        relations=(RelationEvidence(0, 1, strength=0.4, disagreement=0.2),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=positive.total_budget_fes,
        phase1_fes=positive.phase1_fes,
        incumbent=positive.incumbent,
        incumbent_error=positive.incumbent_error,
    )
    context = ActionContext("ctp", positive, problem, ledger, action_seed=3)
    cover = _relation_cover(context)
    result = ActionRegistry().execute(context)

    expected_overlap = tuple(sorted(set(positive.blocks[0]) | set(positive.blocks[1])))
    assert cover[: len(positive.blocks)] == positive.blocks
    assert expected_overlap in cover[len(positive.blocks) :]
    assert result.route.startswith("coverage_")
    assert "relation_cover_polish" in result.route
    tail_fes = int(result.route.rsplit("_then_mmes_tail_", maxsplit=1)[1])
    assert tail_fes > 0


def test_ctp_positive_relation_reserves_at_least_twenty_percent_phase2_tail() -> None:
    problem = _problem()
    checkpoint = PhaseCheckpoint(
        protocol="test-ctp-tail-v1",
        run_seed=17,
        total_budget_fes=3_000,
        phase1_fes=180,
        incumbent=(0.0,) * 40,
        incumbent_error=0.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=(tuple(range(20)), tuple(range(20, 40))),
        relations=(RelationEvidence(0, 1, strength=0.4, disagreement=0.2),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )

    result = ActionRegistry().execute(
        ActionContext("ctp", checkpoint, problem, ledger, action_seed=3)
    )

    tail_fes = int(result.route.rsplit("_then_mmes_tail_", maxsplit=1)[1])
    reserved = int((3_000 - 180) * 0.20)
    relation_cover_sweep = 3 * BLOCK_POPULATION_SIZE
    assert reserved <= tail_fes < reserved + relation_cover_sweep
    assert result.terminal_fes == 3_000


def test_full_block_budget_is_dimension_weighted_but_coverage_is_equal() -> None:
    blocks = ((0,), tuple(range(1, 10)))
    total = 20 * BLOCK_POPULATION_SIZE

    weighted = _allocate_block_budgets(blocks, total, equal_generations=False)
    coverage = _allocate_block_budgets(blocks, total, equal_generations=True)

    assert sum(weighted) == total
    assert weighted[1] > weighted[0]
    assert all(value % BLOCK_POPULATION_SIZE == 0 for value in weighted)
    assert coverage == (total // 2, total // 2)


@pytest.mark.parametrize(
    ("dimension", "expected"),
    ((25, 16), (50, 16), (100, 19)),
)
def test_dynamic_block_population_matches_frozen_smp_parameterization(
    dimension: int,
    expected: int,
) -> None:
    assert _block_population_size(dimension) == expected


def test_smp_repairs_persistent_block_candidates_to_public_bounds() -> None:
    def bounded_objective(x):
        candidates = np.asarray(x, dtype=float)
        assert np.all(candidates >= -1.0)
        assert np.all(candidates <= 1.0)
        return np.sum(candidates**2, axis=-1)

    problem = OptimizationProblem(
        objective=bounded_objective,
        dimension=4,
        lower_bounds=(-1.0,) * 4,
        upper_bounds=(1.0,) * 4,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=25,
        phase1_fes=1,
        incumbent=(1.0,) * 4,
        incumbent_error=4.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1), (2, 3)),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=25,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )

    result = ActionRegistry().execute(
        ActionContext("smp", checkpoint, problem, ledger, action_seed=1)
    )

    assert result.terminal_fes == 25
    assert "global_polish_0" in result.route


def test_smp_positive_relations_reserve_a_global_polish_budget() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=4,
        lower_bounds=(-1.0,) * 4,
        upper_bounds=(1.0,) * 4,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=25,
        phase1_fes=1,
        incumbent=(1.0,) * 4,
        incumbent_error=4.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1), (2, 3)),
        relations=(RelationEvidence(0, 1, strength=0.4, disagreement=0.2),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=25,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )

    result = ActionRegistry().execute(
        ActionContext("smp", checkpoint, problem, ledger, action_seed=1)
    )

    assert result.terminal_fes == 25
    assert "global_polish_12" in result.route


def test_smp_zero_relation_rescue_reuses_early_stopped_budget(monkeypatch) -> None:
    assert execution.ZERO_RELATION_RESCUE_SIGMA_DIVISORS == (
        (1.0,) * 4 + (2.0,) * 4 + (4.0,) * 4 + (8.0,) * 4
    )
    monkeypatch.setattr(execution, "EARLY_STOPPING_EVALUATIONS", 1)
    monkeypatch.setattr(smp, "STATE_RESCUE_MIN_FES", 0)
    problem = OptimizationProblem(
        objective=lambda x: np.ones(np.asarray(x).shape[:-1] or (), dtype=float),
        dimension=4,
        lower_bounds=(-1.0,) * 4,
        upper_bounds=(1.0,) * 4,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=2_000,
        phase1_fes=1,
        incumbent=(0.0,) * 4,
        incumbent_error=1.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1), (2, 3)),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )

    result = ActionRegistry().execute(
        ActionContext("smp", checkpoint, problem, ledger, action_seed=1)
    )

    persistent_visits = int(
        result.route.split("persistent_rescue_visits_", maxsplit=1)[1].split("_", maxsplit=1)[0]
    )
    assert result.terminal_fes == checkpoint.total_budget_fes
    assert "coverage_" in result.route
    assert "cold_rescue_visits_" in result.route
    assert persistent_visits > 0


def test_smp_stale_restarts_restore_the_frozen_local_sigma() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=5,
        lower_bounds=(-100.0,) * 5,
        upper_bounds=(100.0,) * 5,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=13,
        phase1_fes=1,
        incumbent=(1.0,) * 5,
        incumbent_error=5.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1, 2, 3, 4),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=13,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    session = _PersistentBlockSession(
        ActionContext("smp", checkpoint, problem, ledger, action_seed=1),
        checkpoint.blocks[0],
        block_index=0,
        budget_fes=12,
        population_size=12,
    )

    session.restart()

    assert session.optimizer.sigma == 0.5


def test_smp_rescue_ranks_the_block_with_the_largest_residual_slope() -> None:
    weights = np.asarray([1e6, 1e6, 1.0, 1.0])
    problem = OptimizationProblem(
        objective=lambda x: np.sum(weights * np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=4,
        lower_bounds=(-5.0,) * 4,
        upper_bounds=(5.0,) * 4,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=20,
        phase1_fes=1,
        incumbent=(1.0,) * 4,
        incumbent_error=2_000_002.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1), (2, 3)),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=20,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    context = ActionContext("smp", checkpoint, problem, ledger, action_seed=1)

    ranking, consumed = _rank_blocks_by_directional_slope(context, cycle_index=0)

    assert ranking[0] == 0
    assert consumed == 4


def test_block_cma_advances_generation_before_distribution_update() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=5,
        lower_bounds=(-5.0,) * 5,
        upper_bounds=(5.0,) * 5,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=13,
        phase1_fes=1,
        incumbent=(1.0,) * 5,
        incumbent_error=5.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1, 2, 3, 4),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=13,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    context = ActionContext("smp", checkpoint, problem, ledger, action_seed=1)
    session = _PersistentBlockSession(
        context,
        checkpoint.blocks[0],
        block_index=0,
        budget_fes=12,
        population_size=12,
    )
    session.begin_visit()
    original_update = session.optimizer.update_distribution
    observed_generations: list[int] = []

    def record_generation(*args, **kwargs):
        observed_generations.append(session.optimizer._n_generations)
        return original_update(*args, **kwargs)

    session.optimizer.update_distribution = record_generation
    session.advance()

    assert observed_generations == [1]


def test_block_cma_does_not_adapt_an_early_stopped_population() -> None:
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=5,
        lower_bounds=(-5.0,) * 5,
        upper_bounds=(5.0,) * 5,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=25,
        phase1_fes=1,
        incumbent=(1.0,) * 5,
        incumbent_error=5.0,
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1, 2, 3, 4),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=25,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    session = _PersistentBlockSession(
        ActionContext("smp", checkpoint, problem, ledger, action_seed=1),
        checkpoint.blocks[0],
        block_index=0,
        budget_fes=24,
        population_size=12,
    )
    session.begin_visit()
    session.optimizer._base_early_stopping = -math.inf
    session.optimizer._counter_early_stopping = 999

    session.advance(adapt_on_early_stop=False)

    assert session.consumed_fes == 12
    assert session.early_stopped
    assert session.optimizer._n_generations == 0


def test_block_cma_uses_the_frozen_batched_sampling_orientation() -> None:
    incumbent = (0.37, -1.11, 2.03, -0.79, 1.41)
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=5,
        lower_bounds=(-5.0,) * 5,
        upper_bounds=(5.0,) * 5,
    )
    checkpoint = PhaseCheckpoint(
        protocol="test-evidence-v1",
        run_seed=1,
        total_budget_fes=13,
        phase1_fes=1,
        incumbent=incumbent,
        incumbent_error=sum(value**2 for value in incumbent),
        feature_names=("signal",),
        feature_values=(0.0,),
        blocks=((0, 1, 2, 3, 4),),
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=13,
        phase1_fes=1,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    session = _PersistentBlockSession(
        ActionContext("smp", checkpoint, problem, ledger, action_seed=1),
        checkpoint.blocks[0],
        block_index=0,
        budget_fes=12,
        population_size=12,
    )
    session.begin_visit()
    rotation, _ = np.linalg.qr(
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 5.0, 7.0, 11.0, 13.0],
                [3.0, 7.0, 17.0, 19.0, 23.0],
                [5.0, 11.0, 19.0, 29.0, 31.0],
                [7.0, 13.0, 23.0, 31.0, 41.0],
            ]
        )
    )
    session.eigenvectors = rotation
    session.eigenvalues = np.linspace(0.2, 1.0, 5)
    rng_state = copy.deepcopy(session.optimizer.rng_optimization.bit_generator.state)
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = rng_state
    noise = expected_rng.standard_normal((12, 5))
    expected_steps = noise @ (np.diag(session.eigenvalues) @ session.eigenvectors.T)
    observed_steps: list[np.ndarray] = []
    original_update = session.optimizer.update_distribution

    def record_steps(*args, **kwargs):
        observed_steps.append(np.asarray(args[-1]).copy())
        return original_update(*args, **kwargs)

    session.optimizer.update_distribution = record_steps
    session.advance()

    assert len(observed_steps) == 1
    assert np.array_equal(observed_steps[0], expected_steps)
