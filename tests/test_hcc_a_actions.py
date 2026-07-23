from __future__ import annotations

import copy
import importlib
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

HCC_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
if str(HCC_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(HCC_VENDOR_ROOT))
MMESState = importlib.import_module("HCC.NDAs.MMES.state").MMESState

from arac.actions.mmes_resume import (  # noqa: E402
    canonical_mmes_parameters,
    canonical_mmes_state_hash,
    mmes_resume_anchor_hash,
)
from arac.actions.relation_sweep import FULL_SPACE_DIMENSION, RELATION_COUNT  # noqa: E402
from arac.backends.hcc_a_actions import (  # noqa: E402
    NO_WRITEBACK_SEED_NAMESPACE,
    PHASE1_MMES_SEED_NAMESPACE,
    RELATION_CMA_SEED_NAMESPACE,
    build_a_relation_graph,
    compile_a_sweep_action_candidates,
    compile_no_writeback_window_action,
    compile_phase1_mmes_resume_action,
    compile_relation_shared_cma_sweep_action,
    relation_cma_optimizer_seed,
)


CHECKPOINT_HASH = "a" * 64
CHECKPOINT_FE = 300_000


def _rng_state(seed: int) -> dict[str, object]:
    return copy.deepcopy(np.random.default_rng(seed).bit_generator.state)


def _mmes_state(*, ndim: int = FULL_SPACE_DIMENSION):
    population = 4
    mean = np.linspace(-0.5, 0.5, ndim)
    return MMESState(
        x=np.arange(population * ndim, dtype=float).reshape(population, ndim),
        mean=mean.copy(),
        p=np.zeros((1, ndim), dtype=float),
        w=0.25,
        q=np.zeros((4, ndim), dtype=float),
        t=np.arange(4, dtype=float),
        v=np.arange(4, dtype=int),
        y=np.linspace(10.0, 5.0, population),
        sigma=0.75,
        sigma_bak=1.0,
        initial_mean=mean.reshape(1, -1),
        n_individuals=population,
        n_parents=population // 2,
        n_mirror_sampling=(population + 1) // 2,
        n_generations=3,
        n_restart=1,
        list_generations=[2],
        list_fitness=[float("inf"), 8.0, 5.0],
        list_initial_mean=[np.ones(ndim)],
        best_so_far_x=np.zeros(ndim),
        best_so_far_y=2.0,
        n_function_evaluations=CHECKPOINT_FE,
        termination_signal=1,
        fitness=[10.0, 8.0, 5.0],
        recent_best=[(1, 10.0), (7, 5.0), (13, 2.0)],
        rng_initialization_state=_rng_state(11),
        rng_optimization_state=_rng_state(17),
        counter_early_stopping=2,
        base_early_stopping=2.0,
        printed_evaluations=CHECKPOINT_FE,
        time_function_evaluations=0.25,
        runtime=1.5,
    )


def _compile_mmes(
    *,
    state=None,
    incumbent: tuple[float, ...] | None = None,
    incumbent_fitness: float | None = None,
    budget_fes: int = 8,
    target_sweep: int = 7,
    checkpoint_fe: int = CHECKPOINT_FE,
):
    frozen_state = _mmes_state() if state is None else state
    runner_incumbent = (
        tuple(float(value) for value in frozen_state.best_so_far_x)
        if incumbent is None
        else incumbent
    )
    runner_fitness = (
        float(frozen_state.best_so_far_y) if incumbent_fitness is None else incumbent_fitness
    )
    return compile_phase1_mmes_resume_action(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=checkpoint_fe,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        state=frozen_state,
        incumbent=runner_incumbent,
        incumbent_fitness=runner_fitness,
        budget_fes=budget_fes,
        target_sweep=target_sweep,
    )


def test_mmes_compiler_is_deterministic_and_freezes_the_real_state() -> None:
    state = _mmes_state()
    first = _compile_mmes(state=state)
    second = _compile_mmes(state=state)
    frozen_action_hash = first.action_hash

    assert first == second
    assert first.action_hash == second.action_hash
    assert first.state_hash == canonical_mmes_state_hash(state)
    assert first.state_snapshot.payload_hash == second.state_snapshot.payload_hash
    assert first.seed_namespace == PHASE1_MMES_SEED_NAMESPACE

    state.mean[0] += 10.0
    assert first.action_hash == frozen_action_hash
    assert first.state_hash != canonical_mmes_state_hash(state)


def test_mmes_compiler_rejects_runner_incumbent_mismatch() -> None:
    state = _mmes_state()
    runner_incumbent = tuple(float(value) for value in state.best_so_far_x)
    state.best_so_far_x[0] = 1.0

    with pytest.raises(ValueError, match="exactly equal state.best_so_far_x"):
        _compile_mmes(state=state, incumbent=runner_incumbent)


def test_mmes_compiler_rejects_runner_fitness_mismatch() -> None:
    state = _mmes_state()

    with pytest.raises(ValueError, match="exactly equal state.best_so_far_y"):
        _compile_mmes(
            state=state,
            incumbent_fitness=float(state.best_so_far_y) + 1.0,
        )


def test_mmes_compiler_rejects_state_checkpoint_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="checkpoint_fe must exactly equal state.n_function_evaluations",
    ):
        _compile_mmes(checkpoint_fe=CHECKPOINT_FE + 1)


def test_mmes_compiler_rejects_non_aob_dimension() -> None:
    with pytest.raises(ValueError, match="exactly 1000 dimensions"):
        _compile_mmes(state=_mmes_state(ndim=4))


def test_mmes_compiler_freezes_phase_boundary_lifecycle_and_parameters() -> None:
    state = _mmes_state()
    action = _compile_mmes(state=state, target_sweep=7)
    expected_parameters = canonical_mmes_parameters(state)

    assert action.trigger_scope == "phase_boundary"
    assert action.issued_sweep == action.target_sweep == action.expires_sweep == 7
    assert action.ttl_sweeps == 0
    assert action.state_dimension == len(state.best_so_far_x)
    assert action.population_size == state.n_individuals
    assert action.optimizer_parameters == expected_parameters
    assert action.optimizer_parameter_hash == expected_parameters.parameter_hash
    assert action.anchor_hash == mmes_resume_anchor_hash(
        "A4",
        state.best_so_far_x,
        state.best_so_far_y,
    )
    assert action.acceptance_fitness == state.best_so_far_y


def test_mmes_budget_state_and_target_drift_change_action_hash() -> None:
    state = _mmes_state()
    baseline = _compile_mmes(state=state, budget_fes=8, target_sweep=7)
    budget_changed = _compile_mmes(state=state, budget_fes=12, target_sweep=7)
    target_changed = _compile_mmes(state=state, budget_fes=8, target_sweep=8)
    state_changed = state.clone()
    state_changed.mean[0] += 1.0
    frozen_state_changed = _compile_mmes(
        state=state_changed,
        budget_fes=8,
        target_sweep=7,
    )

    assert baseline.state_hash == budget_changed.state_hash == target_changed.state_hash
    assert baseline.action_hash != budget_changed.action_hash
    assert baseline.action_hash != target_changed.action_hash
    assert baseline.state_hash != frozen_state_changed.state_hash
    assert baseline.action_hash != frozen_state_changed.action_hash


def _groups_from_path(
    owner_path: tuple[int, ...] = tuple(range(20)),
) -> tuple[tuple[int, ...], ...]:
    groups: list[set[int]] = [set() for _ in range(20)]
    for dimension in range(RELATION_COUNT * 5, FULL_SPACE_DIMENSION):
        groups[dimension % len(groups)].add(dimension)
    for edge_position, (left, right) in enumerate(zip(owner_path, owner_path[1:])):
        shared = set(range(edge_position * 5, edge_position * 5 + 5))
        groups[left].update(shared)
        groups[right].update(shared)
    return tuple(tuple(sorted(group, reverse=True)) for group in groups)


def _incumbent() -> tuple[float, ...]:
    return tuple(index / 10.0 for index in range(FULL_SPACE_DIMENSION))


def _compile(
    *,
    group_dims: tuple[tuple[int, ...], ...] | None = None,
    checkpoint_hash: str = CHECKPOINT_HASH,
):
    return compile_a_sweep_action_candidates(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=checkpoint_hash,
        group_dims=group_dims or _groups_from_path(),
        incumbent=_incumbent(),
        acceptance_fitness=1000.0,
        issued_sweep=3,
        lower_bound=-100.0,
        upper_bound=100.0,
    )


def test_graph_traverses_from_the_smaller_owner_endpoint() -> None:
    owner_path = (17, 3, 12, 6, 14, 1, 19, 8, 11, 4, 15, 5, 10, 7, 16, 9, 13, 0, 18, 2)
    graph = build_a_relation_graph(_groups_from_path(owner_path))

    expected_owner_path = tuple(reversed(owner_path))
    assert tuple(relation.owner_group_indices for relation in graph.relations) == tuple(
        tuple(sorted(pair)) for pair in zip(expected_owner_path, expected_owner_path[1:])
    )
    assert graph.relations[0].owner_group_indices == (2, 18)
    assert len(graph.relations) == RELATION_COUNT
    assert all(len(relation.shared_variable_indices) == 5 for relation in graph.relations)


def test_dimension_order_inside_groups_is_canonical() -> None:
    descending = _groups_from_path()
    ascending = tuple(tuple(sorted(group)) for group in descending)

    assert build_a_relation_graph(descending) == build_a_relation_graph(ascending)


@pytest.mark.parametrize("group_count", [19, 21])
def test_graph_rejects_the_wrong_group_count(group_count: int) -> None:
    groups = _groups_from_path()
    if group_count == 19:
        groups = groups[:-1]
    else:
        groups = (*groups, (999,))

    with pytest.raises(ValueError, match="exactly 20 groups"):
        build_a_relation_graph(groups)


def test_graph_rejects_wrong_overlap_width() -> None:
    groups = [set(group) for group in _groups_from_path()]
    groups[0].remove(4)

    with pytest.raises(ValueError, match="exactly 5 shared dimensions"):
        build_a_relation_graph(tuple(tuple(group) for group in groups))


def test_graph_rejects_incomplete_full_space_coverage() -> None:
    groups = [set(group) for group in _groups_from_path()]
    groups[0].remove(100)

    with pytest.raises(ValueError, match="cover exactly dimensions 0..999"):
        build_a_relation_graph(tuple(tuple(group) for group in groups))


def test_graph_rejects_a_branched_topology() -> None:
    edges = [(0, 1), (0, 2), (0, 3), *[(index, index + 1) for index in range(3, 19)]]
    groups: list[set[int]] = [set() for _ in range(20)]
    for dimension in range(len(edges) * 5, FULL_SPACE_DIMENSION):
        groups[dimension % len(groups)].add(dimension)
    for position, (left, right) in enumerate(edges):
        block = set(range(position * 5, position * 5 + 5))
        groups[left].update(block)
        groups[right].update(block)

    with pytest.raises(ValueError, match="one simple path"):
        build_a_relation_graph(tuple(tuple(group) for group in groups))


def test_graph_rejects_reused_overlap_variables() -> None:
    groups = [set(group) for group in _groups_from_path()]
    first_block = set(range(5))
    second_block = set(range(5, 10))
    groups[1].difference_update(second_block)
    groups[2].difference_update(second_block)
    groups[1].update(first_block)
    groups[2].update(first_block)
    groups[19].update(second_block)

    with pytest.raises(ValueError, match="edges|disjoint"):
        build_a_relation_graph(tuple(tuple(group) for group in groups))


def test_compiler_freezes_both_actions_from_one_checkpoint_graph() -> None:
    candidates = _compile()
    graph = candidates.graph
    cma = candidates.relation_shared_cma_sweep
    no_writeback = candidates.no_writeback_window

    assert cma.dispatch_checkpoint_hash == no_writeback.dispatch_checkpoint_hash
    assert cma.topology_hash == no_writeback.topology_hash == graph.topology_hash
    assert cma.relation_order_hash == no_writeback.relation_order_hash
    assert tuple(block.relation for block in cma.blocks) == no_writeback.relations
    assert cma.seed_namespace == RELATION_CMA_SEED_NAMESPACE
    assert no_writeback.seed_namespace == NO_WRITEBACK_SEED_NAMESPACE
    assert cma.target_sweep == no_writeback.target_sweep == 4
    assert cma.ttl_sweeps == no_writeback.ttl_sweeps == 1
    assert cma.expires_sweep == no_writeback.expires_sweep == 4
    assert len(cma.blocks) == RELATION_COUNT
    assert all(
        block.initial_mean
        == tuple(_incumbent()[index] for index in block.relation.shared_variable_indices)
        for block in cma.blocks
    )
    assert tuple(block.optimizer_seed for block in cma.blocks) == tuple(
        relation_cma_optimizer_seed(
            problem_id="A4",
            run_seed=117,
            dispatch_checkpoint_hash=CHECKPOINT_HASH,
            relation_position=position,
        )
        for position in range(RELATION_COUNT)
    )
    assert len({block.optimizer_seed for block in cma.blocks}) == RELATION_COUNT

    with pytest.raises(FrozenInstanceError):
        cma.target_sweep = 5  # type: ignore[misc]


def test_individual_compilers_match_the_bundle() -> None:
    expected = _compile()
    cma = compile_relation_shared_cma_sweep_action(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        group_dims=_groups_from_path(),
        incumbent=_incumbent(),
        acceptance_fitness=1000.0,
        issued_sweep=3,
        lower_bound=-100.0,
        upper_bound=100.0,
    )
    no_writeback = compile_no_writeback_window_action(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        group_dims=_groups_from_path(),
        issued_sweep=3,
    )

    assert cma == expected.relation_shared_cma_sweep
    assert no_writeback == expected.no_writeback_window


def test_compilation_is_deterministic_and_checkpoint_sensitive() -> None:
    first = _compile()
    second = _compile()
    changed = _compile(checkpoint_hash="b" * 64)

    assert first == second
    assert (
        first.relation_shared_cma_sweep.action_hash != changed.relation_shared_cma_sweep.action_hash
    )
    assert first.no_writeback_window.action_hash != changed.no_writeback_window.action_hash
    assert tuple(block.optimizer_seed for block in first.relation_shared_cma_sweep.blocks) != tuple(
        block.optimizer_seed for block in changed.relation_shared_cma_sweep.blocks
    )


def test_private_dimension_drift_changes_source_graph_order_and_actions() -> None:
    original_groups = _groups_from_path()
    changed_groups = [set(group) for group in original_groups]
    changed_groups[0].remove(100)
    changed_groups[1].remove(101)
    changed_groups[0].add(101)
    changed_groups[1].add(100)
    changed_groups_tuple = tuple(tuple(group) for group in changed_groups)

    original = _compile(group_dims=original_groups)
    changed = _compile(group_dims=changed_groups_tuple)

    assert original.graph.relations == changed.graph.relations
    assert original.graph.relation_order_hash == changed.graph.relation_order_hash
    assert original.graph.source_hash != changed.graph.source_hash
    assert original.graph.graph_hash != changed.graph.graph_hash
    assert original.graph.order_hash != changed.graph.order_hash
    assert (
        original.relation_shared_cma_sweep.action_hash
        != changed.relation_shared_cma_sweep.action_hash
    )
    assert original.no_writeback_window.action_hash != changed.no_writeback_window.action_hash


def test_group_owner_reordering_changes_the_source_binding() -> None:
    groups = _groups_from_path()
    swapped = (groups[1], groups[0], *groups[2:])

    original = build_a_relation_graph(groups)
    changed = build_a_relation_graph(swapped)

    assert original.source_hash != changed.source_hash
    assert original.graph_hash != changed.graph_hash
    assert original.order_hash != changed.order_hash


def test_seed_binds_problem_run_checkpoint_and_relation_position() -> None:
    base = relation_cma_optimizer_seed(
        problem_id="A4",
        run_seed=117,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        relation_position=0,
    )

    assert base != relation_cma_optimizer_seed(
        problem_id="A5",
        run_seed=117,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        relation_position=0,
    )
    assert base != relation_cma_optimizer_seed(
        problem_id="A4",
        run_seed=118,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        relation_position=0,
    )
    assert base != relation_cma_optimizer_seed(
        problem_id="A4",
        run_seed=117,
        dispatch_checkpoint_hash="b" * 64,
        relation_position=0,
    )
    assert base != relation_cma_optimizer_seed(
        problem_id="A4",
        run_seed=117,
        dispatch_checkpoint_hash=CHECKPOINT_HASH,
        relation_position=1,
    )


def test_compiler_fails_closed_on_relation_seed_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "arac.backends.hcc_a_actions.relation_cma_optimizer_seed",
        lambda **_kwargs: 7,
    )

    with pytest.raises(ValueError, match="optimizer seeds must be unique"):
        _compile()


def test_compiler_module_has_no_benchmark_reference_dependency() -> None:
    import arac.backends.hcc_a_actions as module

    source = module.__loader__.get_source(module.__name__)  # type: ignore[union-attr]
    assert source is not None
    for forbidden in (
        "Benchmark",
        "Pvector",
        "Ovector",
        "xopt",
        "order_grouping_by_aob_topology",
    ):
        assert forbidden not in source
