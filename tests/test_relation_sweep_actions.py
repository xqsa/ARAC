from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from arac.actions.relation_sweep import (
    FULL_SPACE_DIMENSION,
    RELATION_CMA_BLOCK_BUDGET_FES,
    RELATION_CMA_GENERATION_COUNT,
    RELATION_CMA_PARAMETERS_HASH,
    RELATION_CMA_POPULATION_SIZE,
    RELATION_CMA_REFERENCE_VERSION,
    RELATION_CMA_TOTAL_BUDGET_FES,
    FrozenRelationCmaBlock,
    NoWritebackWindowAction,
    NoWritebackWindowExecutionState,
    OwnerContextMemorySnapshot,
    OwnerContextMemoryTransition,
    OwnerMemorySyncReceipt,
    OwnerMemorySyncRequest,
    RelationBlockKey,
    RelationCmaTestOnlyImplementationReceipt,
    RelationSharedCmaExecutionContext,
    RelationSharedCmaSweepAction,
    RelationSharedCmaSweepExecutionState,
    execute_no_writeback_window_action,
    execute_relation_shared_cma_sweep_action,
    full_space_vector_hash,
    no_writeback_window_anchor_hash,
    owner_context_memory_hash,
    ordered_relations_hash,
    relation_cma_anchor_hash,
    shared_values_hash,
)


def _hash(character: str) -> str:
    return character * 64


def _relations() -> tuple[RelationBlockKey, ...]:
    return tuple(
        RelationBlockKey((index, index + 1), tuple(range(5 * index, 5 * index + 5)))
        for index in range(19)
    )


def _blocks(
    incumbent: tuple[float, ...],
) -> tuple[FrozenRelationCmaBlock, ...]:
    return tuple(
        FrozenRelationCmaBlock(
            relation=relation,
            initial_mean=tuple(incumbent[index] for index in relation.shared_variable_indices),
            optimizer_seed=2026071901 + position,
        )
        for position, relation in enumerate(_relations())
    )


def _cma_action() -> RelationSharedCmaSweepAction:
    incumbent = tuple(0.0 for _ in range(FULL_SPACE_DIMENSION))
    blocks = _blocks(incumbent)
    return RelationSharedCmaSweepAction(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=_hash("a"),
        topology_hash=_hash("b"),
        anchor_hash=relation_cma_anchor_hash(
            problem_id="A4",
            run_seed=117,
            checkpoint_fe=300_000,
            dispatch_checkpoint_hash=_hash("a"),
            topology_hash=_hash("b"),
            initial_incumbent=incumbent,
            blocks=blocks,
            issued_sweep=3,
        ),
        initial_incumbent=incumbent,
        initial_incumbent_hash=full_space_vector_hash(incumbent),
        acceptance_fitness=1000.0,
        blocks=blocks,
        relation_order_hash=ordered_relations_hash(_relations()),
        seed_namespace="action-ceiling:A4:117:relation-shared-cma",
        issued_sweep=3,
        target_sweep=4,
        ttl_sweeps=1,
        expires_sweep=4,
        lower_bound=-100.0,
        upper_bound=100.0,
    )


class _FakeCma:
    def __init__(
        self,
        problem: dict[str, object],
        options: dict[str, object],
        candidate: float,
        actual_fes: int,
        *,
        one_batch: bool,
    ) -> None:
        self.objective = problem["fitness_function"]
        self.candidate = candidate
        self.actual_fes = actual_fes
        self.one_batch = one_batch
        self.options = options
        self.ndim_problem = problem["ndim_problem"]
        self.lower_boundary = problem["lower_boundary"]
        self.upper_boundary = problem["upper_boundary"]
        self.max_function_evaluations = options["max_function_evaluations"]
        self.mean = options["mean"]
        self.sigma = options["sigma"]
        self.n_individuals = options["n_individuals"]
        self.is_restart = options["is_restart"]
        self.verbose = options["verbose"]
        self.early_stopping_evaluations = options["early_stopping_evaluations"]
        self.seed_rng = options["seed_rng"]
        self.max_runtime = np.inf
        self.fitness_threshold = -np.inf

    def optimize(self) -> dict[str, object]:
        remaining = self.actual_fes
        evaluated_candidates: list[np.ndarray] = []
        evaluated_fitness: list[np.ndarray] = []
        batch_count = 0
        while remaining:
            batch_size = (
                remaining if self.one_batch else min(RELATION_CMA_POPULATION_SIZE, remaining)
            )
            candidates = np.full((batch_size, 5), self.candidate)
            fitness = np.asarray(  # type: ignore[operator]
                self.objective(candidates),
                dtype=float,
            )
            evaluated_candidates.append(candidates)
            evaluated_fitness.append(fitness)
            remaining -= batch_size
            batch_count += 1
        candidates = np.concatenate(evaluated_candidates)
        fitness = np.concatenate(evaluated_fitness)
        best_index = int(np.argmin(fitness))
        return {
            "best_so_far_x": candidates[best_index],
            "best_so_far_y": float(fitness[best_index]),
            "n_function_evaluations": self.actual_fes,
            "mean": candidates[0],
            "sigma": 0.5,
            "p_s": np.zeros(5),
            "p_c": np.zeros(5),
            "e_va": np.ones(5),
            "e_ve": np.eye(5),
            "_n_restart": 0,
            "_n_generations": batch_count - 1,
        }


def _context(
    action: RelationSharedCmaSweepAction,
    *,
    candidate: float,
    actual_fes: int = RELATION_CMA_BLOCK_BUDGET_FES,
    one_batch: bool = False,
    include_test_receipt: bool = True,
) -> tuple[
    RelationSharedCmaExecutionContext,
    list[object],
    list[OwnerMemorySyncRequest],
    dict[int, tuple[tuple[int, ...], list[float]]],
]:
    factory_calls: list[object] = []
    sync_calls: list[OwnerMemorySyncRequest] = []
    owner_dimensions: dict[int, set[int]] = {index: set() for index in range(20)}
    for relation in _relations():
        for owner in relation.owner_group_indices:
            owner_dimensions[owner].update(relation.shared_variable_indices)
    for owner, dimensions in owner_dimensions.items():
        dimensions.add(100 + owner)
    owner_memories = {
        owner: (tuple(sorted(dimensions)), [0.0] * len(dimensions))
        for owner, dimensions in owner_dimensions.items()
    }

    def memory_hash(owner: int) -> str:
        dimensions, mean = owner_memories[owner]
        return owner_context_memory_hash(owner, dimensions, mean)

    def objective(batch: np.ndarray) -> np.ndarray:
        return 1000.0 - np.sum(np.abs(batch), axis=1)

    def factory(problem: dict[str, object], options: dict[str, object]) -> _FakeCma:
        factory_calls.append((problem, options))
        return _FakeCma(
            problem,
            options,
            candidate,
            actual_fes,
            one_batch=one_batch,
        )

    def synchronize(request: OwnerMemorySyncRequest) -> OwnerMemorySyncReceipt:
        sync_calls.append(request)
        transitions: list[OwnerContextMemoryTransition] = []
        for owner, expected_pre_hash in zip(
            request.relation.owner_group_indices,
            request.pre_owner_context_memory_hashes,
            strict=True,
        ):
            dimensions, mean = owner_memories[owner]
            pre_hash = memory_hash(owner)
            assert pre_hash == expected_pre_hash
            local_positions = {dimension: index for index, dimension in enumerate(dimensions)}
            for dimension, value in zip(
                request.relation.shared_variable_indices,
                request.shared_values,
                strict=True,
            ):
                mean[local_positions[dimension]] = value
            transitions.append(
                OwnerContextMemoryTransition(
                    owner_group_index=owner,
                    pre_context_memory_hash=pre_hash,
                    post_context_memory_hash=memory_hash(owner),
                )
            )
        return OwnerMemorySyncReceipt(
            request_hash=request.request_hash,
            relation=request.relation,
            shared_values_hash=shared_values_hash(request.shared_values),
            owner_transitions=(transitions[0], transitions[1]),
        )

    return (
        RelationSharedCmaExecutionContext(
            objective=objective,
            cmaes_factory=factory,
            synchronize_owner_memory=synchronize,
            current_fe=action.checkpoint_fe,
            current_sweep=action.target_sweep,
            dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
            topology_hash=action.topology_hash,
            incumbent=action.initial_incumbent,
            incumbent_fitness=action.acceptance_fitness,
            required_seed_namespace=action.seed_namespace,
            owner_context_memories=tuple(
                OwnerContextMemorySnapshot(
                    owner_group_index=owner,
                    dimensions=owner_memories[owner][0],
                    mean_values=tuple(owner_memories[owner][1]),
                )
                for owner in sorted(owner_memories)
            ),
            test_only_implementation_receipt=(
                RelationCmaTestOnlyImplementationReceipt(
                    implementation_type=(f"{_FakeCma.__module__}.{_FakeCma.__qualname__}"),
                    reference_version=RELATION_CMA_REFERENCE_VERSION,
                    parameter_hash=RELATION_CMA_PARAMETERS_HASH,
                )
                if include_test_receipt
                else None
            ),
        ),
        factory_calls,
        sync_calls,
        owner_memories,
    )


def test_relation_cma_contract_is_frozen_and_rejects_non_checkpoint_mean() -> None:
    action = _cma_action()

    assert len(action.blocks) == 19
    assert action.total_budget_fes == 4750
    assert action.schema == "arac.action.relation_shared_cma_sweep"
    assert action.schema_version == 1
    with pytest.raises(FrozenInstanceError):
        action.population_size = 12  # type: ignore[misc]

    incumbent = action.initial_incumbent
    bad_blocks = list(action.blocks)
    bad_blocks[0] = FrozenRelationCmaBlock(
        relation=bad_blocks[0].relation,
        initial_mean=(1.0,) * 5,
        optimizer_seed=bad_blocks[0].optimizer_seed,
    )
    with pytest.raises(ValueError, match="checkpoint shared values"):
        RelationSharedCmaSweepAction(
            **{
                **action.__dict__,
                "blocks": tuple(bad_blocks),
                "anchor_hash": relation_cma_anchor_hash(
                    problem_id=action.problem_id,
                    run_seed=action.run_seed,
                    checkpoint_fe=action.checkpoint_fe,
                    dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
                    topology_hash=action.topology_hash,
                    initial_incumbent=incumbent,
                    blocks=bad_blocks,
                    issued_sweep=action.issued_sweep,
                ),
            }
        )

    duplicate_seed_blocks = list(action.blocks)
    duplicate_seed_blocks[1] = replace(
        duplicate_seed_blocks[1],
        optimizer_seed=duplicate_seed_blocks[0].optimizer_seed,
    )
    with pytest.raises(ValueError, match="optimizer seeds must be unique"):
        replace(action, blocks=tuple(duplicate_seed_blocks))


def test_relation_cma_rejects_unpinned_factory_before_objective() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, _, _ = _context(
        action,
        candidate=0.1,
        include_test_receipt=False,
    )
    objective_calls = 0

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal objective_calls
        objective_calls += len(batch)
        return np.zeros(len(batch))

    with pytest.raises(RuntimeError, match="pinned implementation"):
        execute_relation_shared_cma_sweep_action(
            action,
            state,
            replace(context, objective=objective),
        )

    assert len(factory_calls) == 1
    assert objective_calls == 0
    assert state.status == "failed"
    assert state.consumed_fes == 0


@pytest.mark.parametrize(
    ("option_name", "drifted_value", "error_match"),
    (
        ("n_individuals", 11, "population_size drifted"),
        ("sigma", 0.75, "sigma drifted"),
        ("is_restart", True, "restart policy drifted"),
        ("early_stopping_evaluations", 100, "early-stopping policy drifted"),
        ("seed_rng", 999, "seed drifted"),
    ),
)
def test_relation_cma_rejects_static_factory_parameter_drift_before_objective(
    option_name: str,
    drifted_value: object,
    error_match: str,
) -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, _, _, _ = _context(action, candidate=0.1)
    canonical_factory = context.cmaes_factory
    objective_calls = 0

    def drifted_factory(
        problem: dict[str, object],
        options: dict[str, object],
    ) -> _FakeCma:
        drifted_options = dict(options)
        drifted_options[option_name] = drifted_value
        return canonical_factory(problem, drifted_options)

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal objective_calls
        objective_calls += len(batch)
        return np.zeros(len(batch))

    with pytest.raises(RuntimeError, match=error_match):
        execute_relation_shared_cma_sweep_action(
            action,
            state,
            replace(
                context,
                objective=objective,
                cmaes_factory=drifted_factory,
            ),
        )

    assert objective_calls == 0
    assert state.status == "failed"
    assert state.consumed_fes == 0


def test_relation_cma_rejects_one_batch_factory_generation_claim() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, _, _, _ = _context(action, candidate=0.1, one_batch=True)

    with pytest.raises(RuntimeError, match="population-generation schedule"):
        execute_relation_shared_cma_sweep_action(action, state, context)

    assert state.status == "failed"
    assert state.consumed_fes == RELATION_CMA_BLOCK_BUDGET_FES
    assert state.completed_fe == action.checkpoint_fe + RELATION_CMA_BLOCK_BUDGET_FES


def test_relation_cma_executor_consumes_exact_budget_and_syncs_both_owners() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, sync_calls, owner_memories = _context(action, candidate=0.1)

    result = execute_relation_shared_cma_sweep_action(action, state, context)

    assert len(factory_calls) == len(action.blocks) == 19
    assert len(sync_calls) == result.accepted_block_count == 19
    assert result.consumed_fes == RELATION_CMA_TOTAL_BUDGET_FES
    assert state.status == "completed"
    assert state.started_fe == action.checkpoint_fe
    assert state.completed_fe == action.checkpoint_fe + action.total_budget_fes
    assert result.incumbent_fitness == pytest.approx(990.5)
    assert all(block.consumed_fes == 250 for block in result.block_results)
    assert all(
        block.sampled_generation_count == RELATION_CMA_GENERATION_COUNT
        for block in result.block_results
    )
    assert tuple(call.relation for call in sync_calls) == tuple(
        block.relation for block in action.blocks
    )
    for call in sync_calls:
        for owner in call.relation.owner_group_indices:
            dimensions, mean = owner_memories[owner]
            local_positions = {dimension: index for index, dimension in enumerate(dimensions)}
            assert all(
                mean[local_positions[dimension]] == value
                for dimension, value in zip(
                    call.relation.shared_variable_indices,
                    call.shared_values,
                    strict=True,
                )
            )


def test_relation_cma_strict_tie_rejects_without_owner_sync() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, sync_calls, _ = _context(action, candidate=0.0)

    result = execute_relation_shared_cma_sweep_action(action, state, context)

    assert result.accepted_block_count == 0
    assert sync_calls == []
    assert result.incumbent == action.initial_incumbent
    assert result.incumbent_fitness == action.acceptance_fitness

    lifecycle_payload = state.audit_payload(action)
    lifecycle_hash = state.state_hash(action)
    duplicate = execute_relation_shared_cma_sweep_action(action, state, context)
    assert duplicate.abstained is True
    assert duplicate.invalidation_reason == "action_already_consumed"
    assert duplicate.consumed_fes == 0
    assert state.audit_payload(action) == lifecycle_payload
    assert state.state_hash(action) == lifecycle_hash
    assert len(factory_calls) == 19


def test_relation_cma_preflight_mismatch_abstains_before_optimizer() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, _, _ = _context(action, candidate=0.1)

    result = execute_relation_shared_cma_sweep_action(
        action,
        state,
        replace(context, current_sweep=action.expires_sweep + 1),
    )

    assert result.abstained is True
    assert result.invalidation_reason == "action_expired"
    assert state.status == "abstained"
    assert state.invalidation_reason == "action_expired"
    assert state.consumed_fes == 0
    assert factory_calls == []

    repeated = execute_relation_shared_cma_sweep_action(action, state, context)
    assert repeated.abstained is True
    assert repeated.invalidation_reason == "action_invalidated:action_expired"
    assert factory_calls == []


def test_relation_cma_running_state_returns_explicit_abstain_before_optimizer() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    state.start(action, current_fe=action.checkpoint_fe)
    context, factory_calls, _, _ = _context(action, candidate=0.1)
    lifecycle_hash = state.state_hash(action)

    result = execute_relation_shared_cma_sweep_action(action, state, context)

    assert result.abstained is True
    assert result.invalidation_reason == "action_already_running"
    assert result.lifecycle_hash == lifecycle_hash
    assert state.status == "running"
    assert factory_calls == []


def test_relation_cma_fails_on_short_optimizer_execution() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, _, _ = _context(action, candidate=0.1, actual_fes=249)

    with pytest.raises(RuntimeError, match="exact frozen FE budget"):
        execute_relation_shared_cma_sweep_action(action, state, context)

    assert state.status == "failed"
    assert state.started_fe == action.checkpoint_fe
    assert state.completed_fe == action.checkpoint_fe + 249
    assert state.consumed_fes == 249
    assert state.accepted_block_count == 0
    assert state.post_incumbent_hash == action.initial_incumbent_hash
    assert state.failure_reason == (
        "RuntimeError:relation_cma_did_not_consume_its_exact_frozen_fe_budget"
    )
    failed_payload = state.audit_payload(action)
    failed_hash = state.state_hash(action)

    repeated = execute_relation_shared_cma_sweep_action(action, state, context)

    assert repeated.abstained is True
    assert repeated.invalidation_reason == f"action_failed:{state.failure_reason}"
    assert repeated.consumed_fes == 0
    assert len(factory_calls) == 1
    assert state.audit_payload(action) == failed_payload
    assert state.state_hash(action) == failed_hash


def test_relation_cma_failed_ledger_exposes_251_observed_fes() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, _, _ = _context(action, candidate=0.1, actual_fes=251)

    with pytest.raises(RuntimeError, match="exact frozen FE budget"):
        execute_relation_shared_cma_sweep_action(action, state, context)

    assert state.status == "failed"
    assert state.consumed_fes == 251
    assert state.completed_fe == action.checkpoint_fe + 251
    assert state.accepted_block_count == 0
    assert state.post_incumbent_hash == action.initial_incumbent_hash
    assert len(factory_calls) == 1

    repeated = execute_relation_shared_cma_sweep_action(action, state, context)
    assert repeated.abstained is True
    assert repeated.invalidation_reason.startswith("action_failed:")
    assert len(factory_calls) == 1


def test_relation_cma_owner_sync_exception_seals_observed_fe_before_reraise() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, factory_calls, _, _ = _context(action, candidate=0.1)

    def fail_owner_sync(_request: OwnerMemorySyncRequest) -> OwnerMemorySyncReceipt:
        raise RuntimeError("owner sync failed")

    with pytest.raises(RuntimeError, match="owner sync failed"):
        execute_relation_shared_cma_sweep_action(
            action,
            state,
            replace(context, synchronize_owner_memory=fail_owner_sync),
        )

    assert state.status == "failed"
    assert state.consumed_fes == 250
    assert state.completed_fe == action.checkpoint_fe + 250
    assert state.accepted_block_count == 0
    assert state.post_incumbent_hash == action.initial_incumbent_hash
    assert state.failure_reason == "RuntimeError:owner_sync_failed"
    assert len(factory_calls) == 1


def test_relation_cma_rejects_noop_owner_sync_receipt() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, _, _, _ = _context(action, candidate=0.1)

    def no_op_sync(request: OwnerMemorySyncRequest) -> OwnerMemorySyncReceipt:
        transitions = tuple(
            OwnerContextMemoryTransition(
                owner_group_index=owner,
                pre_context_memory_hash=pre_hash,
                post_context_memory_hash=pre_hash,
            )
            for owner, pre_hash in zip(
                request.relation.owner_group_indices,
                request.pre_owner_context_memory_hashes,
                strict=True,
            )
        )
        return OwnerMemorySyncReceipt(
            request_hash=request.request_hash,
            relation=request.relation,
            shared_values_hash=shared_values_hash(request.shared_values),
            owner_transitions=(transitions[0], transitions[1]),
        )

    with pytest.raises(RuntimeError, match="post-context mismatch"):
        execute_relation_shared_cma_sweep_action(
            action,
            state,
            replace(context, synchronize_owner_memory=no_op_sync),
        )

    assert state.status == "failed"
    assert state.consumed_fes == RELATION_CMA_BLOCK_BUDGET_FES
    assert state.accepted_block_count == 0
    assert state.post_incumbent_hash == action.initial_incumbent_hash


def test_relation_cma_malformed_objective_output_still_records_submitted_fes() -> None:
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, _, _, _ = _context(action, candidate=0.1)

    def malformed_objective(_batch: np.ndarray) -> np.ndarray:
        return np.asarray([1.0])

    with pytest.raises(ValueError, match="one finite fitness per candidate"):
        execute_relation_shared_cma_sweep_action(
            action,
            state,
            replace(context, objective=malformed_objective),
        )

    assert state.status == "failed"
    assert state.consumed_fes == RELATION_CMA_POPULATION_SIZE
    assert state.completed_fe == action.checkpoint_fe + RELATION_CMA_POPULATION_SIZE


def test_relation_cma_wrong_action_lifecycle_is_rejected_without_mutation() -> None:
    action = _cma_action()
    other_action = replace(
        action,
        seed_namespace=f"{action.seed_namespace}:other",
    )
    state = RelationSharedCmaSweepExecutionState.for_action(other_action)
    observed_state_hash = state.observed_state_hash()
    context, factory_calls, _, _ = _context(action, candidate=0.1)

    result = execute_relation_shared_cma_sweep_action(action, state, context)

    assert result.abstained is True
    assert result.invalidation_reason == "lifecycle_action_hash_mismatch"
    assert result.observed_lifecycle_action_hash == other_action.action_hash
    assert result.lifecycle_hash == observed_state_hash
    assert state.status == "issued"
    assert state.observed_state_hash() == observed_state_hash
    assert factory_calls == []


def test_relation_cma_runs_real_vendor_full_cma_for_4750_fes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
    monkeypatch.syspath_prepend(str(vendor_root))
    cmaes_class = importlib.import_module("HCC.OPT.CMAES.cmaes").CMAES
    action = _cma_action()
    state = RelationSharedCmaSweepExecutionState.for_action(action)
    context, _, _, _ = _context(action, candidate=0.1)

    def objective(batch: np.ndarray) -> np.ndarray:
        return np.sum(np.square(batch - 1.0), axis=1)

    result = execute_relation_shared_cma_sweep_action(
        action,
        state,
        replace(
            context,
            objective=objective,
            cmaes_factory=cmaes_class,
            test_only_implementation_receipt=None,
        ),
    )

    assert result.consumed_fes == 4750
    assert len(result.block_results) == 19
    assert state.status == "completed"
    assert state.completed_fe == action.checkpoint_fe + 4750
    assert all(block.sampled_generation_count == 25 for block in result.block_results)
    assert all(
        block.optimizer_contract_receipt.implementation_type == "HCC.OPT.CMAES.cmaes.CMAES"
        for block in result.block_results
    )


def _no_writeback_action(
    *,
    seed_namespace: str = "action-ceiling:A4:117:no-writeback-window",
) -> NoWritebackWindowAction:
    relations = _relations()
    return NoWritebackWindowAction(
        problem_id="A4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=_hash("a"),
        topology_hash=_hash("b"),
        anchor_hash=no_writeback_window_anchor_hash(
            problem_id="A4",
            run_seed=117,
            checkpoint_fe=300_000,
            dispatch_checkpoint_hash=_hash("a"),
            topology_hash=_hash("b"),
            relations=relations,
            issued_sweep=3,
            seed_namespace=seed_namespace,
        ),
        relations=relations,
        relation_order_hash=ordered_relations_hash(relations),
        seed_namespace=seed_namespace,
        issued_sweep=3,
        target_sweep=4,
        ttl_sweeps=1,
        expires_sweep=4,
    )


def _consume(
    action: NoWritebackWindowAction,
    state: NoWritebackWindowExecutionState,
    relation: RelationBlockKey,
    *,
    current_fe: int | None = None,
):
    return execute_no_writeback_window_action(
        action,
        state,
        relation=relation,
        current_sweep=4,
        current_fe=(
            action.checkpoint_fe + len(state.consumed_relations) + 1
            if current_fe is None
            else current_fe
        ),
        dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
        topology_hash=action.topology_hash,
        required_seed_namespace=action.seed_namespace,
    )


def test_no_writeback_window_consumes_once_in_frozen_order() -> None:
    action = _no_writeback_action()
    state = NoWritebackWindowExecutionState.for_action(action)

    assert action.schema == "arac.action.no_writeback_window"
    assert action.schema_version == 1
    assert action.budget_fes == 0

    decisions = [_consume(action, state, relation) for relation in action.relations]

    assert all(decision.suppress_writeback for decision in decisions)
    assert state.status == "completed"
    assert state.started_fe == action.checkpoint_fe + 1
    assert state.completed_fe == action.checkpoint_fe + 19
    duplicate = _consume(action, state, action.relations[-1])
    assert duplicate.abstained is True
    assert duplicate.reason == "action_already_consumed"
    assert duplicate.suppress_writeback is False


@pytest.mark.parametrize(
    ("observed_index", "reason"),
    ((1, "relation_order_mismatch"), (0, "duplicate_relation")),
)
def test_no_writeback_window_abstains_on_order_or_duplicate(
    observed_index: int,
    reason: str,
) -> None:
    action = _no_writeback_action()
    state = NoWritebackWindowExecutionState.for_action(action)
    if reason == "duplicate_relation":
        assert _consume(action, state, action.relations[0]).suppress_writeback

    decision = _consume(action, state, action.relations[observed_index])

    assert decision.abstained is True
    assert decision.reason == reason
    assert state.status == "abstained"


def test_no_writeback_window_finish_exposes_missing_relation() -> None:
    action = _no_writeback_action()
    state = NoWritebackWindowExecutionState.for_action(action)
    _consume(action, state, action.relations[0])

    decision = state.finish_sweep(action, current_sweep=4)

    assert decision.abstained is True
    assert decision.reason.startswith("missing_relation:")
    assert state.status == "abstained"


@pytest.mark.parametrize(
    ("first_fe", "second_fe", "reason"),
    (
        (300_000, None, "first_relation_fe_not_after_checkpoint"),
        (300_001, 300_001, "non_monotonic_relation_fe"),
    ),
)
def test_no_writeback_window_abstains_on_invalid_fe_sequence(
    first_fe: int,
    second_fe: int | None,
    reason: str,
) -> None:
    action = _no_writeback_action()
    state = NoWritebackWindowExecutionState.for_action(action)

    first = _consume(action, state, action.relations[0], current_fe=first_fe)
    decision = (
        first
        if first.abstained
        else _consume(
            action,
            state,
            action.relations[1],
            current_fe=second_fe,
        )
    )

    assert decision.abstained is True
    assert decision.reason == reason


def test_no_writeback_wrong_action_lifecycle_is_rejected_without_mutation() -> None:
    action = _no_writeback_action()
    other_action = _no_writeback_action(
        seed_namespace=f"{action.seed_namespace}:other",
    )
    state = NoWritebackWindowExecutionState.for_action(other_action)
    observed_state_hash = state.observed_state_hash()

    decision = _consume(action, state, action.relations[0])

    assert decision.abstained is True
    assert decision.reason == "lifecycle_action_hash_mismatch"
    assert decision.expected_action_hash == action.action_hash
    assert decision.observed_lifecycle_action_hash == other_action.action_hash
    assert decision.observed_state_hash == observed_state_hash
    assert state.status == "issued"
    assert state.observed_state_hash() == observed_state_hash
