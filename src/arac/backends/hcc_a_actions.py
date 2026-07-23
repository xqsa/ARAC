"""Reference-blind A-series relation graph and action compilers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from arac.actions.mmes_resume import (
    FrozenMmesState,
    Phase1MmesResumeAction,
    canonical_mmes_parameters,
    mmes_resume_anchor_hash,
    mmes_vector_hash,
)
from arac.actions.relation_sweep import (
    FULL_SPACE_DIMENSION,
    RELATION_COUNT,
    RELATION_DIMENSION,
    FrozenRelationCmaBlock,
    NoWritebackWindowAction,
    RelationBlockKey,
    RelationSharedCmaSweepAction,
    full_space_vector_hash,
    no_writeback_window_anchor_hash,
    ordered_relations_hash,
    relation_cma_anchor_hash,
)


A_RELATION_GRAPH_SCHEMA = "arac.hcc.a_relation_graph"
A_RELATION_GRAPH_SCHEMA_VERSION = 1
A_RELATION_SOURCE_SCHEMA = "arac.hcc.a_relation_source"
A_RELATION_ORDER_SCHEMA = "arac.hcc.a_relation_order"
RELATION_CMA_SEED_NAMESPACE = "arac:a-series:relation-shared-cma:v1"
NO_WRITEBACK_SEED_NAMESPACE = "arac:a-series:no-writeback-window:v1"
PHASE1_MMES_SEED_NAMESPACE = "arac:phase1-mmes-resume:v1"

_GROUP_COUNT = RELATION_COUNT + 1
_HASH_LENGTH = 64


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value: int, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    converted = int(value)
    if converted < 0 or (maximum is not None and converted > maximum):
        suffix = f" <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a non-negative integer{suffix}")
    return converted


def _problem_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("problem_id must be a non-empty string")
    return value


def _validate_hash(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _canonical_group_dims(
    group_dims: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    groups = tuple(group_dims)
    if len(groups) != _GROUP_COUNT:
        raise ValueError(f"A-series relation graph requires exactly {_GROUP_COUNT} groups")

    canonical: list[tuple[int, ...]] = []
    for group_index, raw_group in enumerate(groups):
        dimensions = tuple(
            _integer(
                value,
                f"group_dims[{group_index}] dimension",
                maximum=FULL_SPACE_DIMENSION - 1,
            )
            for value in raw_group
        )
        if not dimensions:
            raise ValueError("A-series groups must be non-empty")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("A-series groups must not contain duplicate dimensions")
        canonical.append(tuple(sorted(dimensions)))
    return tuple(canonical)


def _source_hash(groups: tuple[tuple[int, ...], ...]) -> str:
    return _canonical_sha256(
        {
            "schema": A_RELATION_SOURCE_SCHEMA,
            "schema_version": A_RELATION_GRAPH_SCHEMA_VERSION,
            "owner_indexed_groups": [
                {"owner": owner, "dimensions": list(dimensions)}
                for owner, dimensions in enumerate(groups)
            ],
        }
    )


@dataclass(frozen=True)
class ARelationGraph:
    """Canonical 20-owner path inferred only from RDDSM memberships."""

    group_dims: tuple[tuple[int, ...], ...]
    relations: tuple[RelationBlockKey, ...]
    source_hash: str
    graph_hash: str
    order_hash: str
    relation_order_hash: str

    @property
    def topology_hash(self) -> str:
        return self.graph_hash


def build_a_relation_graph(
    group_dims: Sequence[Sequence[int]],
) -> ARelationGraph:
    """Build the strict A4 path without consulting benchmark reference metadata."""

    groups = _canonical_group_dims(group_dims)
    group_sets = tuple(set(group) for group in groups)
    covered_dimensions = set().union(*group_sets)
    if covered_dimensions != set(range(FULL_SPACE_DIMENSION)):
        raise ValueError(
            f"A-series groups must cover exactly dimensions 0..{FULL_SPACE_DIMENSION - 1}"
        )
    edge_by_owners: dict[tuple[int, int], RelationBlockKey] = {}
    adjacency = [set() for _ in groups]

    for left in range(_GROUP_COUNT):
        for right in range(left + 1, _GROUP_COUNT):
            shared = tuple(sorted(group_sets[left].intersection(group_sets[right])))
            if not shared:
                continue
            if len(shared) != RELATION_DIMENSION:
                raise ValueError(
                    "every A-series relation must contain exactly "
                    f"{RELATION_DIMENSION} shared dimensions"
                )
            owners = (left, right)
            edge_by_owners[owners] = RelationBlockKey(owners, shared)
            adjacency[left].add(right)
            adjacency[right].add(left)

    if len(edge_by_owners) != RELATION_COUNT:
        raise ValueError(f"A-series relation graph requires exactly {RELATION_COUNT} edges")

    shared_blocks = [set(relation.shared_variable_indices) for relation in edge_by_owners.values()]
    for position, block in enumerate(shared_blocks):
        if any(block.intersection(other) for other in shared_blocks[position + 1 :]):
            raise ValueError("A-series relation blocks must be pairwise disjoint")

    degrees = tuple(len(neighbors) for neighbors in adjacency)
    endpoints = tuple(index for index, degree in enumerate(degrees) if degree == 1)
    if len(endpoints) != 2 or any(degree not in (1, 2) for degree in degrees):
        raise ValueError("A-series owner graph must be one simple path")

    owner_order: list[int] = []
    previous: int | None = None
    current = min(endpoints)
    while True:
        owner_order.append(current)
        forward = adjacency[current] - ({previous} if previous is not None else set())
        if not forward:
            break
        if len(forward) != 1:
            raise ValueError("A-series owner graph must be one simple path")
        previous, current = current, next(iter(forward))
    if len(owner_order) != _GROUP_COUNT:
        raise ValueError("A-series owner graph must be one connected simple path")

    relations = tuple(
        edge_by_owners[tuple(sorted((left, right)))]
        for left, right in zip(owner_order, owner_order[1:])
    )
    source_hash = _source_hash(groups)
    graph_hash = _canonical_sha256(
        {
            "schema": A_RELATION_GRAPH_SCHEMA,
            "schema_version": A_RELATION_GRAPH_SCHEMA_VERSION,
            "source_hash": source_hash,
            "edges": [relation.audit_payload() for relation in sorted(edge_by_owners.values())],
        }
    )
    order_hash = _canonical_sha256(
        {
            "schema": A_RELATION_ORDER_SCHEMA,
            "schema_version": A_RELATION_GRAPH_SCHEMA_VERSION,
            "graph_hash": graph_hash,
            "relations": [relation.audit_payload() for relation in relations],
        }
    )
    return ARelationGraph(
        group_dims=groups,
        relations=relations,
        source_hash=source_hash,
        graph_hash=graph_hash,
        order_hash=order_hash,
        relation_order_hash=ordered_relations_hash(relations),
    )


def relation_cma_optimizer_seed(
    *,
    problem_id: str,
    run_seed: int,
    dispatch_checkpoint_hash: str,
    relation_position: int,
) -> int:
    """Derive one vendor-compatible 32-bit seed from the frozen checkpoint."""

    digest = _canonical_sha256(
        {
            "namespace": RELATION_CMA_SEED_NAMESPACE,
            "problem_id": _problem_id(problem_id),
            "run_seed": _integer(run_seed, "run_seed"),
            "dispatch_checkpoint_hash": _validate_hash(
                dispatch_checkpoint_hash,
                "dispatch_checkpoint_hash",
            ),
            "relation_position": _integer(
                relation_position,
                "relation_position",
                maximum=RELATION_COUNT - 1,
            ),
        }
    )
    return int(digest[:8], 16)


def _frozen_incumbent(values: Sequence[float]) -> tuple[float, ...]:
    incumbent = tuple(float(value) for value in values)
    if len(incumbent) != FULL_SPACE_DIMENSION:
        raise ValueError(f"incumbent must contain exactly {FULL_SPACE_DIMENSION} values")
    if any(not math.isfinite(value) for value in incumbent):
        raise ValueError("incumbent values must be finite")
    return incumbent


def compile_phase1_mmes_resume_action(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    state: object,
    incumbent: Sequence[float],
    incumbent_fitness: float,
    budget_fes: int,
    target_sweep: int,
) -> Phase1MmesResumeAction:
    """Freeze one exact MMES continuation at its Phase1 boundary."""

    problem = _problem_id(problem_id)
    seed = _integer(run_seed, "run_seed")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    checkpoint_hash = _validate_hash(
        dispatch_checkpoint_hash,
        "dispatch_checkpoint_hash",
    )
    budget = _integer(budget_fes, "budget_fes")
    target = _integer(target_sweep, "target_sweep")

    snapshot = FrozenMmesState.capture(state)
    frozen_state = snapshot.clone_state()
    state_checkpoint_fe = _integer(
        getattr(frozen_state, "n_function_evaluations"),
        "state.n_function_evaluations",
    )
    if state_checkpoint_fe != checkpoint:
        raise ValueError("checkpoint_fe must exactly equal state.n_function_evaluations")
    state_incumbent = np.asarray(
        getattr(frozen_state, "best_so_far_x"),
        dtype=np.float64,
    )
    if state_incumbent.shape != (FULL_SPACE_DIMENSION,):
        raise ValueError(
            f"Phase1 MMES state must contain exactly {FULL_SPACE_DIMENSION} dimensions"
        )
    runner_incumbent = np.asarray(incumbent, dtype=np.float64)
    if mmes_vector_hash(runner_incumbent) != mmes_vector_hash(state_incumbent):
        raise ValueError("incumbent must exactly equal state.best_so_far_x")

    state_fitness = float(getattr(frozen_state, "best_so_far_y"))
    runner_fitness = float(incumbent_fitness)
    if not math.isfinite(runner_fitness):
        raise ValueError("incumbent_fitness must be finite")
    if runner_fitness != state_fitness:
        raise ValueError("incumbent_fitness must exactly equal state.best_so_far_y")

    parameters = canonical_mmes_parameters(frozen_state)
    frozen_incumbent = tuple(float(value) for value in state_incumbent)
    return Phase1MmesResumeAction(
        problem_id=problem,
        run_seed=seed,
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=checkpoint_hash,
        anchor_hash=mmes_resume_anchor_hash(
            problem,
            frozen_incumbent,
            state_fitness,
        ),
        state_snapshot=snapshot,
        state_dimension=len(frozen_incumbent),
        population_size=int(getattr(frozen_state, "n_individuals")),
        budget_fes=budget,
        optimizer_parameters=parameters,
        optimizer_parameter_hash=parameters.parameter_hash,
        seed_namespace=PHASE1_MMES_SEED_NAMESPACE,
        acceptance_fitness=state_fitness,
        issued_sweep=target,
        target_sweep=target,
        ttl_sweeps=0,
        expires_sweep=target,
    )


def _compile_relation_shared_cma_sweep_action(
    *,
    graph: ARelationGraph,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    incumbent: Sequence[float],
    acceptance_fitness: float,
    issued_sweep: int,
    lower_bound: float,
    upper_bound: float,
) -> RelationSharedCmaSweepAction:
    problem = _problem_id(problem_id)
    seed = _integer(run_seed, "run_seed")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    checkpoint_hash = _validate_hash(
        dispatch_checkpoint_hash,
        "dispatch_checkpoint_hash",
    )
    issued = _integer(issued_sweep, "issued_sweep")
    frozen_incumbent = _frozen_incumbent(incumbent)
    blocks = tuple(
        FrozenRelationCmaBlock(
            relation=relation,
            initial_mean=tuple(
                frozen_incumbent[index] for index in relation.shared_variable_indices
            ),
            optimizer_seed=relation_cma_optimizer_seed(
                problem_id=problem,
                run_seed=seed,
                dispatch_checkpoint_hash=checkpoint_hash,
                relation_position=position,
            ),
        )
        for position, relation in enumerate(graph.relations)
    )
    anchor_hash = relation_cma_anchor_hash(
        problem_id=problem,
        run_seed=seed,
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=checkpoint_hash,
        topology_hash=graph.topology_hash,
        initial_incumbent=frozen_incumbent,
        blocks=blocks,
        issued_sweep=issued,
    )
    return RelationSharedCmaSweepAction(
        problem_id=problem,
        run_seed=seed,
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=checkpoint_hash,
        topology_hash=graph.topology_hash,
        anchor_hash=anchor_hash,
        initial_incumbent=frozen_incumbent,
        initial_incumbent_hash=full_space_vector_hash(frozen_incumbent),
        acceptance_fitness=float(acceptance_fitness),
        blocks=blocks,
        relation_order_hash=graph.relation_order_hash,
        seed_namespace=RELATION_CMA_SEED_NAMESPACE,
        issued_sweep=issued,
        target_sweep=issued + 1,
        ttl_sweeps=1,
        expires_sweep=issued + 1,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
    )


def compile_relation_shared_cma_sweep_action(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    group_dims: Sequence[Sequence[int]],
    incumbent: Sequence[float],
    acceptance_fitness: float,
    issued_sweep: int,
    lower_bound: float,
    upper_bound: float,
) -> RelationSharedCmaSweepAction:
    """Freeze all 19 CMA blocks from one completed-sweep checkpoint."""

    return _compile_relation_shared_cma_sweep_action(
        graph=build_a_relation_graph(group_dims),
        problem_id=problem_id,
        run_seed=run_seed,
        checkpoint_fe=checkpoint_fe,
        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
        incumbent=incumbent,
        acceptance_fitness=acceptance_fitness,
        issued_sweep=issued_sweep,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def _compile_no_writeback_window_action(
    *,
    graph: ARelationGraph,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    issued_sweep: int,
) -> NoWritebackWindowAction:
    problem = _problem_id(problem_id)
    seed = _integer(run_seed, "run_seed")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    checkpoint_hash = _validate_hash(
        dispatch_checkpoint_hash,
        "dispatch_checkpoint_hash",
    )
    issued = _integer(issued_sweep, "issued_sweep")
    anchor_hash = no_writeback_window_anchor_hash(
        problem_id=problem,
        run_seed=seed,
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=checkpoint_hash,
        topology_hash=graph.topology_hash,
        relations=graph.relations,
        issued_sweep=issued,
        seed_namespace=NO_WRITEBACK_SEED_NAMESPACE,
    )
    return NoWritebackWindowAction(
        problem_id=problem,
        run_seed=seed,
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=checkpoint_hash,
        topology_hash=graph.topology_hash,
        anchor_hash=anchor_hash,
        relations=graph.relations,
        relation_order_hash=graph.relation_order_hash,
        seed_namespace=NO_WRITEBACK_SEED_NAMESPACE,
        issued_sweep=issued,
        target_sweep=issued + 1,
        ttl_sweeps=1,
        expires_sweep=issued + 1,
    )


def compile_no_writeback_window_action(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    group_dims: Sequence[Sequence[int]],
    issued_sweep: int,
) -> NoWritebackWindowAction:
    """Freeze one full no-writeback window from the same A relation graph."""

    return _compile_no_writeback_window_action(
        graph=build_a_relation_graph(group_dims),
        problem_id=problem_id,
        run_seed=run_seed,
        checkpoint_fe=checkpoint_fe,
        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
        issued_sweep=issued_sweep,
    )


@dataclass(frozen=True)
class ASweepActionCandidates:
    graph: ARelationGraph
    relation_shared_cma_sweep: RelationSharedCmaSweepAction
    no_writeback_window: NoWritebackWindowAction


def compile_a_sweep_action_candidates(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    group_dims: Sequence[Sequence[int]],
    incumbent: Sequence[float],
    acceptance_fitness: float,
    issued_sweep: int,
    lower_bound: float,
    upper_bound: float,
) -> ASweepActionCandidates:
    """Compile exactly the two explicit A-sweep candidates from one graph."""

    graph = build_a_relation_graph(group_dims)
    return ASweepActionCandidates(
        graph=graph,
        relation_shared_cma_sweep=_compile_relation_shared_cma_sweep_action(
            graph=graph,
            problem_id=problem_id,
            run_seed=run_seed,
            checkpoint_fe=checkpoint_fe,
            dispatch_checkpoint_hash=dispatch_checkpoint_hash,
            incumbent=incumbent,
            acceptance_fitness=acceptance_fitness,
            issued_sweep=issued_sweep,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
        no_writeback_window=_compile_no_writeback_window_action(
            graph=graph,
            problem_id=problem_id,
            run_seed=run_seed,
            checkpoint_fe=checkpoint_fe,
            dispatch_checkpoint_hash=dispatch_checkpoint_hash,
            issued_sweep=issued_sweep,
        ),
    )


__all__ = [
    "A_RELATION_GRAPH_SCHEMA",
    "A_RELATION_GRAPH_SCHEMA_VERSION",
    "A_RELATION_ORDER_SCHEMA",
    "A_RELATION_SOURCE_SCHEMA",
    "NO_WRITEBACK_SEED_NAMESPACE",
    "PHASE1_MMES_SEED_NAMESPACE",
    "RELATION_CMA_SEED_NAMESPACE",
    "ARelationGraph",
    "ASweepActionCandidates",
    "build_a_relation_graph",
    "compile_a_sweep_action_candidates",
    "compile_no_writeback_window_action",
    "compile_phase1_mmes_resume_action",
    "compile_relation_shared_cma_sweep_action",
    "relation_cma_optimizer_seed",
]
