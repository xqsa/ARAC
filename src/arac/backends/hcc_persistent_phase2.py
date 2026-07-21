"""HCC adapters for one action that persists through the whole Phase2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arac.actions.full_space_sep_cma import (
    CANONICAL_SEP_CMA_PARAMETERIZATION,
    CANONICAL_SEP_CMA_POPULATION_SIZE,
    CANONICAL_SEP_CMA_REFERENCE_VERSION,
    FULL_SPACE_DIMENSION,
    FULL_SPACE_SEP_CMA_ACTION,
    NO_RESTART_POLICY,
    TRIGGER_SCOPE_PHASE_BOUNDARY,
    TRIGGER_SCOPE_RELATION_DISPATCH,
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionState,
    full_space_sep_cma_anchor_hash,
    full_space_vector_hash,
)


PERSISTENT_PHASE2_ARTIFACT_SCHEMA = "phase2-action-v2"
GLOBAL_PHASE2_ARTIFACT_SCHEMA = "phase2-global-action-v1"
PERSISTENT_SEP_CMA_SEED_NAMESPACE = "persistent_phase2_full_space_sep_cma"
FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE = FULL_SPACE_SEP_CMA_ACTION
PERSISTENT_SELECTION_RULE = "max_voi_then_structural_key"

_HASH_LENGTH = 64
_BURST_DISPATCH_CHECKPOINT_SCHEMA = "full-space-sep-cma-dispatch-checkpoint-v1"
_PHASE_BOUNDARY_ACTION_SOURCE_SCHEMA = (
    "full-space-sep-cma-phase-boundary-action-source-v1"
)
_PHASE_BOUNDARY_CHECKPOINT_SCHEMA = (
    "full-space-sep-cma-phase-boundary-checkpoint-v1"
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be finite and non-empty")
    return vector


def persistent_relation_hash(
    owner_group_indices: Sequence[int],
    shared_variable_indices: Sequence[int],
) -> str:
    owners = tuple(int(value) for value in owner_group_indices)
    shared = tuple(int(value) for value in shared_variable_indices)
    if not owners or not shared or any(value < 0 for value in (*owners, *shared)):
        raise ValueError("persistent relation identity must be non-empty and non-negative")
    return _canonical_sha256({"owners": owners, "shared": shared})


def persistent_phase2_checkpoint_hash(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    fitness_prefix: Sequence[float],
    incumbent: Sequence[float],
    topology_hash: str,
    order_hash: str,
    action_set_hash: str,
    start_sweep: int,
) -> str:
    """Bind the exact Phase1 endpoint used by either persistent action."""

    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("problem_id must be non-empty")
    seed = _integer(run_seed, "run_seed")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    sweep = _integer(start_sweep, "start_sweep")
    prefix = _finite_vector(fitness_prefix, "fitness_prefix")
    mean = _finite_vector(incumbent, "incumbent")
    if len(prefix) != checkpoint:
        raise ValueError("fitness_prefix length must equal checkpoint_fe")
    if len(mean) != FULL_SPACE_DIMENSION:
        raise ValueError("persistent Phase2 incumbent must be 1000-dimensional")
    return _canonical_sha256(
        {
            "protocol": PERSISTENT_PHASE2_ARTIFACT_SCHEMA,
            "problem_id": problem_id,
            "run_seed": seed,
            "checkpoint_fe": checkpoint,
            "fitness_prefix_hash": _canonical_sha256(prefix),
            "incumbent_hash": full_space_vector_hash(mean),
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "order_hash": _validate_hash(order_hash, "order_hash"),
            "action_set_hash": _validate_hash(action_set_hash, "action_set_hash"),
            "start_sweep": sweep,
        }
    )


def full_space_sep_cma_dispatch_checkpoint_hash(
    *,
    problem_id: str,
    run_seed: int,
    dispatch_fe: int,
    outer_iter: int,
    group_index: int,
    owner_group_indices: Sequence[int],
    shared_variable_indices: Sequence[int],
    incumbent: Sequence[float],
    fitness_prefix: Sequence[float],
    topology_hash: str,
    order_hash: str,
    action_set_hash: str,
    previous_shared_values: Sequence[float],
    current_shared_values: Sequence[float],
    previous_delta: float,
    current_delta: float,
    completed_group_deltas: Sequence[float],
    completed_group_actual_fes: Sequence[int],
    frozen_burst_budget_fes: int,
    budget_source_sweep: int,
) -> str:
    """Hash the complete relation-dispatch context that freezes one burst."""

    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("problem_id must be non-empty")
    frozen_dispatch_fe = _integer(dispatch_fe, "dispatch_fe")
    owners = tuple(int(value) for value in owner_group_indices)
    shared = tuple(int(value) for value in shared_variable_indices)
    relation_hash = persistent_relation_hash(owners, shared)
    mean = _finite_vector(incumbent, "incumbent")
    if len(mean) != FULL_SPACE_DIMENSION:
        raise ValueError("dispatch incumbent must be 1000-dimensional")
    prefix = _finite_vector(fitness_prefix, "fitness_prefix")
    if len(prefix) != frozen_dispatch_fe:
        raise ValueError("fitness_prefix length must equal dispatch_fe")
    previous_values = _finite_vector(
        previous_shared_values,
        "previous_shared_values",
    )
    current_values = _finite_vector(
        current_shared_values,
        "current_shared_values",
    )
    if len(previous_values) != len(shared) or len(current_values) != len(shared):
        raise ValueError("shared value lengths must match shared_variable_indices")
    previous = float(previous_delta)
    current = float(current_delta)
    completed_deltas = tuple(float(value) for value in completed_group_deltas)
    if any(
        not math.isfinite(value)
        for value in (previous, current, *completed_deltas)
    ):
        raise ValueError("relation and completed-group deltas must be finite")
    completed_fes = tuple(
        _integer(value, "completed_group_actual_fes")
        for value in completed_group_actual_fes
    )
    if len(completed_deltas) != len(completed_fes):
        raise ValueError(
            "completed_group_deltas and completed_group_actual_fes must align"
        )
    budget = _integer(
        frozen_burst_budget_fes,
        "frozen_burst_budget_fes",
        minimum=CANONICAL_SEP_CMA_POPULATION_SIZE,
    )
    return _canonical_sha256(
        {
            "protocol": _BURST_DISPATCH_CHECKPOINT_SCHEMA,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "dispatch_fe": frozen_dispatch_fe,
            "outer_iter": _integer(outer_iter, "outer_iter"),
            "group_index": _integer(group_index, "group_index"),
            "relation": {
                "owners": owners,
                "shared": shared,
                "hash": relation_hash,
            },
            "incumbent_hash": full_space_vector_hash(mean),
            "fitness_prefix_hash": _canonical_sha256(prefix),
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "order_hash": _validate_hash(order_hash, "order_hash"),
            "action_set_hash": _validate_hash(
                action_set_hash,
                "action_set_hash",
            ),
            "previous_shared_values": previous_values,
            "current_shared_values": current_values,
            "previous_delta": previous,
            "current_delta": current,
            "completed_group_deltas": completed_deltas,
            "completed_group_actual_fes": completed_fes,
            "frozen_burst_budget_fes": budget,
            "budget_source_sweep": _integer(
                budget_source_sweep,
                "budget_source_sweep",
            ),
        }
    )


def full_space_sep_cma_phase_boundary_action_source_hash(
    *,
    problem_id: str,
    run_seed: int,
    issued_sweep: int,
    target_sweep: int,
    frozen_burst_budget_fes: int,
    topology_hash: str,
    order_hash: str,
) -> str:
    """Identify one forced global action without inventing a relation."""

    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("problem_id must be non-empty")
    issued = _integer(issued_sweep, "issued_sweep")
    target = _integer(target_sweep, "target_sweep")
    if target != issued + 1:
        raise ValueError("phase-boundary action must target the next sweep")
    budget = _integer(
        frozen_burst_budget_fes,
        "frozen_burst_budget_fes",
        minimum=CANONICAL_SEP_CMA_POPULATION_SIZE,
    )
    return _canonical_sha256(
        {
            "protocol": _PHASE_BOUNDARY_ACTION_SOURCE_SCHEMA,
            "action": FULL_SPACE_SEP_CMA_ACTION,
            "trigger_scope": TRIGGER_SCOPE_PHASE_BOUNDARY,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "issued_sweep": issued,
            "target_sweep": target,
            "frozen_burst_budget_fes": budget,
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "order_hash": _validate_hash(order_hash, "order_hash"),
        }
    )


def full_space_sep_cma_phase_boundary_checkpoint_hash(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    issued_sweep: int,
    target_sweep: int,
    incumbent: Sequence[float],
    fitness_prefix: Sequence[float],
    topology_hash: str,
    order_hash: str,
    action_source_hash: str,
    completed_group_deltas: Sequence[float],
    completed_group_actual_fes: Sequence[int],
    frozen_burst_budget_fes: int,
) -> str:
    """Bind a no-relation action to one complete native sweep boundary."""

    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("problem_id must be non-empty")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    issued = _integer(issued_sweep, "issued_sweep")
    target = _integer(target_sweep, "target_sweep")
    if target != issued + 1:
        raise ValueError("phase-boundary action must target the next sweep")
    mean = _finite_vector(incumbent, "incumbent")
    if len(mean) != FULL_SPACE_DIMENSION:
        raise ValueError("phase-boundary incumbent must be 1000-dimensional")
    prefix = _finite_vector(fitness_prefix, "fitness_prefix")
    if len(prefix) != checkpoint:
        raise ValueError("fitness_prefix length must equal checkpoint_fe")
    completed_deltas = tuple(float(value) for value in completed_group_deltas)
    if any(not math.isfinite(value) for value in completed_deltas):
        raise ValueError("completed_group_deltas must be finite")
    completed_fes = tuple(
        _integer(value, "completed_group_actual_fes")
        for value in completed_group_actual_fes
    )
    if len(completed_deltas) != len(completed_fes) or not completed_fes:
        raise ValueError("completed group deltas and FEs must be non-empty and aligned")
    budget = _integer(
        frozen_burst_budget_fes,
        "frozen_burst_budget_fes",
        minimum=CANONICAL_SEP_CMA_POPULATION_SIZE,
    )
    if sum(completed_fes) != budget:
        raise ValueError("frozen burst budget must equal the completed sweep FEs")
    return _canonical_sha256(
        {
            "protocol": _PHASE_BOUNDARY_CHECKPOINT_SCHEMA,
            "trigger_scope": TRIGGER_SCOPE_PHASE_BOUNDARY,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "checkpoint_fe": checkpoint,
            "issued_sweep": issued,
            "target_sweep": target,
            "incumbent_hash": full_space_vector_hash(mean),
            "fitness_prefix_hash": _canonical_sha256(prefix),
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "order_hash": _validate_hash(order_hash, "order_hash"),
            "action_source_hash": _validate_hash(
                action_source_hash,
                "action_source_hash",
            ),
            "completed_group_deltas": completed_deltas,
            "completed_group_actual_fes": completed_fes,
            "frozen_burst_budget_fes": budget,
        }
    )


def persistent_optimizer_seed(checkpoint_hash: str) -> int:
    frozen = _validate_hash(checkpoint_hash, "checkpoint_hash")
    digest = hashlib.blake2b(
        f"{PERSISTENT_SEP_CMA_SEED_NAMESPACE}:{frozen}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


def full_space_sep_cma_burst_optimizer_seed(
    dispatch_checkpoint_hash: str,
) -> int:
    """Reproduce the exp019 optimizer seed for one frozen dispatch."""

    frozen = _validate_hash(
        dispatch_checkpoint_hash,
        "dispatch_checkpoint_hash",
    )
    return int(
        _canonical_sha256(
            {
                "namespace": FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE,
                "dispatch_checkpoint_hash": frozen,
            }
        )[:16],
        16,
    ) % (2**32)


def _sep_optimizer(
    factory: Callable[..., Any],
    *,
    objective: Callable[[Any], Any],
    action_mean: Sequence[float],
    sigma: float,
    lower: float,
    upper: float,
    budget_fes: int,
    optimizer_seed: int,
) -> Any:
    return factory(
        {
            "fitness_function": objective,
            "ndim_problem": FULL_SPACE_DIMENSION,
            "lower_boundary": float(lower) * np.ones((FULL_SPACE_DIMENSION,)),
            "upper_boundary": float(upper) * np.ones((FULL_SPACE_DIMENSION,)),
        },
        {
            "max_function_evaluations": int(budget_fes),
            "mean": (np.asarray(action_mean, dtype=float),),
            "sigma": float(sigma),
            "n_individuals": CANONICAL_SEP_CMA_POPULATION_SIZE,
            "is_restart": False,
            "verbose": 0,
            "early_stopping_evaluations": np.inf,
            "seed_rng": int(optimizer_seed),
        },
    )


def _compile_full_space_sep_cma_action(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    checkpoint_hash: str,
    trigger_context_hash: str,
    trigger_scope: str,
    incumbent: Sequence[float],
    acceptance_fitness: float,
    sigma: float,
    lower: float,
    upper: float,
    budget_fes: int,
    issued_sweep: int,
    target_sweep: int,
    optimizer_seed: int,
    seed_namespace: str,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
) -> FullSpaceSepCmaAction:
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    budget = _integer(budget_fes, "budget_fes", minimum=1)
    if budget < CANONICAL_SEP_CMA_POPULATION_SIZE:
        raise ValueError("full-space Sep-CMA budget must cover one population")
    mean = _finite_vector(incumbent, "incumbent")
    if len(mean) != FULL_SPACE_DIMENSION:
        raise ValueError("full-space Sep-CMA mean must be 1000-dimensional")
    frozen_checkpoint = _validate_hash(checkpoint_hash, "checkpoint_hash")
    optimizer = _sep_optimizer(
        sepcmaes_factory,
        objective=objective,
        action_mean=mean,
        sigma=sigma,
        lower=lower,
        upper=upper,
        budget_fes=budget,
        optimizer_seed=optimizer_seed,
    )
    state = optimizer.initialize_state()
    parameters = optimizer.parameters
    if (
        parameters.parameterization != CANONICAL_SEP_CMA_PARAMETERIZATION
        or parameters.reference_version != CANONICAL_SEP_CMA_REFERENCE_VERSION
        or parameters.dimension != FULL_SPACE_DIMENSION
        or parameters.population_size != CANONICAL_SEP_CMA_POPULATION_SIZE
    ):
        raise RuntimeError("canonical Sep-CMA parameter snapshot drifted")
    issued = _integer(issued_sweep, "issued_sweep")
    target = _integer(target_sweep, "target_sweep")
    if target != issued + 1:
        raise ValueError("full-space Sep-CMA must start in the next sweep")
    return FullSpaceSepCmaAction(
        problem_id=problem_id,
        run_seed=_integer(run_seed, "run_seed"),
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=frozen_checkpoint,
        trigger_relation_hash=_validate_hash(
            trigger_context_hash,
            "trigger_context_hash",
        ),
        anchor_hash=full_space_sep_cma_anchor_hash(problem_id, mean),
        initial_mean=mean,
        initial_mean_hash=full_space_vector_hash(mean),
        initial_state_hash=state.state_hash,
        initial_sigma=float(sigma),
        lower_bound=float(lower),
        upper_bound=float(upper),
        acceptance_fitness=float(acceptance_fitness),
        population_size=CANONICAL_SEP_CMA_POPULATION_SIZE,
        budget_fes=budget,
        parameterization=CANONICAL_SEP_CMA_PARAMETERIZATION,
        canonical_reference_version=CANONICAL_SEP_CMA_REFERENCE_VERSION,
        canonical_parameters_hash=parameters.parameter_hash,
        optimizer_seed=_integer(optimizer_seed, "optimizer_seed"),
        seed_namespace=seed_namespace,
        restart_policy=NO_RESTART_POLICY,
        issued_sweep=issued,
        target_sweep=target,
        ttl_sweeps=1,
        expires_sweep=target,
        trigger_scope=trigger_scope,
    )


def compile_full_space_sep_cma_burst_action(
    *,
    problem_id: str,
    run_seed: int,
    dispatch_fe: int,
    dispatch_checkpoint_hash: str,
    owner_group_indices: Sequence[int],
    shared_variable_indices: Sequence[int],
    incumbent: Sequence[float],
    acceptance_fitness: float,
    sigma: float,
    lower: float,
    upper: float,
    budget_fes: int,
    issued_sweep: int,
    target_sweep: int,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
) -> FullSpaceSepCmaAction:
    """Compile one exp019-equivalent burst without evaluating the objective."""

    checkpoint_hash = _validate_hash(
        dispatch_checkpoint_hash,
        "dispatch_checkpoint_hash",
    )
    return _compile_full_space_sep_cma_action(
        problem_id=problem_id,
        run_seed=run_seed,
        checkpoint_fe=dispatch_fe,
        checkpoint_hash=checkpoint_hash,
        trigger_context_hash=persistent_relation_hash(
            owner_group_indices,
            shared_variable_indices,
        ),
        trigger_scope=TRIGGER_SCOPE_RELATION_DISPATCH,
        incumbent=incumbent,
        acceptance_fitness=acceptance_fitness,
        sigma=sigma,
        lower=lower,
        upper=upper,
        budget_fes=budget_fes,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        optimizer_seed=full_space_sep_cma_burst_optimizer_seed(checkpoint_hash),
        seed_namespace=FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE,
        objective=objective,
        sepcmaes_factory=sepcmaes_factory,
    )


def compile_full_space_sep_cma_phase_boundary_action(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    checkpoint_hash: str,
    incumbent: Sequence[float],
    acceptance_fitness: float,
    sigma: float,
    lower: float,
    upper: float,
    budget_fes: int,
    issued_sweep: int,
    target_sweep: int,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
) -> FullSpaceSepCmaAction:
    """Compile one global burst at a complete native sweep boundary."""

    frozen_checkpoint = _validate_hash(checkpoint_hash, "checkpoint_hash")
    return _compile_full_space_sep_cma_action(
        problem_id=problem_id,
        run_seed=run_seed,
        checkpoint_fe=checkpoint_fe,
        checkpoint_hash=frozen_checkpoint,
        trigger_context_hash=frozen_checkpoint,
        trigger_scope=TRIGGER_SCOPE_PHASE_BOUNDARY,
        incumbent=incumbent,
        acceptance_fitness=acceptance_fitness,
        sigma=sigma,
        lower=lower,
        upper=upper,
        budget_fes=budget_fes,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        optimizer_seed=full_space_sep_cma_burst_optimizer_seed(
            frozen_checkpoint
        ),
        seed_namespace=FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE,
        objective=objective,
        sepcmaes_factory=sepcmaes_factory,
    )


def compile_persistent_full_space_sep_cma_action(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    checkpoint_hash: str,
    owner_group_indices: Sequence[int],
    shared_variable_indices: Sequence[int],
    incumbent: Sequence[float],
    acceptance_fitness: float,
    sigma: float,
    lower: float,
    upper: float,
    budget_fes: int,
    issued_sweep: int,
    start_sweep: int,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
) -> FullSpaceSepCmaAction:
    """Compile one canonical Sep-CMA instance without evaluating the objective."""

    frozen_checkpoint = _validate_hash(checkpoint_hash, "checkpoint_hash")
    return _compile_full_space_sep_cma_action(
        problem_id=problem_id,
        run_seed=run_seed,
        checkpoint_fe=checkpoint_fe,
        checkpoint_hash=frozen_checkpoint,
        trigger_context_hash=persistent_relation_hash(
            owner_group_indices,
            shared_variable_indices,
        ),
        trigger_scope=TRIGGER_SCOPE_RELATION_DISPATCH,
        incumbent=incumbent,
        acceptance_fitness=acceptance_fitness,
        sigma=sigma,
        lower=lower,
        upper=upper,
        budget_fes=budget_fes,
        issued_sweep=issued_sweep,
        target_sweep=start_sweep,
        optimizer_seed=persistent_optimizer_seed(frozen_checkpoint),
        seed_namespace=PERSISTENT_SEP_CMA_SEED_NAMESPACE,
        objective=objective,
        sepcmaes_factory=sepcmaes_factory,
    )


@dataclass(frozen=True)
class PersistentSepCmaResult:
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    candidate_fitness: float
    accepted: bool
    consumed_fes: int
    final_state_hash: str
    lifecycle: FullSpaceSepCmaExecutionState


@dataclass(frozen=True)
class FullSpaceSepCmaBurstResult:
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    candidate: tuple[float, ...]
    candidate_fitness: float
    accepted: bool
    consumed_fes: int
    final_state_hash: str
    action_hash: str
    lifecycle: FullSpaceSepCmaExecutionState
    lifecycle_hash: str
    candidate_hash: str
    post_incumbent_hash: str
    resume_native: bool = True


@dataclass(frozen=True)
class _SepCmaExecutionOutcome:
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    candidate: tuple[float, ...]
    candidate_fitness: float
    accepted: bool
    consumed_fes: int
    final_state_hash: str
    lifecycle: FullSpaceSepCmaExecutionState
    lifecycle_hash: str
    candidate_hash: str
    post_incumbent_hash: str


def _execute_full_space_sep_cma_action(
    action: FullSpaceSepCmaAction,
    *,
    required_seed_namespace: str,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
    current_fe: int,
    current_sweep: int,
    dispatch_checkpoint_hash: str,
    incumbent: Sequence[float],
) -> _SepCmaExecutionOutcome:
    if action.seed_namespace != required_seed_namespace:
        raise ValueError("action seed namespace does not match the execution mode")
    mean = _finite_vector(incumbent, "incumbent")
    if len(mean) != FULL_SPACE_DIMENSION:
        raise ValueError("full-space Sep-CMA incumbent must be 1000-dimensional")
    if full_space_vector_hash(mean) != action.initial_mean_hash:
        raise ValueError("full-space Sep-CMA incumbent anchor changed")
    lifecycle = FullSpaceSepCmaExecutionState.for_action(action)
    observed_fe = _integer(current_fe, "current_fe")
    lifecycle.start(
        action,
        current_fe=observed_fe,
        current_sweep=_integer(current_sweep, "current_sweep"),
        dispatch_checkpoint_hash=_validate_hash(
            dispatch_checkpoint_hash,
            "dispatch_checkpoint_hash",
        ),
        trigger_relation_hash=action.trigger_relation_hash,
        anchor_hash=full_space_sep_cma_anchor_hash(action.problem_id, mean),
        trigger_scope=action.trigger_scope,
    )
    optimizer = _sep_optimizer(
        sepcmaes_factory,
        objective=objective,
        action_mean=mean,
        sigma=action.initial_sigma,
        lower=action.lower_bound,
        upper=action.upper_bound,
        budget_fes=action.budget_fes,
        optimizer_seed=action.optimizer_seed,
    )
    initial_state = optimizer.initialize_state()
    if initial_state.state_hash != action.initial_state_hash:
        raise RuntimeError("full-space Sep-CMA initial state hash drifted")
    result = optimizer.advance(action.budget_fes)
    consumed = int(result["advanced_function_evaluations"])
    if consumed != action.budget_fes or int(result["n_function_evaluations"]) != consumed:
        raise RuntimeError("full-space Sep-CMA did not consume its exact frozen budget")
    if result["parameter_hash"] != action.canonical_parameters_hash:
        raise RuntimeError("full-space Sep-CMA parameter hash drifted")
    final_state_hash = result["optimizer_state"].state_hash
    lifecycle.complete(
        action,
        consumed_fes=consumed,
        completed_fe=observed_fe + consumed,
        final_state_hash=final_state_hash,
    )
    lifecycle.validate_for(action)
    candidate = _finite_vector(result["best_so_far_x"], "Sep-CMA candidate")
    if len(candidate) != FULL_SPACE_DIMENSION:
        raise RuntimeError("full-space Sep-CMA returned a non-1000D candidate")
    candidate_fitness = float(result["best_so_far_y"])
    if not math.isfinite(candidate_fitness) or candidate_fitness < 0.0:
        raise RuntimeError("full-space Sep-CMA returned invalid fitness")
    accepted = candidate_fitness < action.acceptance_fitness
    post_incumbent = candidate if accepted else mean
    return _SepCmaExecutionOutcome(
        incumbent=post_incumbent,
        incumbent_fitness=(
            candidate_fitness if accepted else action.acceptance_fitness
        ),
        candidate=candidate,
        candidate_fitness=candidate_fitness,
        accepted=accepted,
        consumed_fes=consumed,
        final_state_hash=final_state_hash,
        lifecycle=lifecycle,
        lifecycle_hash=lifecycle.state_hash(action),
        candidate_hash=full_space_vector_hash(candidate),
        post_incumbent_hash=full_space_vector_hash(post_incumbent),
    )


def execute_full_space_sep_cma_burst_action(
    action: FullSpaceSepCmaAction,
    *,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
    current_fe: int,
    current_sweep: int,
    dispatch_checkpoint_hash: str,
    incumbent: Sequence[float],
) -> FullSpaceSepCmaBurstResult:
    """Execute one frozen burst; the caller must resume native HCC afterwards."""

    outcome = _execute_full_space_sep_cma_action(
        action,
        required_seed_namespace=FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE,
        objective=objective,
        sepcmaes_factory=sepcmaes_factory,
        current_fe=current_fe,
        current_sweep=current_sweep,
        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
        incumbent=incumbent,
    )
    return FullSpaceSepCmaBurstResult(
        incumbent=outcome.incumbent,
        incumbent_fitness=outcome.incumbent_fitness,
        candidate=outcome.candidate,
        candidate_fitness=outcome.candidate_fitness,
        accepted=outcome.accepted,
        consumed_fes=outcome.consumed_fes,
        final_state_hash=outcome.final_state_hash,
        action_hash=action.action_hash,
        lifecycle=outcome.lifecycle,
        lifecycle_hash=outcome.lifecycle_hash,
        candidate_hash=outcome.candidate_hash,
        post_incumbent_hash=outcome.post_incumbent_hash,
    )


def execute_persistent_full_space_sep_cma_action(
    action: FullSpaceSepCmaAction,
    *,
    objective: Callable[[Any], Any],
    sepcmaes_factory: Callable[..., Any],
    current_fe: int,
    current_sweep: int,
    checkpoint_hash: str,
    incumbent: Sequence[float],
) -> PersistentSepCmaResult:
    """Consume the entire remaining Phase2 budget; native HCC must not resume."""

    outcome = _execute_full_space_sep_cma_action(
        action,
        required_seed_namespace=PERSISTENT_SEP_CMA_SEED_NAMESPACE,
        objective=objective,
        sepcmaes_factory=sepcmaes_factory,
        current_fe=current_fe,
        current_sweep=current_sweep,
        dispatch_checkpoint_hash=checkpoint_hash,
        incumbent=incumbent,
    )
    return PersistentSepCmaResult(
        incumbent=outcome.incumbent,
        incumbent_fitness=outcome.incumbent_fitness,
        candidate_fitness=outcome.candidate_fitness,
        accepted=outcome.accepted,
        consumed_fes=outcome.consumed_fes,
        final_state_hash=outcome.final_state_hash,
        lifecycle=outcome.lifecycle,
    )


__all__ = [
    "FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE",
    "GLOBAL_PHASE2_ARTIFACT_SCHEMA",
    "PERSISTENT_PHASE2_ARTIFACT_SCHEMA",
    "PERSISTENT_SELECTION_RULE",
    "PERSISTENT_SEP_CMA_SEED_NAMESPACE",
    "FullSpaceSepCmaBurstResult",
    "PersistentSepCmaResult",
    "compile_full_space_sep_cma_burst_action",
    "compile_full_space_sep_cma_phase_boundary_action",
    "compile_persistent_full_space_sep_cma_action",
    "execute_full_space_sep_cma_burst_action",
    "execute_persistent_full_space_sep_cma_action",
    "full_space_sep_cma_burst_optimizer_seed",
    "full_space_sep_cma_dispatch_checkpoint_hash",
    "full_space_sep_cma_phase_boundary_action_source_hash",
    "full_space_sep_cma_phase_boundary_checkpoint_hash",
    "persistent_optimizer_seed",
    "persistent_phase2_checkpoint_hash",
    "persistent_relation_hash",
]
