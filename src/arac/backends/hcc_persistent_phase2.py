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
    NO_RESTART_POLICY,
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionState,
    full_space_sep_cma_anchor_hash,
    full_space_vector_hash,
)


PERSISTENT_PHASE2_ARTIFACT_SCHEMA = "persistent-phase2-action-v1"
PERSISTENT_SEP_CMA_SEED_NAMESPACE = "persistent_phase2_full_space_sep_cma"
PERSISTENT_SELECTION_RULE = "max_voi_then_structural_key"

_HASH_LENGTH = 64


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


def persistent_optimizer_seed(checkpoint_hash: str) -> int:
    frozen = _validate_hash(checkpoint_hash, "checkpoint_hash")
    digest = hashlib.blake2b(
        f"{PERSISTENT_SEP_CMA_SEED_NAMESPACE}:{frozen}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)


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

    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    budget = _integer(budget_fes, "budget_fes", minimum=1)
    if budget < CANONICAL_SEP_CMA_POPULATION_SIZE:
        raise ValueError("persistent Sep-CMA budget must cover one population")
    mean = _finite_vector(incumbent, "incumbent")
    if len(mean) != FULL_SPACE_DIMENSION:
        raise ValueError("persistent Sep-CMA mean must be 1000-dimensional")
    frozen_checkpoint = _validate_hash(checkpoint_hash, "checkpoint_hash")
    optimizer_seed = persistent_optimizer_seed(frozen_checkpoint)
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
    target = _integer(start_sweep, "start_sweep")
    if target != issued + 1:
        raise ValueError("persistent Sep-CMA must start in the next sweep")
    return FullSpaceSepCmaAction(
        problem_id=problem_id,
        run_seed=_integer(run_seed, "run_seed"),
        checkpoint_fe=checkpoint,
        dispatch_checkpoint_hash=frozen_checkpoint,
        trigger_relation_hash=persistent_relation_hash(
            owner_group_indices,
            shared_variable_indices,
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
        optimizer_seed=optimizer_seed,
        seed_namespace=PERSISTENT_SEP_CMA_SEED_NAMESPACE,
        restart_policy=NO_RESTART_POLICY,
        issued_sweep=issued,
        target_sweep=target,
        ttl_sweeps=1,
        expires_sweep=target,
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

    if action.seed_namespace != PERSISTENT_SEP_CMA_SEED_NAMESPACE:
        raise ValueError("action is not a persistent Phase2 Sep-CMA instance")
    mean = _finite_vector(incumbent, "incumbent")
    if full_space_vector_hash(mean) != action.initial_mean_hash:
        raise ValueError("persistent Sep-CMA incumbent anchor changed")
    relation_hash = action.trigger_relation_hash
    lifecycle = FullSpaceSepCmaExecutionState.for_action(action)
    lifecycle.start(
        action,
        current_fe=_integer(current_fe, "current_fe"),
        current_sweep=_integer(current_sweep, "current_sweep"),
        dispatch_checkpoint_hash=_validate_hash(checkpoint_hash, "checkpoint_hash"),
        trigger_relation_hash=relation_hash,
        anchor_hash=full_space_sep_cma_anchor_hash(action.problem_id, mean),
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
        raise RuntimeError("persistent Sep-CMA initial state hash drifted")
    result = optimizer.advance(action.budget_fes)
    consumed = int(result["advanced_function_evaluations"])
    if consumed != action.budget_fes or int(result["n_function_evaluations"]) != consumed:
        raise RuntimeError("persistent Sep-CMA did not consume its exact Phase2 budget")
    if result["parameter_hash"] != action.canonical_parameters_hash:
        raise RuntimeError("persistent Sep-CMA parameter hash drifted")
    final_state_hash = result["optimizer_state"].state_hash
    lifecycle.complete(
        action,
        consumed_fes=consumed,
        completed_fe=action.checkpoint_fe + consumed,
        final_state_hash=final_state_hash,
    )
    lifecycle.validate_for(action)
    candidate = _finite_vector(result["best_so_far_x"], "Sep-CMA candidate")
    if len(candidate) != FULL_SPACE_DIMENSION:
        raise RuntimeError("persistent Sep-CMA returned a non-1000D candidate")
    candidate_fitness = float(result["best_so_far_y"])
    if not math.isfinite(candidate_fitness) or candidate_fitness < 0.0:
        raise RuntimeError("persistent Sep-CMA returned invalid fitness")
    accepted = candidate_fitness < action.acceptance_fitness
    return PersistentSepCmaResult(
        incumbent=candidate if accepted else mean,
        incumbent_fitness=(
            candidate_fitness if accepted else action.acceptance_fitness
        ),
        candidate_fitness=candidate_fitness,
        accepted=accepted,
        consumed_fes=consumed,
        final_state_hash=final_state_hash,
        lifecycle=lifecycle,
    )


__all__ = [
    "PERSISTENT_PHASE2_ARTIFACT_SCHEMA",
    "PERSISTENT_SELECTION_RULE",
    "PERSISTENT_SEP_CMA_SEED_NAMESPACE",
    "PersistentSepCmaResult",
    "compile_persistent_full_space_sep_cma_action",
    "execute_persistent_full_space_sep_cma_action",
    "persistent_optimizer_seed",
    "persistent_phase2_checkpoint_hash",
    "persistent_relation_hash",
]
