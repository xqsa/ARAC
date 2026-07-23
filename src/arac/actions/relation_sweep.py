"""Frozen A-series relation-sweep actions and deterministic executors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arac.actions.action_spec import ActionSpec


RELATION_SHARED_CMA_SWEEP_ACTION = "relation_shared_cma_sweep"
NO_WRITEBACK_WINDOW_ACTION = "no_writeback_window"
RELATION_SHARED_CMA_SWEEP_SCHEMA = "arac.action.relation_shared_cma_sweep"
RELATION_SHARED_CMA_SWEEP_SCHEMA_VERSION = 1
NO_WRITEBACK_WINDOW_SCHEMA = "arac.action.no_writeback_window"
NO_WRITEBACK_WINDOW_SCHEMA_VERSION = 1
FULL_SPACE_DIMENSION = 1000
RELATION_COUNT = 19
RELATION_DIMENSION = 5
RELATION_CMA_POPULATION_SIZE = 10
RELATION_CMA_GENERATION_COUNT = 25
RELATION_CMA_BLOCK_BUDGET_FES = RELATION_CMA_POPULATION_SIZE * RELATION_CMA_GENERATION_COUNT
RELATION_CMA_TOTAL_BUDGET_FES = RELATION_COUNT * RELATION_CMA_BLOCK_BUDGET_FES
RELATION_CMA_INITIAL_SIGMA = 0.5
RELATION_CMA_PARAMETERIZATION = "hcc_ros_hansen_full_cmaes"
RELATION_CMA_REFERENCE_VERSION = (
    "HCC-main/HCC_SRC/HCC/OPT/CMAES/cmaes.py@da0661082b27c3b3ed2547131c42b1b2a0f960db"
)
RELATION_CMA_IMPLEMENTATION_TYPE = "HCC.OPT.CMAES.cmaes.CMAES"
NO_RESTART_POLICY = "none"
NO_EARLY_STOPPING_POLICY = "none"
NO_REPAIR_POLICY = "none"
STRICT_IMPROVEMENT_ACCEPTANCE = "strict_improvement"

RELATION_SHARED_CMA_SWEEP_ACTION_SPEC = ActionSpec(
    name=RELATION_SHARED_CMA_SWEEP_ACTION,
    semantic_surface="relation_shared_cma_sweep",
    parameter_names=(
        "relations",
        "initial_means",
        "optimizer_seeds",
        "initial_sigma",
        "population_size",
        "generation_count",
        "block_budget_fes",
        "restart_policy",
        "early_stopping_policy",
        "repair_policy",
        "acceptance_rule",
    ),
)
NO_WRITEBACK_WINDOW_ACTION_SPEC = ActionSpec(
    name=NO_WRITEBACK_WINDOW_ACTION,
    semantic_surface="relation_writeback_window",
    parameter_names=("relations", "target_sweep", "ttl_sweeps"),
)

_HASH_LENGTH = 64
_WINDOW_STATUSES = frozenset({"issued", "running", "completed", "abstained"})
_SWEEP_STATUSES = frozenset({"issued", "running", "completed", "abstained", "failed"})


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


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _vector(values: object, dimension: int, name: str) -> tuple[float, ...]:
    try:
        vector = tuple(_finite(value, name) for value in values)  # type: ignore[union-attr]
    except TypeError as error:
        raise ValueError(f"{name} must be a one-dimensional numeric sequence") from error
    if len(vector) != dimension:
        raise ValueError(f"{name} must contain exactly {dimension} values")
    return vector


def full_space_vector_hash(values: object) -> str:
    vector = _vector(values, FULL_SPACE_DIMENSION, "full-space vector")
    return _canonical_sha256({"dimension": FULL_SPACE_DIMENSION, "values": vector})


def shared_values_hash(values: object) -> str:
    vector = _vector(values, RELATION_DIMENSION, "shared values")
    return _canonical_sha256({"dimension": RELATION_DIMENSION, "values": vector})


def owner_context_memory_hash(
    owner_group_index: int,
    dimensions: Sequence[int],
    mean_values: Sequence[float],
) -> str:
    """Hash one owner's ordered local optimizer context memory."""

    owner = _integer(owner_group_index, "owner_group_index")
    local_dimensions = tuple(dimensions)
    if not local_dimensions or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in local_dimensions
    ):
        raise ValueError("owner dimensions must be non-empty non-negative integers")
    if local_dimensions != tuple(sorted(set(local_dimensions))):
        raise ValueError("owner dimensions must be sorted and distinct")
    values = _vector(mean_values, len(local_dimensions), "owner mean values")
    return _canonical_sha256(
        {
            "owner_group_index": owner,
            "dimensions": list(local_dimensions),
            "mean_values": list(values),
        }
    )


@dataclass(frozen=True)
class OwnerContextMemorySnapshot:
    """Complete ordered local mean owned by one native group optimizer."""

    owner_group_index: int
    dimensions: tuple[int, ...]
    mean_values: tuple[float, ...]

    def __post_init__(self) -> None:
        owner = _integer(self.owner_group_index, "owner_group_index")
        dimensions = tuple(self.dimensions)
        if not dimensions or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in dimensions
        ):
            raise ValueError("owner dimensions must be non-empty non-negative integers")
        if dimensions != tuple(sorted(set(dimensions))):
            raise ValueError("owner dimensions must be sorted and distinct")
        mean_values = _vector(
            self.mean_values,
            len(dimensions),
            "owner mean values",
        )
        object.__setattr__(self, "owner_group_index", owner)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "mean_values", mean_values)

    @property
    def context_memory_hash(self) -> str:
        return owner_context_memory_hash(
            self.owner_group_index,
            self.dimensions,
            self.mean_values,
        )

    def with_shared_values(
        self,
        relation: RelationBlockKey,
        shared_values: Sequence[float],
    ) -> OwnerContextMemorySnapshot:
        if self.owner_group_index not in relation.owner_group_indices:
            raise ValueError("owner snapshot does not belong to relation")
        values = _vector(shared_values, RELATION_DIMENSION, "shared_values")
        positions = {dimension: index for index, dimension in enumerate(self.dimensions)}
        if any(dimension not in positions for dimension in relation.shared_variable_indices):
            raise ValueError("owner snapshot is missing a relation dimension")
        updated = list(self.mean_values)
        for dimension, value in zip(
            relation.shared_variable_indices,
            values,
            strict=True,
        ):
            updated[positions[dimension]] = value
        return OwnerContextMemorySnapshot(
            owner_group_index=self.owner_group_index,
            dimensions=self.dimensions,
            mean_values=tuple(updated),
        )


@dataclass(frozen=True, order=True)
class RelationBlockKey:
    """Stable identity for one two-owner shared-variable block."""

    owner_group_indices: tuple[int, int]
    shared_variable_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        owners = tuple(self.owner_group_indices)
        shared = tuple(self.shared_variable_indices)
        if len(owners) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in owners
        ):
            raise ValueError("relation requires two non-negative integer owners")
        if owners != tuple(sorted(set(owners))):
            raise ValueError("relation owners must be sorted and distinct")
        if len(shared) != RELATION_DIMENSION or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shared
        ):
            raise ValueError(f"relation requires exactly {RELATION_DIMENSION} shared variables")
        if shared != tuple(sorted(set(shared))):
            raise ValueError("shared relation variables must be sorted and distinct")
        object.__setattr__(self, "owner_group_indices", owners)
        object.__setattr__(self, "shared_variable_indices", shared)

    def audit_payload(self) -> dict[str, object]:
        return {
            "owners": list(self.owner_group_indices),
            "shared_variables": list(self.shared_variable_indices),
        }


def _validate_a_relation_order(relations: Sequence[RelationBlockKey]) -> None:
    if len(relations) != RELATION_COUNT or len(set(relations)) != RELATION_COUNT:
        raise ValueError(f"A-series sweep requires {RELATION_COUNT} distinct relations")

    shared_variables = [
        variable for relation in relations for variable in relation.shared_variable_indices
    ]
    if len(set(shared_variables)) != len(shared_variables):
        raise ValueError("A-series relation blocks must have disjoint shared variables")

    degrees: dict[int, int] = {}
    for relation in relations:
        for owner in relation.owner_group_indices:
            degrees[owner] = degrees.get(owner, 0) + 1
    if len(degrees) != RELATION_COUNT + 1:
        raise ValueError("A-series relations must connect exactly twenty owner groups")
    if sorted(degrees.values()) != [1, 1] + [2] * (RELATION_COUNT - 1):
        raise ValueError("A-series owner graph must be one simple path")

    seen_owners = set(relations[0].owner_group_indices)
    if not any(degrees[owner] == 1 for owner in seen_owners):
        raise ValueError("relation order must start at one endpoint of the owner path")
    for relation in relations[1:]:
        owners = set(relation.owner_group_indices)
        if len(owners & seen_owners) != 1 or len(owners - seen_owners) != 1:
            raise ValueError("relations must follow the frozen native path order")
        seen_owners.update(owners)


def ordered_relations_hash(relations: Sequence[RelationBlockKey]) -> str:
    relation_tuple = tuple(relations)
    _validate_a_relation_order(relation_tuple)
    return _canonical_sha256(
        {
            "relation_count": RELATION_COUNT,
            "relations": [relation.audit_payload() for relation in relation_tuple],
        }
    )


@dataclass(frozen=True)
class FrozenRelationCmaBlock:
    """One 5D optimizer instance frozen before the target sweep starts."""

    relation: RelationBlockKey
    initial_mean: tuple[float, ...]
    optimizer_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.relation, RelationBlockKey):
            raise TypeError("relation must be a RelationBlockKey")
        object.__setattr__(
            self,
            "initial_mean",
            _vector(self.initial_mean, RELATION_DIMENSION, "initial_mean"),
        )
        _integer(self.optimizer_seed, "optimizer_seed")

    @property
    def initial_state_hash(self) -> str:
        return _canonical_sha256(
            {
                "relation": self.relation.audit_payload(),
                "initial_mean": list(self.initial_mean),
                "optimizer_seed": self.optimizer_seed,
                "parameter_hash": RELATION_CMA_PARAMETERS_HASH,
            }
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "relation": self.relation.audit_payload(),
            "initial_mean": list(self.initial_mean),
            "initial_mean_hash": shared_values_hash(self.initial_mean),
            "initial_state_hash": self.initial_state_hash,
            "optimizer_seed": self.optimizer_seed,
        }


_RELATION_CMA_PARAMETERS = {
    "dimension": RELATION_DIMENSION,
    "population_size": RELATION_CMA_POPULATION_SIZE,
    "generation_count": RELATION_CMA_GENERATION_COUNT,
    "block_budget_fes": RELATION_CMA_BLOCK_BUDGET_FES,
    "initial_sigma": RELATION_CMA_INITIAL_SIGMA,
    "parameterization": RELATION_CMA_PARAMETERIZATION,
    "reference_version": RELATION_CMA_REFERENCE_VERSION,
    "restart_policy": NO_RESTART_POLICY,
    "early_stopping_policy": NO_EARLY_STOPPING_POLICY,
    "repair_policy": NO_REPAIR_POLICY,
    "acceptance_rule": STRICT_IMPROVEMENT_ACCEPTANCE,
}
RELATION_CMA_PARAMETERS_HASH = _canonical_sha256(_RELATION_CMA_PARAMETERS)


def relation_cma_anchor_hash(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    topology_hash: str,
    initial_incumbent: Sequence[float],
    blocks: Sequence[FrozenRelationCmaBlock],
    issued_sweep: int,
) -> str:
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string")
    incumbent = _vector(initial_incumbent, FULL_SPACE_DIMENSION, "initial_incumbent")
    block_tuple = tuple(blocks)
    relations = tuple(block.relation for block in block_tuple)
    return _canonical_sha256(
        {
            "action": RELATION_SHARED_CMA_SWEEP_ACTION,
            "schema": RELATION_SHARED_CMA_SWEEP_SCHEMA,
            "schema_version": RELATION_SHARED_CMA_SWEEP_SCHEMA_VERSION,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "checkpoint_fe": _integer(checkpoint_fe, "checkpoint_fe"),
            "dispatch_checkpoint_hash": _validate_hash(
                dispatch_checkpoint_hash,
                "dispatch_checkpoint_hash",
            ),
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "initial_incumbent_hash": full_space_vector_hash(incumbent),
            "relation_order_hash": ordered_relations_hash(relations),
            "initial_means": [list(block.initial_mean) for block in block_tuple],
            "issued_sweep": _integer(issued_sweep, "issued_sweep"),
        }
    )


@dataclass(frozen=True)
class RelationSharedCmaSweepAction:
    """Immutable 19-block full-CMA sweep compiled from one checkpoint."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    dispatch_checkpoint_hash: str
    topology_hash: str
    anchor_hash: str
    initial_incumbent: tuple[float, ...]
    initial_incumbent_hash: str
    acceptance_fitness: float
    blocks: tuple[FrozenRelationCmaBlock, ...]
    relation_order_hash: str
    seed_namespace: str
    issued_sweep: int
    target_sweep: int
    ttl_sweeps: int
    expires_sweep: int
    lower_bound: float
    upper_bound: float
    schema: str = RELATION_SHARED_CMA_SWEEP_SCHEMA
    schema_version: int = RELATION_SHARED_CMA_SWEEP_SCHEMA_VERSION
    initial_sigma: float = RELATION_CMA_INITIAL_SIGMA
    population_size: int = RELATION_CMA_POPULATION_SIZE
    generation_count: int = RELATION_CMA_GENERATION_COUNT
    block_budget_fes: int = RELATION_CMA_BLOCK_BUDGET_FES
    total_budget_fes: int = RELATION_CMA_TOTAL_BUDGET_FES
    parameterization: str = RELATION_CMA_PARAMETERIZATION
    reference_version: str = RELATION_CMA_REFERENCE_VERSION
    parameter_hash: str = RELATION_CMA_PARAMETERS_HASH
    restart_policy: str = NO_RESTART_POLICY
    early_stopping_policy: str = NO_EARLY_STOPPING_POLICY
    repair_policy: str = NO_REPAIR_POLICY
    acceptance_rule: str = STRICT_IMPROVEMENT_ACCEPTANCE

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace.strip():
            raise ValueError("seed_namespace must be a non-empty string")
        if self.schema != RELATION_SHARED_CMA_SWEEP_SCHEMA:
            raise ValueError("unsupported relation CMA action schema")
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != RELATION_SHARED_CMA_SWEEP_SCHEMA_VERSION
        ):
            raise ValueError("unsupported relation CMA action schema version")
        _integer(self.run_seed, "run_seed")
        _integer(self.checkpoint_fe, "checkpoint_fe")
        for name in (
            "dispatch_checkpoint_hash",
            "topology_hash",
            "anchor_hash",
            "initial_incumbent_hash",
            "relation_order_hash",
            "parameter_hash",
        ):
            _validate_hash(getattr(self, name), name)

        incumbent = _vector(
            self.initial_incumbent,
            FULL_SPACE_DIMENSION,
            "initial_incumbent",
        )
        object.__setattr__(self, "initial_incumbent", incumbent)
        if self.initial_incumbent_hash != full_space_vector_hash(incumbent):
            raise ValueError("initial_incumbent_hash does not match initial_incumbent")
        fitness = _finite(self.acceptance_fitness, "acceptance_fitness")
        if fitness < 0.0:
            raise ValueError("acceptance_fitness must be non-negative")
        object.__setattr__(self, "acceptance_fitness", fitness)

        blocks = tuple(self.blocks)
        if any(not isinstance(block, FrozenRelationCmaBlock) for block in blocks):
            raise TypeError("blocks must contain FrozenRelationCmaBlock values")
        if len({block.optimizer_seed for block in blocks}) != len(blocks):
            raise ValueError("relation CMA optimizer seeds must be unique")
        relations = tuple(block.relation for block in blocks)
        _validate_a_relation_order(relations)
        object.__setattr__(self, "blocks", blocks)
        expected_order_hash = ordered_relations_hash(relations)
        if self.relation_order_hash != expected_order_hash:
            raise ValueError("relation_order_hash does not match the frozen block order")
        for block in blocks:
            indices = block.relation.shared_variable_indices
            if indices[-1] >= FULL_SPACE_DIMENSION:
                raise ValueError("relation shared variable is outside the AOB incumbent")
            expected_mean = tuple(incumbent[index] for index in indices)
            if block.initial_mean != expected_mean:
                raise ValueError("block initial_mean must equal its checkpoint shared values")

        lower = _finite(self.lower_bound, "lower_bound")
        upper = _finite(self.upper_bound, "upper_bound")
        if lower >= upper:
            raise ValueError("lower_bound must be smaller than upper_bound")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

        if self.initial_sigma != RELATION_CMA_INITIAL_SIGMA:
            raise ValueError("A-series relation CMA sigma must be 0.5")
        if self.population_size != RELATION_CMA_POPULATION_SIZE:
            raise ValueError("A-series relation CMA population must be 10")
        if self.generation_count != RELATION_CMA_GENERATION_COUNT:
            raise ValueError("A-series relation CMA must sample 25 generations")
        if self.block_budget_fes != RELATION_CMA_BLOCK_BUDGET_FES:
            raise ValueError("each relation CMA block must consume 250 FEs")
        if self.total_budget_fes != RELATION_CMA_TOTAL_BUDGET_FES:
            raise ValueError("the relation CMA sweep must consume 4750 FEs")
        if self.parameterization != RELATION_CMA_PARAMETERIZATION:
            raise ValueError("unsupported relation CMA parameterization")
        if self.reference_version != RELATION_CMA_REFERENCE_VERSION:
            raise ValueError("unsupported relation CMA reference version")
        if self.parameter_hash != RELATION_CMA_PARAMETERS_HASH:
            raise ValueError("relation CMA parameter hash drifted")
        if self.restart_policy != NO_RESTART_POLICY:
            raise ValueError("relation CMA does not permit restarts")
        if self.early_stopping_policy != NO_EARLY_STOPPING_POLICY:
            raise ValueError("relation CMA does not permit early stopping")
        if self.repair_policy != NO_REPAIR_POLICY:
            raise ValueError("relation CMA does not permit candidate repair")
        if self.acceptance_rule != STRICT_IMPROVEMENT_ACCEPTANCE:
            raise ValueError("relation CMA requires strict-improvement acceptance")

        issued = _integer(self.issued_sweep, "issued_sweep")
        target = _integer(self.target_sweep, "target_sweep")
        ttl = _integer(self.ttl_sweeps, "ttl_sweeps", minimum=1)
        expires = _integer(self.expires_sweep, "expires_sweep")
        if ttl != 1 or target != issued + 1 or expires != target:
            raise ValueError("relation CMA action must target only the next sweep")

        expected_anchor = relation_cma_anchor_hash(
            problem_id=self.problem_id,
            run_seed=self.run_seed,
            checkpoint_fe=self.checkpoint_fe,
            dispatch_checkpoint_hash=self.dispatch_checkpoint_hash,
            topology_hash=self.topology_hash,
            initial_incumbent=incumbent,
            blocks=blocks,
            issued_sweep=issued,
        )
        if self.anchor_hash != expected_anchor:
            raise ValueError("anchor_hash does not match the frozen relation sweep")

    def audit_payload(self) -> dict[str, object]:
        return {
            "action": RELATION_SHARED_CMA_SWEEP_ACTION,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "dispatch_checkpoint_hash": self.dispatch_checkpoint_hash,
            "topology_hash": self.topology_hash,
            "anchor_hash": self.anchor_hash,
            "initial_incumbent": list(self.initial_incumbent),
            "initial_incumbent_hash": self.initial_incumbent_hash,
            "acceptance_fitness": self.acceptance_fitness,
            "blocks": [block.audit_payload() for block in self.blocks],
            "relation_order_hash": self.relation_order_hash,
            "seed_namespace": self.seed_namespace,
            "issued_sweep": self.issued_sweep,
            "target_sweep": self.target_sweep,
            "ttl_sweeps": self.ttl_sweeps,
            "expires_sweep": self.expires_sweep,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            **_RELATION_CMA_PARAMETERS,
            "total_budget_fes": self.total_budget_fes,
            "parameter_hash": self.parameter_hash,
        }

    @property
    def action_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


@dataclass(frozen=True)
class OwnerMemorySyncRequest:
    action_hash: str
    block_position: int
    relation: RelationBlockKey
    shared_values: tuple[float, ...]
    pre_incumbent_hash: str
    post_incumbent_hash: str
    pre_owner_context_memory_hashes: tuple[str, str]
    expected_post_owner_context_memory_hashes: tuple[str, str]

    def __post_init__(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        _integer(self.block_position, "block_position")
        if not isinstance(self.relation, RelationBlockKey):
            raise TypeError("relation must be a RelationBlockKey")
        object.__setattr__(
            self,
            "shared_values",
            _vector(self.shared_values, RELATION_DIMENSION, "shared_values"),
        )
        _validate_hash(self.pre_incumbent_hash, "pre_incumbent_hash")
        _validate_hash(self.post_incumbent_hash, "post_incumbent_hash")
        hashes = tuple(self.pre_owner_context_memory_hashes)
        if len(hashes) != 2:
            raise ValueError("owner sync requires two pre-context hashes")
        for value in hashes:
            _validate_hash(value, "pre_owner_context_memory_hash")
        object.__setattr__(self, "pre_owner_context_memory_hashes", hashes)
        expected_hashes = tuple(self.expected_post_owner_context_memory_hashes)
        if len(expected_hashes) != 2:
            raise ValueError("owner sync requires two expected post-context hashes")
        for value in expected_hashes:
            _validate_hash(value, "expected_post_owner_context_memory_hash")
        object.__setattr__(
            self,
            "expected_post_owner_context_memory_hashes",
            expected_hashes,
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "action_hash": self.action_hash,
            "block_position": self.block_position,
            "relation": self.relation.audit_payload(),
            "shared_values": list(self.shared_values),
            "shared_values_hash": shared_values_hash(self.shared_values),
            "pre_incumbent_hash": self.pre_incumbent_hash,
            "post_incumbent_hash": self.post_incumbent_hash,
            "pre_owner_context_memory_hashes": list(self.pre_owner_context_memory_hashes),
            "expected_post_owner_context_memory_hashes": list(
                self.expected_post_owner_context_memory_hashes
            ),
        }

    @property
    def request_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


@dataclass(frozen=True)
class OwnerContextMemoryTransition:
    owner_group_index: int
    pre_context_memory_hash: str
    post_context_memory_hash: str

    def __post_init__(self) -> None:
        _integer(self.owner_group_index, "owner_group_index")
        _validate_hash(self.pre_context_memory_hash, "pre_context_memory_hash")
        _validate_hash(self.post_context_memory_hash, "post_context_memory_hash")


@dataclass(frozen=True)
class OwnerMemorySyncReceipt:
    request_hash: str
    relation: RelationBlockKey
    shared_values_hash: str
    owner_transitions: tuple[OwnerContextMemoryTransition, OwnerContextMemoryTransition]

    def __post_init__(self) -> None:
        _validate_hash(self.request_hash, "request_hash")
        if not isinstance(self.relation, RelationBlockKey):
            raise TypeError("relation must be a RelationBlockKey")
        _validate_hash(self.shared_values_hash, "shared_values_hash")
        transitions = tuple(self.owner_transitions)
        if len(transitions) != 2 or any(
            not isinstance(value, OwnerContextMemoryTransition) for value in transitions
        ):
            raise TypeError("sync receipt requires two owner transitions")
        owners = tuple(value.owner_group_index for value in transitions)
        if owners != self.relation.owner_group_indices:
            raise ValueError("sync receipt must name both relation owners")
        object.__setattr__(self, "owner_transitions", transitions)


@dataclass(frozen=True)
class RelationCmaOptimizerContractReceipt:
    implementation_type: str
    reference_version: str
    parameter_hash: str
    observed_configuration_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_type, str) or not self.implementation_type:
            raise ValueError("implementation_type must be non-empty")
        if self.reference_version != RELATION_CMA_REFERENCE_VERSION:
            raise ValueError("optimizer receipt reference version drifted")
        if self.parameter_hash != RELATION_CMA_PARAMETERS_HASH:
            raise ValueError("optimizer receipt parameter hash drifted")
        _validate_hash(
            self.observed_configuration_hash,
            "observed_configuration_hash",
        )


@dataclass(frozen=True)
class RelationCmaTestOnlyImplementationReceipt:
    """Explicit opt-in for deterministic test doubles; production defaults reject them."""

    implementation_type: str
    reference_version: str = RELATION_CMA_REFERENCE_VERSION
    parameter_hash: str = RELATION_CMA_PARAMETERS_HASH

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_type, str) or not self.implementation_type:
            raise ValueError("implementation_type must be non-empty")
        if self.reference_version != RELATION_CMA_REFERENCE_VERSION:
            raise ValueError("test receipt reference version drifted")
        if self.parameter_hash != RELATION_CMA_PARAMETERS_HASH:
            raise ValueError("test receipt parameter hash drifted")


@dataclass(frozen=True)
class RelationSharedCmaExecutionContext:
    objective: Callable[[Any], Any]
    cmaes_factory: Callable[..., Any]
    synchronize_owner_memory: Callable[[OwnerMemorySyncRequest], OwnerMemorySyncReceipt]
    current_fe: int
    current_sweep: int
    dispatch_checkpoint_hash: str
    topology_hash: str
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    required_seed_namespace: str
    owner_context_memories: tuple[OwnerContextMemorySnapshot, ...]
    test_only_implementation_receipt: RelationCmaTestOnlyImplementationReceipt | None = None

    def __post_init__(self) -> None:
        for callback_name in (
            "objective",
            "cmaes_factory",
            "synchronize_owner_memory",
        ):
            if not callable(getattr(self, callback_name)):
                raise TypeError(f"{callback_name} must be callable")
        _integer(self.current_fe, "current_fe")
        _integer(self.current_sweep, "current_sweep")
        _validate_hash(self.dispatch_checkpoint_hash, "dispatch_checkpoint_hash")
        _validate_hash(self.topology_hash, "topology_hash")
        object.__setattr__(
            self,
            "incumbent",
            _vector(self.incumbent, FULL_SPACE_DIMENSION, "incumbent"),
        )
        fitness = _finite(self.incumbent_fitness, "incumbent_fitness")
        if fitness < 0.0:
            raise ValueError("incumbent_fitness must be non-negative")
        object.__setattr__(self, "incumbent_fitness", fitness)
        if (
            not isinstance(self.required_seed_namespace, str)
            or not self.required_seed_namespace.strip()
        ):
            raise ValueError("required_seed_namespace must be a non-empty string")
        owner_memories = tuple(self.owner_context_memories)
        if len(owner_memories) != RELATION_COUNT + 1:
            raise ValueError("context must bind all twenty owner memories")
        observed_owners: list[int] = []
        for snapshot in owner_memories:
            if not isinstance(snapshot, OwnerContextMemorySnapshot):
                raise TypeError(
                    "owner_context_memories must contain OwnerContextMemorySnapshot values"
                )
            observed_owners.append(snapshot.owner_group_index)
        if observed_owners != sorted(set(observed_owners)):
            raise ValueError("owner context-memory hashes must be sorted and unique")
        object.__setattr__(self, "owner_context_memories", owner_memories)
        receipt = self.test_only_implementation_receipt
        if receipt is not None and not isinstance(
            receipt,
            RelationCmaTestOnlyImplementationReceipt,
        ):
            raise TypeError("test_only_implementation_receipt has the wrong type")


@dataclass(frozen=True)
class RelationCmaBlockExecutionResult:
    relation: RelationBlockKey
    candidate_shared_values: tuple[float, ...]
    candidate_fitness: float
    accepted: bool
    consumed_fes: int
    sampled_generation_count: int
    initial_state_hash: str
    final_state_hash: str
    candidate_hash: str
    post_incumbent_hash: str
    owner_context_memory_transitions: tuple[OwnerContextMemoryTransition, ...]
    optimizer_contract_receipt: RelationCmaOptimizerContractReceipt


@dataclass
class RelationSharedCmaSweepExecutionState:
    """Caller-owned one-shot lifecycle for a relation-CMA sweep."""

    action_hash: str
    status: str = "issued"
    started_fe: int | None = None
    completed_fe: int | None = None
    consumed_fes: int = 0
    accepted_block_count: int = 0
    post_incumbent_hash: str | None = None
    invalidation_reason: str = ""
    failure_reason: str = ""

    def __post_init__(self) -> None:
        self._validate_shape()

    @classmethod
    def for_action(
        cls,
        action: RelationSharedCmaSweepAction,
    ) -> RelationSharedCmaSweepExecutionState:
        return cls(action_hash=action.action_hash)

    def _validate_shape(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        if self.status not in _SWEEP_STATUSES:
            raise ValueError("unsupported relation CMA lifecycle status")
        if self.started_fe is not None:
            _integer(self.started_fe, "started_fe")
        if self.completed_fe is not None:
            _integer(self.completed_fe, "completed_fe")
        _integer(self.consumed_fes, "consumed_fes")
        _integer(self.accepted_block_count, "accepted_block_count")
        if self.post_incumbent_hash is not None:
            _validate_hash(self.post_incumbent_hash, "post_incumbent_hash")
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")
        if not isinstance(self.failure_reason, str):
            raise ValueError("failure_reason must be a string")

    def validate_for(self, action: RelationSharedCmaSweepAction) -> None:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            raise ValueError("relation CMA lifecycle does not match action_hash")
        if self.status == "issued":
            if any(
                value is not None
                for value in (self.started_fe, self.completed_fe, self.post_incumbent_hash)
            ) or any(
                (
                    self.consumed_fes,
                    self.accepted_block_count,
                    self.invalidation_reason,
                    self.failure_reason,
                )
            ):
                raise ValueError("issued relation CMA lifecycle contains outcome data")
        elif self.status == "running":
            if (
                self.started_fe is None
                or self.completed_fe is not None
                or self.invalidation_reason
                or self.failure_reason
                or self.accepted_block_count > RELATION_COUNT
                or self.post_incumbent_hash is None
            ):
                raise ValueError("running relation CMA lifecycle is inconsistent")
        elif self.status == "completed":
            if (
                self.started_fe is None
                or self.completed_fe != self.started_fe + action.total_budget_fes
                or self.consumed_fes != action.total_budget_fes
                or self.accepted_block_count > RELATION_COUNT
                or self.post_incumbent_hash is None
                or self.invalidation_reason
                or self.failure_reason
            ):
                raise ValueError("completed relation CMA lifecycle is inconsistent")
        elif self.status == "abstained":
            if (
                self.started_fe is not None
                or self.completed_fe is not None
                or self.consumed_fes
                or self.accepted_block_count
                or self.post_incumbent_hash is not None
                or not self.invalidation_reason
                or self.failure_reason
            ):
                raise ValueError("abstained relation CMA lifecycle is inconsistent")
        elif (
            self.started_fe is None
            or self.completed_fe != self.started_fe + self.consumed_fes
            or self.accepted_block_count > RELATION_COUNT
            or self.post_incumbent_hash is None
            or self.invalidation_reason
            or not self.failure_reason
        ):
            raise ValueError("failed relation CMA lifecycle is inconsistent")

    def abstain(self, action: RelationSharedCmaSweepAction, reason: str) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("only an issued relation CMA action can abstain")
        if not isinstance(reason, str) or not reason:
            raise ValueError("abstain reason must be non-empty")
        self.status = "abstained"
        self.invalidation_reason = reason

    def start(self, action: RelationSharedCmaSweepAction, *, current_fe: int) -> None:
        self.validate_for(action)
        if self.status != "issued":
            raise ValueError("relation CMA action already consumed")
        self.started_fe = _integer(current_fe, "current_fe")
        self.post_incumbent_hash = action.initial_incumbent_hash
        self.status = "running"

    def observe_evaluations(
        self,
        action: RelationSharedCmaSweepAction,
        count: int,
    ) -> None:
        self.validate_for(action)
        if self.status != "running":
            raise ValueError("only a running relation CMA action can observe FEs")
        observed = _integer(count, "observed evaluations", minimum=1)
        self.consumed_fes += observed
        if self.consumed_fes > action.total_budget_fes:
            raise RuntimeError("relation CMA observed FEs exceed the frozen total")

    def observe_acceptance(
        self,
        action: RelationSharedCmaSweepAction,
        *,
        post_incumbent_hash: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "running":
            raise ValueError("only a running relation CMA action can accept a block")
        if self.accepted_block_count >= RELATION_COUNT:
            raise RuntimeError("relation CMA accepted too many blocks")
        self.accepted_block_count += 1
        self.post_incumbent_hash = _validate_hash(
            post_incumbent_hash,
            "post_incumbent_hash",
        )

    def fail(
        self,
        action: RelationSharedCmaSweepAction,
        *,
        reason: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "running" or self.started_fe is None:
            raise ValueError("only a running relation CMA action can fail")
        if not isinstance(reason, str) or not reason:
            raise ValueError("failure reason must be non-empty")
        self.completed_fe = self.started_fe + self.consumed_fes
        self.failure_reason = reason
        self.status = "failed"
        self.validate_for(action)

    def complete(
        self,
        action: RelationSharedCmaSweepAction,
        *,
        accepted_block_count: int,
        post_incumbent_hash: str,
    ) -> None:
        self.validate_for(action)
        if self.status != "running" or self.started_fe is None:
            raise ValueError("only a running relation CMA action can complete")
        accepted = _integer(accepted_block_count, "accepted_block_count")
        if accepted > RELATION_COUNT:
            raise ValueError("accepted_block_count exceeds relation count")
        if self.consumed_fes != action.total_budget_fes:
            raise RuntimeError("relation CMA lifecycle FE ledger is incomplete")
        if self.accepted_block_count != accepted:
            raise RuntimeError("relation CMA lifecycle acceptance ledger drifted")
        self.completed_fe = self.started_fe + self.consumed_fes
        self.post_incumbent_hash = _validate_hash(
            post_incumbent_hash,
            "post_incumbent_hash",
        )
        self.status = "completed"
        self.validate_for(action)

    def audit_payload(self, action: RelationSharedCmaSweepAction) -> dict[str, object]:
        self.validate_for(action)
        return self.observed_audit_payload()

    def observed_audit_payload(self) -> dict[str, object]:
        self._validate_shape()
        return {
            "action": RELATION_SHARED_CMA_SWEEP_ACTION,
            "schema": RELATION_SHARED_CMA_SWEEP_SCHEMA,
            "schema_version": RELATION_SHARED_CMA_SWEEP_SCHEMA_VERSION,
            "action_hash": self.action_hash,
            "status": self.status,
            "started_fe": self.started_fe,
            "completed_fe": self.completed_fe,
            "consumed_fes": self.consumed_fes,
            "accepted_block_count": self.accepted_block_count,
            "post_incumbent_hash": self.post_incumbent_hash,
            "invalidation_reason": self.invalidation_reason,
            "failure_reason": self.failure_reason,
        }

    def state_hash(self, action: RelationSharedCmaSweepAction) -> str:
        return _canonical_sha256(self.audit_payload(action))

    def observed_state_hash(self) -> str:
        return _canonical_sha256(self.observed_audit_payload())


@dataclass(frozen=True)
class RelationSharedCmaSweepExecutionResult:
    incumbent: tuple[float, ...]
    incumbent_fitness: float
    accepted_block_count: int
    consumed_fes: int
    action_hash: str
    initial_incumbent_hash: str
    post_incumbent_hash: str
    block_results: tuple[RelationCmaBlockExecutionResult, ...]
    lifecycle: RelationSharedCmaSweepExecutionState
    lifecycle_hash: str
    observed_lifecycle_action_hash: str
    abstained: bool = False
    invalidation_reason: str = ""
    resume_native: bool = True


@dataclass(frozen=True)
class PreparedRelationCmaOptimizer:
    optimizer: Any
    contract_receipt: RelationCmaOptimizerContractReceipt


def _optimizer_implementation_type(optimizer: object) -> str:
    optimizer_type = type(optimizer)
    return f"{optimizer_type.__module__}.{optimizer_type.__qualname__}"


def _require_observed_scalar(
    optimizer: object,
    attribute: str,
    expected: float,
) -> None:
    observed = _finite(getattr(optimizer, attribute, None), attribute)
    if observed != expected:
        raise RuntimeError(f"relation CMA optimizer {attribute} drifted")


def _validate_relation_cma_optimizer_contract(
    optimizer: object,
    *,
    initial_mean: tuple[float, ...],
    lower_bound: float,
    upper_bound: float,
    optimizer_seed: int,
    test_only_implementation_receipt: (RelationCmaTestOnlyImplementationReceipt | None),
) -> RelationCmaOptimizerContractReceipt:
    implementation_type = _optimizer_implementation_type(optimizer)
    if implementation_type != RELATION_CMA_IMPLEMENTATION_TYPE:
        test_receipt = test_only_implementation_receipt
        if (
            test_receipt is None
            or test_receipt.implementation_type != implementation_type
            or test_receipt.reference_version != RELATION_CMA_REFERENCE_VERSION
            or test_receipt.parameter_hash != RELATION_CMA_PARAMETERS_HASH
        ):
            raise RuntimeError("relation CMA factory did not construct the pinned implementation")

    options = getattr(optimizer, "options", None)
    if not isinstance(options, dict):
        raise RuntimeError("relation CMA optimizer must expose its frozen options")
    if int(getattr(optimizer, "ndim_problem", -1)) != RELATION_DIMENSION:
        raise RuntimeError("relation CMA optimizer dimension drifted")
    if int(getattr(optimizer, "n_individuals", -1)) != RELATION_CMA_POPULATION_SIZE:
        raise RuntimeError("relation CMA optimizer population_size drifted")
    if int(getattr(optimizer, "max_function_evaluations", -1)) != (RELATION_CMA_BLOCK_BUDGET_FES):
        raise RuntimeError("relation CMA optimizer FE budget drifted")
    _require_observed_scalar(
        optimizer,
        "sigma",
        RELATION_CMA_INITIAL_SIGMA,
    )
    if getattr(optimizer, "is_restart", None) is not False:
        raise RuntimeError("relation CMA optimizer restart policy drifted")
    early_stopping = float(getattr(optimizer, "early_stopping_evaluations", math.nan))
    if early_stopping != math.inf:
        raise RuntimeError("relation CMA optimizer early-stopping policy drifted")
    if int(getattr(optimizer, "seed_rng", -1)) != optimizer_seed:
        raise RuntimeError("relation CMA optimizer seed drifted")
    if int(getattr(optimizer, "verbose", -1)) != 0:
        raise RuntimeError("relation CMA optimizer verbosity drifted")
    if float(getattr(optimizer, "max_runtime", math.nan)) != math.inf:
        raise RuntimeError("relation CMA optimizer runtime termination drifted")
    if float(getattr(optimizer, "fitness_threshold", math.nan)) != -math.inf:
        raise RuntimeError("relation CMA optimizer fitness termination drifted")

    observed_mean = _vector(
        getattr(optimizer, "mean", None),
        RELATION_DIMENSION,
        "optimizer mean",
    )
    if observed_mean != initial_mean:
        raise RuntimeError("relation CMA optimizer initial mean drifted")
    observed_lower = _vector(
        getattr(optimizer, "lower_boundary", None),
        RELATION_DIMENSION,
        "optimizer lower boundary",
    )
    observed_upper = _vector(
        getattr(optimizer, "upper_boundary", None),
        RELATION_DIMENSION,
        "optimizer upper boundary",
    )
    if observed_lower != (lower_bound,) * RELATION_DIMENSION:
        raise RuntimeError("relation CMA optimizer lower boundary drifted")
    if observed_upper != (upper_bound,) * RELATION_DIMENSION:
        raise RuntimeError("relation CMA optimizer upper boundary drifted")

    expected_options: tuple[tuple[str, object], ...] = (
        ("max_function_evaluations", RELATION_CMA_BLOCK_BUDGET_FES),
        ("sigma", RELATION_CMA_INITIAL_SIGMA),
        ("n_individuals", RELATION_CMA_POPULATION_SIZE),
        ("is_restart", False),
        ("verbose", 0),
        ("early_stopping_evaluations", math.inf),
        ("seed_rng", optimizer_seed),
        ("_save_eig", True),
        ("diagonal_only", False),
    )
    if any(options.get(name) != expected for name, expected in expected_options):
        raise RuntimeError("relation CMA optimizer frozen options drifted")
    option_mean = _vector(
        options.get("mean"),
        RELATION_DIMENSION,
        "optimizer option mean",
    )
    if option_mean != initial_mean:
        raise RuntimeError("relation CMA optimizer option mean drifted")

    configuration_hash = _canonical_sha256(
        {
            "implementation_type": implementation_type,
            "reference_version": RELATION_CMA_REFERENCE_VERSION,
            "parameter_hash": RELATION_CMA_PARAMETERS_HASH,
            "dimension": RELATION_DIMENSION,
            "initial_mean": list(observed_mean),
            "lower_boundary": list(observed_lower),
            "upper_boundary": list(observed_upper),
            "optimizer_seed": optimizer_seed,
            "population_size": RELATION_CMA_POPULATION_SIZE,
            "block_budget_fes": RELATION_CMA_BLOCK_BUDGET_FES,
            "initial_sigma": RELATION_CMA_INITIAL_SIGMA,
            "restart_policy": NO_RESTART_POLICY,
            "early_stopping_policy": NO_EARLY_STOPPING_POLICY,
            "runtime_termination_policy": "disabled",
            "fitness_termination_policy": "disabled",
            "repair_policy": NO_REPAIR_POLICY,
            "save_eigendecomposition": True,
            "diagonal_only": False,
        }
    )
    return RelationCmaOptimizerContractReceipt(
        implementation_type=implementation_type,
        reference_version=RELATION_CMA_REFERENCE_VERSION,
        parameter_hash=RELATION_CMA_PARAMETERS_HASH,
        observed_configuration_hash=configuration_hash,
    )


def build_relation_cma_optimizer(
    factory: Callable[..., Any],
    *,
    objective: Callable[[Any], Any],
    initial_mean: Sequence[float],
    lower_bound: float,
    upper_bound: float,
    optimizer_seed: int,
    test_only_implementation_receipt: (RelationCmaTestOnlyImplementationReceipt | None) = None,
) -> PreparedRelationCmaOptimizer:
    """Construct the pinned 5D vendor full-CMA optimizer without reimplementing it."""

    if not callable(factory):
        raise TypeError("CMA-ES factory must be callable")
    if not callable(objective):
        raise TypeError("objective must be callable")
    mean = _vector(initial_mean, RELATION_DIMENSION, "initial_mean")
    lower = _finite(lower_bound, "lower_bound")
    upper = _finite(upper_bound, "upper_bound")
    if lower >= upper:
        raise ValueError("lower_bound must be smaller than upper_bound")
    seed = _integer(optimizer_seed, "optimizer_seed")
    optimizer = factory(
        {
            "fitness_function": objective,
            "ndim_problem": RELATION_DIMENSION,
            "lower_boundary": lower * np.ones((RELATION_DIMENSION,)),
            "upper_boundary": upper * np.ones((RELATION_DIMENSION,)),
        },
        {
            "max_function_evaluations": RELATION_CMA_BLOCK_BUDGET_FES,
            "mean": np.asarray(mean, dtype=float),
            "sigma": RELATION_CMA_INITIAL_SIGMA,
            "n_individuals": RELATION_CMA_POPULATION_SIZE,
            "is_restart": False,
            "verbose": 0,
            "early_stopping_evaluations": np.inf,
            "seed_rng": seed,
            "_save_eig": True,
            "diagonal_only": False,
        },
    )
    receipt = _validate_relation_cma_optimizer_contract(
        optimizer,
        initial_mean=mean,
        lower_bound=lower,
        upper_bound=upper,
        optimizer_seed=seed,
        test_only_implementation_receipt=test_only_implementation_receipt,
    )
    return PreparedRelationCmaOptimizer(
        optimizer=optimizer,
        contract_receipt=receipt,
    )


def _matrix(values: object, dimension: int, name: str) -> list[list[float]]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{name} must be a finite {dimension}x{dimension} matrix")
    return [[float(value) for value in row] for row in matrix]


def _optimizer_final_state_hash(
    result: dict[str, object],
    *,
    sampled_generation_count: int,
) -> str:
    restart_count = int(result.get("_n_restart", -1))
    if restart_count != 0:
        raise RuntimeError("relation CMA restarted despite the frozen no-restart policy")
    return _canonical_sha256(
        {
            "mean": list(_vector(result.get("mean"), RELATION_DIMENSION, "final mean")),
            "sigma": _finite(result.get("sigma"), "final sigma"),  # type: ignore[arg-type]
            "p_s": list(_vector(result.get("p_s"), RELATION_DIMENSION, "p_s")),
            "p_c": list(_vector(result.get("p_c"), RELATION_DIMENSION, "p_c")),
            "e_va": list(_vector(result.get("e_va"), RELATION_DIMENSION, "e_va")),
            "e_ve": _matrix(result.get("e_ve"), RELATION_DIMENSION, "e_ve"),
            "n_function_evaluations": int(result["n_function_evaluations"]),
            "sampled_generation_count": sampled_generation_count,
            "parameter_hash": RELATION_CMA_PARAMETERS_HASH,
        }
    )


def _execute_relation_shared_cma_sweep_action(
    action: RelationSharedCmaSweepAction,
    state: RelationSharedCmaSweepExecutionState,
    context: RelationSharedCmaExecutionContext,
) -> RelationSharedCmaSweepExecutionResult:
    """Run the frozen 19 blocks in order, accepting only strict improvements."""

    if not isinstance(action, RelationSharedCmaSweepAction):
        raise TypeError("action must be a RelationSharedCmaSweepAction")
    if not isinstance(state, RelationSharedCmaSweepExecutionState):
        raise TypeError("state must be a RelationSharedCmaSweepExecutionState")
    if not isinstance(context, RelationSharedCmaExecutionContext):
        raise TypeError("context must be a RelationSharedCmaExecutionContext")
    state._validate_shape()
    if state.action_hash != action.action_hash:
        return RelationSharedCmaSweepExecutionResult(
            incumbent=context.incumbent,
            incumbent_fitness=context.incumbent_fitness,
            accepted_block_count=0,
            consumed_fes=0,
            action_hash=action.action_hash,
            initial_incumbent_hash=action.initial_incumbent_hash,
            post_incumbent_hash=full_space_vector_hash(context.incumbent),
            block_results=(),
            lifecycle=state,
            lifecycle_hash=state.observed_state_hash(),
            observed_lifecycle_action_hash=state.action_hash,
            abstained=True,
            invalidation_reason="lifecycle_action_hash_mismatch",
        )
    state.validate_for(action)
    if state.status != "issued":
        reason = {
            "completed": "action_already_consumed",
            "running": "action_already_running",
            "abstained": f"action_invalidated:{state.invalidation_reason}",
            "failed": f"action_failed:{state.failure_reason}",
        }[state.status]
        return RelationSharedCmaSweepExecutionResult(
            incumbent=context.incumbent,
            incumbent_fitness=context.incumbent_fitness,
            accepted_block_count=0,
            consumed_fes=0,
            action_hash=action.action_hash,
            initial_incumbent_hash=action.initial_incumbent_hash,
            post_incumbent_hash=full_space_vector_hash(context.incumbent),
            block_results=(),
            lifecycle=state,
            lifecycle_hash=state.state_hash(action),
            observed_lifecycle_action_hash=state.action_hash,
            abstained=True,
            invalidation_reason=reason,
        )

    expected_owners = tuple(
        sorted({owner for block in action.blocks for owner in block.relation.owner_group_indices})
    )
    observed_owner_memories = {
        snapshot.owner_group_index: snapshot for snapshot in context.owner_context_memories
    }
    mismatch_reason = ""
    if context.current_fe != action.checkpoint_fe:
        mismatch_reason = "checkpoint_fe_mismatch"
    elif context.current_sweep > action.expires_sweep:
        mismatch_reason = "action_expired"
    elif context.current_sweep != action.target_sweep:
        mismatch_reason = "target_sweep_mismatch"
    elif context.dispatch_checkpoint_hash != action.dispatch_checkpoint_hash:
        mismatch_reason = "dispatch_checkpoint_hash_mismatch"
    elif context.topology_hash != action.topology_hash:
        mismatch_reason = "topology_hash_mismatch"
    elif context.required_seed_namespace != action.seed_namespace:
        mismatch_reason = "seed_namespace_mismatch"
    elif full_space_vector_hash(context.incumbent) != action.initial_incumbent_hash:
        mismatch_reason = "incumbent_anchor_mismatch"
    elif context.incumbent_fitness != action.acceptance_fitness:
        mismatch_reason = "acceptance_fitness_mismatch"
    elif tuple(observed_owner_memories) != expected_owners:
        mismatch_reason = "owner_context_set_mismatch"
    elif any(
        any(
            dimension not in observed_owner_memories[owner].dimensions
            for dimension in block.relation.shared_variable_indices
        )
        for block in action.blocks
        for owner in block.relation.owner_group_indices
    ):
        mismatch_reason = "owner_context_relation_mismatch"
    if mismatch_reason:
        state.abstain(action, mismatch_reason)
        return RelationSharedCmaSweepExecutionResult(
            incumbent=context.incumbent,
            incumbent_fitness=context.incumbent_fitness,
            accepted_block_count=0,
            consumed_fes=0,
            action_hash=action.action_hash,
            initial_incumbent_hash=action.initial_incumbent_hash,
            post_incumbent_hash=full_space_vector_hash(context.incumbent),
            block_results=(),
            lifecycle=state,
            lifecycle_hash=state.state_hash(action),
            observed_lifecycle_action_hash=state.action_hash,
            abstained=True,
            invalidation_reason=mismatch_reason,
        )
    state.start(action, current_fe=context.current_fe)

    incumbent = np.asarray(context.incumbent, dtype=float).copy()
    incumbent_fitness = context.incumbent_fitness
    consumed_fes = 0
    block_results: list[RelationCmaBlockExecutionResult] = []

    for position, block in enumerate(action.blocks):
        indices = np.asarray(block.relation.shared_variable_indices, dtype=int)
        snapshot = incumbent.copy()
        evaluated_fes = 0
        evaluation_batch_sizes: list[int] = []
        best_values: np.ndarray | None = None
        best_fitness = math.inf

        def relation_objective(raw_values: object) -> float | np.ndarray:
            nonlocal evaluated_fes, best_values, best_fitness
            values = np.asarray(raw_values, dtype=float)
            single = values.ndim == 1
            if single:
                values = values.reshape(1, -1)
            if values.ndim != 2 or values.shape[1] != RELATION_DIMENSION:
                raise ValueError("relation CMA candidate batch has the wrong shape")
            if not np.all(np.isfinite(values)):
                raise ValueError("relation CMA candidate batch must be finite")
            candidate_batch = np.repeat(snapshot.reshape(1, -1), len(values), axis=0)
            candidate_batch[:, indices] = values
            raw_output = context.objective(candidate_batch)
            evaluated_fes += len(values)
            evaluation_batch_sizes.append(len(values))
            state.observe_evaluations(action, len(values))
            output = np.asarray(raw_output, dtype=float).reshape(-1)
            if output.shape != (len(values),) or not np.all(np.isfinite(output)):
                raise ValueError("objective must return one finite fitness per candidate")
            for candidate, fitness in zip(values, output, strict=True):
                value = float(fitness)
                if value < best_fitness:
                    best_fitness = value
                    best_values = candidate.copy()
            return float(output[0]) if single else output

        prepared_optimizer = build_relation_cma_optimizer(
            context.cmaes_factory,
            objective=relation_objective,
            initial_mean=block.initial_mean,
            lower_bound=action.lower_bound,
            upper_bound=action.upper_bound,
            optimizer_seed=block.optimizer_seed,
            test_only_implementation_receipt=(context.test_only_implementation_receipt),
        )
        raw_result = prepared_optimizer.optimizer.optimize()
        if not isinstance(raw_result, dict):
            raise RuntimeError("relation CMA optimizer must return a result mapping")
        reported_fes = int(raw_result.get("n_function_evaluations", -1))
        if (
            evaluated_fes != RELATION_CMA_BLOCK_BUDGET_FES
            or reported_fes != RELATION_CMA_BLOCK_BUDGET_FES
        ):
            raise RuntimeError("relation CMA did not consume its exact frozen FE budget")
        sampled_generation_count = len(evaluation_batch_sizes)
        if evaluation_batch_sizes != [RELATION_CMA_POPULATION_SIZE] * RELATION_CMA_GENERATION_COUNT:
            raise RuntimeError(
                "relation CMA did not execute the frozen population-generation schedule"
            )
        reported_generation_updates = int(raw_result.get("_n_generations", -1))
        if reported_generation_updates != sampled_generation_count - 1:
            raise RuntimeError("relation CMA generation ledger drifted")
        if best_values is None:
            raise RuntimeError("relation CMA produced no evaluated candidate")
        reported_candidate = _vector(
            raw_result.get("best_so_far_x"),
            RELATION_DIMENSION,
            "best_so_far_x",
        )
        reported_fitness = _finite(
            raw_result.get("best_so_far_y"),  # type: ignore[arg-type]
            "best_so_far_y",
        )
        candidate = tuple(float(value) for value in best_values)
        if reported_candidate != candidate or reported_fitness != best_fitness:
            raise RuntimeError("relation CMA best-candidate ledger drifted")
        if best_fitness < 0.0:
            raise RuntimeError("relation CMA returned a negative AOB error")

        accepted = best_fitness < incumbent_fitness
        pre_hash = full_space_vector_hash(incumbent)
        owner_transitions: tuple[OwnerContextMemoryTransition, ...] = ()
        if accepted:
            post_incumbent = incumbent.copy()
            post_incumbent[indices] = best_values
            post_hash = full_space_vector_hash(post_incumbent)
            pre_owner_memories = tuple(
                observed_owner_memories[owner] for owner in block.relation.owner_group_indices
            )
            expected_post_owner_memories = tuple(
                memory.with_shared_values(block.relation, candidate)
                for memory in pre_owner_memories
            )
            sync_request = OwnerMemorySyncRequest(
                action_hash=action.action_hash,
                block_position=position,
                relation=block.relation,
                shared_values=candidate,
                pre_incumbent_hash=pre_hash,
                post_incumbent_hash=post_hash,
                pre_owner_context_memory_hashes=tuple(
                    memory.context_memory_hash for memory in pre_owner_memories
                ),
                expected_post_owner_context_memory_hashes=tuple(
                    memory.context_memory_hash for memory in expected_post_owner_memories
                ),
            )
            receipt = context.synchronize_owner_memory(sync_request)
            if not isinstance(receipt, OwnerMemorySyncReceipt):
                raise RuntimeError("owner sync callback must return OwnerMemorySyncReceipt")
            if receipt.request_hash != sync_request.request_hash:
                raise RuntimeError("owner sync receipt request mismatch")
            if receipt.relation != block.relation:
                raise RuntimeError("owner sync receipt relation mismatch")
            if receipt.shared_values_hash != shared_values_hash(candidate):
                raise RuntimeError("owner sync receipt values mismatch")
            owner_transitions = receipt.owner_transitions
            for expected_pre_hash, expected_post_memory, transition in zip(
                sync_request.pre_owner_context_memory_hashes,
                expected_post_owner_memories,
                owner_transitions,
                strict=True,
            ):
                if transition.pre_context_memory_hash != expected_pre_hash:
                    raise RuntimeError("owner sync receipt pre-context mismatch")
                if transition.post_context_memory_hash != expected_post_memory.context_memory_hash:
                    raise RuntimeError("owner sync receipt post-context mismatch")
            for expected_post_memory in expected_post_owner_memories:
                observed_owner_memories[expected_post_memory.owner_group_index] = (
                    expected_post_memory
                )
            incumbent = post_incumbent
            incumbent_fitness = best_fitness
            state.observe_acceptance(action, post_incumbent_hash=post_hash)
        else:
            post_hash = pre_hash

        final_state_hash = _optimizer_final_state_hash(
            raw_result,
            sampled_generation_count=sampled_generation_count,
        )
        block_results.append(
            RelationCmaBlockExecutionResult(
                relation=block.relation,
                candidate_shared_values=candidate,
                candidate_fitness=best_fitness,
                accepted=accepted,
                consumed_fes=RELATION_CMA_BLOCK_BUDGET_FES,
                sampled_generation_count=sampled_generation_count,
                initial_state_hash=block.initial_state_hash,
                final_state_hash=final_state_hash,
                candidate_hash=shared_values_hash(candidate),
                post_incumbent_hash=post_hash,
                owner_context_memory_transitions=owner_transitions,
                optimizer_contract_receipt=(prepared_optimizer.contract_receipt),
            )
        )
        consumed_fes += RELATION_CMA_BLOCK_BUDGET_FES

    if consumed_fes != action.total_budget_fes:
        raise RuntimeError("relation CMA sweep did not consume its frozen total FE budget")
    accepted_block_count = sum(result.accepted for result in block_results)
    post_incumbent_hash = full_space_vector_hash(incumbent)
    state.complete(
        action,
        accepted_block_count=accepted_block_count,
        post_incumbent_hash=post_incumbent_hash,
    )
    return RelationSharedCmaSweepExecutionResult(
        incumbent=tuple(float(value) for value in incumbent),
        incumbent_fitness=incumbent_fitness,
        accepted_block_count=accepted_block_count,
        consumed_fes=consumed_fes,
        action_hash=action.action_hash,
        initial_incumbent_hash=action.initial_incumbent_hash,
        post_incumbent_hash=post_incumbent_hash,
        block_results=tuple(block_results),
        lifecycle=state,
        lifecycle_hash=state.state_hash(action),
        observed_lifecycle_action_hash=state.action_hash,
    )


def _stable_failure_reason(error: Exception) -> str:
    message = "_".join(str(error).strip().lower().split())
    normalized = "".join(
        character if character.isalnum() or character == "_" else "_" for character in message
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")[:160] or "no_message"
    return f"{type(error).__name__}:{normalized}"


def execute_relation_shared_cma_sweep_action(
    action: RelationSharedCmaSweepAction,
    state: RelationSharedCmaSweepExecutionState,
    context: RelationSharedCmaExecutionContext,
) -> RelationSharedCmaSweepExecutionResult:
    """Execute once; seal any post-start exception before re-raising it."""

    try:
        return _execute_relation_shared_cma_sweep_action(action, state, context)
    except Exception as error:
        if (
            isinstance(action, RelationSharedCmaSweepAction)
            and isinstance(state, RelationSharedCmaSweepExecutionState)
            and state.status == "running"
        ):
            state.fail(action, reason=_stable_failure_reason(error))
        raise


def no_writeback_window_anchor_hash(
    *,
    problem_id: str,
    run_seed: int,
    checkpoint_fe: int,
    dispatch_checkpoint_hash: str,
    topology_hash: str,
    relations: Sequence[RelationBlockKey],
    issued_sweep: int,
    seed_namespace: str,
) -> str:
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("problem_id must be a non-empty string")
    if not isinstance(seed_namespace, str) or not seed_namespace.strip():
        raise ValueError("seed_namespace must be a non-empty string")
    return _canonical_sha256(
        {
            "action": NO_WRITEBACK_WINDOW_ACTION,
            "schema": NO_WRITEBACK_WINDOW_SCHEMA,
            "schema_version": NO_WRITEBACK_WINDOW_SCHEMA_VERSION,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "checkpoint_fe": _integer(checkpoint_fe, "checkpoint_fe"),
            "dispatch_checkpoint_hash": _validate_hash(
                dispatch_checkpoint_hash,
                "dispatch_checkpoint_hash",
            ),
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "relation_order_hash": ordered_relations_hash(relations),
            "issued_sweep": _integer(issued_sweep, "issued_sweep"),
            "seed_namespace": seed_namespace,
            "budget_fes": 0,
        }
    )


@dataclass(frozen=True)
class NoWritebackWindowAction:
    """Suppress Eq.8 for every frozen relation in exactly one target sweep."""

    problem_id: str
    run_seed: int
    checkpoint_fe: int
    dispatch_checkpoint_hash: str
    topology_hash: str
    anchor_hash: str
    relations: tuple[RelationBlockKey, ...]
    relation_order_hash: str
    seed_namespace: str
    issued_sweep: int
    target_sweep: int
    ttl_sweeps: int
    expires_sweep: int
    budget_fes: int = 0
    schema: str = NO_WRITEBACK_WINDOW_SCHEMA
    schema_version: int = NO_WRITEBACK_WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace.strip():
            raise ValueError("seed_namespace must be a non-empty string")
        if self.schema != NO_WRITEBACK_WINDOW_SCHEMA:
            raise ValueError("unsupported no-writeback action schema")
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != NO_WRITEBACK_WINDOW_SCHEMA_VERSION
        ):
            raise ValueError("unsupported no-writeback action schema version")
        if isinstance(self.budget_fes, bool) or self.budget_fes != 0:
            raise ValueError("no-writeback action budget_fes must be zero")
        _integer(self.run_seed, "run_seed")
        _integer(self.checkpoint_fe, "checkpoint_fe")
        for name in (
            "dispatch_checkpoint_hash",
            "topology_hash",
            "anchor_hash",
            "relation_order_hash",
        ):
            _validate_hash(getattr(self, name), name)
        relations = tuple(self.relations)
        if any(not isinstance(relation, RelationBlockKey) for relation in relations):
            raise TypeError("relations must contain RelationBlockKey values")
        _validate_a_relation_order(relations)
        object.__setattr__(self, "relations", relations)
        if self.relation_order_hash != ordered_relations_hash(relations):
            raise ValueError("relation_order_hash does not match the frozen order")

        issued = _integer(self.issued_sweep, "issued_sweep")
        target = _integer(self.target_sweep, "target_sweep")
        ttl = _integer(self.ttl_sweeps, "ttl_sweeps", minimum=1)
        expires = _integer(self.expires_sweep, "expires_sweep")
        if ttl != 1 or target != issued + 1 or expires != target:
            raise ValueError("no-writeback window must target only the next sweep")
        expected_anchor = no_writeback_window_anchor_hash(
            problem_id=self.problem_id,
            run_seed=self.run_seed,
            checkpoint_fe=self.checkpoint_fe,
            dispatch_checkpoint_hash=self.dispatch_checkpoint_hash,
            topology_hash=self.topology_hash,
            relations=relations,
            issued_sweep=issued,
            seed_namespace=self.seed_namespace,
        )
        if self.anchor_hash != expected_anchor:
            raise ValueError("anchor_hash does not match the no-writeback window")

    def audit_payload(self) -> dict[str, object]:
        return {
            "action": NO_WRITEBACK_WINDOW_ACTION,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "run_seed": self.run_seed,
            "checkpoint_fe": self.checkpoint_fe,
            "dispatch_checkpoint_hash": self.dispatch_checkpoint_hash,
            "topology_hash": self.topology_hash,
            "anchor_hash": self.anchor_hash,
            "relations": [relation.audit_payload() for relation in self.relations],
            "relation_order_hash": self.relation_order_hash,
            "seed_namespace": self.seed_namespace,
            "issued_sweep": self.issued_sweep,
            "target_sweep": self.target_sweep,
            "ttl_sweeps": self.ttl_sweeps,
            "expires_sweep": self.expires_sweep,
            "budget_fes": self.budget_fes,
        }

    @property
    def action_hash(self) -> str:
        return _canonical_sha256(self.audit_payload())


@dataclass(frozen=True)
class NoWritebackDecision:
    suppress_writeback: bool
    abstained: bool
    reason: str
    relation_position: int | None
    expected_action_hash: str
    observed_lifecycle_action_hash: str
    observed_state_hash: str


@dataclass
class NoWritebackWindowExecutionState:
    """Persist relation order and explicit invalidation for one target sweep."""

    action_hash: str
    status: str = "issued"
    consumed_relations: tuple[RelationBlockKey, ...] = ()
    started_fe: int | None = None
    last_consumed_fe: int | None = None
    completed_fe: int | None = None
    invalidation_reason: str = ""

    def __post_init__(self) -> None:
        self.consumed_relations = tuple(self.consumed_relations)
        self._validate_shape()

    @classmethod
    def for_action(
        cls,
        action: NoWritebackWindowAction,
    ) -> NoWritebackWindowExecutionState:
        return cls(action_hash=action.action_hash)

    def _validate_shape(self) -> None:
        _validate_hash(self.action_hash, "action_hash")
        if self.status not in _WINDOW_STATUSES:
            raise ValueError("unsupported no-writeback lifecycle status")
        if any(not isinstance(relation, RelationBlockKey) for relation in self.consumed_relations):
            raise TypeError("consumed_relations must contain RelationBlockKey values")
        if self.started_fe is not None:
            _integer(self.started_fe, "started_fe")
        if self.last_consumed_fe is not None:
            _integer(self.last_consumed_fe, "last_consumed_fe")
        if self.completed_fe is not None:
            _integer(self.completed_fe, "completed_fe")
        if not isinstance(self.invalidation_reason, str):
            raise ValueError("invalidation_reason must be a string")

    def validate_for(self, action: NoWritebackWindowAction) -> None:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            raise ValueError("no-writeback lifecycle does not match action_hash")
        count = len(self.consumed_relations)
        if tuple(action.relations[:count]) != self.consumed_relations:
            raise ValueError("no-writeback lifecycle relation prefix drifted")
        if self.status == "issued":
            if (
                count
                or self.started_fe is not None
                or self.last_consumed_fe is not None
                or self.completed_fe is not None
                or self.invalidation_reason
            ):
                raise ValueError("issued no-writeback lifecycle contains outcome data")
        elif self.status == "running":
            if (
                not (0 < count < RELATION_COUNT)
                or self.started_fe is None
                or self.last_consumed_fe is None
                or self.completed_fe is not None
            ):
                raise ValueError("running no-writeback lifecycle is inconsistent")
            if self.invalidation_reason:
                raise ValueError("running no-writeback lifecycle has invalidation data")
        elif self.status == "completed":
            if (
                count != RELATION_COUNT
                or self.started_fe is None
                or self.last_consumed_fe is None
                or self.completed_fe != self.last_consumed_fe
            ):
                raise ValueError("completed no-writeback lifecycle is inconsistent")
            if self.invalidation_reason:
                raise ValueError("completed no-writeback lifecycle has invalidation data")
        elif not self.invalidation_reason:
            raise ValueError("abstained no-writeback lifecycle requires a reason")

    def _decision(
        self,
        action: NoWritebackWindowAction,
        *,
        suppress_writeback: bool,
        abstained: bool,
        reason: str,
        relation_position: int | None,
    ) -> NoWritebackDecision:
        return NoWritebackDecision(
            suppress_writeback=suppress_writeback,
            abstained=abstained,
            reason=reason,
            relation_position=relation_position,
            expected_action_hash=action.action_hash,
            observed_lifecycle_action_hash=self.action_hash,
            observed_state_hash=self.observed_state_hash(),
        )

    def _abstain(
        self,
        action: NoWritebackWindowAction,
        reason: str,
    ) -> NoWritebackDecision:
        if not reason:
            raise ValueError("abstain reason must be non-empty")
        self.status = "abstained"
        self.invalidation_reason = reason
        return self._decision(
            action,
            suppress_writeback=False,
            abstained=True,
            reason=reason,
            relation_position=None,
        )

    def consume(
        self,
        action: NoWritebackWindowAction,
        *,
        relation: RelationBlockKey,
        current_sweep: int,
        current_fe: int,
        dispatch_checkpoint_hash: str,
        topology_hash: str,
        required_seed_namespace: str,
    ) -> NoWritebackDecision:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            return self._decision(
                action,
                suppress_writeback=False,
                abstained=True,
                reason="lifecycle_action_hash_mismatch",
                relation_position=None,
            )
        self.validate_for(action)
        if self.status == "completed":
            return self._decision(
                action,
                suppress_writeback=False,
                abstained=True,
                reason="action_already_consumed",
                relation_position=None,
            )
        if self.status == "abstained":
            return self._decision(
                action,
                suppress_writeback=False,
                abstained=True,
                reason=f"action_invalidated:{self.invalidation_reason}",
                relation_position=None,
            )
        if not isinstance(relation, RelationBlockKey):
            return self._abstain(action, "invalid_relation_type")
        sweep = _integer(current_sweep, "current_sweep")
        observed_fe = _integer(current_fe, "current_fe")
        if sweep > action.expires_sweep:
            return self._abstain(action, "window_expired")
        if sweep != action.target_sweep:
            return self._abstain(action, "target_sweep_mismatch")
        if dispatch_checkpoint_hash != action.dispatch_checkpoint_hash:
            return self._abstain(action, "dispatch_checkpoint_hash_mismatch")
        if topology_hash != action.topology_hash:
            return self._abstain(action, "topology_hash_mismatch")
        if required_seed_namespace != action.seed_namespace:
            return self._abstain(action, "seed_namespace_mismatch")
        if self.started_fe is None and observed_fe <= action.checkpoint_fe:
            return self._abstain(action, "first_relation_fe_not_after_checkpoint")
        if self.last_consumed_fe is not None and observed_fe <= self.last_consumed_fe:
            return self._abstain(action, "non_monotonic_relation_fe")

        position = len(self.consumed_relations)
        expected = action.relations[position]
        if relation != expected:
            if relation in self.consumed_relations:
                reason = "duplicate_relation"
            elif relation in action.relations[position + 1 :]:
                reason = "relation_order_mismatch"
            else:
                reason = "unexpected_relation"
            return self._abstain(action, reason)

        if self.started_fe is None:
            self.started_fe = observed_fe
        self.last_consumed_fe = observed_fe
        self.consumed_relations = (*self.consumed_relations, relation)
        self.status = "completed" if len(self.consumed_relations) == RELATION_COUNT else "running"
        if self.status == "completed":
            self.completed_fe = observed_fe
        return self._decision(
            action,
            suppress_writeback=True,
            abstained=False,
            reason="",
            relation_position=position,
        )

    def finish_sweep(
        self,
        action: NoWritebackWindowAction,
        *,
        current_sweep: int,
    ) -> NoWritebackDecision:
        self._validate_shape()
        if self.action_hash != action.action_hash:
            return self._decision(
                action,
                suppress_writeback=False,
                abstained=True,
                reason="lifecycle_action_hash_mismatch",
                relation_position=None,
            )
        self.validate_for(action)
        if self.status == "completed":
            return self._decision(
                action,
                suppress_writeback=False,
                abstained=False,
                reason="",
                relation_position=RELATION_COUNT - 1,
            )
        if self.status == "abstained":
            return self._decision(
                action,
                suppress_writeback=False,
                abstained=True,
                reason=f"action_invalidated:{self.invalidation_reason}",
                relation_position=None,
            )
        if _integer(current_sweep, "current_sweep") != action.target_sweep:
            return self._abstain(action, "target_sweep_mismatch")
        missing = action.relations[len(self.consumed_relations)]
        return self._abstain(
            action, "missing_relation:" + _canonical_sha256(missing.audit_payload())
        )

    def audit_payload(self, action: NoWritebackWindowAction) -> dict[str, object]:
        self.validate_for(action)
        return self.observed_audit_payload()

    def observed_audit_payload(self) -> dict[str, object]:
        self._validate_shape()
        return {
            "action": NO_WRITEBACK_WINDOW_ACTION,
            "schema": NO_WRITEBACK_WINDOW_SCHEMA,
            "schema_version": NO_WRITEBACK_WINDOW_SCHEMA_VERSION,
            "action_hash": self.action_hash,
            "status": self.status,
            "consumed_relations": [
                relation.audit_payload() for relation in self.consumed_relations
            ],
            "started_fe": self.started_fe,
            "last_consumed_fe": self.last_consumed_fe,
            "completed_fe": self.completed_fe,
            "invalidation_reason": self.invalidation_reason,
        }

    def state_hash(self, action: NoWritebackWindowAction) -> str:
        return _canonical_sha256(self.audit_payload(action))

    def observed_state_hash(self) -> str:
        return _canonical_sha256(self.observed_audit_payload())


def execute_no_writeback_window_action(
    action: NoWritebackWindowAction,
    state: NoWritebackWindowExecutionState,
    *,
    relation: RelationBlockKey,
    current_sweep: int,
    current_fe: int,
    dispatch_checkpoint_hash: str,
    topology_hash: str,
    required_seed_namespace: str,
) -> NoWritebackDecision:
    """Suppress one expected relation writeback or explicitly abstain."""

    if not isinstance(action, NoWritebackWindowAction):
        raise TypeError("action must be a NoWritebackWindowAction")
    if not isinstance(state, NoWritebackWindowExecutionState):
        raise TypeError("state must be a NoWritebackWindowExecutionState")
    return state.consume(
        action,
        relation=relation,
        current_sweep=current_sweep,
        current_fe=current_fe,
        dispatch_checkpoint_hash=dispatch_checkpoint_hash,
        topology_hash=topology_hash,
        required_seed_namespace=required_seed_namespace,
    )


__all__ = [
    "FULL_SPACE_DIMENSION",
    "NO_EARLY_STOPPING_POLICY",
    "NO_REPAIR_POLICY",
    "NO_RESTART_POLICY",
    "NO_WRITEBACK_WINDOW_ACTION",
    "NO_WRITEBACK_WINDOW_ACTION_SPEC",
    "NO_WRITEBACK_WINDOW_SCHEMA",
    "NO_WRITEBACK_WINDOW_SCHEMA_VERSION",
    "NoWritebackDecision",
    "NoWritebackWindowAction",
    "NoWritebackWindowExecutionState",
    "OwnerContextMemoryTransition",
    "OwnerContextMemorySnapshot",
    "OwnerMemorySyncReceipt",
    "OwnerMemorySyncRequest",
    "RELATION_CMA_BLOCK_BUDGET_FES",
    "RELATION_CMA_GENERATION_COUNT",
    "RELATION_CMA_INITIAL_SIGMA",
    "RELATION_CMA_IMPLEMENTATION_TYPE",
    "RELATION_CMA_PARAMETERS_HASH",
    "RELATION_CMA_POPULATION_SIZE",
    "RELATION_CMA_TOTAL_BUDGET_FES",
    "RELATION_COUNT",
    "RELATION_DIMENSION",
    "RELATION_SHARED_CMA_SWEEP_ACTION",
    "RELATION_SHARED_CMA_SWEEP_ACTION_SPEC",
    "RELATION_SHARED_CMA_SWEEP_SCHEMA",
    "RELATION_SHARED_CMA_SWEEP_SCHEMA_VERSION",
    "FrozenRelationCmaBlock",
    "RelationBlockKey",
    "RelationCmaBlockExecutionResult",
    "RelationCmaOptimizerContractReceipt",
    "RelationCmaTestOnlyImplementationReceipt",
    "RelationSharedCmaExecutionContext",
    "RelationSharedCmaSweepAction",
    "RelationSharedCmaSweepExecutionResult",
    "RelationSharedCmaSweepExecutionState",
    "PreparedRelationCmaOptimizer",
    "build_relation_cma_optimizer",
    "execute_no_writeback_window_action",
    "execute_relation_shared_cma_sweep_action",
    "full_space_vector_hash",
    "no_writeback_window_anchor_hash",
    "ordered_relations_hash",
    "owner_context_memory_hash",
    "relation_cma_anchor_hash",
    "shared_values_hash",
]
