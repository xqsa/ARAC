"""Shared HCC context identity for explicit Phase2 actions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence


PHASE2_ACTION_SELECTION_RULE = "max_voi_then_structural_key"
PHASE2_SELECTION_CHECKPOINT_SCHEMA = "phase2-action-selection-checkpoint-v1"
AOB_DECISION_DIMENSION = 1000

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


def _vector_hash(values: Sequence[float]) -> str:
    return _canonical_sha256(
        {"dimension": len(values), "values": tuple(float(value) for value in values)}
    )


def phase2_relation_hash(
    owner_group_indices: Sequence[int],
    shared_variable_indices: Sequence[int],
) -> str:
    owners = tuple(int(value) for value in owner_group_indices)
    shared = tuple(int(value) for value in shared_variable_indices)
    if not owners or not shared or any(value < 0 for value in (*owners, *shared)):
        raise ValueError("Phase2 relation identity must be non-empty and non-negative")
    return _canonical_sha256({"owners": owners, "shared": shared})


def phase2_selection_checkpoint_hash(
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
    """Bind the exact Phase1 endpoint shared by explicit Phase2 actions."""

    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("problem_id must be non-empty")
    checkpoint = _integer(checkpoint_fe, "checkpoint_fe")
    prefix = _finite_vector(fitness_prefix, "fitness_prefix")
    mean = _finite_vector(incumbent, "incumbent")
    if len(prefix) != checkpoint:
        raise ValueError("fitness_prefix length must equal checkpoint_fe")
    if len(mean) != AOB_DECISION_DIMENSION:
        raise ValueError("Phase2 incumbent must be 1000-dimensional")
    return _canonical_sha256(
        {
            "protocol": PHASE2_SELECTION_CHECKPOINT_SCHEMA,
            "problem_id": problem_id,
            "run_seed": _integer(run_seed, "run_seed"),
            "checkpoint_fe": checkpoint,
            "fitness_prefix_hash": _canonical_sha256(prefix),
            "incumbent_hash": _vector_hash(mean),
            "topology_hash": _validate_hash(topology_hash, "topology_hash"),
            "order_hash": _validate_hash(order_hash, "order_hash"),
            "action_set_hash": _validate_hash(action_set_hash, "action_set_hash"),
            "start_sweep": _integer(start_sweep, "start_sweep"),
        }
    )


__all__ = [
    "AOB_DECISION_DIMENSION",
    "PHASE2_ACTION_SELECTION_RULE",
    "PHASE2_SELECTION_CHECKPOINT_SCHEMA",
    "phase2_relation_hash",
    "phase2_selection_checkpoint_hash",
]
