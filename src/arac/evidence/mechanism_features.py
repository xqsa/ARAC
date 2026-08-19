"""Pure summaries for candidate mechanism-level evidence features.

The helpers in this module do not evaluate an objective and do not mutate a
checkpoint or an evidence object.  They are intentionally kept separate from
the Phase-I protocol so candidate summaries can be evaluated before being
promoted to the frozen feature schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import operator

from arac.runtime.contracts import RelationEvidence


DISAGREEMENT_FEATURE_NAMES = (
    "relation_disagreement_median",
    "relation_disagreement_std",
    "relation_disagreement_q90",
    "relation_disagreement_max",
    "relation_disagreement_nonzero_fraction",
)

PROGRESS_FEATURE_NAMES = (
    "warmup_log10_gain",
    "structure_log10_gain",
    "tail_log10_gain",
    "late_gain_fraction",
)

# These names are deliberately not included in CANDIDATE_FEATURE_NAMES yet.
# The topology and cover summaries are available for ablations without changing
# the frozen Phase-I checkpoint schema.
TOPOLOGY_FEATURE_NAMES = (
    "weighted_degree_concentration",
    "weighted_degree_entropy",
    "largest_component_fraction",
)

CTP_COVER_FEATURE_NAMES = (
    "ctp_cover_block_inflation",
    "ctp_cover_variable_inflation",
)

CANDIDATE_FEATURE_NAMES = DISAGREEMENT_FEATURE_NAMES + PROGRESS_FEATURE_NAMES
MECHANISM_FEATURE_NAMES = CANDIDATE_FEATURE_NAMES

_EPSILON = 1e-300


@dataclass(frozen=True)
class PhaseProgressErrors:
    """Best-error endpoints for the probe, warmup, structure, and tail phases."""

    probe: float
    warmup: float
    structure: float
    tail: float


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def _relation_tuple(relations: Iterable[RelationEvidence]) -> tuple[RelationEvidence, ...]:
    try:
        values = tuple(relations)
    except TypeError as exc:
        raise ValueError("relations must be an iterable of RelationEvidence") from exc
    for relation in values:
        if not isinstance(relation, RelationEvidence):
            raise TypeError("relations must contain RelationEvidence values")
    return values


def _validate_blocks(blocks: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    try:
        values = tuple(
            tuple(operator.index(index) for index in block)
            for block in blocks
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("blocks must be a sequence of non-empty integer sequences") from exc
    if not values or any(not block for block in values):
        raise ValueError("blocks must be a sequence of non-empty integer sequences")
    if any(index < 0 for block in values for index in block):
        raise ValueError("block indices must be non-negative")
    flattened = tuple(index for block in values for index in block)
    if len(set(flattened)) != len(flattened):
        raise ValueError("blocks must not contain duplicate variable indices")
    return values


def _validate_relation_blocks(
    blocks: tuple[tuple[int, ...], ...],
    relations: tuple[RelationEvidence, ...],
) -> None:
    block_count = len(blocks)
    for relation in relations:
        if relation.right_block >= block_count:
            raise ValueError("relation references an unknown block")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def summarize_relation_disagreement(
    relations: Iterable[RelationEvidence],
) -> tuple[float, ...]:
    """Return the five fixed-order summaries of relation disagreement."""

    values = tuple(
        _finite_nonnegative(relation.disagreement, "relation disagreement")
        for relation in _relation_tuple(relations)
    )
    if not values:
        return (0.0,) * len(DISAGREEMENT_FEATURE_NAMES)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (
        _quantile(values, 0.5),
        math.sqrt(variance),
        _quantile(values, 0.9),
        max(values),
        sum(value > 0.0 for value in values) / len(values),
    )


def _validated_graph_inputs(
    blocks: Sequence[Sequence[int]],
    relations: Iterable[RelationEvidence],
) -> tuple[tuple[tuple[int, ...], ...], tuple[RelationEvidence, ...]]:
    block_values = _validate_blocks(blocks)
    relation_values = _relation_tuple(relations)
    _validate_relation_blocks(block_values, relation_values)
    return block_values, relation_values


def summarize_relation_topology(
    blocks: Sequence[Sequence[int]],
    relations: Iterable[RelationEvidence],
) -> tuple[float, ...]:
    """Return optional weighted-degree and connected-component summaries.

    Degree concentration and entropy use the normalized weighted degree over
    all blocks.  An empty relation graph returns zeros, rather than treating
    isolated blocks as a connected component signal.
    """

    block_values, relation_values = _validated_graph_inputs(blocks, relations)
    block_count = len(block_values)
    degrees = [0.0] * block_count
    parent = list(range(block_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for relation in relation_values:
        weight = _finite_nonnegative(relation.strength, "relation strength") * (
            1.0 + _finite_nonnegative(relation.disagreement, "relation disagreement")
        )
        degrees[relation.left_block] += weight
        degrees[relation.right_block] += weight
        union(relation.left_block, relation.right_block)

    total_degree = sum(degrees)
    if total_degree <= 0.0:
        concentration = 0.0
        entropy = 0.0
    else:
        shares = tuple(degree / total_degree for degree in degrees if degree > 0.0)
        concentration = max(shares)
        raw_entropy = -sum(share * math.log(share) for share in shares)
        entropy = raw_entropy / math.log(block_count) if block_count > 1 else 0.0

    if not relation_values:
        largest_component_fraction = 0.0
    else:
        component_sizes: dict[int, int] = {}
        for index in range(block_count):
            root = find(index)
            component_sizes[root] = component_sizes.get(root, 0) + 1
        largest_component_fraction = max(component_sizes.values()) / block_count

    return (concentration, entropy, largest_component_fraction)


def summarize_ctp_cover(
    blocks: Sequence[Sequence[int]],
    relations: Iterable[RelationEvidence],
) -> tuple[float, ...]:
    """Return excess block and variable inflation of the relation-aware CTP cover."""

    block_values, relation_values = _validated_graph_inputs(blocks, relations)
    base_block_count = len(block_values)
    base_variable_count = sum(len(block) for block in block_values)
    merged_pairs = {
        (relation.left_block, relation.right_block) for relation in relation_values
    }
    expanded_block_count = base_block_count + len(merged_pairs)
    expanded_variable_count = base_variable_count + sum(
        len(block_values[left]) + len(block_values[right])
        for left, right in sorted(merged_pairs)
    )
    return (
        (expanded_block_count - base_block_count) / base_block_count,
        (expanded_variable_count - base_variable_count) / base_variable_count,
    )


def _coerce_progress(
    progress_errors: PhaseProgressErrors | Mapping[str, object] | Sequence[object],
) -> PhaseProgressErrors:
    if isinstance(progress_errors, PhaseProgressErrors):
        values = progress_errors
    elif isinstance(progress_errors, Mapping):
        required = ("probe", "warmup", "structure", "tail")
        missing = [name for name in required if name not in progress_errors]
        if missing:
            raise ValueError("progress_errors is missing: " + ", ".join(missing))
        values = PhaseProgressErrors(*(progress_errors[name] for name in required))
    else:
        try:
            sequence = tuple(progress_errors)
        except TypeError as exc:
            raise ValueError(
                "progress_errors must be four endpoints: probe, warmup, structure, tail"
            ) from exc
        if len(sequence) != 4:
            raise ValueError(
                "progress_errors must be four endpoints: probe, warmup, structure, tail"
            )
        values = PhaseProgressErrors(*sequence)
    return PhaseProgressErrors(
        probe=_finite_nonnegative(values.probe, "probe error"),
        warmup=_finite_nonnegative(values.warmup, "warmup error"),
        structure=_finite_nonnegative(values.structure, "structure error"),
        tail=_finite_nonnegative(values.tail, "tail error"),
    )


def summarize_phase_progress(
    progress_errors: PhaseProgressErrors
    | Mapping[str, object]
    | Sequence[object]
    | None,
) -> tuple[float, ...]:
    """Return phase gains and the fraction earned after warmup.

    Gains are ``log10((before + 1) / (after + 1))`` and are clipped at zero,
    matching the strict-best progress convention used by the runtime.  The
    late fraction is ``(structure_gain + tail_gain) / total_gain``; it is zero
    when no gain is observed.
    """

    if progress_errors is None:
        return (0.0,) * len(PROGRESS_FEATURE_NAMES)
    endpoints = _coerce_progress(progress_errors)

    def gain(before: float, after: float) -> float:
        return max(0.0, math.log10((before + 1.0) / (after + 1.0)))

    warmup_gain = gain(endpoints.probe, endpoints.warmup)
    structure_gain = gain(endpoints.warmup, endpoints.structure)
    tail_gain = gain(endpoints.structure, endpoints.tail)
    total_gain = warmup_gain + structure_gain + tail_gain
    late_fraction = (
        (structure_gain + tail_gain) / total_gain if total_gain > _EPSILON else 0.0
    )
    return (warmup_gain, structure_gain, tail_gain, late_fraction)


def candidate_feature_values(
    relations: Iterable[RelationEvidence],
    progress_errors: PhaseProgressErrors
    | Mapping[str, object]
    | Sequence[object]
    | None = None,
) -> tuple[float, ...]:
    """Return the fixed-order disagreement and progress candidate values."""

    return summarize_relation_disagreement(relations) + summarize_phase_progress(
        progress_errors
    )


def candidate_feature_items(
    relations: Iterable[RelationEvidence],
    progress_errors: PhaseProgressErrors
    | Mapping[str, object]
    | Sequence[object]
    | None = None,
) -> tuple[tuple[str, float], ...]:
    """Return candidate names and values in deterministic schema order."""

    return tuple(zip(CANDIDATE_FEATURE_NAMES, candidate_feature_values(relations, progress_errors)))


def candidate_feature_map(
    relations: Iterable[RelationEvidence],
    progress_errors: PhaseProgressErrors
    | Mapping[str, object]
    | Sequence[object]
    | None = None,
) -> dict[str, float]:
    """Return an insertion-ordered mapping in candidate schema order."""

    return dict(candidate_feature_items(relations, progress_errors))


def summarize_mechanism_features(
    blocks: Sequence[Sequence[int]],
    relations: Iterable[RelationEvidence],
    progress_errors: PhaseProgressErrors
    | Mapping[str, object]
    | Sequence[object]
    | None = None,
) -> tuple[tuple[str, float], ...]:
    """Return the active candidate schema after validating block topology.

    ``blocks`` is validated here because the eventual topology and CTP-cover
    candidates depend on the same partition, even though those candidates are
    intentionally not part of the active aggregate yet.
    """

    _, relation_values = _validated_graph_inputs(blocks, relations)
    return candidate_feature_items(relation_values, progress_errors)


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "CTP_COVER_FEATURE_NAMES",
    "DISAGREEMENT_FEATURE_NAMES",
    "MECHANISM_FEATURE_NAMES",
    "PROGRESS_FEATURE_NAMES",
    "PhaseProgressErrors",
    "TOPOLOGY_FEATURE_NAMES",
    "candidate_feature_items",
    "candidate_feature_map",
    "candidate_feature_values",
    "summarize_ctp_cover",
    "summarize_mechanism_features",
    "summarize_phase_progress",
    "summarize_relation_disagreement",
    "summarize_relation_topology",
]
