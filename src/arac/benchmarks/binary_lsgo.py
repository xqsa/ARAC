"""Deterministic binary overlapping LSGO benchmark primitives.

The construction is ported from the inherited MATLAB generator described in
the ARAC migration design. It is intentionally independent of the continuous
HCC/AOB execution path.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class BinaryLsgoSpec:
    problem_id: str
    nominal_dimension: int
    overlap_count: int
    min_group_size: int
    max_group_size: int
    continuous_groups: bool
    alpha: float
    overlap_distribution_ratio: float
    related_group_ratio: float
    max_repeat_ratio: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.problem_id, str) or not self.problem_id.strip():
            raise ValueError("problem_id must be a non-empty string")
        for name, value in (
            ("nominal_dimension", self.nominal_dimension),
            ("overlap_count", self.overlap_count),
            ("min_group_size", self.min_group_size),
            ("max_group_size", self.max_group_size),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if type(self.continuous_groups) is not bool:
            raise ValueError("continuous_groups must be a boolean")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise ValueError("alpha must be a real number")
        if self.nominal_dimension <= 0:
            raise ValueError("nominal_dimension must be positive")
        if not 0 <= self.overlap_count < self.nominal_dimension:
            raise ValueError("overlap_count must be in [0, nominal_dimension)")
        if self.min_group_size <= 0 or self.max_group_size < self.min_group_size:
            raise ValueError("group size bounds are invalid")
        if self.max_group_size > self.nominal_dimension:
            raise ValueError("max_group_size cannot exceed nominal_dimension")
        if self.max_group_size > self.decision_dimension:
            raise ValueError("max_group_size cannot exceed decision_dimension")
        if not 0 <= self.alpha < 0.9:
            raise ValueError("alpha must be in [0, 0.9)")
        for name, value in (
            ("overlap_distribution_ratio", self.overlap_distribution_ratio),
            ("related_group_ratio", self.related_group_ratio),
            ("max_repeat_ratio", self.max_repeat_ratio),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} ratio must be a real number")
            if not 0 < value <= 1:
                raise ValueError(f"{name} ratio must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def decision_dimension(self) -> int:
        return self.nominal_dimension - self.overlap_count


@dataclass(frozen=True)
class BinaryLsgoTopology:
    groups: tuple[tuple[int, ...], ...]
    group_sizes: tuple[int, ...]
    nominal_dimension: int
    decision_dimension: int
    membership_count: int
    overlap_slot_count: int
    shared_variable_count: int
    variable_occurrence_counts: Mapping[int, int]
    shared_variable_groups: Mapping[int, tuple[int, ...]]
    adjacency_pairs: tuple[tuple[int, int], ...]
    realized_shared_group_count: int
    max_variable_occurrence_count: int
    source_semantics: str = "inherited_matlab_n_minus_overlap"


@dataclass(frozen=True)
class BinaryLsgoProblem:
    spec: BinaryLsgoSpec
    template: tuple[int, ...]
    topology: BinaryLsgoTopology

    @property
    def decision_dimension(self) -> int:
        return self.topology.decision_dimension

    def evaluate(self, vector: Sequence[int]) -> float:
        if len(vector) != self.decision_dimension:
            raise ValueError(
                f"vector length must be {self.decision_dimension}, received {len(vector)}"
            )
        if any(type(bit) not in {int, bool} or bit not in {0, 1} for bit in vector):
            raise ValueError("vector values must be binary integers or booleans")

        total = 0.0
        for group in self.topology.groups:
            matching = sum(vector[index] == self.template[index] for index in group)
            total += _deception_contribution(len(group), matching, self.spec.alpha)
        return -total

    def evaluate_batch(self, vectors: Iterable[Sequence[int]]) -> tuple[float, ...]:
        return tuple(self.evaluate(vector) for vector in vectors)


def generate_binary_lsgo(spec: BinaryLsgoSpec) -> BinaryLsgoProblem:
    rng = random.Random(spec.seed)
    group_sizes = _generate_group_sizes(spec, rng)
    reservations = _allocate_overlap_slots(spec, group_sizes, rng)
    groups = _assign_unique_variables(spec, group_sizes, reservations, rng)
    _assign_repeated_variables(spec, groups, reservations, rng)
    topology = _build_topology(spec, groups)
    template = tuple(rng.randrange(2) for _ in range(spec.decision_dimension))
    return BinaryLsgoProblem(spec=spec, template=template, topology=topology)


def standard_binary_lsgo_specs() -> tuple[BinaryLsgoSpec, ...]:
    """Return the inherited thesis's 18 standard parameter combinations."""

    specs: list[BinaryLsgoSpec] = []
    case_index = 1
    for alpha in (0.1, 0.5, 0.8):
        for min_group_size, max_group_size in ((5, 5), (2, 5)):
            for overlap_count in (100, 200, 300):
                specs.append(
                    BinaryLsgoSpec(
                        problem_id=f"BLSGO-F{case_index:02d}",
                        nominal_dimension=1000,
                        overlap_count=overlap_count,
                        min_group_size=min_group_size,
                        max_group_size=max_group_size,
                        continuous_groups=True,
                        alpha=alpha,
                        overlap_distribution_ratio=0.5,
                        related_group_ratio=0.5,
                        max_repeat_ratio=0.5,
                        seed=case_index,
                    )
                )
                case_index += 1
    return tuple(specs)


def _generate_group_sizes(spec: BinaryLsgoSpec, rng: random.Random) -> list[int]:
    if spec.min_group_size == spec.max_group_size:
        full_groups, residual = divmod(spec.nominal_dimension, spec.min_group_size)
        sizes = [spec.min_group_size] * full_groups
        if residual:
            sizes.append(residual)
        return sizes

    group_count = math.ceil(
        2 * spec.nominal_dimension / (spec.min_group_size + spec.max_group_size)
    )
    if not group_count * spec.min_group_size <= spec.nominal_dimension:
        raise ValueError("min_group_size makes the requested grouping impossible")
    if group_count * spec.max_group_size < spec.nominal_dimension:
        raise ValueError("max_group_size makes the requested grouping impossible")

    sizes = [rng.randint(spec.min_group_size, spec.max_group_size) for _ in range(group_count)]
    difference = spec.nominal_dimension - sum(sizes)
    while difference:
        if difference > 0:
            eligible = [index for index, size in enumerate(sizes) if size < spec.max_group_size]
            step = 1
        else:
            eligible = [index for index, size in enumerate(sizes) if size > spec.min_group_size]
            step = -1
        if not eligible:
            raise ValueError("group size bounds cannot sum to nominal_dimension")
        sizes[rng.choice(eligible)] += step
        difference -= step
    return sizes


def _allocate_overlap_slots(
    spec: BinaryLsgoSpec,
    group_sizes: list[int],
    rng: random.Random,
) -> list[int]:
    reservations = [0] * len(group_sizes)
    remaining = spec.overlap_count
    available = set(range(len(group_sizes)))

    while remaining:
        if not available:
            raise ValueError("overlap_count cannot be allocated across the generated groups")
        maximum_capacity = max(group_sizes[index] for index in available)
        requested_chunk = _round_positive(spec.overlap_count * spec.overlap_distribution_ratio)
        chunk = max(1, min(remaining, maximum_capacity, requested_chunk))
        candidates = [index for index in available if group_sizes[index] >= chunk]
        target = rng.choice(candidates)
        reservations[target] = chunk
        available.remove(target)
        remaining -= chunk
    return reservations


def _assign_unique_variables(
    spec: BinaryLsgoSpec,
    group_sizes: list[int],
    reservations: list[int],
    rng: random.Random,
) -> list[list[int]]:
    variable_order = list(range(spec.decision_dimension))
    if not spec.continuous_groups:
        rng.shuffle(variable_order)

    groups: list[list[int]] = []
    begin = 0
    for size, reserved in zip(group_sizes, reservations, strict=True):
        unique_count = size - reserved
        end = begin + unique_count
        groups.append(variable_order[begin:end])
        begin = end
    if begin != spec.decision_dimension:
        raise ValueError("group allocation did not consume every decision variable")
    return groups


def _assign_repeated_variables(
    spec: BinaryLsgoSpec,
    groups: list[list[int]],
    reservations: list[int],
    rng: random.Random,
) -> None:
    if spec.overlap_count == 0:
        return

    occurrence_counts = [1] * spec.decision_dimension
    duplicate_limit = max(
        1,
        min(
            _round_positive(spec.overlap_count * spec.max_repeat_ratio),
            _round_positive(len(groups) * spec.max_repeat_ratio),
        ),
    )
    maximum_occurrences = 1 + duplicate_limit
    original_groups = tuple(tuple(group) for group in groups)

    for target_index, slot_count in enumerate(reservations):
        if slot_count == 0:
            continue
        eligible_sources = [
            index
            for index, group in enumerate(original_groups)
            if index != target_index and group
        ]
        if not eligible_sources:
            raise ValueError("overlap assignment requires a non-empty source group")
        source_count = max(
            1,
            min(
                len(eligible_sources),
                slot_count,
                _round_positive(slot_count * spec.related_group_ratio),
            ),
        )
        selected_sources = rng.sample(eligible_sources, source_count)
        used = set(groups[target_index])

        for slot_index in range(slot_count):
            preferred = selected_sources[slot_index % len(selected_sources)]
            candidate = _choose_repeat_candidate(
                original_groups,
                preferred_sources=(preferred,),
                used=used,
                occurrence_counts=occurrence_counts,
                maximum_occurrences=maximum_occurrences,
                rng=rng,
            )
            if candidate is None:
                candidate = _choose_repeat_candidate(
                    original_groups,
                    preferred_sources=tuple(eligible_sources),
                    used=used,
                    occurrence_counts=occurrence_counts,
                    maximum_occurrences=maximum_occurrences,
                    rng=rng,
                )
            if candidate is None:
                raise ValueError("overlap constraints leave no valid repeated variable")
            groups[target_index].append(candidate)
            used.add(candidate)
            occurrence_counts[candidate] += 1


def _choose_repeat_candidate(
    original_groups: tuple[tuple[int, ...], ...],
    *,
    preferred_sources: tuple[int, ...],
    used: set[int],
    occurrence_counts: list[int],
    maximum_occurrences: int,
    rng: random.Random,
) -> int | None:
    candidates = [
        variable
        for source_index in preferred_sources
        for variable in original_groups[source_index]
        if variable not in used and occurrence_counts[variable] < maximum_occurrences
    ]
    return None if not candidates else rng.choice(candidates)


def _build_topology(spec: BinaryLsgoSpec, groups: list[list[int]]) -> BinaryLsgoTopology:
    frozen_groups = tuple(tuple(group) for group in groups)
    membership_count = sum(len(group) for group in frozen_groups)
    if membership_count != spec.nominal_dimension:
        raise ValueError("generated membership_count does not match nominal_dimension")
    if any(len(group) != len(set(group)) for group in frozen_groups):
        raise ValueError("a generated group contains a duplicate variable")

    variable_groups: dict[int, list[int]] = {
        variable: [] for variable in range(spec.decision_dimension)
    }
    for group_index, group in enumerate(frozen_groups):
        for variable in group:
            if variable not in variable_groups:
                raise ValueError("generated variable index exceeds decision_dimension")
            variable_groups[variable].append(group_index)
    if any(not containing_groups for containing_groups in variable_groups.values()):
        raise ValueError("a decision variable is absent from every group")

    occurrence_counts = {
        variable: len(containing_groups)
        for variable, containing_groups in variable_groups.items()
    }
    shared_variable_groups = {
        variable: tuple(containing_groups)
        for variable, containing_groups in variable_groups.items()
        if len(containing_groups) > 1
    }
    adjacency_pairs = {
        (left, right)
        for containing_groups in shared_variable_groups.values()
        for position, left in enumerate(containing_groups)
        for right in containing_groups[position + 1 :]
    }
    shared_groups = {
        group_index
        for containing_groups in shared_variable_groups.values()
        for group_index in containing_groups
    }
    overlap_slots = sum(occurrence_counts.values()) - spec.decision_dimension
    if overlap_slots != spec.overlap_count:
        raise ValueError("generated overlap slots do not match overlap_count")

    return BinaryLsgoTopology(
        groups=frozen_groups,
        group_sizes=tuple(len(group) for group in frozen_groups),
        nominal_dimension=spec.nominal_dimension,
        decision_dimension=spec.decision_dimension,
        membership_count=membership_count,
        overlap_slot_count=overlap_slots,
        shared_variable_count=len(shared_variable_groups),
        variable_occurrence_counts=MappingProxyType(occurrence_counts),
        shared_variable_groups=MappingProxyType(shared_variable_groups),
        adjacency_pairs=tuple(sorted(adjacency_pairs)),
        realized_shared_group_count=len(shared_groups),
        max_variable_occurrence_count=max(occurrence_counts.values()),
    )


def _round_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def _deception_contribution(group_size: int, matching: int, alpha: float) -> float:
    if alpha == 0:
        return float(matching)
    local_optimum = 0.9 * group_size
    deception_point = 10 * alpha * group_size / 9
    if matching < deception_point:
        return -(local_optimum / deception_point) * matching + local_optimum
    return (
        (group_size / (group_size - deception_point)) * matching
        - (group_size * deception_point / (group_size - deception_point))
    )


__all__ = [
    "BinaryLsgoProblem",
    "BinaryLsgoSpec",
    "BinaryLsgoTopology",
    "generate_binary_lsgo",
    "standard_binary_lsgo_specs",
]
