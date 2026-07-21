"""Deterministic Wang 2025 extension for binary overlapping LSGO problems."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


WANG2025_SCHEMA_VERSION = "wang2025-overlapping-instance-v1"
WANG2025_MAX_SHARED_MEMBERSHIPS = 2
_SPEC_FIELDS = {
    "dimension",
    "min_group_size",
    "max_group_size",
    "alpha",
    "overlap_count",
    "beta",
    "gamma",
    "permuted",
    "seed",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "max_shared_memberships",
    "spec",
    "template",
    "groups",
    "base_owner_by_variable",
    "instance_hash",
}


def _normalize_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _normalize_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1")
    return normalized


def _matlab_round_positive(value: float) -> int:
    return math.floor(value + 0.5)


@dataclass(frozen=True)
class Wang2025OverlappingSpec:
    """Parameters for one paper-corrected Wang 2025 overlap instance."""

    dimension: int
    min_group_size: int
    max_group_size: int
    alpha: float
    overlap_count: int
    beta: float = 0.5
    gamma: float = 0.5
    permuted: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        dimension = _normalize_integer(self.dimension, "dimension")
        min_group_size = _normalize_integer(self.min_group_size, "min_group_size")
        max_group_size = _normalize_integer(self.max_group_size, "max_group_size")
        overlap_count = _normalize_integer(self.overlap_count, "overlap_count")
        seed = _normalize_integer(self.seed, "seed")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a real number")
        alpha = float(self.alpha)
        beta = _normalize_unit_interval(self.beta, "beta")
        gamma = _normalize_unit_interval(self.gamma, "gamma")

        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if min_group_size <= 0 or max_group_size < min_group_size:
            raise ValueError("group sizes must satisfy 0 < min_group_size <= max_group_size")
        if not math.isfinite(alpha) or not 0.0 <= alpha < 0.9:
            raise ValueError("alpha must satisfy 0 <= alpha < 0.9")
        if overlap_count < 0 or overlap_count > dimension:
            raise ValueError("overlap_count must satisfy 0 <= overlap_count <= dimension")
        if not isinstance(self.permuted, bool):
            raise TypeError("permuted must be a bool")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        membership_count = dimension + overlap_count
        if min_group_size == max_group_size:
            if membership_count % min_group_size:
                raise ValueError(
                    "equal grouping requires dimension + overlap_count to be divisible "
                    "by group size"
                )
            group_count = membership_count // min_group_size
        else:
            group_count = math.ceil(2 * membership_count / (min_group_size + max_group_size))
            if group_count * min_group_size > membership_count:
                raise ValueError("min_group_size is infeasible for the derived group count")
            if group_count * max_group_size < membership_count:
                raise ValueError("max_group_size is infeasible for the derived group count")
        if group_count > dimension:
            raise ValueError("every group requires at least one uniquely owned variable")
        if overlap_count > 0 and group_count < 2:
            raise ValueError("overlap requires at least two groups")

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "min_group_size", min_group_size)
        object.__setattr__(self, "max_group_size", max_group_size)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "overlap_count", overlap_count)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "seed", seed)

    @property
    def membership_count(self) -> int:
        return self.dimension + self.overlap_count

    @property
    def group_count(self) -> int:
        if self.min_group_size == self.max_group_size:
            return self.membership_count // self.min_group_size
        return math.ceil(2 * self.membership_count / (self.min_group_size + self.max_group_size))

    @property
    def overlap_ratio(self) -> float:
        return self.overlap_count / self.dimension


def _group_sizes(
    spec: Wang2025OverlappingSpec,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if spec.min_group_size == spec.max_group_size:
        return (spec.min_group_size,) * spec.group_count

    span = spec.max_group_size - spec.min_group_size
    sizes = np.floor(spec.min_group_size + rng.random(spec.group_count) * span + 0.5)
    sizes = sizes.astype(int)
    while int(sizes.sum()) != spec.membership_count:
        difference = spec.membership_count - int(sizes.sum())
        eligible = np.flatnonzero(
            sizes < spec.max_group_size if difference > 0 else sizes > spec.min_group_size
        )
        selected = int(rng.choice(eligible))
        sizes[selected] += 1 if difference > 0 else -1
    return tuple(int(value) for value in sizes)


def _desired_target_count(spec: Wang2025OverlappingSpec) -> int:
    if spec.overlap_count == 0:
        return 0
    concentrated_load = max(
        1,
        _matlab_round_positive(spec.overlap_count * spec.beta),
    )
    return min(
        spec.overlap_count,
        spec.group_count,
        math.ceil(spec.overlap_count / concentrated_load),
    )


def _minimum_target_count(capacities: Sequence[int], overlap_count: int) -> int:
    total = 0
    for count, capacity in enumerate(sorted(capacities, reverse=True), 1):
        total += capacity
        if total >= overlap_count:
            return count
    raise ValueError("group sizes do not have enough capacity for the requested overlap")


def _duplicate_loads(
    spec: Wang2025OverlappingSpec,
    sizes: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if spec.overlap_count == 0:
        return (0,) * spec.group_count

    capacities = tuple(size - 1 for size in sizes)
    target_count = max(
        _desired_target_count(spec),
        _minimum_target_count(capacities, spec.overlap_count),
    )
    randomized = [int(value) for value in rng.permutation(spec.group_count)]
    ranked = sorted(randomized, key=lambda index: capacities[index], reverse=True)
    targets = ranked[:target_count]
    if sum(capacities[index] for index in targets) < spec.overlap_count:
        raise ValueError("selected groups cannot hold the requested overlap")

    loads = [0] * spec.group_count
    for target in targets:
        loads[target] = 1
    remaining = spec.overlap_count - target_count
    for target in targets:
        added = min(remaining, capacities[target] - 1)
        loads[target] += added
        remaining -= added
    if remaining:
        raise ValueError("failed to allocate all overlap memberships")
    return tuple(loads)


def _desired_source_count(load: int, gamma: float, group_count: int) -> int:
    return max(
        1,
        min(load, group_count - 1, _matlab_round_positive(load * gamma)),
    )


def _normalize_template(
    template: Sequence[int] | np.ndarray,
    dimension: int,
) -> tuple[int, ...]:
    values = np.asarray(template)
    if values.ndim != 1 or values.shape[0] != dimension:
        raise ValueError(f"template must have shape ({dimension},)")
    if not (
        np.issubdtype(values.dtype, np.bool_)
        or np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError("template must contain only binary numeric values")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("template must contain only 0 and 1")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class Wang2025OverlappingProblem:
    """One frozen overlap topology using Wang's Hamming-distance Trap objective."""

    spec: Wang2025OverlappingSpec
    template: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    base_owner_by_variable: tuple[int, ...]
    instance_hash: str = field(init=False)

    def __post_init__(self) -> None:
        template = _normalize_template(self.template, self.spec.dimension)
        groups = tuple(
            tuple(_normalize_integer(index, "group index") for index in group)
            for group in self.groups
        )
        base_owners = tuple(
            _normalize_integer(owner, "base owner") for owner in self.base_owner_by_variable
        )
        if len(groups) != self.spec.group_count:
            raise ValueError("groups do not match the derived group count")
        if any(
            len(group) < self.spec.min_group_size or len(group) > self.spec.max_group_size
            for group in groups
        ):
            raise ValueError("group size is outside the configured bounds")
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("a variable may appear at most once within a group")
        if len(base_owners) != self.spec.dimension:
            raise ValueError("base_owner_by_variable must bind every variable")
        if any(owner < 0 or owner >= len(groups) for owner in base_owners):
            raise ValueError("base owner is outside the group topology")

        memberships: list[list[int]] = [[] for _ in range(self.spec.dimension)]
        for group_index, group in enumerate(groups):
            for variable in group:
                if variable < 0 or variable >= self.spec.dimension:
                    raise ValueError("group variable is outside the decision space")
                memberships[variable].append(group_index)
        if any(not owners for owners in memberships):
            raise ValueError("groups must cover every decision variable")
        if any(len(owners) > WANG2025_MAX_SHARED_MEMBERSHIPS for owners in memberships):
            raise ValueError("Wang 2025 v1 supports at most two owners per variable")
        shared_count = sum(len(owners) == 2 for owners in memberships)
        if shared_count != self.spec.overlap_count:
            raise ValueError("groups do not contain the configured number of shared variables")
        if sum(len(group) for group in groups) != self.spec.membership_count:
            raise ValueError("group memberships do not match dimension + overlap_count")
        if any(base_owners[variable] not in owners for variable, owners in enumerate(memberships)):
            raise ValueError("base owner must contain its variable")

        duplicate_sources: dict[int, set[int]] = {}
        for variable, owners in enumerate(memberships):
            if len(owners) != 2:
                continue
            target = next(owner for owner in owners if owner != base_owners[variable])
            duplicate_sources.setdefault(target, set()).add(base_owners[variable])
        expected_target_count = max(
            _desired_target_count(self.spec),
            _minimum_target_count(
                [len(group) - 1 for group in groups],
                self.spec.overlap_count,
            )
            if self.spec.overlap_count
            else 0,
        )
        if len(duplicate_sources) != expected_target_count:
            raise ValueError("groups do not match the beta-controlled target count")
        for target, sources in duplicate_sources.items():
            duplicate_load = sum(
                1
                for variable, owners in enumerate(memberships)
                if len(owners) == 2 and target in owners and base_owners[variable] != target
            )
            expected_sources = _desired_source_count(
                duplicate_load,
                self.spec.gamma,
                self.spec.group_count,
            )
            if len(sources) != expected_sources:
                raise ValueError("groups do not match the gamma-controlled source count")

        object.__setattr__(self, "template", template)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "base_owner_by_variable", base_owners)
        object.__setattr__(self, "instance_hash", self._calculate_hash())

    @classmethod
    def generate(
        cls,
        spec: Wang2025OverlappingSpec,
        *,
        template: Sequence[int] | np.ndarray | None = None,
    ) -> Wang2025OverlappingProblem:
        """Generate a fixed topology with exactly ``overlap_count`` shared variables."""

        rng = np.random.default_rng(spec.seed)
        generated_template = rng.integers(0, 2, size=spec.dimension, dtype=np.int8)
        frozen_template = _normalize_template(
            generated_template if template is None else template,
            spec.dimension,
        )
        sizes = _group_sizes(spec, rng)
        duplicate_loads = _duplicate_loads(spec, sizes, rng)
        unique_counts = tuple(
            size - duplicate_load
            for size, duplicate_load in zip(sizes, duplicate_loads, strict=True)
        )
        order = rng.permutation(spec.dimension) if spec.permuted else np.arange(spec.dimension)
        groups: list[list[int]] = []
        base_owners = [-1] * spec.dimension
        start = 0
        for group_index, count in enumerate(unique_counts):
            stop = start + count
            group = [int(variable) for variable in order[start:stop]]
            groups.append(group)
            for variable in group:
                base_owners[variable] = group_index
            start = stop
        if start != spec.dimension:
            raise ValueError("unique group allocation did not consume the decision space")

        available = [list(group) for group in groups]
        for pool in available:
            rng.shuffle(pool)
        targets = sorted(
            (index for index, load in enumerate(duplicate_loads) if load),
            key=lambda index: duplicate_loads[index],
            reverse=True,
        )
        for target in targets:
            load = duplicate_loads[target]
            source_count = _desired_source_count(load, spec.gamma, spec.group_count)
            randomized = [int(value) for value in rng.permutation(spec.group_count)]
            eligible = [source for source in randomized if source != target and available[source]]
            ranked = sorted(
                eligible,
                key=lambda source: len(available[source]),
                reverse=True,
            )
            sources = ranked[:source_count]
            if (
                len(sources) != source_count
                or sum(len(available[source]) for source in sources) < load
            ):
                raise ValueError(
                    "gamma-controlled source groups cannot supply the requested overlap"
                )
            chosen: list[int] = []
            for source in sources:
                chosen.append(available[source].pop())
            while len(chosen) < load:
                remaining_sources = [source for source in sources if available[source]]
                if not remaining_sources:
                    raise ValueError("failed to allocate distinct shared variables")
                source = max(remaining_sources, key=lambda value: len(available[value]))
                chosen.append(available[source].pop())
            groups[target].extend(chosen)

        return cls(
            spec=spec,
            template=frozen_template,
            groups=tuple(tuple(group) for group in groups),
            base_owner_by_variable=tuple(base_owners),
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> Wang2025OverlappingProblem:
        if set(manifest) != _MANIFEST_FIELDS:
            raise ValueError("manifest fields do not match the Wang 2025 v1 schema")
        if manifest["schema_version"] != WANG2025_SCHEMA_VERSION:
            raise ValueError("unsupported Wang 2025 manifest schema")
        if manifest["max_shared_memberships"] != WANG2025_MAX_SHARED_MEMBERSHIPS:
            raise ValueError("unsupported Wang 2025 owner multiplicity")
        spec_payload = manifest["spec"]
        if not isinstance(spec_payload, Mapping) or set(spec_payload) != _SPEC_FIELDS:
            raise ValueError("manifest spec fields do not match the Wang 2025 v1 schema")
        problem = cls(
            spec=Wang2025OverlappingSpec(**spec_payload),
            template=tuple(manifest["template"]),
            groups=tuple(tuple(group) for group in manifest["groups"]),
            base_owner_by_variable=tuple(manifest["base_owner_by_variable"]),
        )
        if manifest["instance_hash"] != problem.instance_hash:
            raise ValueError("Wang 2025 manifest hash mismatch")
        return problem

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.groups)

    @property
    def shared_variables(self) -> tuple[int, ...]:
        counts = np.zeros(self.dimension, dtype=int)
        for group in self.groups:
            counts[np.asarray(group, dtype=int)] += 1
        return tuple(int(index) for index in np.flatnonzero(counts == 2))

    @property
    def overlap_relations(self) -> tuple[tuple[int, int, int], ...]:
        memberships: list[list[int]] = [[] for _ in range(self.dimension)]
        for group_index, group in enumerate(self.groups):
            for variable in group:
                memberships[variable].append(group_index)
        return tuple(
            (
                variable,
                self.base_owner_by_variable[variable],
                next(
                    owner
                    for owner in memberships[variable]
                    if owner != self.base_owner_by_variable[variable]
                ),
            )
            for variable in self.shared_variables
        )

    @property
    def global_optimum(self) -> tuple[int, ...]:
        return tuple(1 - value for value in self.template)

    def info(self) -> dict[str, object]:
        return {
            "best": 0.0,
            "dimension": self.dimension,
            "encoding": "binary",
            "group_count": len(self.groups),
            "instance_hash": self.instance_hash,
            "lower": 0,
            "max_shared_memberships": WANG2025_MAX_SHARED_MEMBERSHIPS,
            "overlap_count": self.spec.overlap_count,
            "overlap_ratio": self.spec.overlap_ratio,
            "threshold": 0.0,
            "upper": 1,
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": WANG2025_SCHEMA_VERSION,
            "max_shared_memberships": WANG2025_MAX_SHARED_MEMBERSHIPS,
            "spec": asdict(self.spec),
            "template": list(self.template),
            "groups": [list(group) for group in self.groups],
            "base_owner_by_variable": list(self.base_owner_by_variable),
            "instance_hash": self.instance_hash,
        }

    def group_errors(self, candidates: object) -> np.ndarray:
        values = self._prepare_candidates(candidates)
        contributions = np.empty((values.shape[0], len(self.groups)), dtype=float)
        template = np.asarray(self.template, dtype=np.int8)
        for column, group in enumerate(self.groups):
            indices = np.asarray(group, dtype=int)
            hamming_distance = np.sum(values[:, indices] != template[indices], axis=1)
            reward = self._trap_reward(hamming_distance, len(group))
            contributions[:, column] = len(group) - reward
        return contributions

    def evaluate(self, candidates: object) -> np.ndarray:
        return self.group_errors(candidates).sum(axis=1)

    def legacy_objective(self, candidates: object) -> np.ndarray:
        return self.evaluate(candidates) - self.spec.membership_count

    def __call__(self, candidates: object) -> np.ndarray:
        return self.evaluate(candidates)

    def _prepare_candidates(self, candidates: object) -> np.ndarray:
        try:
            values = np.asarray(candidates)
        except ValueError as error:
            raise ValueError("candidate batch must be rectangular") from error
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"candidate must have shape (n, {self.dimension})")
        if not (
            np.issubdtype(values.dtype, np.bool_)
            or np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.floating)
        ):
            raise ValueError("candidate must contain only binary numeric values")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("candidate must contain only 0 and 1")
        return values.astype(np.int8, copy=False)

    def _trap_reward(
        self,
        hamming_distance: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        if self.spec.alpha == 0.0:
            return hamming_distance.astype(float)
        local_optimum = 0.8 * group_size
        deception_point = 10.0 * self.spec.alpha * group_size / 9.0
        reward = np.where(
            hamming_distance < deception_point,
            local_optimum * (deception_point - hamming_distance) / deception_point,
            group_size * (hamming_distance - deception_point) / (group_size - deception_point),
        )
        return np.where(hamming_distance == group_size, float(group_size), reward)

    def _calculate_hash(self) -> str:
        payload = {
            "schema_version": WANG2025_SCHEMA_VERSION,
            "max_shared_memberships": WANG2025_MAX_SHARED_MEMBERSHIPS,
            "spec": asdict(self.spec),
            "template": list(self.template),
            "groups": [list(group) for group in self.groups],
            "base_owner_by_variable": list(self.base_owner_by_variable),
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
