"""Deterministic Python port of Chen et al.'s 2018 binary LSGO generator."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


CHEN2018_SCHEMA_VERSION = "chen2018-binary-instance-v1"
_SPEC_FIELDS = {
    "dimension",
    "min_group_size",
    "max_group_size",
    "alpha",
    "permuted",
    "seed",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "spec",
    "template",
    "groups",
    "instance_hash",
}


def _normalize_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Chen2018Spec:
    """Parameters that define one reproducible Chen 2018 problem instance."""

    dimension: int
    min_group_size: int
    max_group_size: int
    alpha: float
    permuted: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        dimension = _normalize_integer(self.dimension, "dimension")
        min_group_size = _normalize_integer(self.min_group_size, "min_group_size")
        max_group_size = _normalize_integer(self.max_group_size, "max_group_size")
        seed = _normalize_integer(self.seed, "seed")
        if not isinstance(self.alpha, Real) or isinstance(self.alpha, bool):
            raise TypeError("alpha must be a real number")
        alpha = float(self.alpha)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if min_group_size <= 0 or max_group_size < min_group_size:
            raise ValueError("group sizes must satisfy 0 < min_group_size <= max_group_size")
        if not math.isfinite(alpha) or not 0.0 <= alpha < 0.9:
            raise ValueError("alpha must satisfy 0 <= alpha < 0.9")
        if not isinstance(self.permuted, bool):
            raise TypeError("permuted must be a bool")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        if min_group_size == max_group_size:
            if dimension % min_group_size:
                raise ValueError("equal grouping requires dimension to be divisible by group size")
        else:
            group_count = math.ceil(2 * dimension / (min_group_size + max_group_size))
            if group_count * min_group_size > dimension:
                raise ValueError("min_group_size is infeasible for the derived group count")
            if group_count * max_group_size < dimension:
                raise ValueError("max_group_size is infeasible for the derived group count")

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "min_group_size", min_group_size)
        object.__setattr__(self, "max_group_size", max_group_size)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "seed", seed)

    @property
    def group_count(self) -> int:
        if self.min_group_size == self.max_group_size:
            return self.dimension // self.min_group_size
        return math.ceil(2 * self.dimension / (self.min_group_size + self.max_group_size))


def _group_sizes(spec: Chen2018Spec, rng: np.random.Generator) -> tuple[int, ...]:
    if spec.min_group_size == spec.max_group_size:
        return (spec.min_group_size,) * spec.group_count

    span = spec.max_group_size - spec.min_group_size
    # MATLAB round is half-away-from-zero; every value here is positive.
    sizes = np.floor(spec.min_group_size + rng.random(spec.group_count) * span + 0.5)
    sizes = sizes.astype(int)
    while int(sizes.sum()) != spec.dimension:
        difference = spec.dimension - int(sizes.sum())
        if difference > 0:
            eligible = np.flatnonzero(sizes < spec.max_group_size)
            selected = eligible[:difference]
            sizes[selected] += 1
        else:
            eligible = np.flatnonzero(sizes > spec.min_group_size)
            selected = eligible[:-difference]
            sizes[selected] -= 1
    return tuple(int(value) for value in sizes)


def _normalize_template(template: Sequence[int] | np.ndarray, dimension: int) -> tuple[int, ...]:
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
class Chen2018BinaryProblem:
    """One frozen binary problem with AOB-style zero-optimum error values."""

    spec: Chen2018Spec
    template: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    instance_hash: str = field(init=False)

    def __post_init__(self) -> None:
        template = _normalize_template(self.template, self.spec.dimension)
        groups = tuple(
            tuple(_normalize_integer(index, "group index") for index in group)
            for group in self.groups
        )
        sizes = tuple(len(group) for group in groups)
        if len(groups) != self.spec.group_count:
            raise ValueError("groups do not match the derived group count")
        if any(
            size < self.spec.min_group_size or size > self.spec.max_group_size for size in sizes
        ):
            raise ValueError("group size is outside the configured bounds")
        indices = tuple(index for group in groups for index in group)
        if sorted(indices) != list(range(self.spec.dimension)):
            raise ValueError("groups must partition every variable exactly once")

        object.__setattr__(self, "template", template)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "instance_hash", self._calculate_hash())

    @classmethod
    def generate(
        cls,
        spec: Chen2018Spec,
        *,
        template: Sequence[int] | np.ndarray | None = None,
    ) -> Chen2018BinaryProblem:
        """Generate and freeze the template, grouping, and optional permutation."""

        rng = np.random.default_rng(spec.seed)
        generated_template = rng.integers(0, 2, size=spec.dimension, dtype=np.int8)
        frozen_template = _normalize_template(
            generated_template if template is None else template,
            spec.dimension,
        )
        order = rng.permutation(spec.dimension) if spec.permuted else np.arange(spec.dimension)
        sizes = _group_sizes(spec, rng)
        groups: list[tuple[int, ...]] = []
        start = 0
        for size in sizes:
            stop = start + size
            groups.append(tuple(int(index) for index in order[start:stop]))
            start = stop
        return cls(spec=spec, template=frozen_template, groups=tuple(groups))

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> Chen2018BinaryProblem:
        """Restore an instance and reject missing, extra, stale, or modified fields."""

        if set(manifest) != _MANIFEST_FIELDS:
            raise ValueError("manifest fields do not match the Chen 2018 v1 schema")
        if manifest["schema_version"] != CHEN2018_SCHEMA_VERSION:
            raise ValueError("unsupported Chen 2018 manifest schema")
        spec_payload = manifest["spec"]
        if not isinstance(spec_payload, Mapping) or set(spec_payload) != _SPEC_FIELDS:
            raise ValueError("manifest spec fields do not match the Chen 2018 v1 schema")
        problem = cls(
            spec=Chen2018Spec(**spec_payload),
            template=tuple(manifest["template"]),
            groups=tuple(tuple(group) for group in manifest["groups"]),
        )
        if manifest["instance_hash"] != problem.instance_hash:
            raise ValueError("Chen 2018 manifest hash mismatch")
        return problem

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.groups)

    def info(self) -> dict[str, object]:
        """Return the common metadata needed by benchmark adapters."""

        return {
            "best": 0.0,
            "dimension": self.dimension,
            "encoding": "binary",
            "instance_hash": self.instance_hash,
            "lower": 0,
            "threshold": 0.0,
            "upper": 1,
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": CHEN2018_SCHEMA_VERSION,
            "spec": asdict(self.spec),
            "template": list(self.template),
            "groups": [list(group) for group in self.groups],
            "instance_hash": self.instance_hash,
        }

    def group_errors(self, candidates: object) -> np.ndarray:
        """Return each Trap subgroup's non-negative error contribution."""

        values = self._prepare_candidates(candidates)
        contributions = np.empty((values.shape[0], len(self.groups)), dtype=float)
        template = np.asarray(self.template, dtype=np.int8)
        for column, group in enumerate(self.groups):
            indices = np.asarray(group, dtype=int)
            matches = np.sum(values[:, indices] == template[indices], axis=1)
            reward = self._trap_reward(matches, len(group))
            contributions[:, column] = len(group) - reward
        return contributions

    def evaluate(self, candidates: object) -> np.ndarray:
        """Evaluate one candidate or a batch; the known global optimum is zero."""

        return self.group_errors(candidates).sum(axis=1)

    def legacy_objective(self, candidates: object) -> np.ndarray:
        """Return the original MATLAB minimization value with optimum ``-dimension``."""

        return self.evaluate(candidates) - self.dimension

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

    def _trap_reward(self, matches: np.ndarray, group_size: int) -> np.ndarray:
        if self.spec.alpha == 0.0:
            return matches.astype(float)
        local_optimum = 0.9 * group_size
        deception_point = 10.0 * self.spec.alpha * group_size / 9.0
        reward = np.where(
            matches < deception_point,
            local_optimum * (deception_point - matches) / deception_point,
            group_size * (matches - deception_point) / (group_size - deception_point),
        )
        return np.where(matches == group_size, float(group_size), reward)

    def _calculate_hash(self) -> str:
        payload = {
            "schema_version": CHEN2018_SCHEMA_VERSION,
            "spec": asdict(self.spec),
            "template": list(self.template),
            "groups": [list(group) for group in self.groups],
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
