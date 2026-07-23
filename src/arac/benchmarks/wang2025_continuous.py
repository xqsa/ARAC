"""Continuous, interaction-identifiable extension of Wang 2025 problems."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .wang2025_overlapping import Wang2025OverlappingProblem, Wang2025OverlappingSpec


WANG2025_CONTINUOUS_SCHEMA_VERSION = "wang2025-continuous-interaction-v2"
WANG2025_CONTINUOUS_EXTENSION = (
    "l1-trap-with-vertex-vanishing-endpoint-sensitive-pair-coupling"
)
WANG2025_CONTINUOUS_INTERACTION_STRENGTH = 1.0


@dataclass(frozen=True)
class Wang2025ContinuousProblem:
    """Wrap one frozen binary instance with a continuous objective on [0, 1]^D.

    The L1 distance agrees with Hamming distance at every binary vertex. The
    nonnegative pair term is zero at those vertices, but exposes each intended
    within-group interaction to differential grouping probes in the interior.
    """

    source_problem: Wang2025OverlappingProblem
    objective_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_problem, Wang2025OverlappingProblem):
            raise TypeError("source_problem must be a Wang2025OverlappingProblem")
        object.__setattr__(self, "objective_hash", self._calculate_hash())

    @property
    def spec(self) -> Wang2025OverlappingSpec:
        return self.source_problem.spec

    @property
    def dimension(self) -> int:
        return self.source_problem.dimension

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        return self.source_problem.groups

    @property
    def group_templates(self) -> tuple[tuple[int, ...], ...]:
        return self.source_problem.group_templates

    @property
    def template(self) -> tuple[int, ...]:
        return self.source_problem.template

    @property
    def source_instance_hash(self) -> str:
        return self.source_problem.instance_hash

    @property
    def reference_solution(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.source_problem.reference_solution)

    @property
    def reference_value(self) -> float:
        return float(self.evaluate(self.reference_solution)[0])

    @property
    def global_optimum(self) -> tuple[float, ...] | None:
        source_optimum = self.source_problem.global_optimum
        if source_optimum is None:
            return None
        return tuple(float(value) for value in source_optimum)

    def info(self) -> dict[str, object]:
        source_info = self.source_problem.info()
        source_info.update(
            {
                "schema_version": WANG2025_CONTINUOUS_SCHEMA_VERSION,
                "continuous_extension": WANG2025_CONTINUOUS_EXTENSION,
                "encoding": "continuous",
                "interaction_strength": WANG2025_CONTINUOUS_INTERACTION_STRENGTH,
                "objective_hash": self.objective_hash,
                "source_instance_hash": self.source_instance_hash,
                "lower": 0.0,
                "upper": 1.0,
            }
        )
        return source_info

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": WANG2025_CONTINUOUS_SCHEMA_VERSION,
            "continuous_extension": WANG2025_CONTINUOUS_EXTENSION,
            "interaction_strength": WANG2025_CONTINUOUS_INTERACTION_STRENGTH,
            "source_instance_hash": self.source_instance_hash,
            "objective_hash": self.objective_hash,
        }

    def expected_interaction_matrix(self) -> np.ndarray:
        matrix = np.eye(self.dimension, dtype=bool)
        for group in self.groups:
            indices = np.asarray(group, dtype=int)
            matrix[np.ix_(indices, indices)] = True
        return matrix

    def group_errors(self, candidates: object) -> np.ndarray:
        values = self._prepare_candidates(candidates)
        contributions = np.empty((values.shape[0], len(self.groups)), dtype=float)
        for column, (group, group_template) in enumerate(
            zip(self.groups, self.group_templates, strict=True)
        ):
            indices = np.asarray(group, dtype=int)
            local_template = np.asarray(group_template, dtype=float)
            distances = np.abs(values[:, indices] - local_template)
            l1_distance = np.sum(distances, axis=1)
            reward = self.source_problem._trap_reward(l1_distance, len(group))
            contributions[:, column] = (
                len(group) - reward + self._pair_coupling(distances)
            )
        return contributions

    def evaluate(self, candidates: object) -> np.ndarray:
        return self.group_errors(candidates).sum(axis=1)

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
            raise ValueError("candidate must contain only real numeric values")
        normalized = values.astype(float, copy=False)
        if not np.all(np.isfinite(normalized)) or not np.all(
            (normalized >= 0.0) & (normalized <= 1.0)
        ):
            raise ValueError("candidate must contain finite values in the closed interval [0, 1]")
        return normalized

    @staticmethod
    def _pair_coupling(distances: np.ndarray) -> np.ndarray:
        group_size = distances.shape[1]
        if group_size < 2:
            return np.zeros(distances.shape[0], dtype=float)
        interior = distances * (1.0 - distances)
        pair_sum = np.zeros(distances.shape[0], dtype=float)
        for first in range(group_size - 1):
            remaining_distances = distances[:, first + 1 :]
            remaining_interior = interior[:, first + 1 :]
            pair_sum += distances[:, first] * np.sum(remaining_interior, axis=1)
            pair_sum += interior[:, first] * np.sum(remaining_distances, axis=1)
            pair_sum += interior[:, first] * np.sum(remaining_interior, axis=1)
        pair_mean = 2.0 * pair_sum / (group_size * (group_size - 1))
        return WANG2025_CONTINUOUS_INTERACTION_STRENGTH * pair_mean

    def _calculate_hash(self) -> str:
        payload = {
            "schema_version": WANG2025_CONTINUOUS_SCHEMA_VERSION,
            "continuous_extension": WANG2025_CONTINUOUS_EXTENSION,
            "interaction_strength": WANG2025_CONTINUOUS_INTERACTION_STRENGTH,
            "source_instance_hash": self.source_instance_hash,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
