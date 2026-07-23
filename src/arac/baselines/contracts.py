"""Immutable result contracts shared by the WLOC baseline implementations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


GROUPING_SCHEMA_VERSION = "wloc-grouping-v1"
BASELINE_RESULT_SCHEMA_VERSION = "wloc-baseline-result-v1"


@dataclass(frozen=True, eq=False)
class GroupingResult:
    """Validated decomposition output with explicit FE and provenance metadata."""

    method: str
    dimension: int
    groups: tuple[tuple[int, ...], ...]
    decomposition_fes: int
    allows_overlap: bool
    origin: str
    matrix: np.ndarray | None = field(default=None, repr=False)
    matrix_kind: str | None = None
    grouping_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, Integral):
            raise TypeError("dimension must be an integer")
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive")
        if isinstance(self.decomposition_fes, bool) or not isinstance(
            self.decomposition_fes, Integral
        ):
            raise TypeError("decomposition_fes must be an integer")
        if int(self.decomposition_fes) < 0:
            raise ValueError("decomposition_fes must be non-negative")
        if not isinstance(self.allows_overlap, bool):
            raise TypeError("allows_overlap must be a bool")
        if not isinstance(self.origin, str) or not self.origin.strip():
            raise ValueError("origin must be a non-empty string")

        normalized_groups: list[tuple[int, ...]] = []
        memberships: list[int] = []
        for raw_group in self.groups:
            group: list[int] = []
            for raw_index in raw_group:
                if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
                    raise TypeError("group indices must be integers")
                index = int(raw_index)
                if index < 0 or index >= int(self.dimension):
                    raise ValueError("group index is outside the decision space")
                group.append(index)
            if not group:
                raise ValueError("groups must be non-empty")
            if len(set(group)) != len(group):
                raise ValueError("a variable may appear at most once within a group")
            normalized_groups.append(tuple(group))
            memberships.extend(group)
        if not normalized_groups:
            raise ValueError("at least one group is required")
        if set(memberships) != set(range(int(self.dimension))):
            raise ValueError("groups must cover every decision variable exactly or by overlap")
        if not self.allows_overlap and len(memberships) != int(self.dimension):
            raise ValueError("overlap is not allowed for this grouping")

        matrix = self._normalize_matrix(self.matrix, self.matrix_kind)
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "decomposition_fes", int(self.decomposition_fes))
        object.__setattr__(self, "groups", tuple(normalized_groups))
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "grouping_hash", self._calculate_hash(matrix))

    def _normalize_matrix(
        self,
        matrix: np.ndarray | None,
        matrix_kind: str | None,
    ) -> np.ndarray | None:
        if matrix is None:
            if matrix_kind is not None:
                raise ValueError("matrix_kind requires a matrix")
            return None
        if matrix_kind not in {"interaction", "design"}:
            raise ValueError("matrix_kind must be 'interaction' or 'design'")
        values = np.asarray(matrix)
        if values.ndim != 2 or values.shape[0] != int(self.dimension):
            raise ValueError("matrix must be two-dimensional with one row per variable")
        if matrix_kind == "interaction":
            if values.shape[1] != int(self.dimension):
                raise ValueError("interaction matrix must be square")
            normalized = np.asarray(values, dtype=bool)
            if not np.array_equal(normalized, normalized.T):
                raise ValueError("interaction matrix must be symmetric")
        else:
            if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
                raise ValueError("design matrix must be numeric")
            normalized = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(normalized)):
                raise ValueError("design matrix must contain only finite values")
        frozen = np.array(normalized, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    def _calculate_hash(self, matrix: np.ndarray | None) -> str:
        matrix_payload = None
        if matrix is not None:
            matrix_payload = {
                "dtype": matrix.dtype.str,
                "hash": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
                "shape": list(matrix.shape),
            }
        payload = {
            "schema_version": GROUPING_SCHEMA_VERSION,
            "method": self.method,
            "dimension": int(self.dimension),
            "groups": [list(group) for group in self.groups],
            "decomposition_fes": int(self.decomposition_fes),
            "allows_overlap": self.allows_overlap,
            "origin": self.origin,
            "matrix_kind": self.matrix_kind,
            "matrix": matrix_payload,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BaselineResult:
    """Auditable result shared by every WLOC optimization method."""

    method: str
    backend: str
    dimension: int
    optimizer_seed: int
    optimization_fes: int
    decomposition_fes: int
    best_x: tuple[float, ...]
    best_y: float
    best_so_far_trace: tuple[float, ...]
    initial_mean: tuple[float, ...]
    sigma: float
    repair_policy: str
    repaired_candidate_count: int
    phase_fes: tuple[tuple[str, int], ...]
    grouping_hash: str | None = None
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.method or not self.backend:
            raise ValueError("method and backend must be non-empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.optimization_fes <= 0 or self.decomposition_fes < 0:
            raise ValueError("FE counts are invalid")
        if len(self.best_x) != self.dimension or len(self.initial_mean) != self.dimension:
            raise ValueError("solution and initial mean must match dimension")
        if len(self.best_so_far_trace) != self.optimization_fes:
            raise ValueError("best-so-far trace must contain one value per optimization FE")
        numeric_values = (
            *self.best_x,
            *self.initial_mean,
            *self.best_so_far_trace,
            self.best_y,
            self.sigma,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("baseline result must contain only finite numeric values")
        if any(
            later > earlier
            for earlier, later in zip(
                self.best_so_far_trace,
                self.best_so_far_trace[1:],
                strict=False,
            )
        ):
            raise ValueError("best-so-far trace must be monotonically non-increasing")
        if self.best_y != self.best_so_far_trace[-1]:
            raise ValueError("best_y must equal the final best-so-far value")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.repair_policy != "clip_to_bounds":
            raise ValueError("unsupported repair policy")
        if self.repaired_candidate_count < 0:
            raise ValueError("repaired_candidate_count must be non-negative")
        if sum(count for _, count in self.phase_fes) != self.optimization_fes:
            raise ValueError("phase FE counts must sum to optimization_fes")
        if any(not name or count < 0 for name, count in self.phase_fes):
            raise ValueError("phase FE entries are invalid")
        object.__setattr__(self, "result_hash", self._calculate_hash())

    def _calculate_hash(self) -> str:
        payload = {
            "schema_version": BASELINE_RESULT_SCHEMA_VERSION,
            "method": self.method,
            "backend": self.backend,
            "dimension": self.dimension,
            "optimizer_seed": self.optimizer_seed,
            "optimization_fes": self.optimization_fes,
            "decomposition_fes": self.decomposition_fes,
            "best_x": self.best_x,
            "best_y": self.best_y,
            "best_so_far_trace": self.best_so_far_trace,
            "initial_mean": self.initial_mean,
            "sigma": self.sigma,
            "repair_policy": self.repair_policy,
            "repaired_candidate_count": self.repaired_candidate_count,
            "phase_fes": self.phase_fes,
            "grouping_hash": self.grouping_hash,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
