"""Continuous overlapping-variable benchmark objectives.

Builds a continuous, differentiable objective on top of an
:class:`~arac.benchmarks.overlap_groups.OverlapStructure`.  The global
function is a weighted sum of per-subgroup base functions, exactly as in the
AOB / CEC'2013 lineage::

    f(x) = Σ_i  w_i · base_i( shift_i( rotate_i( x[group_i] ) ) )

Two regimes are supported and are the scientific point of the suite:

``conforming``
    Every subgroup shares one global optimum vector ``Ovector``, so a shared
    variable has the same optimal value in every group that owns it.  A simple
    weighted average (HCC, CBCCO) suffices here.

``conflicting``
    Each subgroup ``i`` has its own optimum vector ``OvectorVec[i]``, so a
    shared variable is pulled toward different values by the groups that own
    it.  No single assignment can satisfy all groups; this is the regime that
    exposes whether a coordination mechanism is principled or merely averages.

The transforms (``transform_osz`` / ``transform_asy``) and base-function forms
match the AOB vendored code so that a conforming instance reproduces the AOB
numeric surface when the grouping degenerates to a sliding window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap_groups import OverlapStructure


__all__ = [
    "BASE_FUNCTIONS",
    "ConflictMode",
    "OverlapObjective",
    "OverlapObjectiveConfig",
    "build_overlap_problem",
]


BASE_FUNCTIONS: tuple[str, ...] = ("sphere", "ackley", "elliptic", "rastrigin", "schwefel")
ConflictMode = str  # "conforming" | "conflicting"


def _transform_osz(values: np.ndarray) -> np.ndarray:
    """Smooth oscillation transform, matching AOB ``transform_osz`` exactly."""

    z = np.asarray(values, dtype=float)
    sign_z = np.sign(z)
    with np.errstate(divide="ignore"):
        hat_z = np.where(z == 0.0, 0.0, np.log(np.abs(z)))
    c1 = np.where(z > 0.0, 10.0, 5.5)
    c2 = np.where(z > 0.0, 7.9, 3.1)
    sin_term = np.sin(c1 * hat_z) + np.sin(c2 * hat_z)
    return sign_z * np.exp(hat_z + 0.049 * sin_term)


def _transform_asy(values: np.ndarray, beta: float) -> np.ndarray:
    """Asymmetry transform, matching AOB ``transform_asy`` exactly."""

    z = np.asarray(values, dtype=float).copy()
    dimension = z.shape[-1]
    if dimension <= 1:
        return z
    positive = z > 0.0
    indices = np.arange(dimension, dtype=float)
    index_shape = (1, dimension) if z.ndim == 2 else (dimension,)
    exponent = 1.0 + beta * indices.reshape(index_shape) / (dimension - 1) * np.sqrt(
        np.maximum(z, 0.0)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(positive, z**exponent, z)


def _rotate(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    converted = np.asarray(values, dtype=float)
    matrix = np.asarray(rotation, dtype=float)
    return matrix @ converted if converted.ndim == 1 else converted @ matrix.T


def _random_rotation(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly distributed rotation matrix via QR of a Gaussian."""

    matrix = rng.standard_normal((size, size))
    orthogonal, upper = np.linalg.qr(matrix)
    diagonal = np.sign(np.diag(upper))
    diagonal[diagonal == 0] = 1.0
    rotation = orthogonal * diagonal
    if abs(np.linalg.det(rotation) - 1.0) > 1.0e-6:
        rotation[:, 0] *= -1.0
    return rotation


def _base_sphere(values: np.ndarray) -> float | np.ndarray:
    converted = np.asarray(values, dtype=float)
    result = np.sum(converted**2, axis=-1)
    return float(result) if converted.ndim == 1 else result


def _base_ackley(values: np.ndarray) -> float | np.ndarray:
    converted = np.asarray(values, dtype=float)
    mean_square = np.mean(converted**2, axis=-1)
    cosine = np.mean(np.cos(2.0 * math.pi * converted), axis=-1)
    result = (
        -20.0 * np.exp(-0.2 * np.sqrt(mean_square))
        - np.exp(cosine)
        + 20.0
        + math.e
    )
    return float(result) if converted.ndim == 1 else result


def _base_elliptic(values: np.ndarray) -> float | np.ndarray:
    converted = np.asarray(values, dtype=float)
    condition = 1.0e6 ** (np.arange(converted.shape[-1]) / max(1, converted.shape[-1] - 1))
    result = np.sum(condition * converted**2, axis=-1)
    return float(result) if converted.ndim == 1 else result


def _base_rastrigin(values: np.ndarray) -> float | np.ndarray:
    converted = np.asarray(values, dtype=float)
    dimension = converted.shape[-1]
    result = 10.0 * dimension + np.sum(
        converted**2 - 10.0 * np.cos(2.0 * math.pi * converted), axis=-1
    )
    return float(result) if converted.ndim == 1 else result


def _base_schwefel(values: np.ndarray) -> float | np.ndarray:
    converted = np.asarray(values, dtype=float)
    result = np.sum(np.cumsum(converted, axis=-1) ** 2, axis=-1)
    return float(result) if converted.ndim == 1 else result


_BASE_TABLE: dict[str, Callable[[np.ndarray], float | np.ndarray]] = {
    "sphere": _base_sphere,
    "ackley": _base_ackley,
    "elliptic": _base_elliptic,
    "rastrigin": _base_rastrigin,
    "schwefel": _base_schwefel,
}


@dataclass(frozen=True)
class OverlapObjectiveConfig:
    """Knobs for building one continuous overlapping objective."""

    base_function: str
    conflict_mode: ConflictMode
    bounds: float
    rotation: bool
    transforms: bool
    seed: int
    interaction_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.base_function not in BASE_FUNCTIONS:
            raise ValueError("unknown base function")
        if self.conflict_mode not in ("conforming", "conflicting"):
            raise ValueError("conflict_mode must be 'conforming' or 'conflicting'")
        if not math.isfinite(self.bounds) or self.bounds <= 0.0:
            raise ValueError("bounds must be finite and positive")
        if not math.isfinite(self.interaction_strength) or self.interaction_strength < 0.0:
            raise ValueError("interaction_strength must be finite and non-negative")


class OverlapObjective:
    """Stateful evaluator: holds the structure and all per-group parameters."""

    def __init__(
        self,
        structure: OverlapStructure,
        config: OverlapObjectiveConfig,
    ) -> None:
        if structure.grouping.dimension <= 0:
            raise ValueError("overlap structure must be non-empty")
        self._structure = structure
        self._config = config
        dimension = structure.grouping.dimension
        optimum_seed, weight_seed, rotation_seed = np.random.SeedSequence(config.seed).spawn(3)
        self._optima = self._sample_optima(
            dimension, structure, config, np.random.default_rng(optimum_seed)
        )
        self._weights = self._sample_weights(structure, np.random.default_rng(weight_seed))
        self._rotations = self._sample_rotations(
            structure, config, np.random.default_rng(rotation_seed)
        )
        self._interaction_pairs = tuple(
            tuple(
                (left, right)
                for left, variable in enumerate(group)
                for right in range(left + 1, len(group))
                if variable in structure.shared_variables
                or group[right] in structure.shared_variables
            )
            for group in structure.groups
        )
        lower = np.full(dimension, -float(config.bounds))
        upper = np.full(dimension, float(config.bounds))
        self._lower = lower
        self._upper = upper

    @staticmethod
    def _sample_optima(
        dimension: int,
        structure: OverlapStructure,
        config: OverlapObjectiveConfig,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """One global optimum (conforming) or one per group (conflicting).

        Returns one per-group shift vector whose length matches the group, so
        that ``converted[group] - optima[i]`` broadcasts cleanly.  For
        ``conforming`` every group sees the same global optimum restricted to
        its own variables; for ``conflicting`` each group draws its own
        independent shift.
        """

        span = 0.8 * float(config.bounds)
        if config.conflict_mode == "conforming":
            global_optimum = rng.uniform(-span, span, size=dimension)
            return [global_optimum[np.asarray(group, dtype=int)] for group in structure.groups]
        optima: list[np.ndarray] = []
        for group in structure.groups:
            local = rng.uniform(-span, span, size=len(group))
            optima.append(local)
        return optima

    @staticmethod
    def _sample_weights(structure: OverlapStructure, rng: np.random.Generator) -> np.ndarray:
        return 1.0 + 9.0 * rng.random(size=len(structure.groups))

    @staticmethod
    def _sample_rotations(
        structure: OverlapStructure,
        config: OverlapObjectiveConfig,
        rng: np.random.Generator,
    ) -> list[np.ndarray | None]:
        if not config.rotation:
            return [None for _ in structure.groups]
        rotations: list[np.ndarray | None] = []
        unique: dict[int, np.ndarray] = {}
        for group in structure.groups:
            size = len(group)
            cache_key = size
            if cache_key not in unique:
                unique[cache_key] = _random_rotation(size, rng)
            rotations.append(unique[cache_key])
        return rotations

    @property
    def structure(self) -> OverlapStructure:
        return self._structure

    @property
    def config(self) -> OverlapObjectiveConfig:
        return self._config

    @property
    def dimension(self) -> int:
        return self._structure.grouping.dimension

    @property
    def interaction_strength(self) -> float:
        """Return the shared-variable quartic coupling strength."""

        return float(self._config.interaction_strength)

    @property
    def interaction_pairs(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return local-coordinate pairs used by the optional interaction term."""

        return self._interaction_pairs

    @property
    def optimum(self) -> float:
        """Return the non-negative objective lower bound.

        The bound is attained by conforming instances and is generally
        unattainable by conflicting instances.
        """

        return 0.0

    @property
    def optimum_is_attainable(self) -> bool:
        """Whether the declared zero lower bound has a compatible global point."""

        return self._config.conflict_mode == "conforming"

    def evaluate(self, x: np.ndarray) -> float | np.ndarray:
        """Evaluate one point or a batch of points.

        Transform chain matches the AOB / CEC'2013 lineage exactly:
        ``shift → rotate → osz → asy → base``.
        """

        converted = np.asarray(x, dtype=float)
        single = converted.ndim == 1
        batch = converted[np.newaxis, :] if single else converted
        if batch.ndim != 2 or batch.shape[1] != self.dimension:
            raise ValueError("decision vector dimension drifted")
        base = _BASE_TABLE[self._config.base_function]
        total = np.zeros(batch.shape[0], dtype=float)
        for index, group in enumerate(self._structure.groups):
            slice_values = self._transform_group(batch, index)
            with np.errstate(over="ignore", invalid="ignore"):
                contribution = float(self._weights[index]) * np.asarray(base(slice_values), dtype=float)
                if self.interaction_strength > 0.0:
                    interaction = np.zeros(batch.shape[0], dtype=float)
                    for left, right in self._interaction_pairs[index]:
                        interaction += slice_values[:, left] ** 2 * slice_values[:, right] ** 2
                    contribution += float(self._weights[index]) * self.interaction_strength * interaction
            contribution = np.where(np.isfinite(contribution), contribution, np.inf)
            total += contribution
        return float(total[0]) if single else total

    def _transform_group(self, converted: np.ndarray, index: int) -> np.ndarray:
        """Apply the canonical ``shift → rotate → osz → asy`` chain to one group."""

        group = self._structure.groups[index]
        indices = np.asarray(group, dtype=int)
        values = converted[indices] if converted.ndim == 1 else converted[:, indices]
        values = values - self._optima[index]
        rotation = self._rotations[index]
        if rotation is not None:
            values = _rotate(values, rotation)
        if self._config.transforms:
            values = _transform_osz(values)
            values = _transform_asy(values, 0.2)
        return values

    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        return self.evaluate(x)

    def optimum_point(self) -> np.ndarray:
        """Return the conforming optimum or a conflicting diagnostic point.

        For ``conforming`` instances this is simply the shared ``Ovector``
        scattered back into full-dimension form.  For ``conflicting`` instances
        there is no such point in general — the best the caller can do is
        minimise the weighted sum of disagreements — so we return the per-group
        optimum of the *heaviest* group, which drives its dominant term to
        zero.  This is intentionally not a guarantee; it exists only for
        diagnostics.
        """

        point = np.zeros(self.dimension)
        if self._config.conflict_mode == "conforming":
            for group, shift in zip(self._structure.groups, self._optima, strict=True):
                for variable, value in zip(group, shift, strict=True):
                    point[variable] = value
            return point
        heaviest = int(np.argmax(self._weights))
        for variable, value in zip(
            self._structure.groups[heaviest], self._optima[heaviest], strict=True
        ):
            point[variable] = value
        return point

    def per_group_contribution(self, x: np.ndarray) -> np.ndarray:
        """Return each group's weighted base value, useful for conflict diagnosis."""

        converted = np.asarray(x, dtype=float)
        if converted.ndim != 1 or converted.shape != (self.dimension,):
            raise ValueError("decision vector dimension drifted")
        base = _BASE_TABLE[self._config.base_function]
        contributions = np.zeros(len(self._structure.groups))
        for index, _group in enumerate(self._structure.groups):
            slice_values = self._transform_group(converted, index)
            contribution = float(base(slice_values))
            if self.interaction_strength > 0.0:
                interaction = sum(
                    float(slice_values[left] ** 2 * slice_values[right] ** 2)
                    for left, right in self._interaction_pairs[index]
                )
                contribution += self.interaction_strength * interaction
            contributions[index] = float(self._weights[index]) * contribution
        return contributions


def build_overlap_problem(
    dimension: int,
    *,
    overlap_budget: int,
    min_group_size: int,
    max_group_size: int,
    base_function: str,
    conflict_mode: ConflictMode,
    bounds: float = 100.0,
    contiguous: bool = True,
    rotation: bool = True,
    transforms: bool = True,
    seed: int = 0,
    num_groups: int | None = None,
    topology: str = "random",
    interaction_strength: float = 0.0,
) -> tuple[OptimizationProblem, OverlapObjective]:
    """Generate one overlapping benchmark instance and its ARAC-facing problem.

    The returned :class:`OptimizationProblem` is the identity-free numeric
    surface consumed by the ARAC runtime; the :class:`OverlapObjective` carries
    the ground-truth structure (groups, optima, weights) needed offline for
    evaluation audits and conflict diagnostics.

    ``num_groups`` fixes the subgroup count K (e.g. 20 for AOB / CEC'2013
    parity); when ``None`` it is inferred from the size range.
    """

    from arac.benchmarks.overlap_groups import generate_overlap_groups

    structure = generate_overlap_groups(
        dimension,
        overlap_budget=overlap_budget,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        contiguous=contiguous,
        seed=seed,
        num_groups=num_groups,
        topology=topology,
    )
    config = OverlapObjectiveConfig(
        base_function=base_function,
        conflict_mode=conflict_mode,
        bounds=bounds,
        rotation=rotation,
        transforms=transforms,
        seed=seed,
        interaction_strength=interaction_strength,
    )
    objective = OverlapObjective(structure, config)
    problem = OptimizationProblem(
        objective=objective.evaluate,
        dimension=dimension,
        lower_bounds=tuple(float(value) for value in objective._lower),
        upper_bounds=tuple(float(value) for value in objective._upper),
        optimum=objective.optimum,
    )
    return problem, objective
