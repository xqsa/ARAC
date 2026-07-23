"""Decomposition methods used by the continuous WLOC comparison suite."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from .contracts import GroupingResult


ScalarObjective = Callable[[np.ndarray], object]


def _bounds(
    dimension: int,
    lower: float | Sequence[float],
    upper: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    lower_values = np.broadcast_to(np.asarray(lower, dtype=float), (dimension,)).copy()
    upper_values = np.broadcast_to(np.asarray(upper, dtype=float), (dimension,)).copy()
    if not np.all(np.isfinite(lower_values)) or not np.all(np.isfinite(upper_values)):
        raise ValueError("bounds must be finite")
    if np.any(lower_values >= upper_values):
        raise ValueError("every lower bound must be smaller than its upper bound")
    return lower_values, upper_values


def _scalar_value(objective: ScalarObjective, candidate: np.ndarray) -> float:
    result = np.asarray(objective(candidate), dtype=float)
    if result.size != 1:
        raise ValueError("decomposition objective must return exactly one value per candidate")
    value = float(result.reshape(-1)[0])
    if not math.isfinite(value):
        raise ValueError("decomposition objective returned a non-finite value")
    return value


def _connected_components(adjacency: np.ndarray) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(adjacency.shape[0]))
    components: list[tuple[int, ...]] = []
    while unseen:
        pending = [min(unseen)]
        unseen.remove(pending[0])
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            neighbors = [
                int(index)
                for index in np.flatnonzero(adjacency[current])
                if int(index) in unseen
            ]
            for neighbor in neighbors:
                unseen.remove(neighbor)
                pending.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def dg2_grouping(
    objective: ScalarObjective,
    dimension: int,
    *,
    lower: float | Sequence[float] = 0.0,
    upper: float | Sequence[float] = 1.0,
) -> GroupingResult:
    """Measure pairwise interactions using DG2 and return graph components."""

    lower_values, upper_values = _bounds(dimension, lower, upper)
    middle = (lower_values + upper_values) / 2.0
    base = lower_values.copy()
    base_value = _scalar_value(objective, base)
    single_values = np.empty(dimension, dtype=float)
    for index in range(dimension):
        candidate = base.copy()
        candidate[index] = middle[index]
        single_values[index] = _scalar_value(objective, candidate)

    pair_values = np.full((dimension, dimension), np.nan, dtype=float)
    signals = np.zeros((dimension, dimension), dtype=float)
    for first in range(dimension - 1):
        for second in range(first + 1, dimension):
            candidate = base.copy()
            candidate[first] = middle[first]
            candidate[second] = middle[second]
            pair_value = _scalar_value(objective, candidate)
            pair_values[first, second] = pair_values[second, first] = pair_value
            delta_base = single_values[first] - base_value
            delta_perturbed = pair_value - single_values[second]
            signals[first, second] = signals[second, first] = abs(
                delta_base - delta_perturbed
            )

    unit_roundoff = np.finfo(float).eps / 2.0

    def gamma(count: float) -> float:
        return count * unit_roundoff / (1.0 - count * unit_roundoff)

    error_lower = np.zeros_like(signals)
    error_upper = np.zeros_like(signals)
    pair_indices = np.triu_indices(dimension, k=1)
    for first, second in zip(*pair_indices, strict=True):
        four_values = (
            base_value,
            single_values[first],
            single_values[second],
            pair_values[first, second],
        )
        error_lower[first, second] = gamma(2.0) * max(
            abs(base_value + pair_values[first, second]),
            abs(single_values[first] + single_values[second]),
        )
        error_upper[first, second] = gamma(math.sqrt(dimension)) * max(
            abs(value) for value in four_values
        )
        error_upper[first, second] = max(
            error_upper[first, second], error_lower[first, second]
        )

    upper_signals = signals[pair_indices]
    upper_lower = error_lower[pair_indices]
    upper_upper = error_upper[pair_indices]
    exact_zero = upper_signals == 0.0
    reliable_separable = (~exact_zero) & (upper_signals <= upper_lower)
    reliable_interacting = (upper_signals > 0.0) & (upper_signals >= upper_upper)
    reliable_count = int(np.count_nonzero(reliable_separable | reliable_interacting))
    denominator = int(np.count_nonzero(exact_zero)) + reliable_count
    if denominator:
        separable_weight = (
            np.count_nonzero(exact_zero) + np.count_nonzero(reliable_separable)
        ) / denominator
        interacting_weight = np.count_nonzero(reliable_interacting) / denominator
    else:
        separable_weight = interacting_weight = 0.5
    adaptive_threshold = separable_weight * upper_lower + interacting_weight * upper_upper
    upper_adjacency = reliable_interacting | (
        (~exact_zero)
        & (~reliable_separable)
        & (~reliable_interacting)
        & (upper_signals > adaptive_threshold)
    )

    adjacency = np.eye(dimension, dtype=bool)
    adjacency[pair_indices] = upper_adjacency
    adjacency |= adjacency.T
    decomposition_fes = 1 + dimension + dimension * (dimension - 1) // 2
    return GroupingResult(
        method="DG2",
        dimension=dimension,
        groups=_connected_components(adjacency),
        decomposition_fes=decomposition_fes,
        allows_overlap=False,
        origin="measured_objective",
        matrix=adjacency,
        matrix_kind="interaction",
    )


def random_grouping(
    dimension: int,
    *,
    seed: int,
    group_count: int = 20,
) -> GroupingResult:
    """Split a random permutation into the 20 disjoint AOB baseline subspaces."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count <= 0:
        raise ValueError("group_count must be a positive integer")
    if group_count > dimension:
        raise ValueError("group_count cannot exceed dimension")
    permutation = np.random.default_rng(seed).permutation(dimension)
    groups = tuple(
        tuple(int(index) for index in chunk)
        for chunk in np.array_split(permutation, group_count)
    )
    return GroupingResult(
        method="Random",
        dimension=dimension,
        groups=groups,
        decomposition_fes=0,
        allows_overlap=False,
        origin="generated_random",
    )


def rdg3_grouping(
    objective: ScalarObjective,
    dimension: int,
    *,
    lower: float | Sequence[float] = 0.0,
    upper: float | Sequence[float] = 1.0,
    nonseparable_threshold: int = 50,
    separable_chunk_size: int = 100,
) -> GroupingResult:
    """Independently implement the RDG3 recursion from Sun et al. (2019)."""

    if (
        isinstance(nonseparable_threshold, bool)
        or not isinstance(nonseparable_threshold, int)
        or nonseparable_threshold < 0
    ):
        raise ValueError("nonseparable_threshold must be a non-negative integer")
    if (
        isinstance(separable_chunk_size, bool)
        or not isinstance(separable_chunk_size, int)
        or separable_chunk_size <= 0
    ):
        raise ValueError("separable_chunk_size must be a positive integer")
    lower_values, upper_values = _bounds(dimension, lower, upper)
    middle = (lower_values + upper_values) / 2.0
    base = lower_values.copy()
    base_value = _scalar_value(objective, base)
    evaluations = 1
    unit_roundoff = np.finfo(float).eps / 2.0
    gamma_count = math.sqrt(dimension) + 2.0
    gamma = gamma_count * unit_roundoff / (1.0 - gamma_count * unit_roundoff)

    def interact(current: tuple[int, ...], candidates: tuple[int, ...]) -> tuple[int, ...]:
        nonlocal evaluations
        upper_current = base.copy()
        upper_current[np.asarray(current, dtype=int)] = upper_values[
            np.asarray(current, dtype=int)
        ]
        lower_middle = base.copy()
        lower_middle[np.asarray(candidates, dtype=int)] = middle[
            np.asarray(candidates, dtype=int)
        ]
        upper_middle = upper_current.copy()
        upper_middle[np.asarray(candidates, dtype=int)] = middle[
            np.asarray(candidates, dtype=int)
        ]
        upper_current_value = _scalar_value(objective, upper_current)
        lower_middle_value = _scalar_value(objective, lower_middle)
        upper_middle_value = _scalar_value(objective, upper_middle)
        evaluations += 3
        delta_base = base_value - upper_current_value
        delta_middle = lower_middle_value - upper_middle_value
        threshold = gamma * sum(
            abs(value)
            for value in (
                base_value,
                upper_current_value,
                lower_middle_value,
                upper_middle_value,
            )
        )
        if abs(delta_base - delta_middle) <= threshold:
            return current
        if len(candidates) == 1:
            return current + candidates
        split = len(candidates) // 2
        left = interact(current, candidates[:split])
        right = interact(current, candidates[split:])
        found = set(left) | set(right)
        return current + tuple(index for index in candidates if index in found)

    remaining = list(range(1, dimension))
    current: tuple[int, ...] = (0,)
    separable: list[int] = []
    nonseparable: list[tuple[int, ...]] = []

    def finalize(group: tuple[int, ...]) -> None:
        if len(group) == 1:
            separable.extend(group)
        else:
            nonseparable.append(group)

    while True:
        if not remaining:
            finalize(current)
            break
        expanded = interact(current, tuple(remaining))
        if len(expanded) >= nonseparable_threshold:
            finalize(expanded)
            found = set(expanded)
            remaining = [index for index in remaining if index not in found]
            if not remaining:
                break
            current = (remaining.pop(0),)
        elif len(expanded) == len(current):
            finalize(current)
            current = (remaining.pop(0),)
        else:
            current = expanded
            found = set(current)
            remaining = [index for index in remaining if index not in found]

    separable_groups = tuple(
        tuple(separable[start : start + separable_chunk_size])
        for start in range(0, len(separable), separable_chunk_size)
    )
    return GroupingResult(
        method="RDG3",
        dimension=dimension,
        groups=tuple(nonseparable) + separable_groups,
        decomposition_fes=evaluations,
        allows_overlap=False,
        origin="measured_objective",
    )


def design_matrix_from_groups(
    dimension: int,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Encode variable-to-component memberships for the RDDSM truth implementation."""

    matrix = np.zeros((dimension, len(groups)), dtype=float)
    memberships: set[int] = set()
    for group_index, group in enumerate(groups):
        for variable in group:
            index = int(variable)
            if index < 0 or index >= dimension:
                raise ValueError("group index is outside the decision space")
            matrix[index, group_index] = 1.0
            memberships.add(index)
    if memberships != set(range(dimension)):
        raise ValueError("groups must cover every decision variable")
    return matrix


def rddsm_grouping(design_matrix: np.ndarray) -> GroupingResult:
    """Run the vendored HCC RDDSM decomposition on a supplied design matrix."""

    from vendor.hcc.HCC.RDDSM import Decomposition

    matrix = np.asarray(design_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("design_matrix must be a non-empty two-dimensional matrix")
    raw_groups = Decomposition(matrix).decomposition()
    groups = tuple(
        sorted(
            (tuple(sorted(int(index) for index in group)) for group in raw_groups),
            key=lambda group: (group[0], len(group), group),
        )
    )
    return GroupingResult(
        method="RDDSM",
        dimension=matrix.shape[0],
        groups=groups,
        decomposition_fes=0,
        allows_overlap=True,
        origin="design_matrix",
        matrix=matrix,
        matrix_kind="design",
    )
