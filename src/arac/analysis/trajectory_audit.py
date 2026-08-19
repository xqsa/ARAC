"""Read-only diagnostics for common-anchor action trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arac.runtime.contracts import ACTION_NAMES


@dataclass(frozen=True)
class TrajectoryAudit:
    """Aggregate diagnostics for one or more identity-blind action probes."""

    run_count: int
    horizon_index: int
    crossover_rate: float
    horizon_rank_correlation: float
    marginal_gain_stability: float


def _validate_run(run: Mapping[str, Sequence[float]], horizon_index: int) -> int:
    if tuple(run) != ACTION_NAMES:
        raise ValueError("trajectory actions must cover ACTION_NAMES in frozen order")
    lengths = {len(run[action]) for action in ACTION_NAMES}
    if len(lengths) != 1:
        raise ValueError("all action trajectories must have the same length")
    length = lengths.pop()
    if length < 3:
        raise ValueError("trajectories must contain at least three points")
    if isinstance(horizon_index, bool) or not 0 < horizon_index < length - 1:
        raise ValueError("horizon_index must be inside the trajectory")
    for action in ACTION_NAMES:
        values = np.asarray(run[action], dtype=float)
        if values.shape != (length,) or not np.all(np.isfinite(values)):
            raise ValueError("trajectories must contain finite scalar values")
    return length


def _rank(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    first = _rank(left)
    second = _rank(right)
    if len(first) < 2:
        return 1.0
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    return 1.0 if denominator == 0.0 else float(np.dot(first_centered, second_centered) / denominator)


def _stability(values: Sequence[float], horizon_index: int) -> float:
    trace = np.asarray(values, dtype=float)
    first = np.maximum(trace[0] - trace[1 : horizon_index + 1], 0.0)
    second = np.maximum(trace[horizon_index] - trace[horizon_index + 1 :], 0.0)
    first_rate = float(np.mean(first))
    second_rate = float(np.mean(second))
    denominator = abs(first_rate) + abs(second_rate) + np.finfo(float).eps
    return max(0.0, min(1.0, 1.0 - abs(first_rate - second_rate) / denominator))


def audit_trajectories(
    runs: Mapping[str, Sequence[float]] | Sequence[Mapping[str, Sequence[float]]],
    *,
    horizon_index: int,
) -> TrajectoryAudit:
    """Measure late winners and gain stability without selecting an action.

    ``horizon_index`` is the last point visible at the proposed commitment
    boundary. Lower objective/error values are treated as better throughout.
    """

    normalized = [runs] if isinstance(runs, Mapping) else list(runs)
    if not normalized:
        raise ValueError("at least one trajectory run is required")
    for run in normalized:
        _validate_run(run, horizon_index)

    crossovers = []
    correlations = []
    stabilities = []
    for run in normalized:
        horizon_values = [float(run[action][horizon_index]) for action in ACTION_NAMES]
        terminal_values = [float(run[action][-1]) for action in ACTION_NAMES]
        horizon_winner = min(range(len(ACTION_NAMES)), key=lambda index: (horizon_values[index], index))
        terminal_winner = min(range(len(ACTION_NAMES)), key=lambda index: (terminal_values[index], index))
        crossovers.append(float(horizon_winner != terminal_winner))
        correlations.append(_spearman(horizon_values, terminal_values))
        stabilities.extend(_stability(run[action], horizon_index) for action in ACTION_NAMES)

    return TrajectoryAudit(
        run_count=len(normalized),
        horizon_index=horizon_index,
        crossover_rate=float(np.mean(crossovers)),
        horizon_rank_correlation=float(np.mean(correlations)),
        marginal_gain_stability=float(np.mean(stabilities)),
    )


__all__ = ["TrajectoryAudit", "audit_trajectories"]
