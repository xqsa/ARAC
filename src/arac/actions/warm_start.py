"""Deterministic execution of a frozen unique-coordinate mean shift."""

from __future__ import annotations

import numpy as np

from arac.actions.action_spec import ActionSpec


WARM_START_ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        name="stagnation_cross_group_warm_start",
        semantic_surface="optimizer_state",
        parameter_names=(
            "current_mean",
            "unique_positions",
            "mean_shift",
            "lower_bound",
            "upper_bound",
        ),
    ),
)


def apply_warm_start_action(
    action_name: str,
    current_mean: np.ndarray,
    unique_positions: tuple[int, ...],
    mean_shift: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Apply an exact shift only to positions declared unique upstream."""

    if action_name != "stagnation_cross_group_warm_start":
        raise ValueError(f"unsupported warm start action: {action_name!r}")
    mean = np.asarray(current_mean, dtype=float).reshape(-1)
    shift = np.asarray(mean_shift, dtype=float).reshape(-1)
    positions = tuple(int(value) for value in unique_positions)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(shift)):
        raise ValueError("warm-start values must be finite")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("unique positions must be non-empty and distinct")
    if len(shift) != len(positions):
        raise ValueError("mean shift must align with unique positions")
    if any(value < 0 or value >= len(mean) for value in positions):
        raise ValueError("unique position is outside the optimizer mean")
    lower = float(lower_bound)
    upper = float(upper_bound)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("warm-start bounds are invalid")
    updated = mean.copy()
    updated[list(positions)] = np.clip(
        updated[list(positions)] + shift,
        lower,
        upper,
    )
    return updated
