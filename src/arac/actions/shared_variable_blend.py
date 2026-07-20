"""Shared-variable-value semantic surface: blend/writeback actions.

These implement the runtime logic that was previously inlined in
hcc_smoke_runner.apply_arac_overlap_action.  The runner now imports
from here; do not duplicate the logic.
"""

from __future__ import annotations

import numpy as np

from arac.actions.action_spec import ActionSpec

# ---------------------------------------------------------------------------
# Low-level blend primitives
# ---------------------------------------------------------------------------


def blend_overlap_values(
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_delta: float,
    current_delta: float,
) -> np.ndarray:
    """HCC native Eq.8 delta-weighted blend."""
    denominator = previous_delta + current_delta
    if denominator == 0:
        return (previous_values + current_values) / 2.0
    return (previous_delta / denominator) * previous_values + (
        current_delta / denominator
    ) * current_values


def clipped_consensus_blend(
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_delta: float,
    current_delta: float,
) -> np.ndarray:
    """Eq.8 blend with weights clipped to [0.35, 0.65] to reduce winner-take-all."""
    denominator = previous_delta + current_delta
    if denominator == 0:
        return (previous_values + current_values) / 2.0
    current_weight = float(np.clip(current_delta / denominator, 0.35, 0.65))
    return (1.0 - current_weight) * previous_values + current_weight * current_values


# ---------------------------------------------------------------------------
# Runtime action dispatch
# ---------------------------------------------------------------------------

NATIVE_EQ8_ACTION = "native_eq8"
TRUE_NO_WRITEBACK_ACTION = "true_no_writeback"


def apply_shared_variable_action(
    action_name: str,
    current_values: np.ndarray,
    shared_values: np.ndarray | None,
) -> np.ndarray | None:
    """Apply exact shared values selected upstream without recomputation."""

    supported = {spec.name for spec in SHARED_VARIABLE_ACTION_SPECS}
    if action_name not in supported:
        raise ValueError(f"unsupported shared-variable action: {action_name!r}")
    current = np.asarray(current_values, dtype=float).reshape(-1)
    if current.size == 0 or not np.all(np.isfinite(current)):
        raise ValueError("current shared values must be finite and non-empty")
    if action_name == TRUE_NO_WRITEBACK_ACTION:
        if shared_values is not None:
            raise ValueError("true_no_writeback cannot carry shared values")
        return None
    candidate = np.asarray(shared_values, dtype=float).reshape(-1)
    if candidate.shape != current.shape or not np.all(np.isfinite(candidate)):
        raise ValueError("frozen shared values must be finite and aligned")
    return candidate.copy()


def apply_legacy_shared_variable_policy(
    action_name: str,
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_delta: float,
    current_delta: float,
) -> np.ndarray | None:
    """Preserve historical rule-tree behavior outside Action Validation.

    Returns None for true_no_writeback (caller must not write back).
    Raises ValueError for unrecognised action names.
    """
    if action_name == TRUE_NO_WRITEBACK_ACTION:
        return None
    if action_name == "repair_shared_variable_binding":
        return current_values if current_delta >= previous_delta else previous_values
    if action_name == "isolate_conflicting_relation":
        return previous_values if previous_delta >= current_delta else current_values
    if action_name == "allow_beneficial_coordination":
        return clipped_consensus_blend(
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
        )
    if action_name in {"conservative_no_action", NATIVE_EQ8_ACTION}:
        return blend_overlap_values(
            previous_values=previous_values,
            current_values=current_values,
            previous_delta=previous_delta,
            current_delta=current_delta,
        )
    raise ValueError(f"unsupported shared-variable action: {action_name!r}")


# ---------------------------------------------------------------------------
# Explicit ActionSpec catalogue for this semantic surface
# ---------------------------------------------------------------------------

SHARED_VARIABLE_ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        name=NATIVE_EQ8_ACTION,
        semantic_surface="shared_variable_value",
        parameter_names=("current_values", "shared_values"),
    ),
    ActionSpec(
        name=TRUE_NO_WRITEBACK_ACTION,
        semantic_surface="shared_variable_value",
        parameter_names=("current_values", "shared_values"),
    ),
    ActionSpec(
        name="repair_shared_variable_binding",
        semantic_surface="shared_variable_value",
        parameter_names=("current_values", "shared_values"),
    ),
    ActionSpec(
        name="isolate_conflicting_relation",
        semantic_surface="shared_variable_value",
        parameter_names=("current_values", "shared_values"),
    ),
    ActionSpec(
        name="allow_beneficial_coordination",
        semantic_surface="shared_variable_value",
        parameter_names=("current_values", "shared_values"),
    ),
)
