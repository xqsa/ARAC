"""Shared offline protocol helpers for CAR actionability audit artifacts.

The actual long continuation is executed by two independent canonical HCC
subprocess lanes.  This module intentionally contains no runtime selector and
no incomplete ``BranchState`` clone implementation.
"""

from __future__ import annotations

import math


CAR_ACTIONABILITY_PROTOCOL_VERSION = "car-actionability-v2"
CAR_ACTIONABILITY_HORIZON_MULTIPLIERS = (1, 3, 9)
CAR_ACTIONABILITY_HORIZON_LABELS = ("closure_1", "budget_3x", "budget_9x")
CAR_ACTIONABILITY_LOG_FLOOR = 1e-300


def log_actionability_advantage(
    fallback_error: float,
    candidate_error: float,
    *,
    log_floor: float = CAR_ACTIONABILITY_LOG_FLOOR,
) -> float:
    """Return ``log(fallback) - log(candidate)`` for offline non-negative errors."""

    fallback = float(fallback_error)
    candidate = float(candidate_error)
    floor = float(log_floor)
    if not all(math.isfinite(value) for value in (fallback, candidate, floor)):
        raise ValueError("actionability errors and log floor must be finite")
    if fallback < 0.0 or candidate < 0.0 or floor <= 0.0:
        raise ValueError("actionability errors must be non-negative")
    return math.log(max(fallback, floor)) - math.log(max(candidate, floor))
