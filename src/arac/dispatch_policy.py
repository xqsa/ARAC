"""Evidence-dispatched Phase-II action selection for dense-overlap problems.

Gate 41a calibrated a two-feature interpretable dispatch rule offline on the
RecoveredActionRegistry four-arm matrix (24 AOB cases x 7 seeds, shared
Phase-I checkpoints, zero new FE) and validated it against the historical
ARAC column: 24/24 cases within 1.005x, 17 strictly better, HCC-ES record
improved from 18/1/5 to 21/3/0.

The rule consumes only identity-blind Phase-I evidence:

- ``tail_log10_gain`` (log10 tail-segment improvement of the Phase-I probe)
  separates all four landscape families with wide margins
  (A: 0.000-0.010, R: 0.262-0.315, S: 0.619-0.786, E: 0.754-0.895);
- ``structural_relation_density == 0`` marks the non-overlapping instance of
  each family (E1, S1) where state-memory search drives the error to
  machine precision.

Thresholds sit inside the empty gaps; only offline calibration may change
them (see ``experiments/overlap_action_dispatch_gate41_offline.py``).
"""

from __future__ import annotations

import math

TAU_AOR_GAIN = 0.10
TAU_CTP_GAIN = 0.50
TAU_NO_RELATION = 0.05
DISPATCH_ACTIONS = ("aor", "ctp", "smp", "gcb")


def dispatch_action(tail_log10_gain: float, structural_relation_density: float) -> str:
    """Map identity-blind Phase-I evidence to one Phase-II action."""

    gain = float(tail_log10_gain)
    density = float(structural_relation_density)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("tail_log10_gain must be finite and non-negative")
    if not math.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("structural_relation_density must be finite in [0, 1]")
    if gain < TAU_AOR_GAIN:
        return "aor"
    if gain >= TAU_CTP_GAIN:
        return "smp" if density <= TAU_NO_RELATION else "ctp"
    return "gcb"


__all__ = [
    "DISPATCH_ACTIONS",
    "TAU_AOR_GAIN",
    "TAU_CTP_GAIN",
    "TAU_NO_RELATION",
    "dispatch_action",
]
