"""Unit tests for the evidence-dispatched action policy."""

from __future__ import annotations

import pytest

from arac.dispatch_policy import dispatch_action


@pytest.mark.parametrize(
    ("gain", "density", "expected"),
    [
        (0.004, 0.017, "aor"),   # A family: flat tail gain
        (0.0096, 0.238, "aor"),  # A family per-seed maximum
        (0.2624, 0.0, "gcb"),    # R1: low gain, no relations still routes to gcb
        (0.3152, 0.147, "gcb"),  # R family per-seed maximum
        (0.7536, 0.0, "smp"),    # E1: high gain, no overlap instance
        (0.6188, 0.0, "smp"),    # S1: high gain, no overlap instance
        (0.7536, 0.147, "ctp"),  # E2-E6
        (0.6188, 0.147, "ctp"),  # S2-S6
    ],
)
def test_dispatch_rule_matches_calibrated_families(gain: float, density: float, expected: str) -> None:
    assert dispatch_action(gain, density) == expected


def test_threshold_boundaries_are_deterministic() -> None:
    assert dispatch_action(0.10, 0.5) == "gcb"
    assert dispatch_action(0.0999, 0.5) == "aor"
    assert dispatch_action(0.50, 0.05) == "smp"
    assert dispatch_action(0.50, 0.0501) == "ctp"


@pytest.mark.parametrize(
    ("gain", "density"),
    [
        (float("nan"), 0.1),
        (-0.1, 0.1),
        (0.5, float("nan")),
        (0.5, -0.1),
        (0.5, 1.5),
    ],
)
def test_dispatch_fails_closed_on_invalid_evidence(gain: float, density: float) -> None:
    with pytest.raises(ValueError):
        dispatch_action(gain, density)
