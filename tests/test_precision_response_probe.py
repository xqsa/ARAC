from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from arac.policy.precision_response_probe import (
    FORBIDDEN_GATE_FIELD_FRAGMENTS,
    PrecisionProbeGateState,
    build_precision_probe_gate_state,
    decide_precision_probe,
    precision_probe_config_from_mapping,
)


ROOT = Path(__file__).parents[1]
VENDOR_ROOT = ROOT / "vendor" / "hcc"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from arac.backends.hcc_cma_proposals import run_paired_hcc_cma_probe


@pytest.fixture(scope="module")
def config():
    payload = json.loads(
        (ROOT / "configs" / "precision_response_loop_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return precision_probe_config_from_mapping(payload)


def _state(config, *, wins: int, large_loss: bool = False):
    normal = [2.0] * wins + [1.0] * (16 - wins)
    precision = [1.0] * wins + [1.1] * (16 - wins)
    if large_loss:
        precision[-1] = 2.0
    return build_precision_probe_gate_state(
        normal_errors=normal,
        precision_errors=precision,
        checkpoint_error=3.0,
        direction_hash_match=True,
        standardized_diversity_ratio=1.0,
        normal_boundary_hit_count=0,
        precision_boundary_hit_count=0,
        config=config,
    )


def test_fixed_gate_accepts_thirteen_wins_and_rejects_twelve(config) -> None:
    accepted = _state(config, wins=13)
    rejected = _state(config, wins=12)

    assert accepted.paired_win_lcb > 0.55
    assert rejected.paired_win_lcb <= 0.55
    assert decide_precision_probe(accepted, config).release is True
    assert decide_precision_probe(rejected, config).reason == "win_lcb_not_positive"


def test_fixed_gate_rejects_one_large_loss(config) -> None:
    state = _state(config, wins=15, large_loss=True)

    assert state.large_loss_count == 1
    assert state.large_loss_ucb > 0.20
    assert decide_precision_probe(state, config).reason == "large_loss_ucb_exceeded"


def test_gate_state_has_no_identity_or_outcome_fields() -> None:
    names = {item.name.lower() for item in fields(PrecisionProbeGateState)}

    assert not {
        name
        for name in names
        if any(fragment in name for fragment in FORBIDDEN_GATE_FIELD_FRAGMENTS)
    }


def test_paired_vendor_probe_uses_same_directions_and_exactly_32_fe() -> None:
    observed = []

    def sphere(candidates: np.ndarray) -> np.ndarray:
        observed.extend(np.asarray(candidates, dtype=float).copy())
        return np.sum(np.square(candidates), axis=1)

    result = run_paired_hcc_cma_probe(
        fitness_function=sphere,
        mean=np.array([0.4, -0.2, 0.1]),
        lower=-100.0,
        upper=100.0,
        normal_sigma=0.5,
        precision_sigma=0.25,
        seed=1234,
    )

    assert result.total_fe == 32
    assert len(observed) == 32
    assert result.normal.actual_fe == result.precision.actual_fe == 16
    assert result.direction_hash_match is True
    assert result.standardized_diversity_ratio == pytest.approx(1.0)
    assert np.array_equal(
        result.normal.standardized_directions,
        result.precision.standardized_directions,
    )
    assert np.allclose(
        result.precision.candidates - np.array([0.4, -0.2, 0.1]),
        0.5 * (result.normal.candidates - np.array([0.4, -0.2, 0.1])),
    )


def test_invalid_probe_error_fails_closed(config) -> None:
    state = build_precision_probe_gate_state(
        normal_errors=[1.0] * 16,
        precision_errors=[-1.0] * 16,
        checkpoint_error=2.0,
        direction_hash_match=True,
        standardized_diversity_ratio=1.0,
        normal_boundary_hit_count=0,
        precision_boundary_hit_count=0,
        config=config,
    )

    assert state.valid is False
    assert decide_precision_probe(state, config).release is False
