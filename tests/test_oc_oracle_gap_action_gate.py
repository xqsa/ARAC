from __future__ import annotations

from pathlib import Path

import pytest

from experiments.oc_oracle_gap_action_gate import (
    ARMS,
    _best_fixed,
    _classify,
    _endpoint_report,
    _oracle,
    run_gate,
)

ARTIFACT = Path("artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json")


def _payload() -> dict[str, object]:
    contexts = []
    for step in range(6):
        context: dict[str, object] = {
            "topology": "chain",
            "overlap_budget": 6,
            "seed": 10 + step,
            "mode": "conforming",
            "selected_component": [0, 1, 2],
            "checkpoint_error": 100.0,
        }
        for index, arm in enumerate(ARMS):
            # arm 4 dominates 5/6 contexts; arm 0 wins the last context by a lot
            gain = 10.0 * (index + 1)
            if step == 5 and index == 0:
                gain = 100.0
            context[arm] = {"end_to_end_gain": gain, "action_gain": gain}
        contexts.append(context)
    return {"contexts": contexts, "protocol": {"production_selector_modified": False}}


def test_oracle_and_best_fixed_tie_break() -> None:
    gains = {arm: 1.0 for arm in ARMS}
    arm, value = _oracle(gains)
    assert arm == ARMS[0]
    assert value == 1.0
    best, _ = _best_fixed(_payload()["contexts"], "end_to_end_gain")  # type: ignore[arg-type]
    assert best == ARMS[4]


def test_classification_thresholds() -> None:
    assert _classify(0.01, 0.06) == "MATERIAL"
    assert _classify(0.01, 0.02) == "PRESENT_BUT_SMALL"
    assert _classify(-0.01, 0.50) == "NOT_MATERIAL"
    assert _classify(0.0, 0.50) == "NOT_MATERIAL"


def test_endpoint_report_regret_math() -> None:
    report = _endpoint_report(_payload(), "end_to_end_gain")  # type: ignore[arg-type]
    # 5 contexts: oracle 50 (arm4), best fixed arm4 regret 0
    # 1 context: oracle 100 (arm0), arm4 gives 50 -> regret 50
    assert report["best_fixed_arm"] == ARMS[4]
    assert report["mean_best_fixed_regret"] == pytest.approx(50.0 / 6.0)
    assert report["mean_normalized_regret"] == pytest.approx(50.0 / 6.0 / 100.0)
    assert report["oracle_identity"]["oracle_arm_distribution"][ARMS[4]] == 5
    assert report["oracle_identity"]["oracle_arm_distribution"][ARMS[0]] == 1


def test_real_artifact_gate() -> None:
    result = run_gate(ARTIFACT)
    assert result["gate_passed"] is True
    assert result["end_to_end"]["classification"] in {
        "MATERIAL",
        "PRESENT_BUT_SMALL",
        "NOT_MATERIAL",
    }
    flip = result["end_to_end"]["oracle_identity"]["flip_rate"]
    assert 0.0 <= flip <= 1.0
    assert result["protocol"]["production_selector_modified"] is False
