from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.oc_lagged_coupling_shadow import (
    ARMS,
    _replay,
    run_gate,
    validate_input,
)


ARTIFACT = Path("artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json")


def _small_payload() -> dict[str, object]:
    contexts = []
    for index, coupling in enumerate((1.0, 9.0)):
        context: dict[str, object] = {
            "topology": "chain",
            "overlap_budget": 6,
            "seed": 10 + index,
            "mode": "conforming",
            "selected_component": [0, 1, 2],
        }
        for arm_index, arm in enumerate(ARMS):
            context[arm] = {
                "coupled_gain": coupling + arm_index,
                "action_gain": coupling + arm_index,
                "end_to_end_gain": coupling + arm_index,
            }
        contexts.append(context)
    return {"contexts": contexts, "protocol": {"action_fes": 32}}


def test_lagged_replay_updates_after_prediction() -> None:
    rows = _replay(_small_payload())

    assert rows[0].prediction_source == "cold_start"
    assert rows[0].predicted_arm is None
    assert rows[1].prediction_source == "prior_ema"
    assert rows[1].predicted_arm == "duplicated_shared_local_competition"
    assert rows[1].prior_ema[-1][1] == 5.0
    assert rows[1].updated_ema[-1][1] == 9.0
    assert rows[1].action_hit is True


def test_real_artifact_replay_is_contract_safe() -> None:
    result = run_gate(ARTIFACT)

    assert result["gate_passed"] is True
    assert result["protocol"]["production_selector_modified"] is False
    assert result["summary"]["context_rows"] == 60
    assert result["summary"]["eligible_rows"] >= 10
    assert result["summary"]["promotion_recommended"] is False
    assert all(row["ticket_fes"] == 32 for row in result["rows"])


def test_input_production_flag_cannot_be_promoted() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["protocol"]["production_selector_modified"] = True

    with pytest.raises(ValueError, match="production selector"):
        validate_input(payload)
