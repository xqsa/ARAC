from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.oc_lagged_coupling_normalized_gate import (
    ARMS,
    VARIANTS,
    _replay,
    _variant_metrics,
    _within_context_rank,
    run_gate,
)
from experiments.oc_lagged_coupling_shadow import validate_input


ARTIFACT = Path("artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json")
V1_ARTIFACT = Path("artifacts/oc_lagged_coupling_shadow/confirmation.json")


def _context(seed: int, mode: str, error: float, coupled: dict[str, float], action: dict[str, float], end: dict[str, float]) -> dict[str, object]:
    context: dict[str, object] = {
        "topology": "chain",
        "overlap_budget": 6,
        "seed": seed,
        "mode": mode,
        "selected_component": [0, 1, 2],
        "checkpoint_error": error,
        "strict_best": True,
        "promotion_applied": False,
        "coupling_receipt_parity": True,
    }
    for arm in ARMS:
        context[arm] = {
            "coupled_gain": coupled[arm],
            "action_gain": action[arm],
            "end_to_end_gain": end[arm],
            "coupling_fes": 1,
            "coupling_archive_preserved": True,
        }
    return context


def _payload(contexts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "arac-oc-action-semantic-gate-v3",
        "context_count": len(contexts),
        "contexts": contexts,
        "protocol": {"action_fes": 32, "coupling_fes": 1, "production_selector_modified": False},
    }


def test_within_context_rank_handles_ties() -> None:
    values = {
        "owner_control": 5.0,
        "shared_core": 1.0,
        "expanded_shared_private": 5.0,
        "duplicated_shared_competition": 3.0,
        "duplicated_shared_local_competition": 1.0,
    }
    ranks = _within_context_rank(values)
    # two arms tie at rank 1-2 (average 1.5 -> (1.5-1)/4 = 0.125)
    assert ranks["shared_core"] == pytest.approx(0.125)
    assert ranks["duplicated_shared_local_competition"] == pytest.approx(0.125)
    assert ranks["duplicated_shared_competition"] == pytest.approx(0.5)
    # two arms tie at rank 4-5 (average 4.5 -> 0.875)
    assert ranks["owner_control"] == pytest.approx(0.875)
    assert ranks["expanded_shared_private"] == pytest.approx(0.875)


def test_error_normalization_neutralizes_cross_context_scale() -> None:
    big = {arm: 1000.0 * (index + 1) for index, arm in enumerate(ARMS)}
    small = {arm: 1.0 * (index + 1) for index, arm in enumerate(ARMS)}
    # context 1 has a huge scale on a bad arm pattern, context 2 a small scale
    # on the same arm pattern; the arm ORDERING is identical after scaling, and
    # the raw EMA is dominated by context 1's magnitude while the normalized
    # EMA weighs both contexts equally.
    contexts = [
        _context(11, "conforming", 1000.0, big, big, big),
        _context(12, "conforming", 1.0, small, small, small),
    ]
    rows = _replay(_payload(contexts))
    raw_block = rows[1].variants["raw"]
    norm_block = rows[1].variants["error_normalized"]
    # both variants keep the same arm ordering here, but raw EMA is dragged by
    # the large context while normalized EMA blends the two scales
    assert raw_block.prior_ema[0][1] is not None
    assert norm_block.prior_ema[0][1] is not None
    assert rows[1].variants["rank"].observed_input[0][1] == pytest.approx(0.0)
    assert all(
        abs(value - expected) < 1e-12
        for (arm, value), expected in zip(
            rows[0].variants["error_normalized"].observed_input,
            [(index + 1) for index in range(len(ARMS))],
        )
    )


def test_prior_only_contract_and_determinism() -> None:
    contexts = [
        _context(
            21,
            "conforming",
            50.0,
            {arm: float(index) for index, arm in enumerate(ARMS)},
            {arm: float(index) for index, arm in enumerate(ARMS)},
            {arm: float(index) for index, arm in enumerate(ARMS)},
        ),
        _context(
            22,
            "conforming",
            25.0,
            {arm: float(len(ARMS) - index) for index, arm in enumerate(ARMS)},
            {arm: float(index) for index, arm in enumerate(ARMS)},
            {arm: float(index) for index, arm in enumerate(ARMS)},
        ),
    ]
    payload = _payload(contexts)
    rows = _replay(payload)
    assert rows == _replay(payload)
    for variant in VARIANTS:
        assert rows[0].variants[variant].prediction_source == "cold_start"
        assert rows[1].variants[variant].prediction_source == "prior_ema"
        prior = dict(rows[1].variants[variant].prior_ema)
        observed = dict(rows[0].variants[variant].observed_input)
        for arm in ARMS:
            assert prior[arm] == pytest.approx(observed[arm])


def test_revoke_rule_triggers_on_perfect_normalized_signal() -> None:
    contexts = []
    for stream in range(5):
        for step in range(12):
            seed = 1000 + stream * 100 + step
            error = 10.0 + step
            coupled = {arm: (index + 1) * error for index, arm in enumerate(ARMS)}
            action = {arm: float(index + 1) for index, arm in enumerate(ARMS)}
            end = {arm: float(index + 1) for index, arm in enumerate(ARMS)}
            component = [stream * 3, stream * 3 + 1, stream * 3 + 2]
            context = _context(seed, "conforming", error, coupled, action, end)
            context["selected_component"] = component
            contexts.append(context)
    payload = _payload(contexts)
    validate_input(payload)
    rows = _replay(payload)
    metrics = _variant_metrics(rows, "error_normalized")
    assert metrics["eligible_rows"] == 55
    assert metrics["end_to_end_ticket_hit_rate"] == 1.0
    assert metrics["end_to_end_hit_rate_ci95"][0] > 0.20
    assert metrics["end_to_end_cohens_kappa"] == pytest.approx(1.0)


def test_real_artifact_gate_passes_and_records_decision() -> None:
    result = run_gate(ARTIFACT)
    assert result["gate_checks"]["raw_reproduces_v1_shadow"] is True
    assert result["gate_checks"]["prior_only_prediction_and_update"] is True
    assert result["gate_passed"] is True
    reference = json.loads(V1_ARTIFACT.read_text(encoding="utf-8"))
    assert result["variant_metrics"]["raw"]["action_ticket_hit_rate"] == pytest.approx(
        reference["summary"]["action_ticket_hit_rate"]
    )
    # the pre-registered decision must be recorded either way
    decision = result["summary"]["scale_confound_revoke"]
    assert isinstance(decision, bool)
    assert result["summary"]["lagged_negative_confirmed"] is (not decision)
