from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from arac.policy.causal_risk_scheduler import (
    CAUSAL_RISK_MODEL_SCHEMA_VERSION,
    FEATURE_SCHEMA_SHA256,
    PRE_ACTION_UTILITY_SCHEMA_VERSION,
    UTILITY_FEATURE_NAMES,
    CausalRiskInvariantError,
    CausalRiskModelBundle,
    PreActionUtilityState,
    UtilityEstimate,
    compute_model_sha256,
    decide_safe_release,
)


def _state(**overrides: float | str) -> PreActionUtilityState:
    payload: dict[str, float | str] = {
        "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
        **{name: 0.5 for name in UTILITY_FEATURE_NAMES},
    }
    payload.update(overrides)
    return PreActionUtilityState.from_runtime_payload(payload)


def _leaf(value: float) -> dict[str, float]:
    return {"value": value}


def _split(
    *,
    feature: str = "remaining_fe_ratio",
    threshold: float = 0.0,
    left: dict[str, object] | None = None,
    right: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left or _leaf(0.20),
        "right": right or _leaf(0.10),
    }


def _bundle_payload(
    *,
    utility_tree: dict[str, object] | None = None,
    risk_tree: dict[str, object] | None = None,
    cp_tree: dict[str, object] | None = None,
    knn_distance_threshold: float = 1.0,
) -> dict[str, object]:
    utility = utility_tree or _leaf(0.20)
    risk = risk_tree or _leaf(0.02)
    payload: dict[str, object] = {
        "schema_version": CAUSAL_RISK_MODEL_SCHEMA_VERSION,
        "feature_schema": {
            "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
            "feature_names": list(UTILITY_FEATURE_NAMES),
            "sha256": FEATURE_SCHEMA_SHA256,
        },
        "ood": {
            "median": [0.5] * len(UTILITY_FEATURE_NAMES),
            "iqr": [0.25] * len(UTILITY_FEATURE_NAMES),
            "minimum": [0.0] * len(UTILITY_FEATURE_NAMES),
            "maximum": [1.0] * len(UTILITY_FEATURE_NAMES),
            "reference_scaled": [[0.0] * len(UTILITY_FEATURE_NAMES)] * 5,
            "knn_k": 5,
            "knn_distance_threshold": knn_distance_threshold,
        },
        "utility": {
            "bootstrap_trees": [utility] * 1000,
            "lcb_quantile": 0.05,
            "conformal_margin": 0.05,
        },
        "catastrophic_risk": {
            "bootstrap_trees": [risk] * 1000,
            "bootstrap_quantile": 0.95,
            "clopper_pearson_tree": cp_tree or _leaf(0.03),
        },
    }
    payload["model_sha256"] = compute_model_sha256(payload)
    return payload


def _bundle(**kwargs: object) -> CausalRiskModelBundle:
    return CausalRiskModelBundle.from_mapping(_bundle_payload(**kwargs))


def test_pre_action_state_is_frozen_strict_and_hashable() -> None:
    state = _state()

    assert tuple(field.name for field in fields(state)) == (
        "schema_version",
        *UTILITY_FEATURE_NAMES,
    )
    assert len(state.feature_sha256) == 64
    assert state.feature_sha256 == _state().feature_sha256
    assert state.feature_sha256 != _state(remaining_fe_ratio=0.6).feature_sha256
    with pytest.raises(FrozenInstanceError):
        state.remaining_fe_ratio = 0.4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"case_id": "E1"}, "forbidden runtime field"),
        ({"graph_fingerprint": "abc"}, "forbidden runtime field"),
        ({"final_error": 1.0}, "forbidden runtime field"),
        ({"new_signal": 1.0}, "unknown runtime field"),
    ],
)
def test_pre_action_state_rejects_forbidden_and_unknown_fields(
    mutation: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
        **{name: 0.5 for name in UTILITY_FEATURE_NAMES},
        **mutation,
    }

    with pytest.raises(CausalRiskInvariantError, match=message):
        PreActionUtilityState.from_runtime_payload(payload)


def test_pre_action_state_rejects_missing_nonfinite_and_wrong_schema() -> None:
    payload: dict[str, object] = {
        "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
        **{name: 0.5 for name in UTILITY_FEATURE_NAMES},
    }
    payload.pop("cc_progress_rate_last")
    with pytest.raises(CausalRiskInvariantError, match="missing runtime field"):
        PreActionUtilityState.from_runtime_payload(payload)

    with pytest.raises(CausalRiskInvariantError, match="must be finite"):
        _state(cc_progress_rate_last=float("nan"))
    with pytest.raises(CausalRiskInvariantError, match="unsupported pre-action schema"):
        _state(schema_version="future-schema")


def test_policy_contracts_do_not_expose_identity_or_outcome_fields() -> None:
    forbidden = {
        "case",
        "case_id",
        "problem_id",
        "seed",
        "family",
        "function_family",
        "graph_fingerprint",
        "component_id",
        "group_index",
        "paper_best",
        "historical_best",
        "objective",
        "incumbent",
        "final_outcome",
        "final_error",
        "relative_gain",
    }
    for contract in (PreActionUtilityState, UtilityEstimate):
        assert not forbidden.intersection(field.name for field in fields(contract))


def test_bundle_loads_from_json_and_runs_pure_tree_inference() -> None:
    bundle = CausalRiskModelBundle.from_json(json.dumps(_bundle_payload()))
    estimate = bundle.estimate(_state())

    assert estimate.tau_hat == pytest.approx(0.20)
    assert estimate.tau_lcb == pytest.approx(0.15)
    assert estimate.catastrophic_risk_ucb == pytest.approx(0.03)
    assert estimate.in_distribution is True
    assert estimate.ood_reasons == ()
    assert estimate.model_hash == bundle.model_sha256
    assert estimate.feature_hash == _state().feature_sha256


def test_tree_splits_use_robust_scaled_features() -> None:
    tree = _split(left=_leaf(0.25), right=_leaf(-0.25))
    bundle = _bundle(utility_tree=tree)

    at_median = bundle.estimate(_state(remaining_fe_ratio=0.5))
    above_median = bundle.estimate(_state(remaining_fe_ratio=0.75))

    assert at_median.tau_hat == pytest.approx(0.25)
    assert above_median.tau_hat == pytest.approx(-0.25)


def test_bundle_rejects_hash_schema_and_tree_depth_tampering() -> None:
    bad_hash = _bundle_payload()
    bad_hash["model_sha256"] = "0" * 64
    with pytest.raises(CausalRiskInvariantError, match="model_sha256 mismatch"):
        CausalRiskModelBundle.from_mapping(bad_hash)

    bad_feature_hash = _bundle_payload()
    feature_schema = bad_feature_hash["feature_schema"]
    assert isinstance(feature_schema, dict)
    feature_schema["sha256"] = "0" * 64
    bad_feature_hash["model_sha256"] = compute_model_sha256(bad_feature_hash)
    with pytest.raises(CausalRiskInvariantError, match="feature schema hash mismatch"):
        CausalRiskModelBundle.from_mapping(bad_feature_hash)

    depth_three = _split(
        left=_split(left=_split(left=_leaf(1.0), right=_leaf(1.0)))
    )
    too_deep = _bundle_payload(utility_tree=depth_three)
    with pytest.raises(CausalRiskInvariantError, match="depth exceeds 2"):
        CausalRiskModelBundle.from_mapping(too_deep)


def test_bundle_rejects_forbidden_tree_features_and_invalid_risk() -> None:
    identity_split = _bundle_payload(
        utility_tree=_split(feature="case_id"),
    )
    with pytest.raises(CausalRiskInvariantError, match="unknown tree feature"):
        CausalRiskModelBundle.from_mapping(identity_split)

    invalid_risk = _bundle_payload(risk_tree=_leaf(1.01))
    with pytest.raises(CausalRiskInvariantError, match=r"within \[0, 1\]"):
        CausalRiskModelBundle.from_mapping(invalid_risk)


def test_ood_rejects_individual_range_and_joint_knn_extrapolation() -> None:
    bundle = _bundle()
    range_ood = bundle.estimate(_state(remaining_fe_ratio=1.01))
    joint_ood = bundle.estimate(
        _state(**{name: 0.75 for name in UTILITY_FEATURE_NAMES})
    )

    assert range_ood.in_distribution is False
    assert "feature_out_of_range:remaining_fe_ratio" in range_ood.ood_reasons
    assert joint_ood.in_distribution is False
    assert "knn_distance_exceeded" in joint_ood.ood_reasons


def test_safe_release_requires_every_gate_and_consumes_only_one_release() -> None:
    bundle = _bundle()
    state = _state()

    released = decide_safe_release(
        candidate_feasible=True,
        component_unlocked=True,
        release_already_consumed=False,
        state=state,
        model_bundle=bundle,
    )
    consumed = decide_safe_release(
        candidate_feasible=True,
        component_unlocked=True,
        release_already_consumed=True,
        state=state,
        model_bundle=bundle,
    )

    assert released.released is True
    assert released.selected_action == "post_retirement_precision_reanchor"
    assert released.reason == "causal_risk_gate_passed"
    assert consumed.released is False
    assert consumed.selected_action == "baseline"
    assert consumed.reason == "abstain_release_already_consumed"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"candidate_feasible": False}, "abstain_candidate_infeasible"),
        ({"component_unlocked": False}, "abstain_component_locked"),
        ({"state": None}, "abstain_pre_action_state_missing"),
        ({"model_bundle": None}, "abstain_model_unavailable"),
        (
            {"state": _state(remaining_fe_ratio=1.01)},
            "abstain_out_of_distribution",
        ),
        (
            {"model_bundle": _bundle(utility_tree=_leaf(0.05))},
            "abstain_causal_lcb_not_positive",
        ),
        (
            {
                "model_bundle": _bundle(
                    risk_tree=_leaf(0.051),
                    cp_tree=_leaf(0.04),
                )
            },
            "abstain_catastrophic_risk_above_limit",
        ),
    ],
)
def test_safe_release_fails_closed(kwargs: dict[str, object], reason: str) -> None:
    inputs: dict[str, object] = {
        "candidate_feasible": True,
        "component_unlocked": True,
        "release_already_consumed": False,
        "state": _state(),
        "model_bundle": _bundle(),
    }
    inputs.update(kwargs)

    decision = decide_safe_release(**inputs)  # type: ignore[arg-type]

    assert decision.released is False
    assert decision.selected_action == "baseline"
    assert decision.reason == reason


def test_safe_release_allows_exact_risk_limit() -> None:
    decision = decide_safe_release(
        candidate_feasible=True,
        component_unlocked=True,
        release_already_consumed=False,
        state=_state(),
        model_bundle=_bundle(
            risk_tree=_leaf(0.05),
            cp_tree=_leaf(0.05),
        ),
    )

    assert decision.released is True


def test_utility_estimate_is_immutable() -> None:
    estimate = _bundle().estimate(_state())

    with pytest.raises(FrozenInstanceError):
        estimate.tau_lcb = 1.0  # type: ignore[misc]
