from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.historical_recovery.current_recovered_four_arm import (
    REPOSITORY_ROOT,
    _contexts,
    _load_verified_checkpoint,
    _manifest,
    load_protocol,
)
from arac.analysis.grouped_outcome_selector import (
    EXPECTED_AOB_CASES,
    GROUPED_GATE_PARAMETERS,
    build_grouped_evaluation,
    build_grouped_folds,
    metrics_pass_gate,
)
from arac.analysis.outcome_selector import ActionOutcome, OutcomeRecord
from arac.evidence.phase1 import PHASE1_FEATURE_NAMES
from arac.runtime.contracts import ACTION_NAMES


def _outcome(case_id: str, seed: int = 117) -> OutcomeRecord:
    return OutcomeRecord(
        case_id=case_id,
        run_seed=seed,
        checkpoint_hash="a" * 64,
        feature_names=PHASE1_FEATURE_NAMES,
        feature_values=tuple(float(index) for index in range(len(PHASE1_FEATURE_NAMES))),
        outcomes=tuple(
            ActionOutcome(
                action_name=action,
                final_error=float(index + 1),
                result_hash=f"{index + 1:x}" * 64,
            )
            for index, action in enumerate(ACTION_NAMES)
        ),
    )


def test_protocol_freezes_recovered_four_arm_matrix() -> None:
    protocol = load_protocol()

    assert len(protocol["cases"]) == 24
    assert len(protocol["seeds"]) == 25
    assert protocol["actions"] == list(ACTION_NAMES)
    assert protocol["phase1_fes"] == 180_000
    assert protocol["phase2_fes"] == 2_820_000
    assert protocol["total_budget_fes"] == 3_000_000
    assert protocol["registry"] == "RecoveredActionRegistry"
    assert protocol["allow_out_of_bounds"] is True
    assert protocol["selector_execution_allowed"] is False


def test_all_retained_checkpoints_match_current_e2e_hashes() -> None:
    protocol = load_protocol()
    contexts = _contexts(
        protocol,
        REPOSITORY_ROOT / "artifacts" / "unused-test-output",
        "a" * 64,
        protocol["cases"],
        protocol["seeds"],
    )
    representative = {context.key: context for context in contexts if context.action_name == "ctp"}

    assert len(representative) == 600
    for context in representative.values():
        checkpoint = _load_verified_checkpoint(context)
        assert checkpoint.run_seed == context.run_seed
        assert checkpoint.phase1_fes == 180_000
        assert checkpoint.total_budget_fes == 3_000_000


def test_preflight_contexts_cover_each_recovered_action() -> None:
    protocol = load_protocol()
    contexts = _contexts(
        protocol,
        REPOSITORY_ROOT / "artifacts" / "unused-test-output",
        "b" * 64,
        protocol["preflight_cases"],
        protocol["preflight_seeds"],
    )

    assert len(contexts) == 16
    assert {context.action_name for context in contexts} == set(ACTION_NAMES)
    assert {context.case_id for context in contexts} == {"A1", "E1", "R1", "S1"}


def test_manifest_binds_complete_checkpoint_and_current_receipt_trees() -> None:
    protocol_path = Path(
        "experiments/historical_recovery/current_recovered_four_arm_protocol.json"
    ).resolve()
    protocol = load_protocol(protocol_path)
    manifest = _manifest(
        protocol_path,
        protocol,
        mode="preflight",
        cases=protocol["preflight_cases"],
        seeds=protocol["preflight_seeds"],
    )

    assert manifest["checkpoint_source"]["file_count"] == 600
    assert manifest["current_e2e_source"]["file_count"] == 600
    assert len(manifest["checkpoint_source"]["tree_sha256"]) == 64
    assert len(manifest["current_e2e_source"]["tree_sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64


def test_representative_recovered_lanes_authorize_matrix_execution() -> None:
    summary_path = REPOSITORY_ROOT / "artifacts/recovered_actions_fixed_action_v1/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["fixed_action_gate_passed"] is True
    assert summary["selector_execution_authorized"] is True
    assert summary["historical_level_pass_count"] == 4
    assert {row["action"] for row in summary["rows"]} == set(ACTION_NAMES)


def test_grouped_folds_keep_every_case_seed_together() -> None:
    records = tuple(
        _outcome(case_id, seed)
        for case_id in ("A1", "E1", "R2", "S2")
        for seed in (117, 118)
    )
    case_folds = build_grouped_folds(records, "leave_one_case_out")
    assert len(case_folds) == 4
    for group, training, test in case_folds:
        assert {records[index].case_id for index in test} == {group}
        assert not ({records[index].case_id for index in training} & {group})
    variant_folds = build_grouped_folds(records, "leave_one_variant_index_out")
    assert len(variant_folds) == 2
    for group, training, test in variant_folds:
        assert {records[index].case_id[-1] for index in test} == {group}
        assert not ({records[index].case_id[-1] for index in training} & {group})


def test_grouped_gate_uses_frozen_regret_limits() -> None:
    metrics = {
        "accuracy": GROUPED_GATE_PARAMETERS["minimum_accuracy"],
        "balanced_accuracy": GROUPED_GATE_PARAMETERS["minimum_balanced_accuracy"],
        "terminal_regret": {
            "mean_log10_regret": GROUPED_GATE_PARAMETERS["maximum_mean_log10_regret"],
            "worst_log10_regret": GROUPED_GATE_PARAMETERS["maximum_worst_log10_regret"],
        },
    }
    assert metrics_pass_gate(metrics) is True
    metrics["terminal_regret"]["worst_log10_regret"] = 0.251
    assert metrics_pass_gate(metrics) is False


def test_grouped_evaluation_rejects_an_incomplete_matrix() -> None:
    records = tuple(_outcome(case_id) for case_id in EXPECTED_AOB_CASES)
    with pytest.raises(ValueError, match="complete AOB-24 by 25-seed matrix"):
        build_grouped_evaluation(records)
