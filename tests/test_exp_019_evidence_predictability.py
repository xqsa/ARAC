from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from arac.policy.action_ceiling import ACTION_CEILING_ARMS, best_action_ceiling_arm
from experiments.pilots.exp_019_conflict_resolution_pilot.evidence_predictability import (
    FORBIDDEN_MODEL_FIELDS,
    DiagnosticDataset,
    _best_available_arm,
    _ordered_prediction_indices,
    _validate_oof_coverage,
    build_feature_frames,
    crossfit_r4_s5_pairwise,
    crossfit_value_model,
    leave_one_seed_out_splits,
    r4_s5_cluster_permutation_test,
    run_evidence_predictability,
    summarize_pairwise,
)


def _context(
    *,
    case: str,
    seed: int,
    context_id: str,
    shared_count: int,
    shared_offset: int = 0,
    stagnation: int = 0,
) -> dict[str, object]:
    shared = "-".join(str(shared_offset + index) for index in range(shared_count))
    anchor = [float(index + 1) for index in range(shared_count)]
    return {
        "problem_id": case,
        "seed": str(seed),
        "context_id": context_id,
        "relation_id": f"g0-1:v{shared}",
        "group_index": "1",
        "population_sizes": "[16, 19, 16]",
        "uniform_group_budgets": "[100, 120, 100]",
        "efficiency_ewma": "[1.0, 2.0, 3.0]",
        "stagnation_streaks": f"[{stagnation}, {stagnation}, {stagnation}]",
        "anchor_values": str(anchor),
        "left_values": str([value + 0.2 for value in anchor]),
        "right_values": str([value - 0.1 for value in anchor]),
        "bridge_values": str([value + 0.05 for value in anchor]),
        "bridge_weights": '{"left_owner": 0.5, "right_owner": 0.5}',
        "selector_arm": "native_eq8",
    }


def _dataset(contexts: list[dict[str, object]], deltas: np.ndarray) -> DiagnosticDataset:
    frame = pd.DataFrame(contexts)
    frame.index = pd.Index([str(row["context_id"]) for row in contexts])
    delta_frame = pd.DataFrame(deltas, index=frame.index, columns=ACTION_CEILING_ARMS)
    winners = []
    winner_deltas = []
    for row in deltas:
        winner, value = best_action_ceiling_arm(
            dict(zip(ACTION_CEILING_ARMS, row, strict=True))
        )
        winners.append(winner)
        winner_deltas.append(value)
    return DiagnosticDataset(
        contexts=frame,
        deltas=delta_frame,
        oracle_arms=tuple(winners),
        oracle_deltas=np.asarray(winner_deltas),
    )


def test_oracle_tie_break_prefers_native() -> None:
    deltas = {arm: -1.0 for arm in ACTION_CEILING_ARMS}
    deltas["native_eq8"] = 0.0
    deltas["gcb"] = 5e-16

    assert best_action_ceiling_arm(deltas) == ("native_eq8", 0.0)

    scores = np.full(len(ACTION_CEILING_ARMS), -1.0)
    scores[ACTION_CEILING_ARMS.index("native_eq8")] = 0.0
    scores[ACTION_CEILING_ARMS.index("gcb")] = 5e-16
    assert _ordered_prediction_indices(scores)[0] == 0
    assert _best_available_arm(
        {"native_eq8": 0.0, "gcb": 5e-16}
    ) == "native_eq8"


def test_analysis_refuses_to_overwrite_source_artifact_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot overwrite source artifacts"):
        run_evidence_predictability(tmp_path, tmp_path)


def test_leave_one_seed_out_keeps_case_seed_clusters_whole() -> None:
    contexts = pd.DataFrame(
        [
            {"problem_id": case, "seed": seed, "context": context}
            for case in ("E3", "R4")
            for seed in (117, 118, 119)
            for context in range(4)
        ]
    )

    splits = leave_one_seed_out_splits(contexts)

    assert [fold for fold, _, _ in splits] == [117, 118, 119]
    for fold_seed, train, test in splits:
        assert set(contexts.iloc[test]["seed"]) == {fold_seed}
        assert fold_seed not in set(contexts.iloc[train]["seed"])
        assert contexts.iloc[test].groupby(["problem_id", "seed"]).size().eq(4).all()


def test_oof_integrity_gate_rejects_duplicate_predictor_context_pair() -> None:
    valid = pd.DataFrame(
        [
            {"predictor": predictor, "context_id": context, "seed": seed, "fold_seed": seed}
            for predictor in ("first", "second")
            for context, seed in (("a", 117), ("b", 118))
        ]
    )
    assert _validate_oof_coverage(
        valid,
        context_ids=("a", "b"),
        expected_predictors=("first", "second"),
    )["passed"] == 1

    duplicated = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        _validate_oof_coverage(
            duplicated,
            context_ids=("a", "b"),
            expected_predictors=("first", "second"),
        )

    with pytest.raises(RuntimeError, match="predictor set drifted"):
        _validate_oof_coverage(
            valid[valid["predictor"] == "first"],
            context_ids=("a", "b"),
            expected_predictors=("first", "second"),
        )


def test_feature_frames_ignore_raw_shared_ids_and_dispatch_stagnation() -> None:
    first = _context(
        case="A4",
        seed=117,
        context_id="first",
        shared_count=3,
        shared_offset=0,
        stagnation=0,
    )
    second = _context(
        case="R4",
        seed=118,
        context_id="second",
        shared_count=3,
        shared_offset=700,
        stagnation=99,
    )
    deltas = np.zeros((2, len(ACTION_CEILING_ARMS)))
    frames = build_feature_frames(_dataset([first, second], deltas))

    for frame in frames.values():
        assert np.allclose(frame.iloc[0], frame.iloc[1])
        assert not any(
            forbidden in name
            for name in frame.columns
            for forbidden in FORBIDDEN_MODEL_FIELDS
        )


def test_nested_value_crossfit_produces_one_oof_prediction_per_context() -> None:
    contexts = []
    rows = []
    for case_index, case in enumerate(("E3", "R4")):
        for seed in (117, 118, 119):
            for relation in range(2):
                contexts.append(
                    _context(
                        case=case,
                        seed=seed,
                        context_id=f"{case}-{seed}-{relation}",
                        shared_count=3 + 2 * case_index,
                    )
                )
                delta = np.full(len(ACTION_CEILING_ARMS), -0.1)
                delta[0] = 0.0
                delta[10 if case == "E3" else 12] = 0.5
                rows.append(delta)
    dataset = _dataset(contexts, np.asarray(rows))
    features = build_feature_frames(dataset)["topology"]

    oof = crossfit_value_model(
        dataset,
        features,
        kind="ridge_value",
        candidates=(0.1, 1.0),
        feature_set="topology",
    )

    assert len(oof) == len(contexts)
    assert oof["context_id"].nunique() == len(contexts)
    assert (oof["seed"] == oof["fold_seed"]).all()
    assert (oof["runtime_authorized"] == 0).all()
    assert (oof["selected_delta"] == 0.5).all()


def test_r4_s5_shared_count_permutation_uses_case_seed_blocks() -> None:
    contexts = []
    rows = []
    for case, shared_count in (("R4", 5), ("S5", 7)):
        for seed in (117, 118, 119):
            for relation in range(2):
                contexts.append(
                    _context(
                        case=case,
                        seed=seed,
                        context_id=f"{case}-{seed}-{relation}",
                        shared_count=shared_count,
                    )
                )
                delta = np.zeros(len(ACTION_CEILING_ARMS))
                delta[ACTION_CEILING_ARMS.index("gcb")] = (
                    0.5 if case == "R4" else -0.5
                )
                delta[ACTION_CEILING_ARMS.index("efficiency_budget_reallocation")] = (
                    -0.5 if case == "R4" else 0.5
                )
                rows.append(delta)
    dataset = _dataset(contexts, np.asarray(rows))
    topology = build_feature_frames(dataset)["topology"]

    result = r4_s5_cluster_permutation_test(
        dataset,
        topology,
        permutations=99,
        seed=7,
    )

    assert result["observed_balanced_accuracy"] == 1.0
    assert result["permutations"] == 99
    assert math.isclose(
        float(result["p_value"]),
        (int(result["exceedances"]) + 1) / 100,
    )


def test_r4_s5_primary_routes_frozen_beneficial_actions_not_context_oracle() -> None:
    contexts = []
    rows = []
    for case, shared_count in (("R4", 5), ("S5", 7)):
        for seed in (117, 118, 119):
            contexts.append(
                _context(
                    case=case,
                    seed=seed,
                    context_id=f"{case}-{seed}",
                    shared_count=shared_count,
                )
            )
            delta = np.zeros(len(ACTION_CEILING_ARMS))
            delta[ACTION_CEILING_ARMS.index("gcb")] = (
                0.6 if case == "R4" or seed == 119 else 0.4
            )
            delta[ACTION_CEILING_ARMS.index("efficiency_budget_reallocation")] = (
                -0.2 if case == "R4" else 0.5
            )
            rows.append(delta)
    dataset = _dataset(contexts, np.asarray(rows))
    features = build_feature_frames(dataset)

    oof = crossfit_r4_s5_pairwise(dataset, features, logistic_cs=(0.1, 1.0))
    stump = oof[oof["predictor"] == "shared_count_stump"]
    summary = summarize_pairwise(stump, bootstrap_replicates=50, bootstrap_seed=7)

    assert stump["correct"].eq(1).all()
    assert set(stump[stump["problem_id"] == "R4"]["predicted_arm"]) == {
        "gcb"
    }
    assert set(stump[stump["problem_id"] == "S5"]["predicted_arm"]) == {
        "efficiency_budget_reallocation"
    }
    assert stump["preference_correct"].sum() == len(stump) - 1
    overall = summary[summary["scope"] == "all"].iloc[0]
    assert overall["routing_accuracy"] == 1.0
    assert overall["positive_rate"] == 1.0
    assert overall["catastrophic_count"] == 0
