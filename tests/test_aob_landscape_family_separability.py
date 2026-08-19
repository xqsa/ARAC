from __future__ import annotations

import numpy as np

from experiments.historical_recovery.aob_landscape_family_separability import (
    load_dataset,
    render_report,
    run_audit,
    variant_folds,
)


def test_variant_folds_hold_out_every_family_at_one_unseen_index() -> None:
    cases = [
        (f"{family}{variant}", variant)
        for variant in range(1, 7)
        for family in ("A", "E", "R", "S")
        for _ in range(2)
    ]
    folds = variant_folds([variant for _, variant in cases], range(1, 7))

    for held_out, (train_indices, test_indices) in enumerate(folds, start=1):
        assert len(train_indices) == 40
        assert len(test_indices) == 8
        assert not np.intersect1d(train_indices, test_indices).size
        test_cases = {cases[index][0] for index in test_indices}
        assert test_cases == {f"{family}{held_out}" for family in "AERS"}


def test_repository_dataset_is_complete_identity_blind_and_current_bound() -> None:
    dataset = load_dataset()

    assert dataset.features.shape == (600, 40)
    assert len(set(dataset.case_ids)) == 24
    assert len(set(zip(dataset.case_ids, dataset.seeds, strict=True))) == 600
    assert len(dataset.checkpoint_hashes) == 600
    assert len(dataset.current_receipt_hashes) == 600
    assert np.isfinite(dataset.features).all()
    assert not {"case_id", "family", "run_seed", "selected_action"}.intersection(
        dataset.feature_names
    )


def test_audit_uses_fixed_six_fold_protocol_and_reports_scientific_boundary() -> None:
    audit, predictions = run_audit()

    assert audit["input_audit"]["optimizer_or_objective_evaluations_executed"] is False
    assert audit["primary_feature_group"] == "all_40"
    assert set(audit["feature_groups"]) == {
        "landscape_probe_30",
        "structure_5",
        "progress_5",
        "all_40",
        "landscape_shape_28",
        "all_without_level_38",
    }
    assert len(predictions) == 6 * 600
    for result in audit["feature_groups"].values():
        assert len(result["folds"]) == 6
        assert sum(sum(row) for row in result["confusion_matrix"]) == 600
        assert all(fold["training_context_count"] == 500 for fold in result["folds"])
        assert all(fold["test_context_count"] == 100 for fold in result["folds"])
    mapped = audit["mapped_action_label_selection"]
    assert mapped["family_action_mapping"] == {
        "A": "aor",
        "E": "smp",
        "R": "gcb",
        "S": "ctp",
    }
    assert mapped["accuracy"] == audit["feature_groups"]["all_40"]["accuracy"]
    assert audit["separability_gate"]["passed"] is True
    assert mapped["correct_context_count"] == 587
    assert mapped["case_majority_correct_count"] == 24
    assert audit["feature_groups"]["all_40"]["balanced_accuracy"] == 0.978333333333
    assert audit["feature_groups"]["structure_5"]["balanced_accuracy"] == 0.411666666667
    assert audit["feature_groups"]["all_without_level_38"]["balanced_accuracy"] == (
        0.928333333333
    )
    assert audit["feature_groups"]["all_without_level_38"]["analysis_role"] == (
        "post_hoc_scale_sensitivity"
    )
    report = render_report(audit)
    assert "New optimizer/objective evaluations: **0**" in report
    assert "Does not establish that the historical family-to-action mapping is optimal" in report
