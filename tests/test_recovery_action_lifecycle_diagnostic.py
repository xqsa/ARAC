from __future__ import annotations

import math

from experiments.historical_recovery.recovery_action_lifecycle_diagnostic import (
    EXPECTED_ACTIONS,
    EXPECTED_CASES,
    EXPECTED_SEEDS,
    EXPECTED_VARIANTS,
    _paired_ratio,
    load_protocol,
    summarize,
)


def _row(case_id: str, seed: int, variant: str, error: float) -> dict[str, object]:
    return {
        "case_id": case_id,
        "run_seed": seed,
        "variant": variant,
        "checkpoint_hash": f"{case_id}-{seed}",
        "terminal_fes": 3_000_000,
        "final_error": error,
        "route": f"{variant}-route",
    }


def test_protocol_freezes_expected_diagnostic_matrix() -> None:
    protocol = load_protocol()

    assert tuple(protocol["cases"]) == EXPECTED_CASES
    assert tuple(protocol["seeds"]) == EXPECTED_SEEDS
    assert tuple(protocol["variants"]) == EXPECTED_VARIANTS
    assert protocol["max_workers"] == 24
    assert protocol["reference_thresholds_used_for_decision"] is False
    assert protocol["patch_enabled"] is False
    assert protocol["soft_routing_enabled"] is False
    assert protocol["selector_enabled"] is False
    assert protocol["action_by_case"] == EXPECTED_ACTIONS


def test_paired_ratio_handles_zero_historical_error() -> None:
    assert _paired_ratio(0.0, 0.0) == 1.0
    assert _paired_ratio(1.0, 0.0) is None
    assert math.isclose(_paired_ratio(12.0, 3.0) or 0.0, 4.0)


def test_summary_requires_same_checkpoint_and_reports_direction() -> None:
    rows = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            rows.append(_row(case_id, seed, "current", 20.0))
            rows.append(_row(case_id, seed, "historical_compatible", 10.0))

    summary = summarize(rows, load_protocol())

    assert summary["context_count"] == 110
    assert summary["pair_count"] == 55
    assert summary["same_checkpoint_per_pair"] is True
    assert summary["exact_terminal_fes"] is True
    assert summary["diagnostic_conclusion"]["performance_superiority_claim_authorized"] is False
    assert all(row["historical_compatible_better"] for row in summary["pairs"])
    assert all(
        math.isclose(case["current_to_historical_geometric_mean_ratio"], 2.0)
        for case in summary["case_summaries"]
    )


def test_summary_detects_checkpoint_mismatch_in_paired_rows() -> None:
    rows = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            rows.append(_row(case_id, seed, "current", 20.0))
            row = _row(case_id, seed, "historical_compatible", 10.0)
            if case_id == "E2" and seed == 117:
                row["checkpoint_hash"] = "different"
            rows.append(row)

    summary = summarize(rows, load_protocol())

    assert summary["same_checkpoint_per_pair"] is False
