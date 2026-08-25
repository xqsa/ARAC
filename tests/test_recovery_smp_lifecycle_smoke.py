from __future__ import annotations

import math

from experiments.historical_recovery.recovery_smp_lifecycle_smoke import (
    EXPECTED_CASES,
    EXPECTED_SEEDS,
    EXPECTED_VARIANTS,
    load_protocol,
    summarize,
)


def _row(case_id: str, seed: int, variant: str, error: float, route: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "run_seed": seed,
        "variant": variant,
        "checkpoint_hash": f"{case_id}-{seed}",
        "terminal_fes": 3_000_000,
        "final_error": error,
        "current_route": route,
        "route": route,
        "clip_offspring": True,
        "lifecycle_profile": (
            "historical_compatible_smp_v1_clip_offspring_true"
            if variant == "current_recovered"
            else "frozen_historical_smp_source_v1_clip_offspring_true"
        ),
    }


def test_protocol_freezes_25_paired_smp_contexts() -> None:
    protocol = load_protocol()
    assert tuple(protocol["cases"]) == EXPECTED_CASES
    assert tuple(protocol["seeds"]) == EXPECTED_SEEDS
    assert tuple(protocol["variants"]) == EXPECTED_VARIANTS
    assert protocol["max_workers"] == 24
    assert protocol["patch_enabled"] is False


def test_summary_requires_reachable_current_lifecycle() -> None:
    rows = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            rows.append(_row(case_id, seed, "current_recovered", 10.0, "recovered_historical_compatible_smp_v1_clip_offspring_true_stateful_rescue_1_global_polish_2"))
            rows.append(_row(case_id, seed, "historical_compatible", 10.0, "stateful_rescue_1_global_polish_2"))

    summary = summarize(rows, load_protocol())

    assert summary["context_count"] == 50
    assert summary["pair_count"] == 25
    assert summary["smoke_gate_passed"] is True
    assert summary["same_checkpoint_per_pair"] is True
    assert all(math.isclose(case["current_to_historical_geometric_mean_ratio"], 1.0) for case in summary["case_summaries"])


def test_summary_rejects_noop_tail() -> None:
    rows = []
    for case_id in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            rows.append(_row(case_id, seed, "current_recovered", 10.0, "recovered_stateful_noop_10"))
            rows.append(_row(case_id, seed, "historical_compatible", 10.0, "stateful_rescue_1_global_polish_2"))

    assert summarize(rows, load_protocol())["smoke_gate_passed"] is False
