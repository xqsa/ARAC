from __future__ import annotations

from pathlib import Path

import pytest

from experiments.historical_recovery.run_aor_hcc_isolated_replay import (
    AOR_CASES,
    EXACT_ACTION_SHA256,
    HISTORICAL_SEEDS,
    LANE_OUTPUT_ROOT,
    OUTPUT_ROOT,
    _field_checks,
    _input_byte_hash_match,
    _read_json,
    build_preflight,
    recover_exact_action_source,
    verify_lane,
    verify_reproduction,
)


def test_aor_hcc_preflight_is_scoped_to_one_historical_trajectory() -> None:
    preflight = build_preflight()

    assert preflight["authorization_scope"] == "one_trajectory_only"
    assert preflight["target"] == {
        "case": "A1",
        "seed": 117,
        "max_fes": 3_000_000,
    }
    assert preflight["historical_reference"]["final_error"] == 78047.92159464832
    assert preflight["source"]["action_sha256"] == EXACT_ACTION_SHA256
    assert preflight["source"]["action_reversed_patch_count"] == 4
    assert preflight["receipt_environment_binding"] is False


def test_recovered_aor_action_source_has_expected_hash() -> None:
    source, provenance = recover_exact_action_source()

    assert len(source) == 12_203
    assert provenance["sha256"] == EXACT_ACTION_SHA256
    assert b"def execute_aor_action(" in source
    assert b"from arac.actions.gcb import CANONICAL_SEP_CMA_PARAMETERS_HASH" in source


def test_aor_lane_matrix_and_archived_summary_comparator_are_frozen() -> None:
    assert AOR_CASES == ("A1", "A2", "A3", "A4", "A5", "A6")
    assert HISTORICAL_SEEDS == tuple(range(117, 142))
    archived = _read_json(
        Path("results/exp_057_a_series_aor_25seed/a1-a6-25seed-v1/")
        / "runs/A1/seed_117/run_summary.json"
    )

    assert all(_field_checks(archived, archived).values())
    changed = {**archived, "final_error": archived["final_error"] + 1.0}
    assert _field_checks(changed, archived)["final_error"] is False
    reformatted = {**archived, "aob_data_sha256": "0" * 64}
    assert all(_field_checks(reformatted, archived).values())
    assert _input_byte_hash_match(reformatted, archived) is False


@pytest.mark.skipif(
    not (Path(OUTPUT_ROOT) / "reproduction_summary.json").is_file(),
    reason="isolated AOR reproduction has not been executed",
)
def test_completed_aor_hcc_reproduction_verifies_exactly() -> None:
    verification = verify_reproduction()

    assert verification["verification_passed"] is True
    assert verification["checks"]["all_exact_fields_match"] is True


@pytest.mark.skipif(
    not (Path(LANE_OUTPUT_ROOT) / "lane_summary.json").is_file(),
    reason="full AOR HCC lane has not been executed",
)
def test_completed_aor_hcc_lane_verifies_exactly() -> None:
    verification = verify_lane()

    assert verification["verification_passed"] is True
    assert verification["exact_match_count"] == 150
