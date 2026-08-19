from __future__ import annotations

from types import SimpleNamespace

from experiments.historical_recovery import run_smp_hcc_lane as lane


def test_lane_matrix_and_case_mapping() -> None:
    targets = [(case, seed) for case in lane.CASES for seed in lane.SEEDS]

    assert len(targets) == 150
    assert len(set(targets)) == 150
    runner = lane._configure_runner(SimpleNamespace())
    assert runner.CASE_IDS == {
        "E1": 1,
        "E2": 2,
        "E3": 3,
        "E4": 4,
        "E5": 5,
        "E6": 6,
    }
    assert runner.RUNNER_PATH == lane.PARSER_WRAPPER


def test_authorization_binds_partial_exact_archive_and_full_gate() -> None:
    authorization = lane.build_authorization()

    assert authorization["target"]["trajectory_count"] == 150
    assert authorization["archived_exact_reference_count"] == 10
    assert authorization["prior_v1_valid_reuse_count"] == 50
    assert authorization["prior_v1_parser_failure_count"] == 100
    assert authorization["target"]["seeds"] == list(range(117, 142))
    assert authorization["max_jobs"] == 24
    assert authorization["reference_means"]["E1"]["displayed_mean"] == "5.69E+05"
    assert authorization["reference_means"]["E6"]["displayed_mean"] == "3.19E+07"


def test_representative_replay_is_exact_and_valid() -> None:
    summary = lane._read_json(lane.representative.OUTPUT_ROOT / "reproduction_summary.json")

    assert lane._representative_verified() is True
    assert summary["integrity_passed"] is True
    assert summary["exact_historical_value_match"] is True
    assert summary["final_error"] == lane.representative.HISTORICAL_FINAL_ERROR
