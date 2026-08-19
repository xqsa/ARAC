from __future__ import annotations

from experiments.historical_recovery import run_ctp_hcc_isolated_replay as representative
from experiments.historical_recovery import run_ctp_hcc_lane as lane


def test_lane_matrix_and_commands_match_exp058() -> None:
    targets = [(case, seed) for case in lane.CASES for seed in lane.SEEDS]

    assert len(targets) == 150
    assert len(set(targets)) == 150
    command = representative._trajectory_command(
        case="S6",
        seed=141,
        output_root=lane._trajectory_root("S6", 141),
    )
    assert command[command.index("--ids") + 1] == "6"
    assert command[command.index("--seed") + 1] == "141"
    assert command[command.index("--max-fes") + 1] == "3000000"
    assert command[command.index("--s-series-action") + 1] == "ctp_stable"
    assert command[command.index("--timestamp") + 1].endswith("-s6-seed141")


def test_representative_evidence_is_exact() -> None:
    evidence = lane._validate_evidence(
        "S1",
        117,
        lane.REPRESENTATIVE_RESULT_DIRECTORY,
        representative.OUTPUT_ROOT / "stderr.log",
    )

    assert evidence["trajectory_exact"] is True
    assert evidence["final_error"] == 0.005747882589844738
    assert evidence["exact_budget_match"] is True


def test_lane_authorization_binds_full_historical_matrix() -> None:
    authorization = lane.build_authorization()

    assert authorization["target"]["trajectory_count"] == 150
    assert authorization["target"]["cases"] == list(lane.CASES)
    assert authorization["target"]["seeds"] == list(range(117, 142))
    assert authorization["target"]["max_fes"] == 3_000_000
    assert authorization["max_jobs"] == 24
    assert authorization["runner_sha256"] == representative.EXACT_RUNNER_SHA256
