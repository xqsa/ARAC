from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest


def test_exp_005_cli_defaults_to_3m_fe_canonical_controller() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
        ]
    )

    assert args.max_fes == 3_000_000
    assert args.lane_profile == "canonical_evidence_controller_v1"
    assert args.seeds == [1, 2, 3]
    assert args.problems == [
        "E1",
        "E2",
        "E3",
        "E4",
        "E6",
        "S2",
        "S3",
        "S6",
        "R1",
        "R2",
        "R3",
        "A4",
        "A5",
    ]
    assert str(args.aob_data_root).endswith("vendor\\hcc\\AOB\\AOBG\\datafile")


def test_exp_005_canonical_cli_has_no_offline_threshold_arguments() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(["--output-dir", "out", "--hcc-root", "E:/HCC-main"])

    assert "paper_best_matrix" not in vars(args)
    assert "paper_best" not in vars(args)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "out",
                "--hcc-root",
                "E:/HCC-main",
                "--paper-best-matrix",
                "offline.csv",
            ]
        )


def test_exp_005_canonical_request_and_subprocess_command_are_reference_blind(
    tmp_path: Path,
) -> None:
    from arac.backends.hcc import (
        HccAobExecutionRequest,
        build_hcc_aob_smoke_command,
    )

    request = HccAobExecutionRequest(
        problem_id="R3",
        seed=3,
        max_fes=3_000_000,
        output_dir=tmp_path / "canonical-r3",
        arac_action="arac_evidence_action_controller_v31",
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
    )
    command = build_hcc_aob_smoke_command(request)
    forbidden_tokens = {
        "paper_best",
        "historical",
        "final_error",
        "relative_gain",
        "problem_family",
        "function_family",
        "328000",
    }
    request_payload = repr(asdict(request)).lower()
    command_payload = "\n".join(command.argv).lower()

    assert forbidden_tokens.isdisjoint(HccAobExecutionRequest.__dataclass_fields__)
    assert all(token not in request_payload for token in forbidden_tokens)
    assert all(token not in command_payload for token in forbidden_tokens)


def test_exp_005_cli_accepts_landscape_escape_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "--lane-profile",
            "landscape_escape",
        ]
    )

    assert args.lane_profile == "landscape_escape"
    assert args.problems == ["A1", "A2", "A3", "A4", "A5", "A6"]


def test_exp_005_cli_accepts_repair_landscape_escape_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "A5",
            "--lane-profile",
            "repair_landscape_escape",
        ]
    )

    assert args.lane_profile == "repair_landscape_escape"
    assert args.problems == ["A5"]


def test_exp_005_cli_accepts_repair_refine_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "A5",
            "--lane-profile",
            "repair_refine",
        ]
    )

    assert args.lane_profile == "repair_refine"
    assert args.problems == ["A5"]


def test_exp_005_cli_accepts_evidence_routed_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "--lane-profile",
            "evidence_routed_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_only"
    assert args.problems == ["E5", "R4", "S5"]


def test_exp_005_cli_accepts_evidence_routed_v2_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v2_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v2_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_evidence_routed_v21_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v21_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v21_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_evidence_routed_v22_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v22_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v22_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_evidence_routed_v23_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v23_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v23_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_evidence_routed_v24_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v24_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v24_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_evidence_routed_v25_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v25_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v25_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_paper_best_win_push_v2_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E6",
            "S6",
            "A5",
            "--lane-profile",
            "paper_best_win_push_v2",
        ]
    )

    assert args.lane_profile == "paper_best_win_push_v2"
    assert args.problems == ["E6", "S6", "A5"]


def test_exp_005_cli_accepts_historical_anchor_refine_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E2",
            "R2",
            "R3",
            "--lane-profile",
            "historical_anchor_refine_push",
        ]
    )

    assert args.lane_profile == "historical_anchor_refine_push"
    assert args.problems == ["E2", "R2", "R3"]


def test_exp_005_cli_accepts_historical_13_preserve_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E1",
            "R3",
            "S6",
            "A5",
            "--lane-profile",
            "historical_13_preserve_push",
        ]
    )

    assert args.lane_profile == "historical_13_preserve_push"
    assert args.problems == ["E1", "R3", "S6", "A5"]


def test_exp_005_cli_accepts_historical_13_fast_preserve_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E1",
            "E2",
            "R3",
            "A5",
            "--lane-profile",
            "historical_13_fast_preserve",
        ]
    )

    assert args.lane_profile == "historical_13_fast_preserve"
    assert args.problems == ["E1", "E2", "R3", "A5"]


def test_exp_005_cli_accepts_historical_13_runtime_composite_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E1",
            "E2",
            "R3",
            "A5",
            "--lane-profile",
            "historical_13_runtime_composite",
        ]
    )

    assert args.lane_profile == "historical_13_runtime_composite"
    assert args.problems == ["E1", "E2", "R3", "A5"]


def test_exp_005_cli_accepts_historical_13_runtime_composite_v2_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E6",
            "S6",
            "--lane-profile",
            "historical_13_runtime_composite_v2",
        ]
    )

    assert args.lane_profile == "historical_13_runtime_composite_v2"
    assert args.problems == ["E6", "S6"]


def test_exp_005_cli_accepts_evidence_action_controller_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E2",
            "R3",
            "S6",
            "A5",
            "--lane-profile",
            "evidence_action_controller_v1",
        ]
    )

    assert args.lane_profile == "evidence_action_controller_v1"
    assert args.problems == ["E2", "R3", "S6", "A5"]


def test_exp_005_cli_accepts_evidence_action_controller_v2_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "S6",
            "R3",
            "A5",
            "--lane-profile",
            "evidence_action_controller_v2",
        ]
    )

    assert args.lane_profile == "evidence_action_controller_v2"
    assert args.problems == ["S6", "R3", "A5"]


def test_exp_005_cli_accepts_evidence_action_controller_v3_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "S6",
            "R3",
            "A5",
            "--lane-profile",
            "evidence_action_controller_v3",
        ]
    )

    assert args.lane_profile == "evidence_action_controller_v3"
    assert args.problems == ["S6", "R3", "A5"]


def test_exp_005_cli_accepts_evidence_action_controller_v31_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "S6",
            "R3",
            "A5",
            "--lane-profile",
            "evidence_action_controller_v31",
        ]
    )

    assert args.lane_profile == "evidence_action_controller_v31"
    assert args.problems == ["S6", "R3", "A5"]


def test_exp_005_cli_accepts_evidence_routed_v26_only_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E5",
            "R4",
            "S5",
            "A5",
            "--lane-profile",
            "evidence_routed_v26_only",
        ]
    )

    assert args.lane_profile == "evidence_routed_v26_only"
    assert args.problems == ["E5", "R4", "S5", "A5"]


def test_exp_005_cli_accepts_paper_best_win_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "E6",
            "A1",
            "A2",
            "A3",
            "A5",
            "A6",
            "--lane-profile",
            "paper_best_win_push",
        ]
    )

    assert args.lane_profile == "paper_best_win_push"
    assert args.problems == ["E6", "A1", "A2", "A3", "A5", "A6"]


def test_exp_005_cli_accepts_precision_refine_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "A1",
            "A2",
            "A3",
            "A6",
            "--lane-profile",
            "precision_refine_push",
        ]
    )

    assert args.lane_profile == "precision_refine_push"
    assert args.problems == ["A1", "A2", "A3", "A6"]


def test_exp_005_cli_accepts_phase_rescue_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "R3",
            "R4",
            "E5",
            "--lane-profile",
            "phase_rescue_push",
        ]
    )

    assert args.lane_profile == "phase_rescue_push"
    assert args.problems == ["R3", "R4", "E5"]


def test_exp_005_cli_accepts_repair_phase_rescue_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "R3",
            "R4",
            "E5",
            "--lane-profile",
            "repair_phase_rescue_push",
        ]
    )

    assert args.lane_profile == "repair_phase_rescue_push"
    assert args.problems == ["R3", "R4", "E5"]


def test_exp_005_cli_accepts_cc_harm_sep_refresh_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "R3",
            "R4",
            "R5",
            "R6",
            "--lane-profile",
            "cc_harm_sep_refresh",
        ]
    )

    assert args.lane_profile == "cc_harm_sep_refresh"
    assert args.problems == ["R3", "R4", "R5", "R6"]


def test_exp_005_cli_accepts_separable_cmaes_push_profile() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--hcc-root",
            "E:/HCC-main",
            "--problems",
            "R5",
            "R6",
            "--lane-profile",
            "separable_cmaes_push",
        ]
    )

    assert args.lane_profile == "separable_cmaes_push"
    assert args.problems == ["R5", "R6"]


def test_exp_005_cli_accepts_diagonal_search_state_backend() -> None:
    from experiments.final.exp_005_hcc_final_protocol_pilot.run import parse_args

    args = parse_args(
        [
            "--output-dir",
            "out",
            "--problems",
            "R3",
            "--search-state-backend",
            "diagonal_cma",
        ]
    )

    assert args.search_state_backend == "diagonal_cma"
