import csv
import json
from pathlib import Path

import pytest

from experiments.exp_011_binary_lsgo_diagnostic.run import (
    DIAGNOSTIC_PROBLEM_IDS,
    LANES,
    OPTIMIZER_SEEDS,
    classify_diagnostic_signals,
    main,
    parse_args,
    run_diagnostic,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((3, 3, 2, 5), "optimizer_limited"),
        ((2, 3, 3, 3), "policy_limited"),
        ((3, 3, 3, 3), "mixed"),
        ((2, 2, 2, 2), "inconclusive"),
    ],
)
def test_classification_covers_all_reachable_labels(counts, expected):
    assert classify_diagnostic_signals(*counts).label == expected


def test_fixed_protocol_writes_40_same_budget_rows(tmp_path: Path):
    output = run_diagnostic(tmp_path / "diagnostic", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    summaries = read_csv(output / "case_summary.csv")
    diagnosis = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))

    assert tuple(DIAGNOSTIC_PROBLEM_IDS) == ("BLSGO-F08", "BLSGO-F15")
    assert tuple(OPTIMIZER_SEEDS) == (
        20260713,
        20260714,
        20260715,
        20260716,
        20260717,
    )
    assert tuple(LANES) == (
        "native_single_bit",
        "native_group_block",
        "forced_isolate",
        "arac_policy",
    )
    assert len(rows) == 40
    assert len(summaries) == 2
    assert {row["total_fe"] for row in rows} == {"40"}
    assert {row["same_budget_violation"] for row in rows} == {"0"}
    assert {row["claim_allowed"] for row in rows} == {"0"}
    assert len(diagnosis["case_diagnoses"]) == 2


def test_lanes_share_initial_state_and_phase_one_objective(tmp_path: Path):
    rows = read_csv(run_diagnostic(tmp_path / "diagnostic", total_fes=40) / "run_results.csv")
    for problem_id in DIAGNOSTIC_PROBLEM_IDS:
        for seed in OPTIMIZER_SEEDS:
            matched = [
                row
                for row in rows
                if row["problem_id"] == problem_id
                and int(row["optimizer_seed"]) == seed
            ]
            assert len({row["initial_vector_hash"] for row in matched}) == 1
            assert len({row["phase_one_objective"] for row in matched}) == 1


def test_operator_and_action_identities_remain_separate(tmp_path: Path):
    rows = read_csv(run_diagnostic(tmp_path / "diagnostic", total_fes=40) / "run_results.csv")
    by_lane = {lane: [row for row in rows if row["lane_id"] == lane] for lane in LANES}

    assert {row["proposal_operator"] for row in by_lane["native_group_block"]} == {
        "group_block"
    }
    assert {row["selected_action_name"] for row in by_lane["native_group_block"]} == {
        "conservative_no_action"
    }
    assert {row["proposal_operator"] for row in by_lane["forced_isolate"]} == {
        "single_bit"
    }
    assert {row["selected_action_name"] for row in by_lane["forced_isolate"]} == {
        "isolate_conflicting_relation"
    }


def test_artifacts_are_byte_deterministic(tmp_path: Path):
    first = run_diagnostic(tmp_path / "first", total_fes=40)
    second = run_diagnostic(tmp_path / "second", total_fes=40)
    for filename in (
        "run_results.csv",
        "case_summary.csv",
        "diagnosis.json",
        "manifest.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_cli_parses_output_and_runs_test_budget(tmp_path: Path):
    output = tmp_path / "cli"
    args = parse_args(["--output-dir", str(output), "--total-fes", "40"])
    assert Path(args.output_dir) == output
    assert args.total_fes == 40
    assert main(["--output-dir", str(output), "--total-fes", "40"]) == output
    assert (output / "diagnosis.json").is_file()


@pytest.mark.parametrize("invalid_budget", [0, 1, True, 1.5, "40"])
def test_runner_rejects_invalid_budget(tmp_path: Path, invalid_budget):
    with pytest.raises(ValueError, match="total_fes"):
        run_diagnostic(tmp_path / "invalid", total_fes=invalid_budget)
