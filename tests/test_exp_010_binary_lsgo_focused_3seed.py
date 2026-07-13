import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import median

import pytest

from arac.evaluation import SameBudgetLedger
from experiments.exp_010_binary_lsgo_focused_3seed.run import (
    FOCUSED_PROBLEM_IDS,
    LANES,
    OPTIMIZER_SEEDS,
    FocusedRunRecord,
    build_case_summaries,
    build_promotion_gate,
    execute_focused_matrix,
    main,
    parse_args,
    run_focused_pilot,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_fixed_protocol_writes_45_same_budget_rows(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 45
    assert tuple(FOCUSED_PROBLEM_IDS) == (
        "BLSGO-F07",
        "BLSGO-F08",
        "BLSGO-F09",
        "BLSGO-F14",
        "BLSGO-F15",
    )
    assert tuple(OPTIMIZER_SEEDS) == (20260713, 20260714, 20260715)
    assert tuple(LANES) == (
        "native_baseline",
        "arac_policy",
        "shuffled_evidence_negative_control",
    )
    assert {(row["problem_id"], int(row["optimizer_seed"])) for row in rows} == {
        (problem_id, seed)
        for problem_id in FOCUSED_PROBLEM_IDS
        for seed in OPTIMIZER_SEEDS
    }
    assert {row["total_fe"] for row in rows} == {"40"}
    assert {row["same_budget_violation"] for row in rows} == {"0"}
    assert manifest["execution_count"] == 45


def test_case_summary_matches_run_rows(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    rows = read_csv(output / "run_results.csv")
    summaries = read_csv(output / "case_summary.csv")
    for summary in summaries:
        gains = [
            float(row["offline_gain_vs_native"])
            for row in rows
            if row["problem_id"] == summary["problem_id"]
            and row["lane_id"] == "arac_policy"
        ]
        assert float(summary["median_relative_gain"]) == pytest.approx(median(gains))
        assert float(summary["minimum_relative_gain"]) == pytest.approx(min(gains))


def test_noncanonical_budget_cannot_pass_promotion_gate(tmp_path: Path):
    output = run_focused_pilot(tmp_path / "pilot", total_fes=40)
    gate = json.loads((output / "promotion_gate.json").read_text(encoding="utf-8"))
    assert gate["canonical_budget"]["passed"] is False
    assert gate["canonical_budget"]["reason"] == "configured_total_fes_is_not_2000"
    assert gate["overall_pass"] is False


def passing_gate_records() -> list[FocusedRunRecord]:
    source_records, _ = execute_focused_matrix(40)
    records: list[FocusedRunRecord] = []
    for record in source_records:
        result = replace(
            record.result,
            ledger=SameBudgetLedger(
                phase_i_fe=400,
                phase_ii_fe=1600,
                budget_limit=2000,
                fresh_execution=True,
            ),
            optimizer_consumed=(
                record.result.lane_id == "arac_policy"
                and record.result.problem_id in {"BLSGO-F08", "BLSGO-F15"}
            ),
        )
        records.append(
            replace(
                record,
                result=result,
                offline_gain_vs_native=(
                    0.01 if record.result.lane_id == "arac_policy" else 0.0
                ),
                forbidden_runtime_fields=(),
                negative_evidence_changed=(
                    record.result.lane_id == "shuffled_evidence_negative_control"
                ),
                claim_allowed=False,
            )
        )
    return records


def test_constructed_records_can_pass_every_promotion_gate():
    records = passing_gate_records()
    gate = build_promotion_gate(
        records,
        build_case_summaries(records),
        configured_total_fes=2000,
    )
    assert gate["overall_pass"] is True


@pytest.mark.parametrize(
    ("failure", "gate_name"),
    [
        ("action_frequency", "target_action_frequency"),
        ("median_gain", "target_action_median_gain"),
        ("catastrophic", "no_catastrophic_loss"),
        ("same_budget", "same_budget"),
        ("runtime_boundary", "runtime_boundary"),
        ("negative_control", "negative_control"),
    ],
)
def test_each_promotion_gate_fails_independently(failure: str, gate_name: str):
    records = passing_gate_records()
    if failure == "action_frequency":
        changed = 0
        for index, record in enumerate(records):
            if (
                record.result.problem_id == "BLSGO-F08"
                and record.result.lane_id == "arac_policy"
                and changed < 2
            ):
                records[index] = replace(
                    record,
                    result=replace(record.result, optimizer_consumed=False),
                )
                changed += 1
    elif failure == "median_gain":
        records = [
            replace(record, offline_gain_vs_native=-0.01)
            if record.result.problem_id == "BLSGO-F08"
            and record.result.lane_id == "arac_policy"
            else record
            for record in records
        ]
    elif failure == "catastrophic":
        target = next(
            index
            for index, record in enumerate(records)
            if record.result.problem_id == "BLSGO-F07"
            and record.result.lane_id == "arac_policy"
        )
        records[target] = replace(records[target], offline_gain_vs_native=-0.20)
    elif failure == "same_budget":
        records[0] = replace(
            records[0],
            result=replace(
                records[0].result,
                ledger=SameBudgetLedger(400, 1599, 2000, True),
            ),
        )
    elif failure == "runtime_boundary":
        records[0] = replace(records[0], forbidden_runtime_fields=("final_error",))
    elif failure == "negative_control":
        target = next(
            index
            for index, record in enumerate(records)
            if record.result.lane_id == "shuffled_evidence_negative_control"
        )
        records[target] = replace(records[target], negative_evidence_changed=False)

    gate = build_promotion_gate(
        records,
        build_case_summaries(records),
        configured_total_fes=2000,
    )
    assert gate[gate_name]["passed"] is False
    assert gate["overall_pass"] is False


def test_focused_pilot_artifacts_are_byte_deterministic(tmp_path: Path):
    first = run_focused_pilot(tmp_path / "first", total_fes=40)
    second = run_focused_pilot(tmp_path / "second", total_fes=40)
    for filename in (
        "run_results.csv",
        "case_summary.csv",
        "promotion_gate.json",
        "manifest.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_cli_parses_output_and_runs_test_budget(tmp_path: Path):
    output = tmp_path / "cli"
    args = parse_args(["--output-dir", str(output), "--total-fes", "40"])
    assert Path(args.output_dir) == output
    assert args.total_fes == 40
    assert main(["--output-dir", str(output), "--total-fes", "40"]) == output
    assert (output / "promotion_gate.json").is_file()


@pytest.mark.parametrize("invalid_budget", [0, 1, True, 1.5, "40"])
def test_runner_rejects_invalid_budget(tmp_path: Path, invalid_budget):
    with pytest.raises(ValueError, match="total_fes"):
        run_focused_pilot(tmp_path / "invalid", total_fes=invalid_budget)
