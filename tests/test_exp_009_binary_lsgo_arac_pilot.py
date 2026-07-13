import csv
import json
from pathlib import Path

from arac.evidence import FORBIDDEN_RUNTIME_FIELDS
from experiments.exp_009_binary_lsgo_arac_pilot.run import run_pilot


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_pilot_writes_three_same_budget_lanes_for_all_cases(tmp_path: Path):
    output = run_pilot(tmp_path / "pilot", total_fes=40)
    results = read_csv(output / "execution_results.csv")
    ledger = read_csv(output / "same_budget_ledger.csv")
    evidence = read_csv(output / "runtime_evidence.csv")
    trace = read_csv(output / "action_trace.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(results) == len(ledger) == len(evidence) == len(trace) == 54
    assert {row["lane_id"] for row in results} == {
        "native_baseline",
        "arac_policy",
        "shuffled_evidence_negative_control",
    }
    assert {row["problem_id"] for row in results} == {
        f"BLSGO-F{index:02d}" for index in range(1, 19)
    }
    assert {row["total_fe"] for row in ledger} == {"40"}
    assert {row["same_budget_violation"] for row in ledger} == {"0"}
    assert all(row["claim_allowed"] == "0" for row in results)
    assert manifest["benchmark_case_count"] == 18
    assert manifest["lane_count"] == 3


def test_pilot_is_byte_deterministic_and_keeps_runtime_boundary(tmp_path: Path):
    first = run_pilot(tmp_path / "first", total_fes=40)
    second = run_pilot(tmp_path / "second", total_fes=40)
    artifact_names = (
        "execution_results.csv",
        "action_trace.csv",
        "same_budget_ledger.csv",
        "runtime_evidence.csv",
        "manifest.json",
    )
    for artifact_name in artifact_names:
        assert (first / artifact_name).read_bytes() == (second / artifact_name).read_bytes()

    with (first / "runtime_evidence.csv").open(newline="", encoding="utf-8") as handle:
        field_names = set(csv.DictReader(handle).fieldnames or ())
    assert field_names.isdisjoint(FORBIDDEN_RUNTIME_FIELDS)
    result_rows = read_csv(first / "execution_results.csv")
    assert all(row["claim_allowed"] == "0" for row in result_rows)
    assert all(
        row["lane_id"] != "shuffled_evidence_negative_control"
        or row["claim_allowed"] == "0"
        for row in result_rows
    )
    evidence_rows = read_csv(first / "runtime_evidence.csv")
    evidence_by_lane = {
        (row["problem_id"], row["lane_id"]): row for row in evidence_rows
    }
    assert any(
        evidence_by_lane[(problem_id, "arac_policy")]["priority_spread"]
        != evidence_by_lane[(problem_id, "shuffled_evidence_negative_control")][
            "priority_spread"
        ]
        for problem_id in {row["problem_id"] for row in evidence_rows}
    )
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert len(set(manifest["input_hashes"].values())) == 18
    assert set(manifest["code_hashes"]) == {"benchmark", "backend", "runner"}
