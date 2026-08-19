"""Close the AOB-24 displayed-mean gate from the four recovered HCC lanes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "hcc-action-historical-recovery"
OUTPUT_ROOT = TASK_ROOT / "raw" / "aob24-hcc-historical-gate-v1"
REFERENCE_TABLE = REPOSITORY_ROOT / "output" / "pdf" / "aob_arac_method_comparison_corrected.csv"
REFERENCE_COLUMN = "ARAC Mean +/- Std"
PYTHON_EXECUTABLE = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"
ACTION_ORDER = ("smp", "ctp", "gcb", "aor")
CASE_MAPPING = {
    "smp": tuple(f"E{index}" for index in range(1, 7)),
    "ctp": tuple(f"S{index}" for index in range(1, 7)),
    "gcb": tuple(f"R{index}" for index in range(1, 7)),
    "aor": tuple(f"A{index}" for index in range(1, 7)),
}
LANES = {
    "smp": {
        "summary": TASK_ROOT / "raw" / "smp-e1-e6-25seed-hcc-reproduction-v2" / "lane_summary.json",
        "schema": "arac-exp052-smp-hcc-lane-result-v2",
        "verify": (
            "experiments.historical_recovery.run_smp_hcc_lane",
            "verify",
        ),
    },
    "ctp": {
        "summary": TASK_ROOT / "raw" / "ctp-s1-s6-25seed-hcc-reproduction-v1" / "lane_summary.json",
        "schema": "arac-exp058-ctp-hcc-lane-result-v1",
        "verify": (
            "experiments.historical_recovery.run_ctp_hcc_lane",
            "verify",
        ),
    },
    "gcb": {
        "summary": TASK_ROOT / "raw" / "gcb-r1-r6-25seed-hcc-reproduction-v2" / "lane_summary.json",
        "schema": "arac-exp059-gcb-hcc-lane-result-v1",
        "verify": (
            "experiments.historical_recovery.run_gcb_hcc_lane",
            "verify",
        ),
    },
    "aor": {
        "summary": TASK_ROOT / "raw" / "aor-a1-a6-25seed-hcc-reproduction-v2" / "lane_summary.json",
        "schema": "arac-exp057-aor-hcc-lane-result-v2",
        "verify": (
            "experiments.historical_recovery.run_aor_hcc_isolated_replay",
            "verify-lane",
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reference_rows() -> dict[str, dict[str, Any]]:
    expected_cases = {case for cases in CASE_MAPPING.values() for case in cases}
    with REFERENCE_TABLE.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = {str(row["case"]): row for row in csv.DictReader(handle)}
    if not expected_cases.issubset(all_rows):
        raise ValueError("AOB reference table case set drifted")
    source = {case: all_rows[case] for case in expected_cases}
    rows = {}
    for case in sorted(expected_cases):
        cell = str(source[case][REFERENCE_COLUMN])
        displayed_mean = cell.split("+/-", maxsplit=1)[0].strip()
        rows[case] = {
            "displayed_mean": displayed_mean,
            "numeric_mean": float(displayed_mean),
        }
    return rows


def _load_lane_summaries() -> dict[str, dict[str, Any]]:
    summaries = {}
    for action in ACTION_ORDER:
        specification = LANES[action]
        path = specification["summary"]
        assert isinstance(path, Path)
        summary = _read_json(path)
        if summary.get("schema_version") != specification["schema"]:
            raise ValueError(f"{action} lane summary schema drifted")
        if summary.get("target_count") != 150 or summary.get("record_count") != 150:
            raise ValueError(f"{action} lane matrix is incomplete")
        if summary.get("lane_recovered") is not True:
            raise ValueError(f"{action} lane is not recovered")
        case_summaries = summary.get("case_summaries")
        if not isinstance(case_summaries, list):
            raise ValueError(f"{action} case summaries are missing")
        if {row.get("case") for row in case_summaries} != set(CASE_MAPPING[action]):
            raise ValueError(f"{action} case summary set drifted")
        if any(row.get("trajectory_count") != 25 for row in case_summaries):
            raise ValueError(f"{action} case trajectory count drifted")
        summaries[action] = summary
    return summaries


def _evaluate_summaries(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    references = _reference_rows()
    evaluations = []
    for action in ACTION_ORDER:
        by_case = {row["case"]: row for row in summaries[action]["case_summaries"]}
        for case in CASE_MAPPING[action]:
            observed_mean = float(by_case[case]["mean"])
            observed_sample_std = float(by_case[case]["sample_std"])
            displayed_mean = f"{observed_mean:.2E}"
            reference = references[case]
            evaluations.append(
                {
                    "case": case,
                    "action": action,
                    "trajectory_count": int(by_case[case]["trajectory_count"]),
                    "observed_mean": observed_mean,
                    "observed_sample_std": observed_sample_std,
                    "observed_displayed_mean": displayed_mean,
                    "reference_displayed_mean": reference["displayed_mean"],
                    "raw_mean_no_higher": observed_mean <= reference["numeric_mean"],
                    "displayed_mean_no_higher": float(displayed_mean) <= reference["numeric_mean"],
                }
            )
    return evaluations


def build_authorization() -> dict[str, Any]:
    summaries = _load_lane_summaries()
    lane_bindings = {}
    for action in ACTION_ORDER:
        summary_path = LANES[action]["summary"]
        assert isinstance(summary_path, Path)
        lane_bindings[action] = {
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "authorization_sha256": summaries[action]["authorization_sha256"],
            "verification_command": list(LANES[action]["verify"]),
        }
    body = {
        "schema_version": "arac-aob24-hcc-historical-gate-authorization-v1",
        "scope": "aggregate_the_four_recovered_hcc_action_lanes_only",
        "case_mapping": {key: list(value) for key, value in CASE_MAPPING.items()},
        "trajectory_count": 600,
        "reference_table_path": str(REFERENCE_TABLE),
        "reference_table_sha256": _sha256(REFERENCE_TABLE),
        "comparison_rule": "observed_mean_no_higher_at_displayed_3_significant_digits",
        "raw_comparison_reported_but_not_used_for_displayed_table_gate": True,
        "lane_bindings": lane_bindings,
        "output_root": str(OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": _canonical_sha256(body)}


def _run_verifier(action: str) -> dict[str, Any]:
    module, command = LANES[action]["verify"]
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [str(PYTHON_EXECUTABLE), "-m", str(module), str(command)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1_800,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {}
    passed = completed.returncode == 0 and payload.get("verification_passed") is True
    return {
        "action": action,
        "command": [str(module), str(command)],
        "returncode": completed.returncode,
        "verification_passed": passed,
        "verification": payload,
        "stderr": completed.stderr,
    }


def run_gate() -> dict[str, Any]:
    authorization = build_authorization()
    if OUTPUT_ROOT.exists():
        raise ValueError(f"AOB-24 gate output already exists: {OUTPUT_ROOT}")
    _write_json_atomic(OUTPUT_ROOT / "authorization.json", authorization)
    verifications = [_run_verifier(action) for action in ACTION_ORDER]
    summaries = _load_lane_summaries()
    evaluations = _evaluate_summaries(summaries)
    summary_hashes_unchanged = all(
        _sha256(Path(binding["summary_path"])) == binding["summary_sha256"]
        for binding in authorization["lane_bindings"].values()
    )
    passed_count = sum(row["displayed_mean_no_higher"] for row in evaluations)
    raw_passed_count = sum(row["raw_mean_no_higher"] for row in evaluations)
    body = {
        "schema_version": "arac-aob24-hcc-historical-gate-result-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization["authorization_sha256"],
        "independent_lane_verifications": verifications,
        "lane_summary_hashes_unchanged": summary_hashes_unchanged,
        "case_count": len(evaluations),
        "displayed_gate_passed_count": passed_count,
        "raw_mean_no_higher_count": raw_passed_count,
        "case_evaluations": evaluations,
        "gate_passed": all(row["verification_passed"] for row in verifications)
        and summary_hashes_unchanged
        and len(evaluations) == 24
        and passed_count == 24,
    }
    result = {**body, "result_sha256": _canonical_sha256(body)}
    _write_json_atomic(OUTPUT_ROOT / "gate_result.json", result)
    return result


def verify_gate() -> dict[str, Any]:
    expected_authorization = build_authorization()
    authorization = _read_json(OUTPUT_ROOT / "authorization.json")
    result = _read_json(OUTPUT_ROOT / "gate_result.json")
    result_body = {key: value for key, value in result.items() if key != "result_sha256"}
    summaries = _load_lane_summaries()
    evaluations = _evaluate_summaries(summaries)
    checks = {
        "authorization_match": authorization == expected_authorization,
        "result_hash_valid": _canonical_sha256(result_body) == result.get("result_sha256"),
        "case_evaluations_match": result.get("case_evaluations") == evaluations,
        "all_lane_verifications_passed": all(
            row.get("verification_passed") is True
            for row in result.get("independent_lane_verifications", [])
        ),
        "all_24_displayed_means_no_higher": len(evaluations) == 24
        and all(row["displayed_mean_no_higher"] for row in evaluations),
        "gate_passed": result.get("gate_passed") is True,
    }
    return {
        "schema_version": "arac-aob24-hcc-historical-gate-verification-v1",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "displayed_gate_passed_count": sum(row["displayed_mean_no_higher"] for row in evaluations),
        "raw_mean_no_higher_count": sum(row["raw_mean_no_higher"] for row in evaluations),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(build_authorization(), sort_keys=True))
        return 0
    if args.command == "verify":
        result = verify_gate()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["verification_passed"] else 1
    result = run_gate()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
