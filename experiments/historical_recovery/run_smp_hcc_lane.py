"""Recover the historical SMP E1-E6 25-seed lane with the exact HCC runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from types import ModuleType
from typing import Any

from experiments.historical_recovery import run_exp052_isolated_replay as representative


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "hcc-action-historical-recovery"
PRIOR_OUTPUT_ROOT = TASK_ROOT / "raw" / "smp-e1-e6-25seed-hcc-reproduction-v1"
OUTPUT_ROOT = TASK_ROOT / "raw" / "smp-e1-e6-25seed-hcc-reproduction-v2"
PARSER_WRAPPER = Path(__file__).with_name("smp_hcc_e_series_runner.py")
SMOKE_ROOT = TASK_ROOT / "raw" / "smp-e2-seed117-wrapper-smoke-v2"
SMOKE_RESULT_DIRECTORY = SMOKE_ROOT / "smp-e2-seed117-wrapper-smoke-v2" / "elliptic"
REFERENCE_TABLE = REPOSITORY_ROOT / "output" / "pdf" / "aob_arac_method_comparison_corrected.csv"
ARCHIVED_ROOT = REPOSITORY_ROOT / "results" / "exp_052_e_series_smp_paired_gate" / "validation"
CASES = tuple(f"E{index}" for index in range(1, 7))
SEEDS = tuple(range(117, 142))
ARCHIVED_CASES = ("E1", "E3")
ARCHIVED_SEEDS = tuple(range(117, 122))
MAX_FES = 3_000_000
MAX_JOBS = 24
CONDITION = "candidate_smp"
PRIOR_EXPERIMENT_ID = "exp_052_smp_hcc_e1_e6_25seed_reproduction_v1"
EXPERIMENT_ID = "exp_052_smp_hcc_e1_e6_25seed_reproduction_v2"
REFERENCE_COLUMN = "ARAC Mean +/- Std"
EXPECTED_REFERENCE_LABELS = {
    "E1": "5.69E+05",
    "E2": "5.62E+06",
    "E3": "1.34E+07",
    "E4": "2.61E+07",
    "E5": "2.98E+07",
    "E6": "3.19E+07",
}

_RUNNER: ModuleType | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _configure_runner(runner: ModuleType) -> ModuleType:
    runner.CASE_IDS = {case: index for index, case in enumerate(CASES, start=1)}
    runner.RUNNER_PATH = PARSER_WRAPPER
    return runner


def _load_runner() -> ModuleType:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = _configure_runner(representative._load_historical_runner())
    return _RUNNER


def _reference_means() -> dict[str, dict[str, Any]]:
    with REFERENCE_TABLE.open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["case"]): row for row in csv.DictReader(handle)}
    references: dict[str, dict[str, Any]] = {}
    for case, expected_label in EXPECTED_REFERENCE_LABELS.items():
        raw = str(rows[case][REFERENCE_COLUMN])
        label = raw.split("+/-", maxsplit=1)[0].strip()
        if label != expected_label:
            raise ValueError(f"displayed SMP reference drifted for {case}: {label}")
        references[case] = {"displayed_mean": label, "numeric_mean": float(label)}
    return references


def _archived_result_directory(case: str, seed: int) -> Path | None:
    if case not in ARCHIVED_CASES or seed not in ARCHIVED_SEEDS:
        return None
    trajectory_id = f"exp_052_e_series_smp_paired_gate-{case.lower()}-{CONDITION}-seed{seed}"
    return ARCHIVED_ROOT / "runs" / case / CONDITION / f"seed_{seed}" / trajectory_id / "elliptic"


def _archived_matrix_sha256() -> str:
    rows = []
    for case in ARCHIVED_CASES:
        for seed in ARCHIVED_SEEDS:
            root = _archived_result_directory(case, seed)
            assert root is not None
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "run_summary_sha256": representative._sha256(root / "run_summary.json"),
                    "action_sha256": representative._sha256(root / "smp_action.json"),
                    "receipt_sha256": representative._sha256(
                        root / "exp052_execution_receipt.json"
                    ),
                }
            )
    return representative._canonical_sha256(rows)


def _representative_verified() -> bool:
    expected_authorization = representative.build_preflight()
    authorization = _read_json(representative.OUTPUT_ROOT / "authorization.json")
    summary = _read_json(representative.OUTPUT_ROOT / "reproduction_summary.json")
    return (
        authorization == expected_authorization
        and summary.get("authorization_sha256") == authorization["authorization_sha256"]
        and summary.get("integrity_passed") is True
        and summary.get("exact_historical_value_match") is True
        and summary.get("final_error") == representative.HISTORICAL_FINAL_ERROR
    )


def _smoke_evidence() -> dict[str, Any]:
    summary_path = SMOKE_RESULT_DIRECTORY / "run_summary.json"
    action_path = SMOKE_RESULT_DIRECTORY / "smp_action.json"
    summary = _read_json(summary_path)
    action = _read_json(action_path)
    if (
        summary.get("problem_id") != "E2"
        or summary.get("seed") != 117
        or summary.get("configured_max_fes") != 100_000
        or summary.get("fitness_evaluations") != 100_000
        or summary.get("group_optimizer_mode") != "smp"
        or action.get("schema_version") != "smp-action-v1"
        or int(action.get("restore_count", 0)) <= 0
        or int(action.get("abstain_count", -1)) != 0
    ):
        raise ValueError("E2 parser-wrapper smoke evidence is invalid")
    return {
        "case": "E2",
        "seed": 117,
        "max_fes": 100_000,
        "final_error": float(summary["final_error"]),
        "restore_count": int(action["restore_count"]),
        "reset_count": int(action["reset_count"]),
        "abstain_count": int(action["abstain_count"]),
        "run_summary_sha256": representative._sha256(summary_path),
        "action_sha256": representative._sha256(action_path),
    }


def build_authorization() -> dict[str, Any]:
    if not _representative_verified():
        raise ValueError("representative SMP replay must pass before lane expansion")
    prior_summary_path = PRIOR_OUTPUT_ROOT / "lane_summary.json"
    prior_summary = _read_json(prior_summary_path)
    if (
        prior_summary.get("schema_version") != "arac-exp052-smp-hcc-lane-result-v1"
        or prior_summary.get("record_count") != 150
        or prior_summary.get("valid_count") != 50
        or prior_summary.get("failed_count") != 100
        or prior_summary.get("mismatch_count") != 0
        or prior_summary.get("archived_exact_reference_count") != 10
        or prior_summary.get("lane_recovered") is not False
    ):
        raise ValueError("SMP V1 parser-failure evidence is incomplete")
    references = _reference_means()
    body = {
        "schema_version": "arac-exp052-smp-hcc-lane-authorization-v2",
        "authorization_scope": "smp_e1_e6_seeds117_141_parser_expansion_only",
        "claim_boundary": (
            "the_exact_exp052_runner_is_unchanged;the_wrapper_only_expands_the_elliptic_"
            "id_parser_whitelist;ten_archived_trajectories_support_exact_replay"
        ),
        "target": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "max_fes": MAX_FES,
            "trajectory_count": len(CASES) * len(SEEDS),
        },
        "archived_exact_reference_count": len(ARCHIVED_CASES) * len(ARCHIVED_SEEDS),
        "archived_matrix_sha256": _archived_matrix_sha256(),
        "prior_v1_summary_sha256": representative._sha256(prior_summary_path),
        "prior_v1_valid_reuse_count": 50,
        "prior_v1_parser_failure_count": 100,
        "candidate_manifest_sha256": representative._sha256(representative.CANDIDATE_MANIFEST),
        "exact_exp052_runner_sha256": representative._sha256(
            representative.CANDIDATE_ROOT / "scripts" / "hcc_smoke_runner.py"
        ),
        "parser_wrapper_sha256": representative._sha256(PARSER_WRAPPER),
        "parser_wrapper_smoke": _smoke_evidence(),
        "config_sha256": representative._sha256(representative.HISTORICAL_CONFIG),
        "reference_table_sha256": representative._sha256(REFERENCE_TABLE),
        "reference_means": references,
        "aggregate_gate": "observed_mean_no_higher_at_displayed_3_significant_digits",
        "max_jobs": MAX_JOBS,
        "output_root": str(OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": representative._canonical_sha256(body)}


def _trajectory_root(case: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / case / CONDITION / f"seed_{seed}"


def _spec(runner: ModuleType, case: str, seed: int, output_root: Path) -> Any:
    return runner.RunSpec(
        experiment_id=EXPERIMENT_ID,
        case=case,
        condition=CONDITION,
        seed=seed,
        max_fes=MAX_FES,
        output_root=output_root,
    )


def _representative_spec(runner: ModuleType) -> Any:
    return runner.RunSpec(
        experiment_id="exp_052_e_series_smp_paired_gate",
        case=representative.CASE,
        condition=CONDITION,
        seed=representative.SEED,
        max_fes=MAX_FES,
        output_root=representative.OUTPUT_ROOT,
    )


def _prior_spec(runner: ModuleType, case: str, seed: int) -> Any:
    return runner.RunSpec(
        experiment_id=PRIOR_EXPERIMENT_ID,
        case=case,
        condition=CONDITION,
        seed=seed,
        max_fes=MAX_FES,
        output_root=PRIOR_OUTPUT_ROOT,
    )


def _validate_direct_exp052_result(runner: ModuleType, spec: Any) -> dict[str, Any]:
    summary = _read_json(spec.result_directory / "run_summary.json")
    expected_summary = {
        "problem_id": spec.case,
        "seed": spec.seed,
        "configured_max_fes": MAX_FES,
        "fitness_evaluations": MAX_FES,
        "group_optimizer_mode": "smp",
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("direct EXP-052 summary fields drifted")
    final_error = float(summary["final_error"])
    if not math.isfinite(final_error) or final_error < 0.0:
        raise ValueError("direct EXP-052 final error is invalid")
    runner._validate_budget_and_inputs(spec)
    action = _read_json(spec.result_directory / "smp_action.json")
    if action.get("schema_version") != "smp-action-v1":
        raise ValueError("direct EXP-052 SMP action schema drifted")
    restore_count = int(action["restore_count"])
    abstain_count = int(action["abstain_count"])
    if restore_count <= 0 or abstain_count != 0:
        raise ValueError("direct EXP-052 SMP lifecycle is invalid")
    receipt = _read_json(spec.result_directory / "exp052_execution_receipt.json")
    expected_receipt = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "case": spec.case,
        "condition": CONDITION,
        "seed": spec.seed,
        "configured_max_fes": MAX_FES,
        "config_sha256": representative._sha256(representative.HISTORICAL_CONFIG),
        "runner_sha256": representative._sha256(
            representative.CANDIDATE_ROOT / "scripts" / "hcc_smoke_runner.py"
        ),
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ValueError("direct EXP-052 campaign receipt drifted")
    if float(receipt["final_error"]) != final_error:
        raise ValueError("direct EXP-052 receipt result drifted")
    return {
        "ok": True,
        "final_error": final_error,
        "fitness_evaluations": MAX_FES,
        "restore_count": restore_count,
        "reset_count": int(action["reset_count"]),
        "abstain_count": abstain_count,
    }


def _evidence(
    runner: ModuleType,
    spec: Any,
    *,
    direct_exp052_runner: bool = False,
) -> dict[str, Any]:
    config_hash = representative._sha256(representative.HISTORICAL_CONFIG)
    validated = (
        _validate_direct_exp052_result(runner, spec)
        if direct_exp052_runner
        else runner.validate_existing(spec, config_hash, require_restore=True)
    )
    result_directory = spec.result_directory
    stderr_path = spec.run_directory / "stderr.log"
    summary_path = result_directory / "run_summary.json"
    action_path = result_directory / "smp_action.json"
    campaign_receipt_path = result_directory / "exp052_execution_receipt.json"
    summary = _read_json(summary_path)
    action = _read_json(action_path)
    stderr_empty = stderr_path.is_file() and stderr_path.read_text(encoding="utf-8") == ""
    integrity_passed = (
        validated.get("ok") is True
        and validated.get("fitness_evaluations") == MAX_FES
        and int(validated.get("restore_count", 0)) > 0
        and int(validated.get("abstain_count", -1)) == 0
        and stderr_empty
        and math.isfinite(float(validated["final_error"]))
    )

    archived_root = _archived_result_directory(spec.case, spec.seed)
    archived_reference_available = archived_root is not None
    archived_summary_exact = None
    archived_action_exact = None
    archived_summary_sha256 = None
    archived_action_sha256 = None
    if archived_root is not None:
        archived_summary_path = archived_root / "run_summary.json"
        archived_action_path = archived_root / "smp_action.json"
        archived_summary_exact = summary == _read_json(archived_summary_path)
        archived_action_exact = action == _read_json(archived_action_path)
        archived_summary_sha256 = representative._sha256(archived_summary_path)
        archived_action_sha256 = representative._sha256(archived_action_path)

    archive_gate = not archived_reference_available or (
        archived_summary_exact is True and archived_action_exact is True
    )
    return {
        "case": spec.case,
        "seed": spec.seed,
        "result_directory": str(result_directory),
        "stderr_path": str(stderr_path),
        "final_error": float(validated["final_error"]),
        "fitness_evaluations": int(validated["fitness_evaluations"]),
        "restore_count": int(validated["restore_count"]),
        "reset_count": int(validated["reset_count"]),
        "abstain_count": int(validated["abstain_count"]),
        "stderr_empty": stderr_empty,
        "integrity_passed": integrity_passed,
        "archived_reference_available": archived_reference_available,
        "archived_summary_exact": archived_summary_exact,
        "archived_action_exact": archived_action_exact,
        "archived_exact": archive_gate if archived_reference_available else None,
        "trajectory_valid": integrity_passed and archive_gate,
        "run_summary_sha256": representative._sha256(summary_path),
        "action_sha256": representative._sha256(action_path),
        "campaign_receipt_sha256": representative._sha256(campaign_receipt_path),
        "stderr_sha256": representative._sha256(stderr_path),
        "archived_run_summary_sha256": archived_summary_sha256,
        "archived_action_sha256": archived_action_sha256,
    }


def _write_trajectory_receipt(
    authorization_sha256: str,
    evidence: dict[str, Any],
    *,
    execution_source: str,
) -> Path:
    body = {
        "schema_version": "arac-exp052-smp-hcc-lane-trajectory-receipt-v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization_sha256,
        "execution_source": execution_source,
        **evidence,
    }
    receipt = {**body, "receipt_sha256": representative._canonical_sha256(body)}
    path = (
        _trajectory_root(str(evidence["case"]), int(evidence["seed"])) / "trajectory_receipt.json"
    )
    representative._write_json_atomic(path, receipt)
    return path


def _receipt_valid(
    authorization_sha256: str,
    evidence: dict[str, Any],
) -> bool:
    path = (
        _trajectory_root(str(evidence["case"]), int(evidence["seed"])) / "trajectory_receipt.json"
    )
    if not path.is_file():
        return False
    receipt = _read_json(path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if representative._canonical_sha256(body) != receipt.get("receipt_sha256"):
        return False
    if receipt.get("authorization_sha256") != authorization_sha256:
        return False
    for key, value in evidence.items():
        if receipt.get(key) != value:
            return False
    return True


def _record(evidence: dict[str, Any], status: str, receipt_valid: bool) -> dict[str, Any]:
    return {
        "case": evidence["case"],
        "seed": evidence["seed"],
        "status": status,
        "final_error": evidence["final_error"],
        "fitness_evaluations": evidence["fitness_evaluations"],
        "restore_count": evidence["restore_count"],
        "reset_count": evidence["reset_count"],
        "abstain_count": evidence["abstain_count"],
        "stderr_empty": evidence["stderr_empty"],
        "integrity_passed": evidence["integrity_passed"],
        "archived_reference_available": evidence["archived_reference_available"],
        "archived_exact": evidence["archived_exact"],
        "trajectory_receipt_valid": receipt_valid,
        "trajectory_valid": evidence["trajectory_valid"] and receipt_valid,
    }


def _run_trajectory(
    runner: ModuleType,
    case: str,
    seed: int,
    authorization_sha256: str,
) -> dict[str, Any]:
    trajectory_root = _trajectory_root(case, seed)
    try:
        if case == representative.CASE and seed == representative.SEED:
            spec = _representative_spec(runner)
            evidence = _evidence(runner, spec, direct_exp052_runner=True)
            receipt_path = _write_trajectory_receipt(
                authorization_sha256,
                evidence,
                execution_source="representative_reuse",
            )
            return _record(evidence, "reused_representative", receipt_path.is_file())

        if case in ARCHIVED_CASES:
            spec = _prior_spec(runner, case, seed)
            evidence = _evidence(runner, spec, direct_exp052_runner=True)
            receipt_path = _write_trajectory_receipt(
                authorization_sha256,
                evidence,
                execution_source="prior_v1_reuse",
            )
            return _record(evidence, "reused_prior_v1", receipt_path.is_file())

        spec = _spec(runner, case, seed, OUTPUT_ROOT)
        if (spec.result_directory / "run_summary.json").is_file():
            evidence = _evidence(runner, spec)
            receipt_path = _write_trajectory_receipt(
                authorization_sha256,
                evidence,
                execution_source="existing_complete",
            )
            return _record(evidence, "reused_existing", receipt_path.is_file())
        if trajectory_root.exists() and any(trajectory_root.iterdir()):
            raise ValueError("incomplete trajectory directory already exists")

        result = runner.run_one(
            spec,
            representative._sha256(representative.HISTORICAL_CONFIG),
            str(representative.PYTHON_EXECUTABLE),
            require_restore=True,
        )
        if result.get("ok") is not True:
            raise RuntimeError(str(result.get("error", "historical EXP-052 runner failed")))
        evidence = _evidence(runner, spec)
        receipt_path = _write_trajectory_receipt(
            authorization_sha256,
            evidence,
            execution_source="fresh",
        )
        status = "completed" if evidence["trajectory_valid"] else "mismatch"
        return _record(evidence, status, receipt_path.is_file())
    except Exception as error:
        failure = {
            "case": case,
            "seed": seed,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "trajectory_valid": False,
            "trajectory_receipt_valid": False,
        }
        representative._write_json_atomic(
            trajectory_root / "reproduction_failure.json",
            failure,
        )
        return failure


def _case_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = _reference_means()
    summaries = []
    for case in CASES:
        values = [
            float(row["final_error"])
            for row in records
            if row.get("case") == case and row.get("trajectory_valid") is True
        ]
        mean = statistics.mean(values) if len(values) == len(SEEDS) else None
        sample_std = statistics.stdev(values) if len(values) == len(SEEDS) else None
        displayed_mean = None if mean is None else f"{mean:.2E}"
        displayed_numeric_mean = None if displayed_mean is None else float(displayed_mean)
        reference = references[case]
        summaries.append(
            {
                "case": case,
                "trajectory_count": len(values),
                "mean": mean,
                "sample_std": sample_std,
                "displayed_mean": displayed_mean,
                "reference_displayed_mean": reference["displayed_mean"],
                "raw_mean_no_higher": mean is not None and mean <= reference["numeric_mean"],
                "displayed_mean_no_higher": displayed_numeric_mean is not None
                and displayed_numeric_mean <= reference["numeric_mean"],
            }
        )
    return summaries


def run_lane(*, jobs: int = MAX_JOBS) -> dict[str, Any]:
    if jobs < 1 or jobs > MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}")
    authorization = build_authorization()
    authorization_path = OUTPUT_ROOT / "authorization.json"
    if authorization_path.is_file():
        if _read_json(authorization_path) != authorization:
            raise ValueError("SMP lane authorization drifted")
    else:
        if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
            raise ValueError("SMP lane output exists without authorization")
        representative._write_json_atomic(authorization_path, authorization)

    runner = _load_runner()
    targets = [(case, seed) for case in CASES for seed in SEEDS]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {
            executor.submit(
                _run_trajectory,
                runner,
                case,
                seed,
                authorization["authorization_sha256"],
            ): (case, seed)
            for case, seed in targets
        }
        for completed_count, future in enumerate(
            concurrent.futures.as_completed(pending),
            start=1,
        ):
            record = future.result()
            records.append(record)
            print(
                f"SMP lane {completed_count}/{len(targets)} "
                f"{record['case']}/seed{record['seed']} {record['status']}",
                flush=True,
            )
            representative._write_json_atomic(
                OUTPUT_ROOT / "run_progress.json",
                {
                    "schema_version": "arac-exp052-smp-hcc-lane-progress-v2",
                    "authorization_sha256": authorization["authorization_sha256"],
                    "completed_count": completed_count,
                    "target_count": len(targets),
                    "records": sorted(records, key=lambda row: (row["case"], row["seed"])),
                },
            )

    records.sort(key=lambda row: (row["case"], row["seed"]))
    summaries = _case_summaries(records)
    valid_count = sum(row.get("trajectory_valid") is True for row in records)
    archived_exact_count = sum(row.get("archived_exact") is True for row in records)
    summary = {
        "schema_version": "arac-exp052-smp-hcc-lane-result-v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization["authorization_sha256"],
        "target_count": len(targets),
        "record_count": len(records),
        "valid_count": valid_count,
        "archived_exact_reference_count": archived_exact_count,
        "failed_count": sum(row["status"] == "failed" for row in records),
        "mismatch_count": sum(row["status"] == "mismatch" for row in records),
        "case_summaries": summaries,
        "lane_recovered": valid_count == len(targets)
        and archived_exact_count == len(ARCHIVED_CASES) * len(ARCHIVED_SEEDS)
        and all(row["displayed_mean_no_higher"] for row in summaries),
    }
    representative._write_json_atomic(OUTPUT_ROOT / "lane_summary.json", summary)
    return summary


def verify_lane() -> dict[str, Any]:
    expected_authorization = build_authorization()
    authorization = _read_json(OUTPUT_ROOT / "authorization.json")
    runner = _load_runner()
    records = []
    for case in CASES:
        for seed in SEEDS:
            try:
                if case == representative.CASE and seed == representative.SEED:
                    spec = _representative_spec(runner)
                    direct_exp052_runner = True
                elif case in ARCHIVED_CASES:
                    spec = _prior_spec(runner, case, seed)
                    direct_exp052_runner = True
                else:
                    spec = _spec(runner, case, seed, OUTPUT_ROOT)
                    direct_exp052_runner = False
                evidence = _evidence(
                    runner,
                    spec,
                    direct_exp052_runner=direct_exp052_runner,
                )
                receipt_valid = _receipt_valid(
                    expected_authorization["authorization_sha256"], evidence
                )
                records.append(
                    _record(
                        evidence,
                        "verified" if receipt_valid else "invalid_receipt",
                        receipt_valid,
                    )
                )
            except Exception as error:
                records.append(
                    {
                        "case": case,
                        "seed": seed,
                        "status": "missing_or_invalid",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "trajectory_valid": False,
                        "trajectory_receipt_valid": False,
                    }
                )
    summaries = _case_summaries(records)
    valid_count = sum(row.get("trajectory_valid") is True for row in records)
    archived_exact_count = sum(row.get("archived_exact") is True for row in records)
    checks = {
        "authorization_match": authorization == expected_authorization,
        "trajectory_count_match": len(records) == len(CASES) * len(SEEDS),
        "all_trajectories_valid": valid_count == len(CASES) * len(SEEDS),
        "all_receipts_valid": all(row.get("trajectory_receipt_valid") is True for row in records),
        "all_archived_references_exact": archived_exact_count
        == len(ARCHIVED_CASES) * len(ARCHIVED_SEEDS),
        "all_displayed_means_no_higher": all(row["displayed_mean_no_higher"] for row in summaries),
    }
    return {
        "schema_version": "arac-exp052-smp-hcc-lane-verification-v2",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "valid_count": valid_count,
        "archived_exact_reference_count": archived_exact_count,
        "case_summaries": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    parser.add_argument("--jobs", type=int, default=MAX_JOBS)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(build_authorization(), sort_keys=True))
        return 0
    if args.command == "verify":
        result = verify_lane()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["verification_passed"] else 1
    result = run_lane(jobs=args.jobs)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["lane_recovered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
