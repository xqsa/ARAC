"""Reproduce the historical EXP-059 GCB R1-R6 25-seed lane under exact HCC."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import statistics
from pathlib import Path
from typing import Any

from experiments.historical_recovery import run_gcb_hcc_isolated_replay as representative


CASES = representative.GCB_CASES
SEEDS = representative.HISTORICAL_SEEDS
MAX_FES = representative.MAX_FES
MAX_JOBS = 24
PRIOR_OUTPUT_ROOT = representative.TASK_ROOT / "raw" / "gcb-r1-r6-25seed-hcc-reproduction-v1"
OUTPUT_ROOT = representative.TASK_ROOT / "raw" / "gcb-r1-r6-25seed-hcc-reproduction-v2"
SOURCE_TREE = representative.SOURCE_TREE
RUNNER_PATH = SOURCE_TREE / representative.RUNNER_RELATIVE_PATH
REPRESENTATIVE_RESULT_DIRECTORY = representative._result_directory(
    representative.RUN_ROOT,
    representative.CASE,
    representative.SEED,
)


def _trajectory_root(case: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / case / f"seed_{seed}"


def _result_directory(case: str, seed: int) -> Path:
    return representative._result_directory(_trajectory_root(case, seed), case, seed)


def _archived_directory(case: str, seed: int) -> Path:
    return representative._archived_directory(case, seed)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join((str(SOURCE_TREE / "src"), str(SOURCE_TREE))),
        }
    )
    return environment


def _matrix_reference_sha256() -> str:
    records = []
    for case in CASES:
        for seed in SEEDS:
            root = _archived_directory(case, seed)
            records.append(
                {
                    "case": case,
                    "seed": seed,
                    "run_summary_sha256": representative._sha256(root / "run_summary.json"),
                    "action_sha256": representative._sha256(root / "gcb_action.json"),
                    "authorization_sha256": representative._sha256(
                        root / "gcb_stable_authorization.json"
                    ),
                    "campaign_receipt_sha256": representative._sha256(
                        root / "exp059_execution_receipt.json"
                    ),
                }
            )
    return representative._canonical_sha256(records)


def build_authorization() -> dict[str, Any]:
    verification = representative.verify_reproduction()
    if verification.get("verification_passed") is not True:
        raise ValueError("representative GCB replay must pass before lane expansion")
    body = {
        "schema_version": "arac-exp059-gcb-hcc-lane-authorization-v1",
        "authorization_scope": "gcb_r1_r6_historical_matrix_only",
        "target": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "max_fes": MAX_FES,
            "trajectory_count": len(CASES) * len(SEEDS),
        },
        "representative_authorization_sha256": representative._read_json(
            representative.OUTPUT_ROOT / "authorization.json"
        )["authorization_sha256"],
        "representative_receipt_sha256": representative._sha256(
            representative.OUTPUT_ROOT / "reproduction_receipt.json"
        ),
        "source_manifest_sha256": representative._sha256(
            representative.OUTPUT_ROOT / "source_manifest.json"
        ),
        "prior_attempt_authorization_sha256": representative._sha256(
            PRIOR_OUTPUT_ROOT / "authorization.json"
        ),
        "prior_attempt_lane_summary_sha256": representative._sha256(
            PRIOR_OUTPUT_ROOT / "lane_summary.json"
        ),
        "runner_sha256": representative._sha256(RUNNER_PATH),
        "historical_matrix_reference_sha256": _matrix_reference_sha256(),
        "max_jobs": MAX_JOBS,
        "result_gate": "all_summary_action_authorization_and_receipt_fields_exact",
        "output_root": str(OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": representative._canonical_sha256(body)}


def _validate_evidence(
    case: str,
    seed: int,
    result_directory: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    archived_directory = _archived_directory(case, seed)
    observed_summary = representative._read_json(result_directory / "run_summary.json")
    archived_summary = representative._read_json(archived_directory / "run_summary.json")
    observed_action = representative._read_json(result_directory / "gcb_action.json")
    archived_action = representative._read_json(archived_directory / "gcb_action.json")
    observed_authorization = representative._read_json(
        result_directory / "gcb_stable_authorization.json"
    )
    archived_authorization = representative._read_json(
        archived_directory / "gcb_stable_authorization.json"
    )
    archived_receipt = representative._read_json(
        archived_directory / "exp059_execution_receipt.json"
    )
    summary_checks = representative._compare(observed_summary, archived_summary)
    action_checks = representative._compare(observed_action, archived_action)
    authorization_checks = representative._compare(observed_authorization, archived_authorization)
    receipt_checks = representative._receipt_checks(observed_summary, archived_receipt)
    stderr_empty = stderr_path.is_file() and stderr_path.read_text(encoding="utf-8") == ""
    exact_budget_match = observed_summary.get("fitness_evaluations") == MAX_FES
    evidence = {
        "case": case,
        "seed": seed,
        "final_error": observed_summary.get("final_error"),
        "summary_exact": all(summary_checks.values()),
        "action_exact": all(action_checks.values()),
        "authorization_exact": all(authorization_checks.values()),
        "historical_receipt_exact": all(receipt_checks.values()),
        "exact_budget_match": exact_budget_match,
        "stderr_empty": stderr_empty,
        "failed_summary_fields": [key for key, value in summary_checks.items() if not value],
        "failed_action_fields": [key for key, value in action_checks.items() if not value],
        "failed_authorization_fields": [
            key for key, value in authorization_checks.items() if not value
        ],
        "failed_historical_receipt_fields": [
            key for key, value in receipt_checks.items() if not value
        ],
        "run_summary_sha256": representative._sha256(result_directory / "run_summary.json"),
        "action_sha256": representative._sha256(result_directory / "gcb_action.json"),
        "authorization_artifact_sha256": representative._sha256(
            result_directory / "gcb_stable_authorization.json"
        ),
        "stderr_sha256": representative._sha256(stderr_path),
        "archived_run_summary_sha256": representative._sha256(
            archived_directory / "run_summary.json"
        ),
        "archived_action_sha256": representative._sha256(archived_directory / "gcb_action.json"),
        "archived_authorization_sha256": representative._sha256(
            archived_directory / "gcb_stable_authorization.json"
        ),
        "archived_campaign_receipt_sha256": representative._sha256(
            archived_directory / "exp059_execution_receipt.json"
        ),
    }
    evidence["trajectory_exact"] = all(
        evidence[key]
        for key in (
            "summary_exact",
            "action_exact",
            "authorization_exact",
            "historical_receipt_exact",
            "exact_budget_match",
            "stderr_empty",
        )
    )
    return evidence


def _write_trajectory_receipt(
    authorization_sha256: str,
    evidence: dict[str, Any],
    *,
    command: list[str],
    execution_source: str,
    result_directory: Path,
    stderr_path: Path,
) -> Path:
    body = {
        "schema_version": "arac-exp059-gcb-hcc-lane-trajectory-receipt-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization_sha256,
        "case": evidence["case"],
        "seed": evidence["seed"],
        "execution_source": execution_source,
        "command": command,
        "result_directory": str(result_directory),
        "stderr_path": str(stderr_path),
        **{
            key: evidence[key]
            for key in (
                "final_error",
                "summary_exact",
                "action_exact",
                "authorization_exact",
                "historical_receipt_exact",
                "exact_budget_match",
                "stderr_empty",
                "trajectory_exact",
                "run_summary_sha256",
                "action_sha256",
                "authorization_artifact_sha256",
                "stderr_sha256",
                "archived_run_summary_sha256",
                "archived_action_sha256",
                "archived_authorization_sha256",
                "archived_campaign_receipt_sha256",
            )
        },
    }
    receipt = {**body, "receipt_sha256": representative._canonical_sha256(body)}
    path = _trajectory_root(str(evidence["case"]), int(evidence["seed"])) / "trajectory_receipt.json"
    representative._write_json_atomic(path, receipt)
    return path


def _resolve_existing_evidence(case: str, seed: int) -> tuple[Path, Path, str] | None:
    trajectory_root = _trajectory_root(case, seed)
    direct_result = _result_directory(case, seed)
    if (direct_result / "run_summary.json").is_file():
        return direct_result, trajectory_root / "stderr.log", "existing_complete"
    prior_root = PRIOR_OUTPUT_ROOT / "runs" / case / f"seed_{seed}"
    prior_result = representative._result_directory(prior_root, case, seed)
    if (prior_result / "run_summary.json").is_file():
        return prior_result, prior_root / "stderr.log", "prior_v1_reuse"
    receipt_path = trajectory_root / "trajectory_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = representative._read_json(receipt_path)
    if receipt.get("execution_source") != "representative_reuse":
        raise ValueError("existing lane receipt has no local result directory")
    result_directory = Path(str(receipt.get("result_directory", ""))).resolve()
    stderr_path = Path(str(receipt.get("stderr_path", ""))).resolve()
    representative_root = representative.OUTPUT_ROOT.resolve()
    if representative_root not in result_directory.parents or representative_root not in stderr_path.parents:
        raise ValueError("representative reuse result escaped its evidence root")
    return result_directory, stderr_path, "representative_reuse"


def _verify_trajectory_receipt(
    case: str,
    seed: int,
    authorization_sha256: str,
    evidence: dict[str, Any],
) -> bool:
    receipt_path = _trajectory_root(case, seed) / "trajectory_receipt.json"
    if not receipt_path.is_file():
        return False
    receipt = representative._read_json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if representative._canonical_sha256(body) != receipt.get("receipt_sha256"):
        return False
    if receipt.get("authorization_sha256") != authorization_sha256:
        return False
    if receipt.get("case") != case or receipt.get("seed") != seed:
        return False
    for key in (
        "final_error",
        "summary_exact",
        "action_exact",
        "authorization_exact",
        "historical_receipt_exact",
        "exact_budget_match",
        "stderr_empty",
        "trajectory_exact",
        "run_summary_sha256",
        "action_sha256",
        "authorization_artifact_sha256",
        "stderr_sha256",
        "archived_run_summary_sha256",
        "archived_action_sha256",
        "archived_authorization_sha256",
        "archived_campaign_receipt_sha256",
    ):
        if receipt.get(key) != evidence.get(key):
            return False
    return True


def _record_from_evidence(evidence: dict[str, Any], *, status: str, receipt_valid: bool) -> dict[str, Any]:
    return {
        "case": evidence["case"],
        "seed": evidence["seed"],
        "status": status,
        "final_error": evidence["final_error"],
        "summary_exact": evidence["summary_exact"],
        "action_exact": evidence["action_exact"],
        "authorization_exact": evidence["authorization_exact"],
        "historical_receipt_exact": evidence["historical_receipt_exact"],
        "exact_budget_match": evidence["exact_budget_match"],
        "stderr_empty": evidence["stderr_empty"],
        "trajectory_receipt_valid": receipt_valid,
        "trajectory_exact": evidence["trajectory_exact"] and receipt_valid,
        "failed_summary_fields": evidence["failed_summary_fields"],
        "failed_action_fields": evidence["failed_action_fields"],
        "failed_authorization_fields": evidence["failed_authorization_fields"],
        "failed_historical_receipt_fields": evidence["failed_historical_receipt_fields"],
    }


def _run_trajectory(case: str, seed: int, authorization_sha256: str) -> dict[str, Any]:
    trajectory_root = _trajectory_root(case, seed)
    command = representative._trajectory_command(case=case, seed=seed, output_root=trajectory_root)
    try:
        existing = _resolve_existing_evidence(case, seed)
        if existing is not None:
            result_directory, stderr_path, execution_source = existing
            evidence = _validate_evidence(case, seed, result_directory, stderr_path)
            receipt_path = _write_trajectory_receipt(
                authorization_sha256,
                evidence,
                command=command,
                execution_source=execution_source,
                result_directory=result_directory,
                stderr_path=stderr_path,
            )
            return _record_from_evidence(
                evidence,
                status=f"reused_{execution_source}",
                receipt_valid=receipt_path.is_file(),
            )
        if trajectory_root.exists() and any(trajectory_root.iterdir()):
            raise ValueError("incomplete trajectory directory already exists")
        if case == representative.CASE and seed == representative.SEED:
            stderr_path = representative.OUTPUT_ROOT / "stderr.log"
            evidence = _validate_evidence(
                case,
                seed,
                REPRESENTATIVE_RESULT_DIRECTORY,
                stderr_path,
            )
            receipt_path = _write_trajectory_receipt(
                authorization_sha256,
                evidence,
                command=representative._trajectory_command(
                    case=case,
                    seed=seed,
                    output_root=representative.RUN_ROOT,
                ),
                execution_source="representative_reuse",
                result_directory=REPRESENTATIVE_RESULT_DIRECTORY,
                stderr_path=stderr_path,
            )
            return _record_from_evidence(
                evidence,
                status="reused_representative",
                receipt_valid=receipt_path.is_file(),
            )

        trajectory_root.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        completed = subprocess.run(
            command,
            cwd=SOURCE_TREE,
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=7_200,
            check=False,
        )
        trajectory_root.mkdir(parents=True, exist_ok=True)
        stdout_path = trajectory_root / "stdout.log"
        stderr_path = trajectory_root / "stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"historical EXP-059 runner exited {completed.returncode}")
        result_directory = _result_directory(case, seed)
        evidence = _validate_evidence(case, seed, result_directory, stderr_path)
        receipt_path = _write_trajectory_receipt(
            authorization_sha256,
            evidence,
            command=command,
            execution_source="fresh",
            result_directory=result_directory,
            stderr_path=stderr_path,
        )
        return _record_from_evidence(
            evidence,
            status="completed" if evidence["trajectory_exact"] else "mismatch",
            receipt_valid=receipt_path.is_file(),
        )
    except Exception as error:
        failure = {
            "case": case,
            "seed": seed,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "trajectory_exact": False,
            "trajectory_receipt_valid": False,
        }
        representative._write_json_atomic(trajectory_root / "reproduction_failure.json", failure)
        return failure


def _case_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    record_map = {(str(row["case"]), int(row["seed"])): row for row in records}
    summaries = []
    for case in CASES:
        observed = [
            float(record_map[(case, seed)]["final_error"])
            for seed in SEEDS
            if record_map.get((case, seed), {}).get("trajectory_exact") is True
        ]
        archived = [
            float(representative._read_json(_archived_directory(case, seed) / "run_summary.json")["final_error"])
            for seed in SEEDS
        ]
        summaries.append(
            {
                "case": case,
                "trajectory_count": len(observed),
                "mean": statistics.mean(observed) if observed else None,
                "sample_std": statistics.stdev(observed) if len(observed) > 1 else None,
                "archived_mean": statistics.mean(archived),
                "archived_sample_std": statistics.stdev(archived),
                "exact_aggregate_match": observed == archived,
            }
        )
    return summaries


def run_lane(*, jobs: int = MAX_JOBS) -> dict[str, Any]:
    if jobs < 1 or jobs > MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}")
    authorization = build_authorization()
    authorization_path = OUTPUT_ROOT / "authorization.json"
    if authorization_path.is_file():
        if representative._read_json(authorization_path) != authorization:
            raise ValueError("GCB lane authorization drifted")
    elif OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise ValueError("GCB lane output exists without authorization")
    else:
        representative._write_json_atomic(authorization_path, authorization)

    targets = [(case, seed) for case in CASES for seed in SEEDS]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {
            executor.submit(_run_trajectory, case, seed, authorization["authorization_sha256"]): (case, seed)
            for case, seed in targets
        }
        for completed_count, future in enumerate(concurrent.futures.as_completed(pending), start=1):
            record = future.result()
            records.append(record)
            print(f"GCB lane {completed_count}/{len(targets)} {record['case']}/seed{record['seed']} {record['status']}", flush=True)
            representative._write_json_atomic(
                OUTPUT_ROOT / "run_progress.json",
                {
                    "schema_version": "arac-exp059-gcb-hcc-lane-progress-v1",
                    "authorization_sha256": authorization["authorization_sha256"],
                    "completed_count": completed_count,
                    "target_count": len(targets),
                    "records": sorted(records, key=lambda item: (item["case"], item["seed"])),
                },
            )

    records.sort(key=lambda item: (item["case"], item["seed"]))
    exact_count = sum(record.get("trajectory_exact") is True for record in records)
    case_summaries = _case_summaries(records)
    summary = {
        "schema_version": "arac-exp059-gcb-hcc-lane-result-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization["authorization_sha256"],
        "target_count": len(targets),
        "record_count": len(records),
        "exact_match_count": exact_count,
        "failed_count": sum(record["status"] == "failed" for record in records),
        "mismatch_count": sum(record["status"] == "mismatch" for record in records),
        "case_summaries": case_summaries,
        "lane_recovered": exact_count == len(targets)
        and all(item["exact_aggregate_match"] for item in case_summaries),
    }
    representative._write_json_atomic(OUTPUT_ROOT / "lane_summary.json", summary)
    return summary


def verify_lane() -> dict[str, Any]:
    expected_authorization = build_authorization()
    authorization = representative._read_json(OUTPUT_ROOT / "authorization.json")
    records = []
    for case in CASES:
        for seed in SEEDS:
            try:
                resolved = _resolve_existing_evidence(case, seed)
                if resolved is None:
                    raise FileNotFoundError("trajectory evidence is missing")
                result_directory, stderr_path, _ = resolved
                evidence = _validate_evidence(case, seed, result_directory, stderr_path)
                receipt_valid = _verify_trajectory_receipt(case, seed, expected_authorization["authorization_sha256"], evidence)
                records.append(_record_from_evidence(evidence, status="verified" if receipt_valid else "invalid_receipt", receipt_valid=receipt_valid))
            except Exception as error:
                records.append({"case": case, "seed": seed, "status": "failed", "trajectory_exact": False, "trajectory_receipt_valid": False, "error_type": type(error).__name__, "error": str(error)})
    case_summaries = _case_summaries(records)
    exact_count = sum(item.get("trajectory_exact") is True for item in records)
    result = {
        "schema_version": "arac-exp059-gcb-hcc-lane-verification-v1",
        "authorization_match": authorization == expected_authorization,
        "trajectory_count_match": len(records) == len(CASES) * len(SEEDS),
        "exact_match_count": exact_count,
        "all_trajectories_exact": exact_count == len(records) == len(CASES) * len(SEEDS),
        "all_trajectory_receipts_valid": all(item.get("trajectory_receipt_valid") is True for item in records),
        "case_summaries": case_summaries,
        "all_aggregates_exact": all(item["exact_aggregate_match"] for item in case_summaries),
    }
    result["verification_passed"] = all(result[key] for key in ("authorization_match", "trajectory_count_match", "all_trajectories_exact", "all_trajectory_receipts_valid", "all_aggregates_exact"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--jobs", type=int, default=MAX_JOBS)
    args = parser.parse_args(argv)
    result = run_lane(jobs=args.jobs) if args.command == "run" else verify_lane()
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("verification_passed", result.get("lane_recovered", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
