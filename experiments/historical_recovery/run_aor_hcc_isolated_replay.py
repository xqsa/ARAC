"""Reproduce the historical EXP-057 A1/seed117 trajectory with HCC intact."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Any

from experiments.historical_recovery.audit_exp052_environment import build_report
from experiments.historical_recovery.recover_session_sources import (
    read_patch_events,
    recover_sources_at_boundary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "historical-level-recovery"
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "hcc-action-historical-recovery"
CANDIDATE_ROOT = HISTORICAL_TASK_ROOT / "raw" / "replay-tree-candidate-v1"
CANDIDATE_MANIFEST = CANDIDATE_ROOT / "candidate_manifest.json"
EXACT_WORKER = (
    HISTORICAL_TASK_ROOT
    / "raw"
    / "session-source-recovery"
    / "exact-content"
    / "2d870d14fa536dee488d45a69abea19e50e86dc20748b026d5fc4a16afcb4165"
    / "experiments"
    / "pilots"
    / "exp_057_a_series_aor_25seed"
    / "_worker.py"
)
ARCHIVED_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "exp_057_a_series_aor_25seed"
    / "a1-a6-25seed-v1"
    / "runs"
    / "A1"
    / "seed_117"
)
ARCHIVED_SUMMARY = ARCHIVED_ROOT / "run_summary.json"
ARCHIVED_RECEIPT = ARCHIVED_ROOT / "execution_receipt.json"
OUTPUT_ROOT = TASK_ROOT / "raw" / "aor-a1-seed117-hcc-reproduction-v1"
SOURCE_TREE = OUTPUT_ROOT / "source-tree"
RUN_ROOT = OUTPUT_ROOT / "run"
ABORTED_LANE_ROOT = TASK_ROOT / "raw" / "aor-a1-a6-25seed-hcc-reproduction-v1"
LANE_OUTPUT_ROOT = TASK_ROOT / "raw" / "aor-a1-a6-25seed-hcc-reproduction-v2"
PYTHON_EXECUTABLE = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"

AOR_ACTION_PATH = "src/arac/actions/aor.py"
AOR_WORKER_PATH = "experiments/pilots/exp_057_a_series_aor_25seed/_worker.py"
EXACT_ACTION_SHA256 = "c2dba76bee9822c5126aa17753f0858c3abff80ecdf1b3a6c3b5d48a1b2df136"
EXACT_WORKER_SHA256 = "2d870d14fa536dee488d45a69abea19e50e86dc20748b026d5fc4a16afcb4165"
HCC_SOURCE_HASHES = {
    "vendor/hcc/HCC/OPT/CMAES/cmaes.py": (
        "128a208adfee4f7ec548b5db6415280fa323693726bab7caca9f22be9447e5c1"
    ),
    "vendor/hcc/HCC/OPT/CMAES/es.py": (
        "6c771ec1099ec54af337720664c2c32c728123c89aef462bcb78e9da671e0d13"
    ),
    "vendor/hcc/HCC/OPT/CMAES/optimizer.py": (
        "14f586c12b8860575abc4538d4d01bc021362177a9da6abd8275cbf2889603fd"
    ),
    "vendor/hcc/HCC/OPT/CMAES/sepcmaes.py": (
        "43c104d8560a01ffdad4d073e808a6bd557ef2c4dfae20664aba996db580ecd3"
    ),
}
CASE = "A1"
SEED = 117
MAX_FES = 3_000_000
HISTORICAL_FINAL_ERROR = 78047.92159464832
AOR_CASES = tuple(f"A{index}" for index in range(1, 7))
HISTORICAL_SEEDS = tuple(range(117, 142))
LANE_JOBS = 24
EXACT_RESULT_FIELDS = (
    "backend",
    "best_x_sha256",
    "case",
    "configured_max_fes",
    "dimension",
    "final_error",
    "fitness_evaluations",
    "initial_mean",
    "lower",
    "mapping_sha256",
    "objective_fitness_evaluations",
    "optimizer_route",
    "parameter_sha256",
    "policy_action",
    "policy_protocol",
    "population_size",
    "restart",
    "result_sha256",
    "route_sha256",
    "seed",
    "sigma",
    "upper",
    "worker_protocol_version",
)
INPUT_BYTE_HASH_FIELD = "aob_data_sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def recover_exact_action_source() -> tuple[bytes, dict[str, Any]]:
    rows, artifacts = recover_sources_at_boundary(
        read_patch_events(),
        (AOR_ACTION_PATH,),
        boundary_line=0,
        output_prefix="aor-historical-start-sources",
    )
    if len(rows) != 1 or rows[0].get("recovered") is not True:
        raise ValueError("historical AOR action source was not recovered")
    row = rows[0]
    if row.get("sha256") != EXACT_ACTION_SHA256:
        raise ValueError("historical AOR action source hash drifted")
    matches = [value for path, value in artifacts.items() if path.endswith(AOR_ACTION_PATH)]
    if len(matches) != 1 or _sha256_bytes(matches[0]) != EXACT_ACTION_SHA256:
        raise ValueError("historical AOR action source bytes drifted")
    return matches[0], row


def build_preflight() -> dict[str, Any]:
    if not PYTHON_EXECUTABLE.is_file():
        raise FileNotFoundError(f"project virtual environment missing: {PYTHON_EXECUTABLE}")
    if _sha256(EXACT_WORKER) != EXACT_WORKER_SHA256:
        raise ValueError("exact EXP-057 worker hash drifted")
    action_source, action_provenance = recover_exact_action_source()

    candidate = _read_json(CANDIDATE_MANIFEST)
    overrides = candidate.get("exact_overrides")
    if not isinstance(overrides, dict) or len(overrides) != 41:
        raise ValueError("historical source candidate is incomplete")
    for relative, expected in HCC_SOURCE_HASHES.items():
        if _sha256(CANDIDATE_ROOT / relative) != expected:
            raise ValueError(f"historical HCC source hash drifted: {relative}")

    archived_summary = _read_json(ARCHIVED_SUMMARY)
    archived_receipt = _read_json(ARCHIVED_RECEIPT)
    if archived_receipt.get("worker_sha256") != EXACT_WORKER_SHA256:
        raise ValueError("archived receipt does not bind the exact AOR worker")
    if archived_summary.get("final_error") != HISTORICAL_FINAL_ERROR:
        raise ValueError("archived AOR target drifted")
    if archived_summary.get("fitness_evaluations") != MAX_FES:
        raise ValueError("archived AOR FE budget drifted")

    environment = build_report()
    if environment.get("session_observed_environment_binding") is not True:
        raise ValueError("historical environment evidence is incomplete")
    if environment.get("all_pinned_packages_match") is not True:
        raise ValueError("current package pins do not match the historical environment")

    body = {
        "schema_version": "arac-exp057-aor-hcc-reproduction-authorization-v1",
        "authorization_scope": "one_trajectory_only",
        "claim_boundary": (
            "session_reconstructed_sources_and_version_level_environment_"
            "not_receipt_environment_bound"
        ),
        "target": {"case": CASE, "seed": SEED, "max_fes": MAX_FES},
        "historical_reference": {
            "final_error": HISTORICAL_FINAL_ERROR,
            "summary_sha256": _sha256(ARCHIVED_SUMMARY),
            "receipt_sha256": _sha256(ARCHIVED_RECEIPT),
        },
        "source": {
            "worker_sha256": EXACT_WORKER_SHA256,
            "action_sha256": _sha256_bytes(action_source),
            "action_reversed_patch_count": action_provenance["reversed_patch_count"],
            "hcc_source_hashes": HCC_SOURCE_HASHES,
            "candidate_manifest_sha256": _sha256(CANDIDATE_MANIFEST),
        },
        "session_environment_manifest_sha256": environment["session_environment_manifest_sha256"],
        "receipt_environment_binding": environment["receipt_environment_binding"],
        "output_root": str(OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": _canonical_sha256(body)}


def _materialize_source_tree(action_source: bytes) -> dict[str, Any]:
    shutil.copytree(
        CANDIDATE_ROOT,
        SOURCE_TREE,
        copy_function=os.link,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    action_path = SOURCE_TREE / AOR_ACTION_PATH
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_bytes(action_source)
    worker_path = SOURCE_TREE / AOR_WORKER_PATH
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXACT_WORKER, worker_path)
    hashes = {
        AOR_ACTION_PATH: _sha256(action_path),
        AOR_WORKER_PATH: _sha256(worker_path),
        **{relative: _sha256(SOURCE_TREE / relative) for relative in HCC_SOURCE_HASHES},
    }
    expected = {
        AOR_ACTION_PATH: EXACT_ACTION_SHA256,
        AOR_WORKER_PATH: EXACT_WORKER_SHA256,
        **HCC_SOURCE_HASHES,
    }
    if hashes != expected:
        raise ValueError("materialized AOR/HCC source tree hash drifted")
    manifest = {
        "schema_version": "arac-exp057-aor-hcc-source-tree-v1",
        "base_candidate_manifest_sha256": _sha256(CANDIDATE_MANIFEST),
        "source_hashes": hashes,
    }
    _write_json_atomic(OUTPUT_ROOT / "source_manifest.json", manifest)
    return manifest


def _launch_command() -> list[str]:
    return _worker_command(case=CASE, seed=SEED, output_root=RUN_ROOT)


def _worker_command(*, case: str, seed: int, output_root: Path) -> list[str]:
    return [
        str(PYTHON_EXECUTABLE),
        str(SOURCE_TREE / AOR_WORKER_PATH),
        "--case",
        case,
        "--seed",
        str(seed),
        "--max-fes",
        str(MAX_FES),
        "--output-dir",
        str(output_root),
        "--data-root",
        str(SOURCE_TREE / "vendor" / "hcc" / "AOB" / "AOBG" / "datafile"),
    ]


def _archived_summary_path(case: str, seed: int) -> Path:
    return ARCHIVED_ROOT.parents[1] / case / f"seed_{seed}" / "run_summary.json"


def _field_checks(observed: dict[str, Any], archived: dict[str, Any]) -> dict[str, bool]:
    return {field: observed.get(field) == archived.get(field) for field in EXACT_RESULT_FIELDS}


def _input_byte_hash_match(observed: dict[str, Any], archived: dict[str, Any]) -> bool:
    return observed.get(INPUT_BYTE_HASH_FIELD) == archived.get(INPUT_BYTE_HASH_FIELD)


def run_reproduction() -> dict[str, Any]:
    authorization = build_preflight()
    if OUTPUT_ROOT.exists():
        raise ValueError(f"isolated output already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    _write_json_atomic(OUTPUT_ROOT / "authorization.json", authorization)
    try:
        action_source, _ = recover_exact_action_source()
        source_manifest = _materialize_source_tree(action_source)
        command = _launch_command()
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
        completed = subprocess.run(
            command,
            cwd=SOURCE_TREE,
            env=environment,
            capture_output=True,
            text=True,
            timeout=7_200,
            check=False,
        )
        (OUTPUT_ROOT / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (OUTPUT_ROOT / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"historical AOR worker exited {completed.returncode}")

        observed = _read_json(RUN_ROOT / "run_summary.json")
        archived = _read_json(ARCHIVED_SUMMARY)
        field_checks = _field_checks(observed, archived)
        input_byte_hash_match = _input_byte_hash_match(observed, archived)
        summary = {
            "schema_version": "arac-exp057-aor-hcc-reproduction-result-v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization_sha256": authorization["authorization_sha256"],
            "source_manifest_sha256": _canonical_sha256(source_manifest),
            "command": command,
            "returncode": completed.returncode,
            "field_checks": field_checks,
            "all_exact_fields_match": all(field_checks.values()),
            "input_byte_hash_match": input_byte_hash_match,
            "final_error": observed.get("final_error"),
            "historical_final_error": HISTORICAL_FINAL_ERROR,
            "absolute_delta": abs(float(observed["final_error"]) - HISTORICAL_FINAL_ERROR),
            "exact_budget_match": observed.get("fitness_evaluations") == MAX_FES,
            "verdict": (
                "exact_historical_value_reproduced"
                if all(field_checks.values()) and input_byte_hash_match
                else "historical_value_not_reproduced"
            ),
        }
        _write_json_atomic(OUTPUT_ROOT / "reproduction_summary.json", summary)
        return summary
    except Exception as error:
        _write_json_atomic(
            OUTPUT_ROOT / "reproduction_failure.json",
            {
                "schema_version": "arac-exp057-aor-hcc-reproduction-failure-v1",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "authorization_sha256": authorization["authorization_sha256"],
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def verify_reproduction() -> dict[str, Any]:
    expected_authorization = build_preflight()
    authorization = _read_json(OUTPUT_ROOT / "authorization.json")
    if authorization != expected_authorization:
        raise ValueError("AOR reproduction authorization drifted")
    source_manifest = _read_json(OUTPUT_ROOT / "source_manifest.json")
    source_hashes = source_manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("AOR source manifest is invalid")
    source_hashes_valid = all(
        _sha256(SOURCE_TREE / relative) == expected for relative, expected in source_hashes.items()
    )
    summary = _read_json(OUTPUT_ROOT / "reproduction_summary.json")
    observed = _read_json(RUN_ROOT / "run_summary.json")
    archived = _read_json(ARCHIVED_SUMMARY)
    field_checks = _field_checks(observed, archived)
    input_byte_hash_match = _input_byte_hash_match(observed, archived)
    checks = {
        "authorization_match": summary.get("authorization_sha256")
        == authorization["authorization_sha256"],
        "source_hashes_valid": source_hashes_valid,
        "all_exact_fields_match": all(field_checks.values()),
        "input_byte_hash_match": input_byte_hash_match,
        "exact_historical_value_match": observed.get("final_error") == HISTORICAL_FINAL_ERROR,
        "exact_budget_match": observed.get("fitness_evaluations") == MAX_FES,
        "worker_completed": _read_json(RUN_ROOT / "worker_status.json").get("status")
        == "completed",
    }
    return {
        "schema_version": "arac-exp057-aor-hcc-reproduction-verification-v1",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "field_checks": field_checks,
        "final_error": observed.get("final_error"),
        "fitness_evaluations": observed.get("fitness_evaluations"),
    }


def build_lane_authorization() -> dict[str, Any]:
    representative = verify_reproduction()
    if representative.get("verification_passed") is not True:
        raise ValueError("representative AOR replay must pass before lane expansion")
    body = {
        "schema_version": "arac-exp057-aor-hcc-lane-authorization-v2",
        "authorization_scope": "aor_a1_a6_historical_matrix_only",
        "target": {
            "cases": list(AOR_CASES),
            "seeds": list(HISTORICAL_SEEDS),
            "max_fes": MAX_FES,
            "trajectory_count": len(AOR_CASES) * len(HISTORICAL_SEEDS),
        },
        "representative_authorization_sha256": _read_json(OUTPUT_ROOT / "authorization.json")[
            "authorization_sha256"
        ],
        "source_manifest_sha256": _sha256(OUTPUT_ROOT / "source_manifest.json"),
        "worker_sha256": _sha256(SOURCE_TREE / AOR_WORKER_PATH),
        "action_sha256": _sha256(SOURCE_TREE / AOR_ACTION_PATH),
        "jobs": LANE_JOBS,
        "result_gate": "all_stable_execution_fields_exact",
        "input_byte_provenance": (
            "reported_separately_because_archived_aob_files_were_later_reformatted"
        ),
        "reusable_aborted_lane_root": str(ABORTED_LANE_ROOT),
        "output_root": str(LANE_OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": _canonical_sha256(body)}


def _lane_environment() -> dict[str, str]:
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


def _run_lane_trajectory(case: str, seed: int) -> dict[str, Any]:
    output_root = LANE_OUTPUT_ROOT / "runs" / case / f"seed_{seed}"
    archived_path = _archived_summary_path(case, seed)
    try:
        archived = _read_json(archived_path)
        observed_path = output_root / "run_summary.json"
        if observed_path.is_file():
            observed = _read_json(observed_path)
            checks = _field_checks(observed, archived)
            if not all(checks.values()):
                raise ValueError("existing trajectory does not match its archived result")
            return {
                "case": case,
                "seed": seed,
                "status": "reused_exact",
                "final_error": observed["final_error"],
                "all_exact_fields_match": True,
                "input_byte_hash_match": _input_byte_hash_match(observed, archived),
            }
        if output_root.exists():
            raise ValueError("incomplete trajectory directory already exists")

        prior_path = ABORTED_LANE_ROOT / "runs" / case / f"seed_{seed}" / "run_summary.json"
        if prior_path.is_file():
            observed = _read_json(prior_path)
            checks = _field_checks(observed, archived)
            if all(checks.values()):
                reuse_receipt = {
                    "schema_version": "arac-exp057-aor-hcc-lane-reuse-v1",
                    "case": case,
                    "seed": seed,
                    "source_summary": str(prior_path),
                    "source_summary_sha256": _sha256(prior_path),
                    "all_exact_execution_fields_match": True,
                    "input_byte_hash_match": _input_byte_hash_match(observed, archived),
                }
                _write_json_atomic(output_root / "reuse_receipt.json", reuse_receipt)
                return {
                    "case": case,
                    "seed": seed,
                    "status": "reused_exact_from_aborted_v1",
                    "final_error": observed["final_error"],
                    "all_exact_fields_match": True,
                    "input_byte_hash_match": reuse_receipt["input_byte_hash_match"],
                }

        output_root.parent.mkdir(parents=True, exist_ok=True)
        command = _worker_command(case=case, seed=seed, output_root=output_root)
        completed = subprocess.run(
            command,
            cwd=SOURCE_TREE,
            env=_lane_environment(),
            capture_output=True,
            text=True,
            timeout=7_200,
            check=False,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_root / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"historical AOR worker exited {completed.returncode}")
        observed = _read_json(observed_path)
        checks = _field_checks(observed, archived)
        return {
            "case": case,
            "seed": seed,
            "status": "completed" if all(checks.values()) else "mismatch",
            "final_error": observed["final_error"],
            "all_exact_fields_match": all(checks.values()),
            "input_byte_hash_match": _input_byte_hash_match(observed, archived),
            "failed_fields": [field for field, passed in checks.items() if not passed],
        }
    except Exception as error:
        failure = {
            "case": case,
            "seed": seed,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "all_exact_fields_match": False,
            "input_byte_hash_match": False,
        }
        _write_json_atomic(output_root / "reproduction_failure.json", failure)
        return failure


def _lane_case_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for case in AOR_CASES:
        observed = [
            float(record["final_error"])
            for record in records
            if record["case"] == case and record.get("all_exact_fields_match") is True
        ]
        archived = [
            float(_read_json(_archived_summary_path(case, seed))["final_error"])
            for seed in HISTORICAL_SEEDS
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
                "input_byte_hash_mismatch_count": sum(
                    record.get("input_byte_hash_match") is False
                    for record in records
                    if record["case"] == case and record.get("all_exact_fields_match") is True
                ),
            }
        )
    return summaries


def run_lane(*, jobs: int = LANE_JOBS) -> dict[str, Any]:
    if jobs < 1 or jobs > LANE_JOBS:
        raise ValueError(f"jobs must be between 1 and {LANE_JOBS}")
    authorization = build_lane_authorization()
    authorization_path = LANE_OUTPUT_ROOT / "authorization.json"
    if authorization_path.is_file():
        if _read_json(authorization_path) != authorization:
            raise ValueError("AOR lane authorization drifted")
    else:
        if LANE_OUTPUT_ROOT.exists() and any(LANE_OUTPUT_ROOT.iterdir()):
            raise ValueError("AOR lane output exists without authorization")
        _write_json_atomic(authorization_path, authorization)

    targets = [(case, seed) for case in AOR_CASES for seed in HISTORICAL_SEEDS]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {
            executor.submit(_run_lane_trajectory, case, seed): (case, seed)
            for case, seed in targets
        }
        for completed_count, future in enumerate(
            concurrent.futures.as_completed(pending),
            start=1,
        ):
            record = future.result()
            records.append(record)
            print(
                f"AOR lane {completed_count}/{len(targets)} "
                f"{record['case']}/seed{record['seed']} {record['status']}",
                flush=True,
            )
            _write_json_atomic(
                LANE_OUTPUT_ROOT / "run_progress.json",
                {
                    "schema_version": "arac-exp057-aor-hcc-lane-progress-v2",
                    "authorization_sha256": authorization["authorization_sha256"],
                    "completed_count": completed_count,
                    "target_count": len(targets),
                    "records": sorted(records, key=lambda item: (item["case"], item["seed"])),
                },
            )

    records.sort(key=lambda item: (item["case"], item["seed"]))
    exact_count = sum(record.get("all_exact_fields_match") is True for record in records)
    case_summaries = _lane_case_summaries(records)
    summary = {
        "schema_version": "arac-exp057-aor-hcc-lane-result-v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization["authorization_sha256"],
        "target_count": len(targets),
        "record_count": len(records),
        "exact_match_count": exact_count,
        "failed_count": sum(record["status"] == "failed" for record in records),
        "mismatch_count": sum(record["status"] == "mismatch" for record in records),
        "input_byte_hash_mismatch_count": sum(
            record.get("input_byte_hash_match") is False
            and record.get("all_exact_fields_match") is True
            for record in records
        ),
        "case_summaries": case_summaries,
        "lane_recovered": exact_count == len(targets)
        and all(item["exact_aggregate_match"] for item in case_summaries),
    }
    _write_json_atomic(LANE_OUTPUT_ROOT / "lane_summary.json", summary)
    return summary


def verify_lane() -> dict[str, Any]:
    expected_authorization = build_lane_authorization()
    authorization = _read_json(LANE_OUTPUT_ROOT / "authorization.json")
    authorization_match = authorization == expected_authorization
    records = []
    for case in AOR_CASES:
        for seed in HISTORICAL_SEEDS:
            observed_path = LANE_OUTPUT_ROOT / "runs" / case / f"seed_{seed}" / "run_summary.json"
            if not observed_path.is_file():
                reuse_path = observed_path.with_name("reuse_receipt.json")
                if reuse_path.is_file():
                    reuse = _read_json(reuse_path)
                    source_path = Path(str(reuse.get("source_summary", ""))).resolve()
                    if (
                        ABORTED_LANE_ROOT.resolve() not in source_path.parents
                        or not source_path.is_file()
                        or _sha256(source_path) != reuse.get("source_summary_sha256")
                    ):
                        raise ValueError("AOR lane reuse receipt is invalid")
                    observed_path = source_path
                else:
                    records.append(
                        {
                            "case": case,
                            "seed": seed,
                            "status": "missing",
                            "all_exact_fields_match": False,
                            "input_byte_hash_match": False,
                        }
                    )
                    continue
            observed = _read_json(observed_path)
            archived = _read_json(_archived_summary_path(case, seed))
            checks = _field_checks(observed, archived)
            records.append(
                {
                    "case": case,
                    "seed": seed,
                    "status": "verified" if all(checks.values()) else "mismatch",
                    "final_error": observed.get("final_error"),
                    "all_exact_fields_match": all(checks.values()),
                    "input_byte_hash_match": _input_byte_hash_match(observed, archived),
                }
            )
    case_summaries = _lane_case_summaries(records)
    exact_count = sum(record["all_exact_fields_match"] for record in records)
    checks = {
        "authorization_match": authorization_match,
        "trajectory_count_match": len(records) == 150,
        "all_trajectories_exact": exact_count == 150,
        "all_aggregates_exact": all(item["exact_aggregate_match"] for item in case_summaries),
    }
    return {
        "schema_version": "arac-exp057-aor-hcc-lane-verification-v2",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "exact_match_count": exact_count,
        "input_byte_hash_mismatch_count": sum(
            record.get("input_byte_hash_match") is False
            and record.get("all_exact_fields_match") is True
            for record in records
        ),
        "case_summaries": case_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "run", "verify", "run-lane", "verify-lane"),
    )
    parser.add_argument("--jobs", type=int, default=LANE_JOBS)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(build_preflight(), sort_keys=True))
        return 0
    if args.command == "verify":
        verification = verify_reproduction()
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["verification_passed"] else 1
    if args.command == "verify-lane":
        verification = verify_lane()
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["verification_passed"] else 1
    if args.command == "run-lane":
        summary = run_lane(jobs=args.jobs)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["lane_recovered"] else 1
    summary = run_reproduction()
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_exact_fields_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
