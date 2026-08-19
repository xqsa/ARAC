"""Reproduce the historical EXP-058 CTP S1/seed117 trajectory with HCC intact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from experiments.historical_recovery.audit_exp052_environment import build_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "historical-level-recovery"
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "hcc-action-historical-recovery"
CANDIDATE_ROOT = HISTORICAL_TASK_ROOT / "raw" / "replay-tree-candidate-v1"
CANDIDATE_MANIFEST = CANDIDATE_ROOT / "candidate_manifest.json"
EXACT_RUNNER = (
    HISTORICAL_TASK_ROOT
    / "raw"
    / "session-source-recovery"
    / "exact-runner"
    / "9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594"
    / "scripts"
    / "hcc_smoke_runner.py"
)
SESSION_RECONSTRUCTED_GCB_STABLE = (
    TASK_ROOT / "raw" / "ctp-dependency-sources" / "src" / "arac" / "actions" / "gcb_stable.py"
)
ARCHIVED_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "exp_058_ctp_stable_v2_25seed"
    / "validation"
    / "runs"
    / "S1"
    / "seed_117"
    / "exp_058_ctp_stable_v2_25seed-s1-seed117"
    / "schwefel"
)
ARCHIVED_RUNS_ROOT = (
    REPOSITORY_ROOT / "results" / "exp_058_ctp_stable_v2_25seed" / "validation" / "runs"
)
OUTPUT_ROOT = TASK_ROOT / "raw" / "ctp-s1-seed117-hcc-reproduction-v2"
SOURCE_TREE = OUTPUT_ROOT / "source-tree"
RUN_ROOT = OUTPUT_ROOT / "run"
LANE_OUTPUT_ROOT = TASK_ROOT / "raw" / "ctp-s1-s6-25seed-hcc-reproduction-v1"
TIMESTAMP = "exp_058_ctp_stable_v2_25seed-s1-seed117"
PYTHON_EXECUTABLE = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"

MAX_FES = 3_000_000
CASE = "S1"
SEED = 117
FUNCTION = "schwefel"
FUNCTION_ID = 1
CTP_CASES = ("S1", "S2", "S3", "S4", "S5", "S6")
CASE_IDS = {case: index for index, case in enumerate(CTP_CASES, start=1)}
HISTORICAL_SEEDS = tuple(range(117, 142))
LANE_JOBS = 24
EXACT_RUNNER_SHA256 = "9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594"
EXPECTED_CTP_SHA256 = "0b71a603fb4eb065099146dfb335f5927ab1f1b5171018bf58a5ce99eb9d79ee"
EXPECTED_CTP_STABLE_SHA256 = "d78084481cac9ff1a5ed4d6899eb677efa5bada500dac035b1846f27305499b6"
EXPECTED_GCB_STABLE_SHA256 = "f01d25ca0427284247cf9734c1311e044e1be3b6193207c7a15a0fffa99bf2ee"
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
RUNNER_RELATIVE_PATH = "scripts/hcc_smoke_runner.py"
CTP_RELATIVE_PATH = "src/arac/actions/ctp.py"
CTP_STABLE_RELATIVE_PATH = "src/arac/actions/ctp_stable.py"
GCB_STABLE_RELATIVE_PATH = "src/arac/actions/gcb_stable.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _archived(path_name: str) -> dict[str, Any]:
    return _read_json(ARCHIVED_ROOT / path_name)


def _environment_gate() -> dict[str, Any]:
    environment = build_report()
    if environment.get("session_observed_environment_binding") is not True:
        raise ValueError("historical environment evidence is incomplete")
    if environment.get("all_pinned_packages_match") is not True:
        raise ValueError("current package pins do not match historical environment")
    return environment


def build_preflight() -> dict[str, Any]:
    required = (
        PYTHON_EXECUTABLE,
        CANDIDATE_MANIFEST,
        EXACT_RUNNER,
        SESSION_RECONSTRUCTED_GCB_STABLE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing historical replay input: " + ", ".join(missing))
    if _sha256(EXACT_RUNNER) != EXACT_RUNNER_SHA256:
        raise ValueError("exact EXP-058 runner hash drifted")
    if _sha256(SESSION_RECONSTRUCTED_GCB_STABLE) != EXPECTED_GCB_STABLE_SHA256:
        raise ValueError("session-reconstructed GCB-Stable source hash drifted")
    candidate = _read_json(CANDIDATE_MANIFEST)
    overrides = candidate.get("exact_overrides")
    if not isinstance(overrides, dict) or len(overrides) != 41:
        raise ValueError("historical candidate source closure is incomplete")
    for relative, expected in HCC_SOURCE_HASHES.items():
        if _sha256(CANDIDATE_ROOT / relative) != expected:
            raise ValueError(f"historical HCC source hash drifted: {relative}")
    for relative, expected in {
        CTP_RELATIVE_PATH: EXPECTED_CTP_SHA256,
        CTP_STABLE_RELATIVE_PATH: EXPECTED_CTP_STABLE_SHA256,
    }.items():
        if _sha256(CANDIDATE_ROOT / relative) != expected:
            raise ValueError(f"historical CTP source hash drifted: {relative}")

    archived_summary = _archived("run_summary.json")
    archived_receipt = _archived("exp058_execution_receipt.json")
    archived_action = _archived("ctp_stable_action.json")
    if archived_receipt.get("runner_sha256") != EXACT_RUNNER_SHA256:
        raise ValueError("archived receipt does not bind exact EXP-058 runner")
    if archived_receipt.get("runtime_action") != "ctp_stable":
        raise ValueError("archived CTP action drifted")
    if archived_summary.get("fitness_evaluations") != MAX_FES:
        raise ValueError("archived CTP FE budget drifted")
    if archived_summary.get("runtime_action") != "ctp_stable":
        raise ValueError("archived CTP summary action drifted")
    if archived_action.get("terminal_fe") != MAX_FES:
        raise ValueError("archived CTP action terminal budget drifted")
    environment = _environment_gate()
    body = {
        "schema_version": "arac-exp058-ctp-hcc-reproduction-authorization-v2",
        "authorization_scope": "one_trajectory_only",
        "claim_boundary": (
            "session_reconstructed_sources_and_version_level_environment_"
            "not_receipt_environment_bound"
        ),
        "target": {
            "case": CASE,
            "function": FUNCTION,
            "function_id": FUNCTION_ID,
            "seed": SEED,
            "max_fes": MAX_FES,
        },
        "historical_reference": {
            "summary_sha256": _sha256(ARCHIVED_ROOT / "run_summary.json"),
            "receipt_sha256": _sha256(ARCHIVED_ROOT / "exp058_execution_receipt.json"),
            "action_sha256": _sha256(ARCHIVED_ROOT / "ctp_stable_action.json"),
            "final_error": archived_summary["final_error"],
        },
        "source": {
            "runner_sha256": EXACT_RUNNER_SHA256,
            "candidate_manifest_sha256": _sha256(CANDIDATE_MANIFEST),
            "ctp_sha256": EXPECTED_CTP_SHA256,
            "ctp_stable_sha256": EXPECTED_CTP_STABLE_SHA256,
            "gcb_stable_sha256": EXPECTED_GCB_STABLE_SHA256,
            "gcb_stable_provenance": "three_session_patch_events_before_exp058",
            "hcc_source_hashes": HCC_SOURCE_HASHES,
        },
        "session_environment_manifest_sha256": environment["session_environment_manifest_sha256"],
        "receipt_environment_binding": environment["receipt_environment_binding"],
        "output_root": str(OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": _canonical_sha256(body)}


def _materialize_source_tree() -> dict[str, Any]:
    shutil.copytree(
        CANDIDATE_ROOT,
        SOURCE_TREE,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    runner_path = SOURCE_TREE / RUNNER_RELATIVE_PATH
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXACT_RUNNER, runner_path)
    gcb_stable_path = SOURCE_TREE / GCB_STABLE_RELATIVE_PATH
    gcb_stable_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SESSION_RECONSTRUCTED_GCB_STABLE, gcb_stable_path)
    source_hashes = {
        RUNNER_RELATIVE_PATH: _sha256(runner_path),
        CTP_RELATIVE_PATH: _sha256(SOURCE_TREE / CTP_RELATIVE_PATH),
        CTP_STABLE_RELATIVE_PATH: _sha256(SOURCE_TREE / CTP_STABLE_RELATIVE_PATH),
        GCB_STABLE_RELATIVE_PATH: _sha256(gcb_stable_path),
        **{relative: _sha256(SOURCE_TREE / relative) for relative in HCC_SOURCE_HASHES},
    }
    expected = {
        RUNNER_RELATIVE_PATH: EXACT_RUNNER_SHA256,
        CTP_RELATIVE_PATH: EXPECTED_CTP_SHA256,
        CTP_STABLE_RELATIVE_PATH: EXPECTED_CTP_STABLE_SHA256,
        GCB_STABLE_RELATIVE_PATH: EXPECTED_GCB_STABLE_SHA256,
        **HCC_SOURCE_HASHES,
    }
    if source_hashes != expected:
        raise ValueError("materialized CTP/HCC source tree hash drifted")
    manifest = {
        "schema_version": "arac-exp058-ctp-hcc-source-tree-v2",
        "base_candidate_manifest_sha256": _sha256(CANDIDATE_MANIFEST),
        "source_hashes": source_hashes,
    }
    _write_json_atomic(OUTPUT_ROOT / "source_manifest.json", manifest)
    return manifest


def _command() -> list[str]:
    return _trajectory_command(case=CASE, seed=SEED, output_root=RUN_ROOT)


def _trajectory_command(*, case: str, seed: int, output_root: Path) -> list[str]:
    return [
        str(PYTHON_EXECUTABLE),
        str(SOURCE_TREE / RUNNER_RELATIVE_PATH),
        "--functions",
        FUNCTION,
        "--ids",
        str(CASE_IDS[case]),
        "--output-root",
        str(output_root),
        "--aob-data-root",
        str(SOURCE_TREE / "vendor" / "hcc" / "AOB" / "AOBG" / "datafile"),
        "--timestamp",
        _trajectory_id(case, seed),
        "--seed",
        str(seed),
        "--max-fes",
        str(MAX_FES),
        "--arac-action",
        "native_eq8",
        "--budget-accounting",
        "strict",
        "--search-state-backend",
        "phase_i_mmes",
        "--relation-policy",
        "controller_v31",
        "--evidence-overlay-mode",
        "off",
        "--runtime-probe-repair-mode",
        "hard_repair",
        "--group-optimizer-mode",
        "full_cmaes",
        "--s-series-action",
        "ctp_stable",
        "--terminal-noop-fill",
        "--verbose",
        "0",
        "--skip-plots",
        "--no-cmaes-restart",
    ]


def _trajectory_id(case: str, seed: int) -> str:
    return f"exp_058_ctp_stable_v2_25seed-{case.lower()}-seed{seed}"


def _result_directory(output_root: Path, case: str, seed: int) -> Path:
    return output_root / _trajectory_id(case, seed) / FUNCTION


def _archived_result_directory(case: str, seed: int) -> Path:
    return ARCHIVED_RUNS_ROOT / case / f"seed_{seed}" / _trajectory_id(case, seed) / FUNCTION


def _run_summary_path() -> Path:
    return RUN_ROOT / TIMESTAMP / FUNCTION / "run_summary.json"


def _action_path() -> Path:
    return RUN_ROOT / TIMESTAMP / FUNCTION / "ctp_stable_action.json"


def _compare_json(observed: dict[str, Any], archived: dict[str, Any]) -> dict[str, bool]:
    keys = sorted(set(observed) | set(archived))
    return {key: observed.get(key) == archived.get(key) for key in keys}


def _historical_receipt_checks(
    observed: dict[str, Any], archived_receipt: dict[str, Any]
) -> dict[str, bool]:
    return {
        "runner_source_bound": archived_receipt.get("runner_sha256") == EXACT_RUNNER_SHA256,
        "runtime_action": archived_receipt.get("runtime_action")
        == observed.get("runtime_action")
        == "ctp_stable",
        "runtime_action_hash": archived_receipt.get("runtime_action_hash")
        == observed.get("runtime_action_hash"),
        "final_error": archived_receipt.get("final_error") == observed.get("final_error"),
        "configured_max_fes": archived_receipt.get("configured_max_fes")
        == observed.get("configured_max_fes")
        == MAX_FES,
    }


def _finalize_completed_run(
    authorization: dict[str, Any],
    command: list[str],
    *,
    recovered_from_controller_failure: bool,
) -> dict[str, Any]:
    observed = _read_json(_run_summary_path())
    archived = _archived("run_summary.json")
    action = _read_json(_action_path())
    archived_action = _archived("ctp_stable_action.json")
    archived_receipt_path = ARCHIVED_ROOT / "exp058_execution_receipt.json"
    archived_receipt = _read_json(archived_receipt_path)
    summary_checks = _compare_json(observed, archived)
    action_checks = _compare_json(action, archived_action)
    historical_receipt_checks = _historical_receipt_checks(observed, archived_receipt)
    stderr_empty = (OUTPUT_ROOT / "stderr.log").read_text(encoding="utf-8") == ""

    receipt_body = {
        "schema_version": "arac-exp058-ctp-hcc-reproduction-receipt-v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization["authorization_sha256"],
        "source_manifest_sha256": _sha256(OUTPUT_ROOT / "source_manifest.json"),
        "historical_campaign_receipt_sha256": _sha256(archived_receipt_path),
        "historical_campaign_receipt_generated_by_runner": False,
        "runner_sha256": EXACT_RUNNER_SHA256,
        "command": command,
        "runtime_action": observed.get("runtime_action"),
        "runtime_action_hash": observed.get("runtime_action_hash"),
        "configured_max_fes": observed.get("configured_max_fes"),
        "fitness_evaluations": observed.get("fitness_evaluations"),
        "final_error": observed.get("final_error"),
        "run_summary_sha256": _sha256(_run_summary_path()),
        "action_sha256": _sha256(_action_path()),
        "stdout_sha256": _sha256(OUTPUT_ROOT / "stdout.log"),
        "stderr_sha256": _sha256(OUTPUT_ROOT / "stderr.log"),
        "recovered_from_post_run_controller_failure": recovered_from_controller_failure,
    }
    if recovered_from_controller_failure:
        receipt_body["controller_failure_sha256"] = _sha256(
            OUTPUT_ROOT / "reproduction_failure.json"
        )
    reproduction_receipt = {
        **receipt_body,
        "receipt_sha256": _canonical_sha256(receipt_body),
    }
    reproduction_receipt_path = OUTPUT_ROOT / "reproduction_receipt.json"
    _write_json_atomic(reproduction_receipt_path, reproduction_receipt)

    result = {
        "schema_version": "arac-exp058-ctp-hcc-reproduction-result-v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization["authorization_sha256"],
        "source_manifest_sha256": _sha256(OUTPUT_ROOT / "source_manifest.json"),
        "reproduction_receipt_sha256": _sha256(reproduction_receipt_path),
        "command": command,
        "returncode": 0,
        "summary_checks": summary_checks,
        "action_checks": action_checks,
        "historical_receipt_checks": historical_receipt_checks,
        "all_summary_fields_exact": all(summary_checks.values()),
        "all_action_fields_exact": all(action_checks.values()),
        "all_historical_receipt_fields_exact": all(historical_receipt_checks.values()),
        "final_error": observed.get("final_error"),
        "historical_final_error": archived.get("final_error"),
        "exact_budget_match": observed.get("fitness_evaluations") == MAX_FES,
        "stderr_empty": stderr_empty,
        "recovered_from_post_run_controller_failure": recovered_from_controller_failure,
    }
    result["verification_passed"] = all(
        result[key]
        for key in (
            "all_summary_fields_exact",
            "all_action_fields_exact",
            "all_historical_receipt_fields_exact",
            "exact_budget_match",
            "stderr_empty",
        )
    )
    _write_json_atomic(OUTPUT_ROOT / "reproduction_summary.json", result)
    return result


def run_reproduction() -> dict[str, Any]:
    authorization = build_preflight()
    if OUTPUT_ROOT.exists():
        raise ValueError(f"isolated output already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    _write_json_atomic(OUTPUT_ROOT / "authorization.json", authorization)
    try:
        _materialize_source_tree()
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
        command = _command()
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
            raise RuntimeError(f"historical EXP-058 runner exited {completed.returncode}")
        return _finalize_completed_run(
            authorization,
            command,
            recovered_from_controller_failure=False,
        )
    except Exception as error:
        _write_json_atomic(
            OUTPUT_ROOT / "reproduction_failure.json",
            {
                "schema_version": "arac-exp058-ctp-hcc-reproduction-failure-v2",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "authorization_sha256": authorization["authorization_sha256"],
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def finalize_reproduction() -> dict[str, Any]:
    authorization = build_preflight()
    if _read_json(OUTPUT_ROOT / "authorization.json") != authorization:
        raise ValueError("existing V2 authorization does not match current preflight")
    failure = _read_json(OUTPUT_ROOT / "reproduction_failure.json")
    if failure.get("error_type") != "FileNotFoundError" or not str(
        failure.get("error", "")
    ).endswith("exp058_execution_receipt.json'"):
        raise ValueError("existing V2 failure is not the known post-run receipt lookup")
    if _sha256(SOURCE_TREE / RUNNER_RELATIVE_PATH) != EXACT_RUNNER_SHA256:
        raise ValueError("existing V2 runner source hash drifted")
    return _finalize_completed_run(
        authorization,
        _command(),
        recovered_from_controller_failure=True,
    )


def verify_reproduction() -> dict[str, Any]:
    expected_authorization = build_preflight()
    authorization = _read_json(OUTPUT_ROOT / "authorization.json")
    summary = _read_json(OUTPUT_ROOT / "reproduction_summary.json")
    receipt_path = OUTPUT_ROOT / "reproduction_receipt.json"
    receipt = _read_json(receipt_path)
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    checks = {
        "authorization_match": authorization == expected_authorization,
        "source_manifest_hash_valid": _sha256(OUTPUT_ROOT / "source_manifest.json")
        == summary.get("source_manifest_sha256"),
        "runner_hash_valid": _sha256(SOURCE_TREE / RUNNER_RELATIVE_PATH) == EXACT_RUNNER_SHA256,
        "reproduction_receipt_hash_valid": _sha256(receipt_path)
        == summary.get("reproduction_receipt_sha256"),
        "reproduction_receipt_content_valid": _canonical_sha256(receipt_body)
        == receipt.get("receipt_sha256"),
        "run_summary_hash_valid": _sha256(_run_summary_path()) == receipt.get("run_summary_sha256"),
        "action_hash_valid": _sha256(_action_path()) == receipt.get("action_sha256"),
        "verification_passed": summary.get("verification_passed") is True,
        "exact_budget_match": summary.get("exact_budget_match") is True,
        "stderr_empty": summary.get("stderr_empty") is True,
    }
    return {
        "schema_version": "arac-exp058-ctp-hcc-reproduction-verification-v2",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "final_error": summary.get("final_error"),
        "historical_final_error": summary.get("historical_final_error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "finalize", "verify"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(build_preflight(), sort_keys=True))
        return 0
    if args.command == "verify":
        result = verify_reproduction()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["verification_passed"] else 1
    if args.command == "finalize":
        result = finalize_reproduction()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["verification_passed"] else 1
    result = run_reproduction()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
