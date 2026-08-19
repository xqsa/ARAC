"""Run the single session-provenance EXP-052 E1/SMP/seed117 reproduction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from experiments.historical_recovery.audit_exp052_environment import build_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPOSITORY_ROOT / ".codex-tasks" / "historical-level-recovery"
CANDIDATE_ROOT = TASK_ROOT / "raw" / "replay-tree-candidate-v1"
CANDIDATE_MANIFEST = CANDIDATE_ROOT / "candidate_manifest.json"
OUTPUT_ROOT = TASK_ROOT / "raw" / "exp052-e1-seed117-session-reproduction-v1"
HISTORICAL_RUNNER = (
    CANDIDATE_ROOT
    / "experiments"
    / "pilots"
    / "exp_052_e_series_smp_paired_gate"
    / "run.py"
)
HISTORICAL_CONFIG = HISTORICAL_RUNNER.with_name("config.json")
PYTHON_EXECUTABLE = REPOSITORY_ROOT / ".venv" / "Scripts" / "python.exe"

CASE = "E1"
CONDITION = "candidate_smp"
SEED = 117
MAX_FES = 3_000_000
HISTORICAL_FINAL_ERROR = 5.983267874603139e-7
HISTORICAL_P90 = 1.8255606813339802


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
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


def build_preflight() -> dict[str, Any]:
    if not PYTHON_EXECUTABLE.is_file():
        raise FileNotFoundError(f"project virtual environment missing: {PYTHON_EXECUTABLE}")
    candidate = _read_json(CANDIDATE_MANIFEST)
    if candidate.get("schema_version") != "arac-exp052-isolated-replay-candidate-v1":
        raise ValueError("candidate manifest schema drifted")
    overrides = candidate.get("exact_overrides")
    if not isinstance(overrides, dict) or len(overrides) != 41:
        raise ValueError("candidate must contain exactly 41 source overrides")
    mismatches = []
    for relative, metadata in overrides.items():
        path = CANDIDATE_ROOT / relative
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"candidate source hashes drifted: {mismatches}")
    validation = candidate.get("candidate_validation", {})
    required_validation = (
        "compileall_passed",
        "runner_import_passed",
        "exp052_import_passed",
    )
    if not all(validation.get(key) is True for key in required_validation):
        raise ValueError("candidate compile/import validation is incomplete")

    environment = build_report()
    if environment.get("session_observed_environment_binding") is not True:
        raise ValueError("session-observed environment binding is incomplete")
    if environment.get("all_pinned_packages_match") is not True:
        raise ValueError("current candidate package pins do not match history")

    body = {
        "schema_version": "arac-exp052-session-reproduction-authorization-v1",
        "authorization_scope": "one_trajectory_only",
        "claim_boundary": "version_level_session_provenance_not_bitwise_receipt_bound",
        "target": {
            "case": CASE,
            "condition": CONDITION,
            "seed": SEED,
            "max_fes": MAX_FES,
        },
        "historical_reference": {
            "final_error": HISTORICAL_FINAL_ERROR,
            "p90": HISTORICAL_P90,
        },
        "candidate_root": str(CANDIDATE_ROOT),
        "candidate_manifest_sha256": _sha256(CANDIDATE_MANIFEST),
        "exact_override_count": len(overrides),
        "source_hashes_valid": True,
        "historical_config_sha256": _sha256(HISTORICAL_CONFIG),
        "historical_runner_sha256": _sha256(
            CANDIDATE_ROOT / "scripts" / "hcc_smoke_runner.py"
        ),
        "session_environment_manifest_sha256": environment[
            "session_environment_manifest_sha256"
        ],
        "session_observed_environment_binding": True,
        "receipt_environment_binding": environment["receipt_environment_binding"],
        "output_root": str(OUTPUT_ROOT),
    }
    return {**body, "authorization_sha256": _canonical_sha256(body)}


def _load_historical_runner() -> ModuleType:
    loaded_arac = [name for name in sys.modules if name == "arac" or name.startswith("arac.")]
    if loaded_arac:
        raise RuntimeError(f"production arac modules already loaded: {loaded_arac[:5]}")
    candidate_src = str(CANDIDATE_ROOT / "src")
    sys.path.insert(0, str(CANDIDATE_ROOT))
    sys.path.insert(0, candidate_src)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (candidate_src, str(CANDIDATE_ROOT), existing_pythonpath)
        if item
    )
    spec = importlib.util.spec_from_file_location("_arac_exp052_historical_runner", HISTORICAL_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load historical runner: {HISTORICAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    arac_module = sys.modules.get("arac")
    arac_path = Path(str(getattr(arac_module, "__file__", ""))).resolve()
    if CANDIDATE_ROOT.resolve() not in arac_path.parents:
        raise RuntimeError(f"historical runner imported non-candidate arac: {arac_path}")
    return module


def run_reproduction() -> dict[str, Any]:
    authorization = build_preflight()
    if OUTPUT_ROOT.exists():
        raise ValueError(f"isolated output already exists: {OUTPUT_ROOT}")
    _write_json_atomic(OUTPUT_ROOT / "authorization.json", authorization)

    try:
        runner = _load_historical_runner()
        runner.load_config(HISTORICAL_CONFIG)
        spec = runner.RunSpec(
            experiment_id="exp_052_e_series_smp_paired_gate",
            case=CASE,
            condition=CONDITION,
            seed=SEED,
            max_fes=MAX_FES,
            output_root=OUTPUT_ROOT,
        )
        result = runner.run_one(
            spec,
            _sha256(HISTORICAL_CONFIG),
            str(PYTHON_EXECUTABLE),
            require_restore=True,
        )
        final_error = float(result["final_error"]) if result.get("ok") else math.inf
        integrity_passed = (
            result.get("ok") is True
            and int(result.get("fitness_evaluations", -1)) == MAX_FES
            and int(result.get("restore_count", 0)) > 0
            and int(result.get("abstain_count", -1)) == 0
        )
        exact_value_match = final_error == HISTORICAL_FINAL_ERROR
        record_recovered = integrity_passed and final_error <= HISTORICAL_FINAL_ERROR
        p90_passed = integrity_passed and final_error <= HISTORICAL_P90
        summary = {
            "schema_version": "arac-exp052-session-reproduction-result-v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization_sha256": authorization["authorization_sha256"],
            "claim_boundary": authorization["claim_boundary"],
            "target": authorization["target"],
            "historical_reference": authorization["historical_reference"],
            "result": result,
            "final_error": final_error,
            "exact_historical_value_match": exact_value_match,
            "absolute_delta_from_historical": abs(final_error - HISTORICAL_FINAL_ERROR),
            "ratio_to_historical": final_error / HISTORICAL_FINAL_ERROR,
            "integrity_passed": integrity_passed,
            "seed117_record_recovered_or_exceeded": record_recovered,
            "historical_p90_passed": p90_passed,
            "verdict": (
                "exact_historical_value_reproduced"
                if exact_value_match and integrity_passed
                else "historical_seed117_record_recovered_or_exceeded"
                if record_recovered
                else "historical_p90_only_passed"
                if p90_passed
                else "historical_level_not_recovered"
            ),
        }
        _write_json_atomic(OUTPUT_ROOT / "reproduction_summary.json", summary)
        return summary
    except Exception as error:
        _write_json_atomic(
            OUTPUT_ROOT / "reproduction_failure.json",
            {
                "schema_version": "arac-exp052-session-reproduction-failure-v1",
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
        raise ValueError("reproduction authorization drifted")
    summary = _read_json(OUTPUT_ROOT / "reproduction_summary.json")
    if summary.get("authorization_sha256") != authorization["authorization_sha256"]:
        raise ValueError("reproduction summary authorization hash drifted")

    runner = _load_historical_runner()
    spec = runner.RunSpec(
        experiment_id="exp_052_e_series_smp_paired_gate",
        case=CASE,
        condition=CONDITION,
        seed=SEED,
        max_fes=MAX_FES,
        output_root=OUTPUT_ROOT,
    )
    validated = runner.validate_existing(
        spec,
        _sha256(HISTORICAL_CONFIG),
        require_restore=True,
    )
    checks = {
        "target_match": summary.get("target") == authorization["target"],
        "integrity_passed": summary.get("integrity_passed") is True,
        "exact_historical_value_match": summary.get("final_error")
        == HISTORICAL_FINAL_ERROR
        == validated.get("final_error"),
        "exact_budget_match": validated.get("fitness_evaluations") == MAX_FES,
        "restore_observed": int(validated.get("restore_count", 0)) > 0,
        "no_abstention": int(validated.get("abstain_count", -1)) == 0,
    }
    return {
        "schema_version": "arac-exp052-session-reproduction-verification-v1",
        "verification_passed": all(checks.values()),
        "checks": checks,
        "authorization_sha256": authorization["authorization_sha256"],
        "final_error": validated["final_error"],
        "fitness_evaluations": validated["fitness_evaluations"],
        "restore_count": validated["restore_count"],
        "reset_count": validated["reset_count"],
        "abstain_count": validated["abstain_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(build_preflight(), sort_keys=True))
        return 0
    if args.command == "verify":
        verification = verify_reproduction()
        print(json.dumps(verification, sort_keys=True))
        return 0 if verification["verification_passed"] else 1
    summary = run_reproduction()
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["seed117_record_recovered_or_exceeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
