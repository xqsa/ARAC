"""Execute and aggregate the frozen exp_018 RDDSM evidence-overlay pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from arac.backends.hcc import (
    DEFAULT_AOB_DATA_ROOT,
    EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT,
    HCC_VENDOR_ROOT,
    HccAobExecutionRequest,
    HccAobExecutionResult,
    required_aob_data_files,
    run_hcc_aob_smoke_execution,
)
from arac.evidence import FORBIDDEN_RUNTIME_FIELDS

from .protocol import (
    AGGREGATE_ARTIFACTS,
    CHECKPOINT_FIELDS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DELAYED_FIELDS,
    GateInputs,
    LANE_MODE_PAIRS,
    PLAN_FIELDS,
    PROBE_FIELDS,
    PROTOCOL_VERSION,
    REPOSITORY_ROOT,
    RUN_ID,
    SHADOW_FIELDS,
    SOURCE_MODE,
    RunSpec,
    build_execution_request,
    build_promotion_gate,
    build_run_matrix,
    config_sha256,
    load_config,
    stage_jobs,
)

RAW_ARTIFACTS = {
    "checkpoint": ("checkpoint.csv", CHECKPOINT_FIELDS),
    "plan": ("plan.csv", PLAN_FIELDS),
    "probe_evidence": ("probe_evidence.csv", PROBE_FIELDS),
    "delayed_outcomes": ("delayed_outcomes.csv", DELAYED_FIELDS),
    "shadow_decisions": ("shadow_decisions.csv", SHADOW_FIELDS),
}
IDENTITY_FIELDS = (
    "run_id",
    "stage",
    "cohort_id",
    "trajectory_id",
    "triplet_id",
    "lane_id",
    "evidence_overlay_mode",
    "problem_id",
    "seed",
    "max_fes",
)
LEDGER_FIELDS = (
    *IDENTITY_FIELDS,
    "phase_i_fe",
    "phase_ii_fe",
    "cc_phase_fe",
    "rescue_fe",
    "refresh_fe",
    "search_state_fe",
    "precision_probe_fe",
    "evidence_overlay_fe",
    "separable_continuation_fe",
    "overhead_fe",
    "total_fe",
    "budget_limit",
    "actual_fe_used",
    "terminal_tolerance_rule",
    "terminal_tolerance_fe",
    "same_budget_violation",
    "fresh_execution",
    "ledger_closed",
)
RUN_RESULT_FIELDS = (
    *IDENTITY_FIELDS,
    "status",
    "fresh_optimizer_execution",
    "source_mode",
    "result_source",
    "native_terminal_error",
    "all_evaluation_best_error",
    "backend_reported_final_error",
    "actual_fe_used",
    "evidence_overlay_fe",
    "terminal_tolerance_rule",
    "terminal_tolerance_fe",
    "applicable",
    "abstain_reason",
    "runtime_authorized",
    "overlay_manifest_path",
    "overlay_manifest_sha256",
    "error",
)
ANTI_LEAKAGE_FIELDS = (
    *IDENTITY_FIELDS,
    "artifact_path",
    "runtime_input_fields",
    "forbidden_field",
    "found_in_runtime_payload",
    "aob_truth_runtime_used",
    "runtime_authorized",
    "runtime_dispatch_allowed",
    "audit_status",
)
AOB_FIELDS = (
    *IDENTITY_FIELDS,
    "file",
    "path",
    "sha256_before",
    "sha256_after",
    "unchanged",
)
def _source_bundle_files() -> tuple[str, ...]:
    roots = (
        REPOSITORY_ROOT / "src" / "arac",
        REPOSITORY_ROOT / "vendor" / "hcc",
        REPOSITORY_ROOT
        / "experiments"
        / "pilots"
        / "exp_018_rddsm_evidence_overlay_pilot",
    )
    files = {
        "pyproject.toml",
        "configs/rddsm_evidence_overlay_pilot_v1.json",
        "scripts/hcc_smoke_runner.py",
        "experiments/__init__.py",
        "experiments/pilots/__init__.py",
    }
    for root in roots:
        files.update(
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in root.rglob("*.py")
        )
    return tuple(sorted(files))


SOURCE_BUNDLE_FILES = _source_bundle_files()




@dataclass(frozen=True)
class ExecutionRecord:
    spec: RunSpec
    request: HccAobExecutionRequest
    result: HccAobExecutionResult | None
    error: str = ""


@dataclass
class CollectedArtifacts:
    run_results: list[dict[str, object]]
    ledger_rows: list[dict[str, object]]
    checkpoints: dict[str, dict[str, object]]
    plan_rows: list[dict[str, object]]
    probe_rows: list[dict[str, object]]
    delayed_rows: list[dict[str, object]]
    shadow_rows: list[dict[str, object]]
    aob_rows: list[dict[str, object]]
    anti_leakage_rows: list[dict[str, object]]
    per_run_manifests: list[dict[str, object]]
    blockers: list[str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    required_fields: Sequence[str] = (),
) -> None:
    extras = sorted(
        set().union(*(set(row) for row in rows)) - set(required_fields)
    ) if rows else []
    fieldnames = list(dict.fromkeys((*required_fields, *extras)))
    if not fieldnames:
        raise ValueError(f"CSV field contract is empty: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _identity(spec: RunSpec) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "stage": spec.stage,
        "cohort_id": spec.cohort_id,
        "trajectory_id": spec.trajectory_id,
        "triplet_id": spec.triplet_id,
        "lane_id": spec.lane.lane_id,
        "evidence_overlay_mode": spec.lane.evidence_overlay_mode,
        "problem_id": spec.problem_id,
        "seed": spec.seed,
        "max_fes": spec.max_fes,
    }


def _contained_path(base: Path, candidate: Path) -> Path:
    resolved_base = base.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError(f"artifact escapes trajectory output: {resolved}")
    return resolved


def _manifest_artifact_path(
    manifest_path: Path,
    run_output: Path,
    manifest: Mapping[str, object],
    key: str,
    expected_name: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or key not in artifacts:
        raise ValueError(f"overlay manifest is missing artifact mapping: {key}")
    value = artifacts[key]
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"overlay artifact path must be relative: {key}")
    resolved = _contained_path(run_output, manifest_path.parent / value)
    if resolved.name != expected_name:
        raise ValueError(
            f"overlay artifact basename mismatch for {key}: {resolved.name}"
        )
    return resolved


def _one_recursive(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern} below {root}; found={len(matches)}"
        )
    return matches[0]


def _validate_manifest(
    manifest: Mapping[str, object],
    spec: RunSpec,
) -> list[str]:
    failures: list[str] = []
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "problem_id": spec.problem_id,
        "seed": spec.seed,
        "evidence_overlay_mode": spec.lane.evidence_overlay_mode,
        "configured_max_fes": spec.max_fes,
        "source_mode": SOURCE_MODE,
        "terminal_tolerance_rule": "maximum_native_group_population",
        "fresh_optimizer_execution": 1,
        "runtime_authorized": 0,
        "optimizer_calls": 0,
        "rng_calls": 0,
        "observer_integrity": 1,
    }
    for field, value in expected.items():
        if str(manifest.get(field)) != str(value):
            failures.append(f"manifest_{field}_mismatch")
    for field in (
        "native_terminal_error",
        "all_evaluation_best_error",
        "terminal_tolerance_fe",
        "terminal_tolerance_rule",
        "applicable",
        "abstain_reason",
    ):
        if field not in manifest:
            failures.append(f"manifest_missing_{field}")
    if not isinstance(manifest.get("runtime_input_fields"), list):
        failures.append("manifest_runtime_input_fields_invalid")
    if not isinstance(manifest.get("artifacts"), dict):
        failures.append("manifest_artifacts_invalid")
    if not isinstance(manifest.get("artifact_sha256"), dict):
        failures.append("manifest_artifact_sha256_invalid")
    try:
        objective_calls = int(str(manifest["objective_calls"]))
        overlay_fe = int(str(manifest["evidence_overlay_fe"]))
        applicable = int(str(manifest["applicable"]))
        selected_count = int(str(manifest["selected_relation_count"]))
        delayed_expected = int(str(manifest["delayed_label_expected"]))
        delayed_closed = int(str(manifest["delayed_label_closed"]))
    except (KeyError, TypeError, ValueError):
        failures.append("manifest_overlay_counters_invalid")
    else:
        if objective_calls != overlay_fe or overlay_fe not in {0, 16}:
            failures.append("manifest_overlay_fe_mismatch")
        if applicable == 1 and (
            overlay_fe != 16
            or selected_count != 4
            or delayed_expected != 8
            or delayed_closed != 8
        ):
            failures.append("manifest_applicable_bundle_invalid")
    return failures


def _prefix_rows(
    rows: Sequence[Mapping[str, object]],
    spec: RunSpec,
    blockers: list[str],
    artifact_key: str,
) -> list[dict[str, object]]:
    identity = _identity(spec)
    prefixed: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        for field in ("problem_id", "seed", "evidence_overlay_mode"):
            if field in row and str(row[field]) != str(identity[field]):
                blockers.append(
                    f"{spec.trajectory_id}:{artifact_key}:row{index}:{field}_mismatch"
                )
        if "mode" in row and str(row["mode"]) != spec.lane.evidence_overlay_mode:
            blockers.append(
                f"{spec.trajectory_id}:{artifact_key}:row{index}:mode_mismatch"
            )
        prefixed.append({**row, **identity})
    return prefixed


def _expected_aob_bindings(
    data_root: Path | str,
    problem_ids: Sequence[str],
) -> dict[str, dict[str, tuple[Path, str]]]:
    root = Path(data_root).resolve()
    bindings: dict[str, dict[str, tuple[Path, str]]] = {}
    for problem_id in sorted(set(problem_ids)):
        try:
            function_id = int(problem_id[1:])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AOB problem id: {problem_id}") from exc
        files: dict[str, tuple[Path, str]] = {}
        for path in required_aob_data_files(root, function_id):
            resolved = path.resolve()
            files[resolved.name] = (
                resolved,
                _sha256(resolved) if resolved.is_file() else "missing",
            )
        bindings[problem_id] = files
    return bindings


def _aob_manifest_failures(
    rows: Sequence[Mapping[str, object]],
    expected: Mapping[str, tuple[Path, str]],
) -> list[str]:
    failures: list[str] = []
    observed_names = [str(row.get("file", "")) for row in rows]
    if len(observed_names) != len(set(observed_names)):
        failures.append("duplicate_file")
    missing = sorted(set(expected) - set(observed_names))
    extra = sorted(set(observed_names) - set(expected))
    if missing:
        failures.append(f"missing_files={','.join(missing)}")
    if extra:
        failures.append(f"unexpected_files={','.join(extra)}")
    for row in rows:
        filename = str(row.get("file", ""))
        binding = expected.get(filename)
        if binding is None:
            continue
        expected_path, expected_hash = binding
        try:
            observed_path = Path(str(row.get("path", ""))).resolve()
        except (OSError, ValueError):
            observed_path = Path()
        if observed_path != expected_path:
            failures.append(f"{filename}:path_mismatch")
        if expected_hash == "missing":
            failures.append(f"{filename}:canonical_file_missing")
        if str(row.get("sha256_before", "")) != expected_hash:
            failures.append(f"{filename}:before_hash_mismatch")
        if str(row.get("sha256_after", "")) != expected_hash:
            failures.append(f"{filename}:after_hash_mismatch")
        if str(row.get("unchanged", "")) != "1":
            failures.append(f"{filename}:changed")
    return failures


def _result_actual_fe(result: HccAobExecutionResult) -> int:
    if result.optimizer_final_fe_used is None:
        return result.fe_used
    return result.optimizer_final_fe_used


def _ledger_row(
    record: ExecutionRecord,
    manifest: Mapping[str, object] | None,
) -> dict[str, object]:
    identity = _identity(record.spec)
    result = record.result
    if result is None:
        return {
            **identity,
            **{field: "" for field in LEDGER_FIELDS if field not in identity},
            "budget_limit": record.spec.max_fes,
            "same_budget_violation": 1,
            "fresh_execution": 0,
            "ledger_closed": 0,
        }
    actual = _result_actual_fe(result)
    phase_i = min(actual, max(0, result.global_phase_fe or 0))
    stages = {
        "cc_phase_fe": max(0, result.cc_phase_fe or 0),
        "rescue_fe": max(0, result.rescue_fe or 0),
        "refresh_fe": max(0, result.refresh_fe or 0),
        "search_state_fe": max(0, result.search_state_fe or 0),
        "precision_probe_fe": max(0, result.precision_probe_fe or 0),
        "evidence_overlay_fe": max(0, result.evidence_overlay_fe or 0),
        "separable_continuation_fe": max(0, result.separable_continuation_fe or 0),
    }
    known = phase_i + sum(stages.values())
    overhead = max(0, actual - known) if result.overhead_fe is None else max(0, result.overhead_fe)
    closed = known + overhead == actual
    tolerance = manifest.get("terminal_tolerance_fe", "") if manifest else ""
    return {
        **identity,
        "phase_i_fe": phase_i,
        "phase_ii_fe": actual - phase_i,
        **stages,
        "overhead_fe": overhead,
        "total_fe": actual,
        "budget_limit": result.max_fes,
        "actual_fe_used": actual,
        "terminal_tolerance_rule": (
            "" if manifest is None else manifest.get("terminal_tolerance_rule", "")
        ),
        "terminal_tolerance_fe": tolerance,
        "same_budget_violation": int(actual > result.max_fes),
        "fresh_execution": int(result.fresh_optimizer_execution),
        "ledger_closed": int(closed),
    }


def _anti_leakage_row(
    spec: RunSpec,
    manifest_path: Path | None,
    manifest: Mapping[str, object] | None,
    config: Mapping[str, object],
) -> dict[str, object]:
    runtime_fields = manifest.get("runtime_input_fields", []) if manifest else []
    runtime_fields = runtime_fields if isinstance(runtime_fields, list) else []
    configured = config.get("forbidden_runtime_inputs", [])
    forbidden = set(FORBIDDEN_RUNTIME_FIELDS)
    if isinstance(configured, list):
        forbidden.update(str(field) for field in configured)
    found = sorted(forbidden.intersection(str(field) for field in runtime_fields))
    aob_truth_used = int(bool(manifest and manifest.get("aob_truth_runtime_used", 0)))
    runtime_authorized = int(bool(manifest and manifest.get("runtime_authorized", 0)))
    passed = bool(manifest) and not found and not aob_truth_used and not runtime_authorized
    return {
        **_identity(spec),
        "artifact_path": "" if manifest_path is None else str(manifest_path),
        "runtime_input_fields": ";".join(str(field) for field in runtime_fields),
        "forbidden_field": ";".join(found),
        "found_in_runtime_payload": int(bool(found)),
        "aob_truth_runtime_used": aob_truth_used,
        "runtime_authorized": runtime_authorized,
        "runtime_dispatch_allowed": 0,
        "audit_status": "pass" if passed else "fail",
    }


def _collect_record(
    record: ExecutionRecord,
    config: Mapping[str, object],
    collected: CollectedArtifacts,
    expected_aob: Mapping[str, tuple[Path, str]],
) -> None:
    spec = record.spec
    run_output = Path(record.request.output_dir).resolve()
    manifest_path: Path | None = None
    manifest: dict[str, object] | None = None
    if record.error:
        collected.blockers.append(f"{spec.trajectory_id}:execution_error:{record.error}")
    if record.result is None:
        collected.blockers.append(f"{spec.trajectory_id}:missing_execution_result")
    else:
        try:
            manifest_path = _one_recursive(
                run_output,
                f"{spec.problem_id}_evidence_overlay_manifest.json",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("overlay manifest must be an object")
            collected.blockers.extend(
                f"{spec.trajectory_id}:{failure}"
                for failure in _validate_manifest(manifest, spec)
            )
            if record.result is not None and int(
                record.result.evidence_overlay_fe or 0
            ) != int(str(manifest.get("evidence_overlay_fe", -1))):
                collected.blockers.append(
                    f"{spec.trajectory_id}:result_manifest_overlay_fe_mismatch"
                )
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
            collected.blockers.append(f"{spec.trajectory_id}:manifest_invalid:{exc}")
            manifest_path = None
            manifest = None

    raw_rows: dict[str, list[dict[str, str]]] = {}
    if manifest is not None and manifest_path is not None:
        collected.per_run_manifests.append(
            {
                "trajectory_id": spec.trajectory_id,
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            }
        )
        for key, (suffix, required_fields) in RAW_ARTIFACTS.items():
            try:
                path = _manifest_artifact_path(
                    manifest_path,
                    run_output,
                    manifest,
                    key,
                    f"{spec.problem_id}_evidence_overlay_{suffix}",
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                hashes = manifest.get("artifact_sha256")
                if not isinstance(hashes, dict):
                    raise ValueError("artifact_sha256 mapping is missing")
                expected_hash = hashes.get(key, hashes.get(path.name))
                if not isinstance(expected_hash, str) or expected_hash != _sha256(path):
                    raise ValueError("artifact sha256 mismatch")
                header, rows = _read_csv(path)
                missing = sorted(set(required_fields) - set(header))
                if missing:
                    raise ValueError(f"missing columns={','.join(missing)}")
                raw_rows[key] = rows
            except (FileNotFoundError, OSError, ValueError) as exc:
                collected.blockers.append(f"{spec.trajectory_id}:{key}_invalid:{exc}")
                raw_rows[key] = []
        checkpoint_rows = raw_rows["checkpoint"]
        if len(checkpoint_rows) != 1:
            collected.blockers.append(
                f"{spec.trajectory_id}:checkpoint_row_count={len(checkpoint_rows)}"
            )
        else:
            prefixed_checkpoint = _prefix_rows(
                checkpoint_rows,
                spec,
                collected.blockers,
                "checkpoint",
            )
            collected.checkpoints[spec.trajectory_id] = prefixed_checkpoint[0]
        collected.plan_rows.extend(
            _prefix_rows(raw_rows["plan"], spec, collected.blockers, "plan")
        )
        collected.probe_rows.extend(
            _prefix_rows(raw_rows["probe_evidence"], spec, collected.blockers, "probe_evidence")
        )
        collected.delayed_rows.extend(
            _prefix_rows(raw_rows["delayed_outcomes"], spec, collected.blockers, "delayed_outcomes")
        )
        collected.shadow_rows.extend(
            _prefix_rows(raw_rows["shadow_decisions"], spec, collected.blockers, "shadow_decisions")
        )
        try:
            aob_path = _one_recursive(run_output, f"{spec.problem_id}_aob_input_manifest.csv")
            header, rows = _read_csv(aob_path)
            missing = sorted(set(AOB_FIELDS[len(IDENTITY_FIELDS):]) - set(header))
            if missing:
                raise ValueError(f"missing columns={','.join(missing)}")
            failures = _aob_manifest_failures(rows, expected_aob)
            if failures:
                raise ValueError(";".join(failures))
            collected.aob_rows.extend(
                _prefix_rows(rows, spec, collected.blockers, "aob_input_manifest")
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            collected.blockers.append(f"{spec.trajectory_id}:aob_manifest_invalid:{exc}")

    result = record.result
    actual_fe = "" if result is None else _result_actual_fe(result)
    collected.run_results.append(
        {
            **_identity(spec),
            "status": "execution_exception" if result is None else result.status,
            "fresh_optimizer_execution": (
                0 if result is None else int(result.fresh_optimizer_execution)
            ),
            "source_mode": "" if manifest is None else manifest.get("source_mode", ""),
            "result_source": "" if result is None else result.result_source,
            "native_terminal_error": (
                "" if manifest is None else manifest.get("native_terminal_error", "")
            ),
            "all_evaluation_best_error": (
                ""
                if manifest is None
                else manifest.get("all_evaluation_best_error", "")
            ),
            "backend_reported_final_error": "" if result is None else result.final_error,
            "actual_fe_used": actual_fe,
            "evidence_overlay_fe": (
                "" if result is None else (result.evidence_overlay_fe or 0)
            ),
            "terminal_tolerance_rule": (
                "" if manifest is None else manifest.get("terminal_tolerance_rule", "")
            ),
            "terminal_tolerance_fe": (
                "" if manifest is None else manifest.get("terminal_tolerance_fe", "")
            ),
            "applicable": 0 if manifest is None else manifest.get("applicable", 0),
            "abstain_reason": "" if manifest is None else manifest.get("abstain_reason", ""),
            "runtime_authorized": (
                0 if manifest is None else manifest.get("runtime_authorized", 1)
            ),
            "overlay_manifest_path": "" if manifest_path is None else str(manifest_path),
            "overlay_manifest_sha256": "" if manifest_path is None else _sha256(manifest_path),
            "error": record.error,
        }
    )
    ledger = _ledger_row(record, manifest)
    collected.ledger_rows.append(ledger)
    if str(ledger.get("ledger_closed")) != "1":
        collected.blockers.append(f"{spec.trajectory_id}:fe_ledger_not_closed")
    collected.anti_leakage_rows.append(
        _anti_leakage_row(spec, manifest_path, manifest, config)
    )


def collect_artifacts(
    records: Sequence[ExecutionRecord],
    config: Mapping[str, object],
    *,
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
) -> CollectedArtifacts:
    collected = CollectedArtifacts([], [], {}, [], [], [], [], [], [], [], [])
    expected_aob = _expected_aob_bindings(
        aob_data_root,
        [record.spec.problem_id for record in records],
    )
    for record in records:
        _collect_record(
            record,
            config,
            collected,
            expected_aob[record.spec.problem_id],
        )
    collected.blockers = list(dict.fromkeys(collected.blockers))
    return collected


def _source_bundle() -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in SOURCE_BUNDLE_FILES:
        path = REPOSITORY_ROOT / relative
        files[relative] = _sha256(path) if path.is_file() else "missing"
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256_of_canonical_relative_path_to_file_sha256_mapping",
        "files": files,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _aob_data_bundle(data_root: Path | str) -> dict[str, object]:
    root = Path(data_root).resolve()
    files: dict[str, str] = {}
    for function_id in (1, 3, 4, 5):
        for path in required_aob_data_files(root, function_id):
            resolved = path.resolve()
            files[resolved.name] = _sha256(resolved) if resolved.is_file() else "missing"
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256_of_canonical_filename_to_file_sha256_mapping",
        "root": str(root),
        "files": files,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "semantic_use": "integrity_binding_only_not_runtime_dispatch",
    }


def _aob_truth_overlap_variables(
    data_root: Path,
    function_id: int,
) -> tuple[set[int], dict[str, str]]:
    info_path = data_root / f"F{function_id}-info.txt"
    permutation_path = data_root / f"F{function_id}-p.txt"
    with info_path.open(encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    if not isinstance(info, dict):
        raise ValueError(f"AOB info must be a mapping: {info_path}")
    permutation: list[int] = []
    with permutation_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            permutation.extend(
                int(float(value.strip())) - 1
                for value in row
                if value.strip()
            )
    subgroups = info.get("subgroups")
    if not isinstance(subgroups, list) or not subgroups:
        raise ValueError("AOB subgroups must be a non-empty list")
    if len(set(permutation)) != len(permutation):
        raise ValueError("AOB permutation contains duplicate variables")
    groups: list[set[int]] = []
    start = 0
    overlap = int(info["overlap_degree"])
    for index, size_value in enumerate(subgroups):
        size = int(size_value)
        if (
            size <= 0
            or overlap < 0
            or (index != len(subgroups) - 1 and overlap >= size)
        ):
            raise ValueError("AOB subgroup size/overlap is invalid")
        end = start + size
        if end > len(permutation):
            raise ValueError("AOB subgroup layout exceeds the permutation")
        groups.append(set(permutation[start:end]))
        if index != len(subgroups) - 1:
            start = end - overlap
    if end != len(permutation):
        raise ValueError("AOB subgroup layout does not cover the permutation")
    counts: dict[int, int] = {}
    for group in groups:
        for variable in group:
            counts[variable] = counts.get(variable, 0) + 1
    truth = {variable for variable, count in counts.items() if count > 1}
    return truth, {
        info_path.name: _sha256(info_path),
        permutation_path.name: _sha256(permutation_path),
    }


def _offline_topology_evaluation(
    plan_rows: Sequence[Mapping[str, object]],
    aob_data_root: Path | str,
) -> dict[str, object]:
    root = Path(aob_data_root).resolve()
    by_problem: dict[str, set[int]] = {}
    parse_failures: list[str] = []
    for row in plan_rows:
        problem_id = str(row.get("problem_id", ""))
        values = str(row.get("shared_variables", ""))
        if not problem_id or not values:
            continue
        try:
            parsed = {int(value) for value in values.split(";") if value}
        except ValueError:
            parse_failures.append(f"{problem_id}:invalid_shared_variables:{values}")
            continue
        by_problem.setdefault(problem_id, set()).update(parsed)
    rows: list[dict[str, object]] = []
    for problem_id in ("E1", "E3", "A4", "S5"):
        predicted = by_problem.get(problem_id, set())
        try:
            truth, source_hashes = _aob_truth_overlap_variables(
                root,
                int(problem_id[1]),
            )
        except (FileNotFoundError, KeyError, OSError, ValueError, yaml.YAMLError) as exc:
            parse_failures.append(f"{problem_id}:truth_unavailable:{exc}")
            continue
        true_positive = len(predicted & truth)
        precision = (
            1.0 if not predicted and not truth else true_positive / max(1, len(predicted))
        )
        recall = 1.0 if not truth else true_positive / len(truth)
        rows.append(
            {
                "problem_id": problem_id,
                "predicted_shared_variable_count": len(predicted),
                "aob_truth_shared_variable_count": len(truth),
                "true_positive_count": true_positive,
                "precision": precision,
                "recall": recall,
                "truth_source_sha256": source_hashes,
            }
        )
    return {
        "scope": "offline_reference_only_after_optimizer_execution",
        "runtime_inputs": [],
        "used_for_runtime": False,
        "used_for_gate": False,
        "used_for_promotion": False,
        "truth_fields": ["Pvector", "subgroups"],
        "definition": "shared_variable_set_precision_recall",
        "by_problem": rows,
        "macro_precision": (
            None if not rows else statistics.fmean(float(row["precision"]) for row in rows)
        ),
        "macro_recall": (
            None if not rows else statistics.fmean(float(row["recall"]) for row in rows)
        ),
        "blockers": parse_failures,
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _lane_summary_rows(
    specs: Sequence[RunSpec],
    run_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_lane: dict[str, list[Mapping[str, object]]] = {}
    for lane_id, _mode in LANE_MODE_PAIRS:
        by_lane[lane_id] = [row for row in run_rows if str(row.get("lane_id")) == lane_id]
    rows: list[dict[str, object]] = []
    for lane_id, mode in LANE_MODE_PAIRS:
        lane_rows = by_lane[lane_id]
        terminal_errors: list[float] = []
        for row in lane_rows:
            try:
                terminal_errors.append(float(str(row.get("native_terminal_error", ""))))
            except ValueError:
                continue
        rows.append(
            {
                "run_id": RUN_ID,
                "stage": specs[0].stage if specs else "",
                "lane_id": lane_id,
                "evidence_overlay_mode": mode,
                "expected_run_count": sum(
                    spec.lane.lane_id == lane_id for spec in specs
                ),
                "observed_run_count": len(lane_rows),
                "completed_run_count": sum(
                    str(row.get("status")) == "completed" for row in lane_rows
                ),
                "fresh_run_count": sum(
                    str(row.get("fresh_optimizer_execution")) == "1"
                    for row in lane_rows
                ),
                "applicable_run_count": sum(
                    str(row.get("applicable")) == "1" for row in lane_rows
                ),
                "evidence_overlay_fe": sum(
                    int(str(row.get("evidence_overlay_fe") or 0))
                    for row in lane_rows
                ),
                "median_native_terminal_error": (
                    "" if not terminal_errors else f"{statistics.median(terminal_errors):.17e}"
                ),
                "runtime_authorized": 0,
            }
        )
    return rows


def _aggregate_manifest(
    output: Path,
    stage: str,
    config_path: Path,
    specs: Sequence[RunSpec],
    collected: CollectedArtifacts,
    gate: Mapping[str, object],
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "stage": stage,
        "date": "2026-07-17",
        "executor": "Codex",
        "source_mode": SOURCE_MODE,
        "observer_only": True,
        "runtime_authorized": False,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": gate["config_sha256"],
        },
        "source_git_commit": _git_commit(),
        "source_bundle": gate["source_bundle"],
        "source_integrity": gate["source_integrity"],
        "runtime_binding": gate["runtime_binding"],
        "aob_data_bundle": gate["aob_data_bundle"],
        "aob_data_integrity": gate["aob_data_integrity"],
        "expected_run_count": len(specs),
        "observed_run_count": len(collected.run_results),
        "lanes": [lane for lane, _mode in LANE_MODE_PAIRS],
        "matrix": [
            {
                "trajectory_id": spec.trajectory_id,
                "cohort_id": spec.cohort_id,
                "problem_id": spec.problem_id,
                "seed": spec.seed,
                "max_fes": spec.max_fes,
                "lane_id": spec.lane.lane_id,
                "evidence_overlay_mode": spec.lane.evidence_overlay_mode,
            }
            for spec in specs
        ],
        "per_run_manifests": collected.per_run_manifests,
        "promotion_status": gate["status"],
        "offline_reference_topology_evaluation": gate.get("metrics", {}).get(
            "offline_reference_topology"
        ),
        "artifacts": list(AGGREGATE_ARTIFACTS),
        "result_root": str(output.resolve()),
    }


def _write_manifest_markdown(
    output: Path,
    stage: str,
    specs: Sequence[RunSpec],
    gate: Mapping[str, object],
) -> None:
    blockers = gate.get("blockers", [])
    text = (
        "# exp_018 RDDSM Evidence Overlay Pilot\n\n"
        f"- Date: 2026-07-17\n"
        f"- Executor: Codex\n"
        f"- Stage: `{stage}`\n"
        f"- Source mode: `{SOURCE_MODE}`\n"
        f"- Expected trajectories: {len(specs)}\n"
        f"- Promotion status: `{gate['status']}`\n"
        "- Runtime authorization: `0` (observer-only)\n"
        "- Phase-II topology: frozen\n"
        "- AOB truth use: offline audit only\n\n"
        "The original RDDSM topology remains the structural partition. Active probes are "
        "one-shot Phase-I observations and never update the incumbent, CMA state, native RNG, "
        "controller, or cooperative context.\n\n"
        f"Gate blockers: `{';'.join(str(item) for item in blockers) if blockers else 'none'}`\n"
    )
    (output / "run_manifest.md").write_text(text, encoding="utf-8")


def write_aggregate(
    output: Path,
    *,
    stage: str,
    config_path: Path,
    config: Mapping[str, object],
    specs: Sequence[RunSpec],
    records: Sequence[ExecutionRecord],
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
    frozen_source_bundle: Mapping[str, object] | None = None,
    frozen_config_sha256: str | None = None,
    frozen_aob_data_bundle: Mapping[str, object] | None = None,
) -> dict[str, object]:
    collected = collect_artifacts(
        records,
        config,
        aob_data_root=aob_data_root,
    )
    start_bundle = (
        _source_bundle()
        if frozen_source_bundle is None
        else dict(frozen_source_bundle)
    )
    start_config_sha256 = (
        config_sha256(config_path)
        if frozen_config_sha256 is None
        else str(frozen_config_sha256)
    )
    end_bundle = _source_bundle()
    end_config_sha256 = config_sha256(config_path)
    start_aob_bundle = (
        _aob_data_bundle(aob_data_root)
        if frozen_aob_data_bundle is None
        else dict(frozen_aob_data_bundle)
    )
    end_aob_bundle = _aob_data_bundle(aob_data_root)
    if end_bundle != start_bundle:
        collected.blockers.append("source_bundle_drift_during_stage")
    if end_config_sha256 != start_config_sha256:
        collected.blockers.append("config_drift_during_stage")
    if end_aob_bundle != start_aob_bundle:
        collected.blockers.append("aob_data_drift_during_stage")
    gate = build_promotion_gate(
        stage,
        config,
        specs,
        GateInputs(
            run_results=collected.run_results,
            ledger_rows=collected.ledger_rows,
            checkpoint_rows=collected.checkpoints,
            plan_rows=collected.plan_rows,
            probe_rows=collected.probe_rows,
            delayed_rows=collected.delayed_rows,
            shadow_rows=collected.shadow_rows,
            aob_rows=collected.aob_rows,
            anti_leakage_rows=collected.anti_leakage_rows,
            integrity_blockers=collected.blockers,
        ),
    )
    gate_metrics = gate.get("metrics")
    if not isinstance(gate_metrics, dict):
        raise RuntimeError("promotion gate metrics must be an object")
    gate_metrics["offline_reference_topology"] = _offline_topology_evaluation(
        collected.plan_rows,
        aob_data_root,
    )
    gate["config_sha256"] = start_config_sha256
    gate["source_bundle"] = start_bundle
    gate["source_integrity"] = {
        "source_bundle_unchanged": end_bundle == start_bundle,
        "config_unchanged": end_config_sha256 == start_config_sha256,
        "end_source_bundle_sha256": end_bundle.get("sha256"),
        "end_config_sha256": end_config_sha256,
    }
    gate["runtime_binding"] = {
        "hcc_root": str(Path(HCC_VENDOR_ROOT).resolve()),
        "aob_data_root": str(Path(DEFAULT_AOB_DATA_ROOT).resolve()),
        "canonical_roots_required": True,
        "subprocess_environment": EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT,
    }
    gate["aob_data_bundle"] = start_aob_bundle
    gate["aob_data_integrity"] = {
        "unchanged": end_aob_bundle == start_aob_bundle,
        "end_sha256": end_aob_bundle.get("sha256"),
        "semantic_use": "integrity_binding_only_not_runtime_dispatch",
    }
    gate["source_git_commit"] = _git_commit()
    _write_csv(
        output / "run_results.csv",
        collected.run_results,
        required_fields=RUN_RESULT_FIELDS,
    )
    _write_csv(
        output / "same_budget_ledger.csv",
        collected.ledger_rows,
        required_fields=LEDGER_FIELDS,
    )
    _write_csv(
        output / "probe_plan.csv",
        collected.plan_rows,
        required_fields=(*IDENTITY_FIELDS, *PLAN_FIELDS),
    )
    _write_csv(
        output / "probe_evidence.csv",
        collected.probe_rows,
        required_fields=(*IDENTITY_FIELDS, *PROBE_FIELDS),
    )
    _write_csv(
        output / "delayed_outcomes.csv",
        collected.delayed_rows,
        required_fields=(*IDENTITY_FIELDS, *DELAYED_FIELDS),
    )
    _write_csv(
        output / "shadow_decisions.csv",
        collected.shadow_rows,
        required_fields=(*IDENTITY_FIELDS, *SHADOW_FIELDS),
    )
    _write_csv(output / "aob_input_manifest.csv", collected.aob_rows, required_fields=AOB_FIELDS)
    _write_csv(
        output / "anti_leakage_audit.csv",
        collected.anti_leakage_rows,
        required_fields=ANTI_LEAKAGE_FIELDS,
    )
    _write_csv(
        output / "lane_summary.csv",
        _lane_summary_rows(specs, collected.run_results),
        required_fields=(
            "run_id",
            "stage",
            "lane_id",
            "evidence_overlay_mode",
            "expected_run_count",
            "observed_run_count",
            "completed_run_count",
            "fresh_run_count",
            "applicable_run_count",
            "evidence_overlay_fe",
            "median_native_terminal_error",
            "runtime_authorized",
        ),
    )
    _write_json(output / "promotion_gate.json", gate)
    _write_json(
        output / "manifest.json",
        _aggregate_manifest(output, stage, config_path, specs, collected, gate),
    )
    _write_manifest_markdown(output, stage, specs, gate)
    return gate


def _validate_prior_smoke_gate(path: Path, config_path: Path) -> None:
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"mechanism requires a valid prior smoke gate: {exc}") from exc
    if not isinstance(gate, dict):
        raise RuntimeError("mechanism requires a smoke gate JSON object")
    current_bundle = _source_bundle()
    current_config_sha256 = config_sha256(config_path)
    current_aob_bundle = _aob_data_bundle(DEFAULT_AOB_DATA_ROOT)
    source_integrity = gate.get("source_integrity")
    if not isinstance(source_integrity, dict):
        source_integrity = {}
    aob_data_integrity = gate.get("aob_data_integrity")
    if not isinstance(aob_data_integrity, dict):
        aob_data_integrity = {}
    if (
        gate.get("protocol_version") != PROTOCOL_VERSION
        or gate.get("stage") != "smoke"
        or gate.get("status") != "smoke_pass"
        or gate.get("config_sha256") != current_config_sha256
        or gate.get("source_bundle") != current_bundle
        or source_integrity.get("source_bundle_unchanged") is not True
        or source_integrity.get("config_unchanged") is not True
        or source_integrity.get("end_source_bundle_sha256")
        != current_bundle.get("sha256")
        or source_integrity.get("end_config_sha256") != current_config_sha256
        or gate.get("runtime_binding")
        != {
            "hcc_root": str(Path(HCC_VENDOR_ROOT).resolve()),
            "aob_data_root": str(Path(DEFAULT_AOB_DATA_ROOT).resolve()),
            "canonical_roots_required": True,
            "subprocess_environment": EVIDENCE_OVERLAY_SUBPROCESS_ENVIRONMENT,
        }
        or gate.get("aob_data_bundle") != current_aob_bundle
        or aob_data_integrity.get("unchanged") is not True
        or aob_data_integrity.get("end_sha256") != current_aob_bundle.get("sha256")
    ):
        raise RuntimeError("mechanism prior smoke gate binding failed")


def _execute_one(
    spec: RunSpec,
    request: HccAobExecutionRequest,
    runner: Callable[[HccAobExecutionRequest], HccAobExecutionResult],
) -> ExecutionRecord:
    try:
        result = runner(request)
        if not isinstance(result, HccAobExecutionResult):
            raise TypeError("execution runner returned the wrong result type")
        return ExecutionRecord(spec, request, result)
    except Exception as exc:  # Preserve the failed trajectory in the fail-closed aggregate.
        return ExecutionRecord(spec, request, None, f"{type(exc).__name__}:{exc}")


def run_pilot(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    stage: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    source_mode: str = SOURCE_MODE,
    execution_runner: Callable[
        [HccAobExecutionRequest], HccAobExecutionResult
    ] = run_hcc_aob_smoke_execution,
    python_executable: str = sys.executable,
    hcc_root: Path | str = HCC_VENDOR_ROOT,
    aob_data_root: Path | str = DEFAULT_AOB_DATA_ROOT,
    jobs: int | None = None,
    smoke_gate: Path | str | None = None,
) -> Path:
    if source_mode != SOURCE_MODE:
        raise ValueError(f"source_mode must be {SOURCE_MODE}")
    resolved_config = Path(config_path).resolve()
    config = load_config(resolved_config)
    specs = build_run_matrix(config, stage)
    configured_jobs = stage_jobs(config, stage)
    worker_count = configured_jobs if jobs is None else int(jobs)
    if worker_count <= 0:
        raise ValueError("jobs must be positive")
    if stage == "mechanism" and worker_count != 24:
        raise ValueError("mechanism jobs is frozen at 24")
    resolved_hcc_root = Path(hcc_root).resolve()
    resolved_aob_root = Path(aob_data_root).resolve()
    if resolved_hcc_root != Path(HCC_VENDOR_ROOT).resolve():
        raise ValueError("exp_018 requires the canonical HCC vendor root")
    if resolved_aob_root != Path(DEFAULT_AOB_DATA_ROOT).resolve():
        raise ValueError("exp_018 requires the canonical AOB data root")
    root = Path(output_root).resolve()
    output = root / stage
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fresh exp_018 stage output must be empty: {output}")
    if stage == "mechanism":
        gate_path = (
            Path(smoke_gate).resolve()
            if smoke_gate
            else root / "smoke" / "promotion_gate.json"
        )
        _validate_prior_smoke_gate(gate_path, resolved_config)
    frozen_source_bundle = _source_bundle()
    frozen_config_hash = config_sha256(resolved_config)
    frozen_aob_bundle = _aob_data_bundle(resolved_aob_root)
    output.mkdir(parents=True, exist_ok=True)
    requests = []
    for spec in specs:
        run_output = (
            output
            / "_runs"
            / spec.cohort_id
            / spec.problem_id
            / f"seed_{spec.seed}"
            / spec.lane.lane_id
        )
        requests.append(
            (
                spec,
                build_execution_request(
                    spec,
                    run_output,
                    config=config,
                    python_executable=python_executable,
                    hcc_root=resolved_hcc_root,
                    aob_data_root=resolved_aob_root,
                ),
            )
        )
    records_by_id: dict[str, ExecutionRecord] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_execute_one, spec, request, execution_runner): spec.trajectory_id
            for spec, request in requests
        }
        for future in as_completed(futures):
            records_by_id[futures[future]] = future.result()
    ordered_records = [records_by_id[spec.trajectory_id] for spec in specs]
    write_aggregate(
        output,
        stage=stage,
        config_path=resolved_config,
        config=config,
        specs=specs,
        records=ordered_records,
        aob_data_root=resolved_aob_root,
        frozen_source_bundle=frozen_source_bundle,
        frozen_config_sha256=frozen_config_hash,
        frozen_aob_data_bundle=frozen_aob_bundle,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen exp_018 evidence-overlay pilot.")
    parser.add_argument("--stage", required=True, choices=("smoke", "mechanism"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--source-mode", default=SOURCE_MODE)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--hcc-root", default=str(HCC_VENDOR_ROOT))
    parser.add_argument("--aob-data-root", default=str(DEFAULT_AOB_DATA_ROOT))
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--smoke-gate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_pilot(
        output_root=Path(args.output_dir),
        stage=str(args.stage),
        config_path=Path(args.config),
        source_mode=str(args.source_mode),
        python_executable=str(args.python_executable),
        hcc_root=Path(args.hcc_root),
        aob_data_root=Path(args.aob_data_root),
        jobs=args.jobs,
        smoke_gate=None if args.smoke_gate is None else Path(args.smoke_gate),
    )


if __name__ == "__main__":
    main()
