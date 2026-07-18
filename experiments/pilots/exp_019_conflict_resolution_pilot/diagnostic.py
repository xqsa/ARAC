"""Observer-only oracle diagnostic for exp_019 synthetic conflict twins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from arac.backends.hcc_evidence_overlay import (
    CHECKPOINT_FIELDS,
    DELAYED_OUTCOME_FIELDS,
    EVIDENCE_OVERLAY_PROTOCOL_VERSION,
    PLAN_FIELDS,
    PROBE_CANDIDATES,
    PROBE_EVIDENCE_FIELDS,
    RUNTIME_INPUT_FIELDS,
    SHADOW_DECISION_FIELDS,
)
from arac.policy.evidence_overlay import UTILITY_EPSILON

from .benchmark import (
    MANIFEST_NAME as SYNTHETIC_MANIFEST_NAME,
    REPO_ROOT,
    SYNTHETIC_DATA_DIR,
    VENDOR_DATA_DIR,
    VENDOR_ROOT,
    validate_synthetic_bundle,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "diagnostic_config.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "exp_019_conflict_resolution_pilot"
EXP018_CONFIG_PATH = REPO_ROOT / "configs" / "rddsm_evidence_overlay_pilot_v1.json"
PROTOCOL_VERSION = "exp019-conflict-oracle-diagnostic-v1"
CONFIG_SCHEMA_VERSION = "exp019-conflict-diagnostic-config-v1"
CASE_FUNCTIONS = {
    "E3": ("elliptic", 3, "E3_conflict_variant_synthetic"),
    "A4": ("ackley", 4, "A4_conflict_variant_synthetic"),
    "S5": ("schwefel", 5, "S5_conflict_variant_synthetic"),
}
ORACLE_SEEDS = (117, 118, 119, 120, 121)
ORACLE_MAX_FES = 3_000_000
ORACLE_JOBS = 12
SMOKE_MAX_FES = 100_000
MATERIAL_THRESHOLD = math.log(1.01)
LARGE_LOSS_THRESHOLD = -math.log(1.20)
CONFIDENCE = 0.95
EXPECTED_PROBE_FE = 16
EXPECTED_RELATIONS = 4
EXPECTED_RUNTIME_FIELDS = tuple(RUNTIME_INPUT_FIELDS)

BUDGET_FIELDS = (
    "problem_id",
    "budget_accounting",
    "max_fes",
    "optimizer_reported_fe",
    "fitness_record_fe",
    "budget_aligned_fe",
    "same_budget_violation",
    "global_phase_fe",
    "cc_phase_fe",
    "rescue_fe",
    "refresh_fe",
    "search_state_fe",
    "separable_continuation_fe",
    "overhead_fe",
    "evidence_overlay_fe",
)
AOB_INPUT_FIELDS = (
    "problem_id",
    "file",
    "path",
    "sha256_before",
    "sha256_after",
    "unchanged",
)
RELATION_RESULT_FIELDS = (
    "side",
    "problem_id",
    "seed",
    "trajectory_id",
    "relation_id",
    "left_reliability",
    "right_reliability",
    "baseline_owner",
    "baseline_fitness",
    "bridge_fitness",
    "delta",
    "best_owner",
    "best_owner_fitness",
    "best_owner_delta",
    "material_win",
    "large_loss",
)
TRAJECTORY_RESULT_FIELDS = (
    "side",
    "problem_id",
    "seed",
    "trajectory_id",
    "relation_count",
    "trajectory_delta",
    "best_owner_trajectory_delta",
    "material_win",
    "large_loss",
)


@dataclass(frozen=True)
class TrajectorySpec:
    stage: str
    problem_id: str
    seed: int
    max_fes: int

    @property
    def function_name(self) -> str:
        return CASE_FUNCTIONS[self.problem_id][0]

    @property
    def function_id(self) -> int:
        return CASE_FUNCTIONS[self.problem_id][1]

    @property
    def variant_id(self) -> str:
        return CASE_FUNCTIONS[self.problem_id][2]

    @property
    def trajectory_id(self) -> str:
        side = "conform" if self.stage == "conform_control" else "conflict"
        return (
            f"{side}:{self.stage}:{self.problem_id}:"
            f"seed{self.seed}:{self.max_fes}fe"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_csv(path: Path, fields: Sequence[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fields is not None and tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"CSV schema mismatch: {path}")
        return list(reader)


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if nonnegative and converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _contained(base: Path, candidate: Path) -> Path:
    root = base.resolve()
    path = candidate.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes its bound root: {candidate}")
    return path


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("diagnostic protocol version is not frozen")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("diagnostic config schema is not supported")
    if config.get("observer_only") is not True or config.get("runtime_authorized") is not False:
        raise ValueError("diagnostic config must remain observer-only")
    smoke = config.get("smoke")
    oracle = config.get("oracle")
    conform = config.get("conform_control")
    gate = config.get("gate")
    if not all(isinstance(section, dict) for section in (smoke, oracle, conform, gate)):
        raise ValueError("diagnostic config sections are incomplete")
    if smoke != {
        "case": "A4_conflict_variant_synthetic",
        "seeds": [1],
        "max_fes": SMOKE_MAX_FES,
    }:
        raise ValueError("smoke matrix differs from the frozen design")
    if (
        oracle.get("cases") != [CASE_FUNCTIONS[case][2] for case in CASE_FUNCTIONS]
        or oracle.get("seeds") != list(ORACLE_SEEDS)
        or oracle.get("max_fes") != ORACLE_MAX_FES
        or oracle.get("jobs") != ORACLE_JOBS
        or oracle.get("selected_relations_per_trajectory") != EXPECTED_RELATIONS
        or oracle.get("baseline") != "higher_reliability_owner_then_left"
        or oracle.get("bridge_weight") != "one_plus_reliability"
        or oracle.get("bridge_owner_cap") != 0.65
    ):
        raise ValueError("oracle matrix differs from the frozen design")
    if (
        conform.get("result_root")
        != "results/exp_018_rddsm_evidence_overlay_pilot/mechanism"
        or conform.get("lane_id") != "b_rddsm_evidence_overlay"
        or conform.get("evidence_overlay_mode") != "paired_owner"
        or conform.get("cases") != list(CASE_FUNCTIONS)
        or conform.get("seeds") != list(ORACLE_SEEDS)
        or conform.get("max_fes") != ORACLE_MAX_FES
    ):
        raise ValueError("conform control matrix differs from the frozen design")
    if gate.get("epsilon") != UTILITY_EPSILON:
        raise ValueError("diagnostic epsilon differs from the existing utility epsilon")
    if gate.get("material_win_log_ratio") != MATERIAL_THRESHOLD:
        raise ValueError("material-win threshold is not frozen")
    if gate.get("large_loss_log_ratio") != LARGE_LOSS_THRESHOLD:
        raise ValueError("large-loss threshold is not frozen")
    if gate.get("wilson_confidence") != CONFIDENCE:
        raise ValueError("Wilson confidence is not frozen")
    forbidden = config.get("forbidden_runtime_fields")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("forbidden runtime fields are missing")
    return config


def build_specs(stage: str) -> tuple[TrajectorySpec, ...]:
    if stage == "smoke":
        return (TrajectorySpec(stage, "A4", 1, SMOKE_MAX_FES),)
    if stage == "oracle":
        return tuple(
            TrajectorySpec(stage, problem_id, seed, ORACLE_MAX_FES)
            for problem_id in CASE_FUNCTIONS
            for seed in ORACLE_SEEDS
        )
    raise ValueError("stage must be smoke or oracle")


def select_baseline_owner(left_reliability: float, right_reliability: float) -> str:
    left = _finite(left_reliability, "left reliability")
    right = _finite(right_reliability, "right reliability")
    if not 0.0 <= left <= 1.0 or not 0.0 <= right <= 1.0:
        raise ValueError("owner reliability must be in [0, 1]")
    return "left_owner" if left >= right else "right_owner"


def wilson_bounds(
    successes: int,
    total: int,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    count = _integer(successes, "successes", minimum=0)
    sample_size = _integer(total, "total", minimum=1)
    if count > sample_size:
        raise ValueError("successes cannot exceed total")
    level = _finite(confidence, "confidence")
    if not 0.5 < level < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    z = NormalDist().inv_cdf(level)
    proportion = count / sample_size
    denominator = 1.0 + z * z / sample_size
    center = (proportion + z * z / (2.0 * sample_size)) / denominator
    margin = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / sample_size
        + z * z / (4.0 * sample_size * sample_size)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _source_files() -> tuple[Path, ...]:
    files = set(EXPERIMENT_DIR.glob("*.py"))
    files.add(CONFIG_PATH)
    files.update(SYNTHETIC_DATA_DIR.glob("*"))
    files.update((REPO_ROOT / "src" / "arac").rglob("*.py"))
    files.add(REPO_ROOT / "scripts" / "hcc_smoke_runner.py")
    files.update((VENDOR_ROOT / "AOB").glob("*.py"))
    files.update((VENDOR_ROOT / "HCC").rglob("*.py"))
    return tuple(sorted((path.resolve() for path in files if path.is_file()), key=str))


def source_bundle() -> dict[str, object]:
    files = {_relative(path): _sha256(path) for path in _source_files()}
    return {"files": files, "sha256": _canonical_sha256(files)}


def _forbidden_runtime_hits(fields: object, forbidden: Sequence[str]) -> list[str]:
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ValueError("runtime_input_fields must be a list of strings")
    fragments = tuple(str(item).casefold() for item in forbidden)
    return [
        field
        for field in fields
        if any(fragment in field.casefold() for fragment in fragments)
    ]


def _validate_common_rows(
    rows: Sequence[Mapping[str, str]],
    spec: TrajectorySpec,
    *,
    expected_mode: str = "paired_owner",
) -> None:
    if not rows:
        raise ValueError(f"raw artifact rows are empty: {spec.trajectory_id}")
    for row in rows:
        if (
            row.get("problem_id") != spec.problem_id
            or _integer(row.get("seed"), "row seed") != spec.seed
            or row.get("mode") != expected_mode
            or _integer(row.get("runtime_authorized"), "runtime_authorized") != 0
        ):
            raise ValueError(f"raw artifact identity mismatch: {spec.trajectory_id}")


def _validate_budget(path: Path, spec: TrajectorySpec, tolerance: int) -> int:
    rows = _read_csv(path, BUDGET_FIELDS)
    if len(rows) != 1:
        raise ValueError(f"budget summary must contain one row: {path}")
    row = rows[0]
    actual = _integer(row["fitness_record_fe"], "fitness_record_fe", minimum=1)
    if (
        row["problem_id"] != spec.problem_id
        or row["budget_accounting"] != "strict"
        or _integer(row["max_fes"], "max_fes") != spec.max_fes
        or _integer(row["budget_aligned_fe"], "budget_aligned_fe") != actual
        or _integer(row["same_budget_violation"], "same_budget_violation") != 0
        or _integer(row["evidence_overlay_fe"], "evidence_overlay_fe") != EXPECTED_PROBE_FE
        or actual > spec.max_fes
        or actual < spec.max_fes - tolerance
    ):
        raise ValueError(f"budget ledger is not closed: {spec.trajectory_id}")
    return actual


def _synthetic_manifest() -> dict[str, Any]:
    return _read_json(SYNTHETIC_DATA_DIR / SYNTHETIC_MANIFEST_NAME)


def _validate_aob_inputs(path: Path, spec: TrajectorySpec) -> None:
    rows = _read_csv(path, AOB_INPUT_FIELDS)
    vendor_hashes = _synthetic_manifest()["variants"][spec.variant_id]["vendor_files"]
    if {row["file"] for row in rows} != set(vendor_hashes):
        raise ValueError(f"AOB input file set mismatch: {spec.trajectory_id}")
    for row in rows:
        expected_hash = vendor_hashes[row["file"]]
        bound_path = _contained(VENDOR_DATA_DIR, Path(row["path"]))
        if (
            row["problem_id"] != spec.problem_id
            or bound_path.name != row["file"]
            or row["sha256_before"] != expected_hash
            or row["sha256_after"] != expected_hash
            or row["unchanged"] != "1"
            or _sha256(bound_path) != expected_hash
        ):
            raise ValueError(f"AOB input hash mismatch: {spec.trajectory_id}")


def _artifact_path(manifest_path: Path, manifest: Mapping[str, object], key: str) -> Path:
    artifacts = manifest.get("artifacts")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not isinstance(hashes, dict):
        raise ValueError("overlay manifest artifact binding is missing")
    value = artifacts.get(key)
    if not isinstance(value, str) or Path(value).name != value:
        raise ValueError(f"overlay artifact path is invalid: {key}")
    path = _contained(manifest_path.parent, manifest_path.parent / value)
    if not path.is_file() or hashes.get(value) != _sha256(path):
        raise ValueError(f"overlay artifact hash mismatch: {key}")
    return path


def validate_trajectory_bundle(
    spec: TrajectorySpec,
    manifest_path: Path,
    *,
    forbidden_runtime_fields: Sequence[str],
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    if expected_manifest_sha256 is not None and _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"overlay manifest hash mismatch: {spec.trajectory_id}")
    manifest = _read_json(manifest_path)
    runtime_fields = manifest.get("runtime_input_fields")
    runtime_hits = _forbidden_runtime_hits(runtime_fields, forbidden_runtime_fields)
    state = manifest.get("state_fingerprints")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"state fingerprints are missing: {spec.trajectory_id}")
    state_unchanged = all(
        isinstance(value, dict) and value.get("before") == value.get("after")
        for value in state.values()
    )
    if (
        manifest.get("protocol_version") != EVIDENCE_OVERLAY_PROTOCOL_VERSION
        or manifest.get("problem_id") != spec.problem_id
        or _integer(manifest.get("seed"), "manifest seed") != spec.seed
        or _integer(manifest.get("configured_max_fes"), "configured max FEs") != spec.max_fes
        or manifest.get("source_mode") != "fresh_runtime_probe"
        or manifest.get("evidence_overlay_mode") != "paired_owner"
        or manifest.get("barrier_status") != "probed"
        or manifest.get("barrier_reason") != "four_point_probe_complete"
        or _integer(manifest.get("applicable"), "applicable") != 1
        or _integer(manifest.get("selected_relation_count"), "selected relation count")
        != EXPECTED_RELATIONS
        or _integer(manifest.get("objective_calls"), "objective calls") != EXPECTED_PROBE_FE
        or _integer(manifest.get("evidence_overlay_fe"), "overlay FEs") != EXPECTED_PROBE_FE
        or _integer(manifest.get("fresh_optimizer_execution"), "fresh execution") != 1
        or _integer(manifest.get("native_state_unchanged"), "native state unchanged") != 1
        or _integer(manifest.get("observer_integrity"), "observer integrity") != 1
        or _integer(manifest.get("runtime_authorized"), "runtime authorized") != 0
        or _integer(manifest.get("aob_truth_runtime_used"), "AOB truth use") != 0
        or _integer(manifest.get("optimizer_calls"), "optimizer calls") != 0
        or _integer(manifest.get("rng_calls"), "RNG calls") != 0
        or manifest.get("failure") is not None
        or manifest.get("runtime_fingerprint_before") != manifest.get("runtime_fingerprint_after")
        or tuple(runtime_fields or ()) != EXPECTED_RUNTIME_FIELDS
        or runtime_hits
        or not state_unchanged
    ):
        raise ValueError(f"overlay manifest integrity failed: {spec.trajectory_id}")

    paths = {
        "checkpoint": _artifact_path(manifest_path, manifest, "checkpoint"),
        "plan": _artifact_path(manifest_path, manifest, "plan"),
        "probe_evidence": _artifact_path(manifest_path, manifest, "probe_evidence"),
        "delayed_outcomes": _artifact_path(manifest_path, manifest, "delayed_outcomes"),
        "shadow_decisions": _artifact_path(manifest_path, manifest, "shadow_decisions"),
    }
    checkpoint_rows = _read_csv(paths["checkpoint"], CHECKPOINT_FIELDS)
    plan_rows = _read_csv(paths["plan"], PLAN_FIELDS)
    probe_rows = _read_csv(paths["probe_evidence"], PROBE_EVIDENCE_FIELDS)
    delayed_rows = _read_csv(paths["delayed_outcomes"], DELAYED_OUTCOME_FIELDS)
    shadow_rows = _read_csv(paths["shadow_decisions"], SHADOW_DECISION_FIELDS)
    for rows in (checkpoint_rows, plan_rows, probe_rows, delayed_rows, shadow_rows):
        _validate_common_rows(rows, spec)

    selected = [row for row in plan_rows if row["selected"] == "1"]
    selected_ids = {row["relation_id"] for row in selected}
    if len(selected) != EXPECTED_RELATIONS or len(selected_ids) != EXPECTED_RELATIONS:
        raise ValueError(f"selected relation bundle is incomplete: {spec.trajectory_id}")
    for row in selected:
        for field in ("left_owner_reliability", "right_owner_reliability"):
            value = _finite(row[field], field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"owner reliability is out of range: {spec.trajectory_id}")

    probes_by_relation: dict[str, list[dict[str, str]]] = {}
    for row in probe_rows:
        probes_by_relation.setdefault(row["relation_id"], []).append(row)
        _finite(row["fitness"], "probe fitness", nonnegative=True)
        if _integer(row["actual_fe"], "probe actual FE") != 1:
            raise ValueError(f"probe FE mismatch: {spec.trajectory_id}")
    if set(probes_by_relation) != selected_ids or len(probe_rows) != EXPECTED_PROBE_FE:
        raise ValueError(f"probe relation pairing is incomplete: {spec.trajectory_id}")
    for relation_id, rows in probes_by_relation.items():
        if {row["candidate"] for row in rows} != set(PROBE_CANDIDATES):
            raise ValueError(f"probe candidates are incomplete: {spec.trajectory_id}:{relation_id}")

    if len(checkpoint_rows) != 1:
        raise ValueError(f"checkpoint row count mismatch: {spec.trajectory_id}")
    history = checkpoint_rows[0]["history_sweeps"].split(";")
    if len(history) != 3 or checkpoint_rows[0]["plan_status"] != "selected":
        raise ValueError(f"checkpoint is incomplete: {spec.trajectory_id}")
    if (
        len(delayed_rows) != 2 * EXPECTED_RELATIONS
        or any(row["label_closed"] != "1" for row in delayed_rows)
        or len(shadow_rows) != EXPECTED_RELATIONS
    ):
        raise ValueError(f"delayed/shadow artifact bundle is incomplete: {spec.trajectory_id}")

    tolerance = _integer(manifest.get("terminal_tolerance_fe"), "terminal tolerance", minimum=0)
    budget_path = manifest_path.with_name(f"{spec.problem_id}_budget_summary.csv")
    aob_path = manifest_path.with_name(f"{spec.problem_id}_aob_input_manifest.csv")
    actual_fe = _validate_budget(budget_path, spec, tolerance)
    _validate_aob_inputs(aob_path, spec)
    files = [manifest_path, *paths.values(), budget_path, aob_path]
    return {
        "spec": spec,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "plan_rows": plan_rows,
        "probe_rows": probe_rows,
        "actual_fe": actual_fe,
        "files": tuple(files),
    }


def analyze_trajectory(
    bundle: Mapping[str, object],
    side: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if side not in {"conflict", "conform"}:
        raise ValueError("side must be conflict or conform")
    spec = bundle["spec"]
    if not isinstance(spec, TrajectorySpec):
        raise TypeError("trajectory bundle spec is invalid")
    plan_rows = bundle["plan_rows"]
    probe_rows = bundle["probe_rows"]
    if not isinstance(plan_rows, list) or not isinstance(probe_rows, list):
        raise TypeError("trajectory bundle rows are invalid")
    selected = {row["relation_id"]: row for row in plan_rows if row["selected"] == "1"}
    probes: dict[str, dict[str, dict[str, str]]] = {}
    for row in probe_rows:
        probes.setdefault(row["relation_id"], {})[row["candidate"]] = row
    if len(selected) != EXPECTED_RELATIONS or set(probes) != set(selected):
        raise ValueError(f"trajectory relation pairing failed: {spec.trajectory_id}")

    relation_results: list[dict[str, object]] = []
    for relation_id in sorted(selected):
        plan = selected[relation_id]
        candidates = probes[relation_id]
        if set(candidates) != set(PROBE_CANDIDATES):
            raise ValueError(f"missing relation candidate: {spec.trajectory_id}:{relation_id}")
        left_reliability = _finite(plan["left_owner_reliability"], "left reliability")
        right_reliability = _finite(plan["right_owner_reliability"], "right reliability")
        baseline_owner = select_baseline_owner(left_reliability, right_reliability)
        left_fitness = _finite(
            candidates["left_owner"]["fitness"],
            "left fitness",
            nonnegative=True,
        )
        right_fitness = _finite(
            candidates["right_owner"]["fitness"], "right fitness", nonnegative=True
        )
        bridge_fitness = _finite(
            candidates["bridge"]["fitness"],
            "bridge fitness",
            nonnegative=True,
        )
        baseline_fitness = (
            left_fitness if baseline_owner == "left_owner" else right_fitness
        )
        if left_fitness <= right_fitness:
            best_owner = "left_owner"
            best_owner_fitness = left_fitness
        else:
            best_owner = "right_owner"
            best_owner_fitness = right_fitness
        delta = math.log(
            (baseline_fitness + UTILITY_EPSILON)
            / (bridge_fitness + UTILITY_EPSILON)
        )
        best_delta = math.log(
            (best_owner_fitness + UTILITY_EPSILON) / (bridge_fitness + UTILITY_EPSILON)
        )
        relation_results.append(
            {
                "side": side,
                "problem_id": spec.problem_id,
                "seed": spec.seed,
                "trajectory_id": spec.trajectory_id,
                "relation_id": relation_id,
                "left_reliability": left_reliability,
                "right_reliability": right_reliability,
                "baseline_owner": baseline_owner,
                "baseline_fitness": baseline_fitness,
                "bridge_fitness": bridge_fitness,
                "delta": delta,
                "best_owner": best_owner,
                "best_owner_fitness": best_owner_fitness,
                "best_owner_delta": best_delta,
                "material_win": int(delta > MATERIAL_THRESHOLD),
                "large_loss": int(delta <= LARGE_LOSS_THRESHOLD),
            }
        )
    trajectory_delta = statistics.median(float(row["delta"]) for row in relation_results)
    best_trajectory_delta = statistics.median(
        float(row["best_owner_delta"]) for row in relation_results
    )
    trajectory = {
        "side": side,
        "problem_id": spec.problem_id,
        "seed": spec.seed,
        "trajectory_id": spec.trajectory_id,
        "relation_count": len(relation_results),
        "trajectory_delta": trajectory_delta,
        "best_owner_trajectory_delta": best_trajectory_delta,
        "material_win": int(trajectory_delta > MATERIAL_THRESHOLD),
        "large_loss": int(trajectory_delta <= LARGE_LOSS_THRESHOLD),
    }
    return relation_results, trajectory


def summarize_side(rows: Sequence[Mapping[str, object]], side: str) -> dict[str, object]:
    if len(rows) != 15:
        raise ValueError(f"{side} requires exactly 15 case-seed units")
    pairs = {(str(row["problem_id"]), _integer(row["seed"], "seed")) for row in rows}
    expected = {(case, seed) for case in CASE_FUNCTIONS for seed in ORACLE_SEEDS}
    if pairs != expected:
        raise ValueError(f"{side} case-seed pairing is incomplete")
    deltas = [_finite(row["trajectory_delta"], "trajectory delta") for row in rows]
    best_deltas = [
        _finite(row["best_owner_trajectory_delta"], "best-owner trajectory delta")
        for row in rows
    ]
    material_wins = sum(delta > MATERIAL_THRESHOLD for delta in deltas)
    large_losses = sum(delta <= LARGE_LOSS_THRESHOLD for delta in deltas)
    win_lcb, win_ucb = wilson_bounds(material_wins, len(rows))
    loss_lcb, loss_ucb = wilson_bounds(large_losses, len(rows))
    return {
        "unit_count": len(rows),
        "material_win_count": material_wins,
        "large_loss_count": large_losses,
        "paired_win_lcb": win_lcb,
        "paired_win_ucb": win_ucb,
        "large_loss_lcb": loss_lcb,
        "large_loss_ucb": loss_ucb,
        "median_delta": statistics.median(deltas),
        "best_owner_sensitivity_median_delta": statistics.median(best_deltas),
        "case_coverage": sorted({str(row["problem_id"]) for row in rows}),
        "seed_coverage": sorted({_integer(row["seed"], "seed") for row in rows}),
    }


def _worker_command(spec: TrajectorySpec, output_dir: Path) -> tuple[str, ...]:
    timestamp = (
        f"exp019-{spec.stage}-{spec.problem_id}-"
        f"seed{spec.seed}-{spec.max_fes}fe"
    )
    return (
        sys.executable,
        "-m",
        "experiments.pilots.exp_019_conflict_resolution_pilot._diagnostic_worker",
        "--case",
        spec.problem_id,
        "--seed",
        str(spec.seed),
        "--max-fes",
        str(spec.max_fes),
        "--output-root",
        str(output_dir),
        "--timestamp",
        timestamp,
    )


def _run_worker(spec: TrajectorySpec, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fresh trajectory output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _worker_command(spec, output_dir)
    python_paths = (str(REPO_ROOT), str(REPO_ROOT / "src"), str(VENDOR_ROOT))
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(python_paths),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=VENDOR_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return {
        "spec": spec,
        "output_dir": output_dir,
        "command": list(command),
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _one_manifest(output_dir: Path, problem_id: str) -> Path:
    paths = sorted(output_dir.rglob(f"{problem_id}_evidence_overlay_manifest.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one overlay manifest under {output_dir}, found {len(paths)}")
    return paths[0]


def _execute_conflict_specs(
    specs: Sequence[TrajectorySpec],
    output: Path,
    jobs: int,
    forbidden: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _run_worker,
                spec,
                output / "_runs" / spec.problem_id / f"seed_{spec.seed}",
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                records[spec.trajectory_id] = future.result()
            except Exception as error:
                records[spec.trajectory_id] = {
                    "spec": spec,
                    "output_dir": output / "_runs" / spec.problem_id / f"seed_{spec.seed}",
                    "returncode": -1,
                    "elapsed_seconds": 0.0,
                    "stdout_tail": "",
                    "stderr_tail": f"{type(error).__name__}:{error}",
                }
    ordered_records = [records[spec.trajectory_id] for spec in specs]
    serializable_records = []
    bundles = []
    for record in ordered_records:
        spec = record["spec"]
        if not isinstance(spec, TrajectorySpec):
            raise TypeError("worker record spec is invalid")
        serializable = {key: value for key, value in record.items() if key != "spec"}
        serializable["trajectory_id"] = spec.trajectory_id
        serializable["problem_id"] = spec.problem_id
        serializable["seed"] = spec.seed
        serializable["max_fes"] = spec.max_fes
        serializable["output_dir"] = _relative(Path(str(record["output_dir"])))
        serializable_records.append(serializable)
        if record["returncode"] != 0:
            raise RuntimeError(
                f"worker failed for {spec.trajectory_id}: {record['stderr_tail']}"
            )
        manifest_path = _one_manifest(Path(str(record["output_dir"])), spec.problem_id)
        bundles.append(
            validate_trajectory_bundle(
                spec,
                manifest_path,
                forbidden_runtime_fields=forbidden,
            )
        )
    return bundles, serializable_records


def _expected_conform_specs() -> tuple[TrajectorySpec, ...]:
    return tuple(
        TrajectorySpec("conform_control", problem_id, seed, ORACLE_MAX_FES)
        for problem_id in CASE_FUNCTIONS
        for seed in ORACLE_SEEDS
    )


def _filter_control_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("problem_id") in CASE_FUNCTIONS
        and row.get("seed") in {str(seed) for seed in ORACLE_SEEDS}
        and row.get("lane_id") == "b_rddsm_evidence_overlay"
        and row.get("evidence_overlay_mode") == "paired_owner"
        and row.get("max_fes") == str(ORACLE_MAX_FES)
    ]


def load_conform_controls(
    config: Mapping[str, object],
    forbidden: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    conform = config["conform_control"]
    if not isinstance(conform, dict):
        raise TypeError("conform control config is invalid")
    root = _contained(REPO_ROOT, REPO_ROOT / str(conform["result_root"]))
    gate_path = root / "promotion_gate.json"
    aggregate_manifest_path = root / "manifest.json"
    gate = _read_json(gate_path)
    aggregate_manifest = _read_json(aggregate_manifest_path)
    if (
        gate.get("stage") != "mechanism"
        or gate.get("protocol_version") != EVIDENCE_OVERLAY_PROTOCOL_VERSION
    ):
        raise ValueError("exp_018 conform gate identity mismatch")
    checks = gate.get("checks")
    required_checks = conform.get("required_integrity_checks")
    if not isinstance(checks, dict) or not isinstance(required_checks, list):
        raise ValueError("exp_018 integrity checks are missing")
    failed_checks = [name for name in required_checks if checks.get(name) is not True]
    source_integrity = gate.get("source_integrity")
    aob_integrity = gate.get("aob_data_integrity")
    if (
        failed_checks
        or not isinstance(source_integrity, dict)
        or source_integrity.get("source_bundle_unchanged") is not True
        or source_integrity.get("config_unchanged") is not True
        or not isinstance(aob_integrity, dict)
        or aob_integrity.get("unchanged") is not True
        or gate.get("config_sha256") != _sha256(EXP018_CONFIG_PATH)
    ):
        raise ValueError(f"exp_018 conform integrity gate failed: {failed_checks}")

    run_results_path = root / "run_results.csv"
    ledger_path = root / "same_budget_ledger.csv"
    anti_path = root / "anti_leakage_audit.csv"
    run_rows = _filter_control_rows(_read_csv(run_results_path))
    ledger_rows = _filter_control_rows(_read_csv(ledger_path))
    anti_rows = _filter_control_rows(_read_csv(anti_path))
    expected_specs = _expected_conform_specs()
    expected_pairs = {(spec.problem_id, spec.seed) for spec in expected_specs}
    if (
        len(run_rows) != 15
        or len(ledger_rows) != 15
        or len(anti_rows) != 15
        or {(row["problem_id"], int(row["seed"])) for row in run_rows} != expected_pairs
        or {(row["problem_id"], int(row["seed"])) for row in ledger_rows} != expected_pairs
        or {(row["problem_id"], int(row["seed"])) for row in anti_rows} != expected_pairs
    ):
        raise ValueError("exp_018 conform case-seed pairing is incomplete")
    if any(
        row.get("status") != "completed"
        or row.get("fresh_optimizer_execution") != "1"
        or row.get("runtime_authorized") != "0"
        or row.get("applicable") != "1"
        or row.get("evidence_overlay_fe") != str(EXPECTED_PROBE_FE)
        for row in run_rows
    ):
        raise ValueError("exp_018 conform run integrity failed")
    if any(
        row.get("ledger_closed") != "1"
        or row.get("same_budget_violation") != "0"
        or row.get("evidence_overlay_fe") != str(EXPECTED_PROBE_FE)
        for row in ledger_rows
    ):
        raise ValueError("exp_018 conform FE ledger failed")
    if any(row.get("audit_status") != "pass" for row in anti_rows):
        raise ValueError("exp_018 conform anti-leakage audit failed")

    per_run = aggregate_manifest.get("per_run_manifests")
    if not isinstance(per_run, list):
        raise ValueError("exp_018 per-run manifest binding is missing")
    manifest_bindings = {
        str(item.get("trajectory_id")): item
        for item in per_run
        if isinstance(item, dict)
    }
    row_by_pair = {(row["problem_id"], int(row["seed"])): row for row in run_rows}
    bundles = []
    source_paths = {
        gate_path,
        aggregate_manifest_path,
        EXP018_CONFIG_PATH,
        *(root / name for name in aggregate_manifest.get("artifacts", [])),
    }
    for spec in expected_specs:
        row = row_by_pair[(spec.problem_id, spec.seed)]
        manifest_path = _contained(root, Path(row["overlay_manifest_path"]))
        expected_hash = row["overlay_manifest_sha256"]
        binding = manifest_bindings.get(row["trajectory_id"])
        if (
            not isinstance(binding, dict)
            or binding.get("path") != str(manifest_path)
            or binding.get("sha256") != expected_hash
        ):
            raise ValueError(f"exp_018 raw manifest binding failed: {spec.trajectory_id}")
        bundle = validate_trajectory_bundle(
            spec,
            manifest_path,
            forbidden_runtime_fields=forbidden,
            expected_manifest_sha256=expected_hash,
        )
        bundles.append(bundle)
        source_paths.update(bundle["files"])
    file_records = [_file_record(path) for path in sorted(source_paths, key=str)]
    source_manifest = {
        "source": "exp_018_mechanism_paired_owner_conform_control",
        "promotion_status": gate.get("status"),
        "required_integrity_checks": required_checks,
        "integrity_checks_passed": True,
        "file_count": len(file_records),
        "files": file_records,
    }
    source_manifest["bundle_sha256"] = _canonical_sha256(file_records)
    return bundles, source_manifest


def _side_checks(conflict: Mapping[str, object], conform: Mapping[str, object]) -> dict[str, bool]:
    return {
        "conflict_paired_win_lcb": float(conflict["paired_win_lcb"]) > 0.5,
        "conflict_median_delta": float(conflict["median_delta"]) > MATERIAL_THRESHOLD,
        "conflict_zero_large_losses": int(conflict["large_loss_count"]) == 0,
        "conform_paired_win_lcb": float(conform["paired_win_lcb"]) <= 0.5,
        "conform_abs_median_delta": abs(float(conform["median_delta"])) <= MATERIAL_THRESHOLD,
        "conform_zero_large_losses": int(conform["large_loss_count"]) == 0,
    }


def _synthetic_source_binding() -> dict[str, object]:
    manifest_path = SYNTHETIC_DATA_DIR / SYNTHETIC_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    files = [_file_record(manifest_path)]
    for entry in manifest["variants"].values():
        files.append(_file_record(SYNTHETIC_DATA_DIR / entry["synthetic_csv"]["path"]))
    files.sort(key=lambda row: str(row["path"]))
    return {"files": files, "bundle_sha256": _canonical_sha256(files)}


def _validate_prior_smoke(
    root: Path,
    config_hash: str,
    current_source: Mapping[str, object],
) -> None:
    gate = _read_json(root / "smoke" / "smoke_gate.json")
    if (
        gate.get("status") != "smoke_pass"
        or gate.get("protocol_version") != PROTOCOL_VERSION
        or gate.get("config_sha256") != config_hash
        or gate.get("source_bundle") != current_source
        or not all(gate.get("checks", {}).values())
    ):
        raise RuntimeError("oracle requires a bound passing smoke gate")


def run_diagnostic(
    stage: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_path: Path = CONFIG_PATH,
) -> Path:
    config = load_config(config_path)
    validate_synthetic_bundle()
    root = output_root.resolve()
    output = root / stage
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fresh diagnostic stage output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(config_path)
    initial_source = source_bundle()
    if stage == "oracle":
        _validate_prior_smoke(root, config_hash, initial_source)
    specs = build_specs(stage)
    forbidden = tuple(str(value) for value in config["forbidden_runtime_fields"])
    jobs = 1 if stage == "smoke" else ORACLE_JOBS
    gate_path = output / ("smoke_gate.json" if stage == "smoke" else "oracle_gate.json")
    try:
        bundles, execution_records = _execute_conflict_specs(
            specs,
            output,
            jobs,
            forbidden,
        )
        final_source = source_bundle()
        source_unchanged = final_source == initial_source
        _write_json(output / "execution_records.json", execution_records)
        conflict_relations = []
        conflict_trajectories = []
        for bundle in bundles:
            relation_rows, trajectory = analyze_trajectory(bundle, "conflict")
            conflict_relations.extend(relation_rows)
            conflict_trajectories.append(trajectory)

        base_gate = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "observer_only": True,
            "runtime_authorized": False,
            "config_sha256": config_hash,
            "source_bundle": initial_source,
            "source_unchanged": source_unchanged,
            "synthetic_source_binding": _synthetic_source_binding(),
            "claim_boundary": (
                "same_checkpoint_immediate_objective_value_not_terminal_writeback_gain"
            ),
        }
        if stage == "smoke":
            checks = {
                "one_fresh_conflict_trajectory": len(bundles) == 1,
                "four_relations": len(conflict_relations) == EXPECTED_RELATIONS,
                "source_unchanged": source_unchanged,
                "observer_only": True,
            }
            _write_csv(
                output / "relation_results.csv",
                conflict_relations,
                RELATION_RESULT_FIELDS,
            )
            _write_csv(
                output / "trajectory_results.csv",
                conflict_trajectories,
                TRAJECTORY_RESULT_FIELDS,
            )
            gate = {
                **base_gate,
                "status": "smoke_pass" if all(checks.values()) else "smoke_fail",
                "checks": checks,
                "blockers": [name for name, passed in checks.items() if not passed],
            }
            _write_json(gate_path, gate)
            return output

        conform_bundles, conform_source = load_conform_controls(config, forbidden)
        conform_relations = []
        conform_trajectories = []
        for bundle in conform_bundles:
            relation_rows, trajectory = analyze_trajectory(bundle, "conform")
            conform_relations.extend(relation_rows)
            conform_trajectories.append(trajectory)
        all_relations = [*conflict_relations, *conform_relations]
        all_trajectories = [*conflict_trajectories, *conform_trajectories]
        conflict_summary = summarize_side(conflict_trajectories, "conflict")
        conform_summary = summarize_side(conform_trajectories, "conform")
        checks = {
            "all_conflict_trajectories_fresh_and_complete": len(bundles) == 15,
            "all_conform_trajectories_bound_and_complete": len(conform_bundles) == 15,
            "source_unchanged": source_unchanged,
            **_side_checks(conflict_summary, conform_summary),
        }
        blockers = [f"gate_failed:{name}" for name, passed in checks.items() if not passed]
        _write_csv(output / "relation_results.csv", all_relations, RELATION_RESULT_FIELDS)
        _write_csv(
            output / "trajectory_results.csv",
            all_trajectories,
            TRAJECTORY_RESULT_FIELDS,
        )
        _write_json(output / "conform_source_manifest.json", conform_source)
        gate = {
            **base_gate,
            "status": "oracle_go" if all(checks.values()) else "oracle_no_go",
            "checks": checks,
            "metrics": {
                "conflict": conflict_summary,
                "conform": conform_summary,
            },
            "thresholds": {
                "material_win": MATERIAL_THRESHOLD,
                "large_loss": LARGE_LOSS_THRESHOLD,
                "wilson_confidence": CONFIDENCE,
            },
            "conform_source_bundle_sha256": conform_source["bundle_sha256"],
            "blockers": blockers,
        }
        _write_json(gate_path, gate)
        return output
    except Exception as error:
        gate = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "status": "diagnostic_invalid",
            "observer_only": True,
            "runtime_authorized": False,
            "config_sha256": config_hash,
            "source_bundle": initial_source,
            "source_unchanged": source_bundle() == initial_source,
            "blockers": [f"{type(error).__name__}:{error}"],
        }
        _write_json(gate_path, gate)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen exp_019 diagnostic.")
    parser.add_argument("--stage", required=True, choices=("smoke", "oracle"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run_diagnostic(str(args.stage))
    gate_name = "smoke_gate.json" if args.stage == "smoke" else "oracle_gate.json"
    gate = _read_json(output / gate_name)
    print(json.dumps({"output": str(output), "status": gate["status"]}, sort_keys=True))
    return 1 if gate["status"] in {"smoke_fail", "diagnostic_invalid"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
