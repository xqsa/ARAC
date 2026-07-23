"""Execute WLOC baseline tasks and emit strict synthetic-only artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from arac.baselines import (
    BASELINE_RESULT_SCHEMA_VERSION,
    GroupingResult,
    design_matrix_from_groups,
    dg2_grouping,
    random_grouping,
    rddsm_grouping,
    rdg3_grouping,
    run_cooperative_cmaes,
    run_hcc_es,
    run_pypop_baseline,
)
from arac.benchmarks import Wang2025ContinuousProblem, get_wang2025_local_escape_case

from .protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ProtocolConfig,
    WLOCBaselineTask,
    build_task,
    stable_hash,
)


ARTIFACT_SCHEMA_VERSION = "wloc-baseline-artifact-v1"
MATRIX_SCHEMA_VERSION = "wloc-baseline-task-matrix-v1"
DECOMPOSITION_METHODS = {
    "DG2-CMAES",
    "Random-CMAES",
    "RDG3-CMAES",
    "RDDSM-CMAES",
}
NDA_METHODS = {"Sep-CMAES", "LM-MA-ES", "LMCMA", "MM-ES"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provided_smoke_grouping(problem: Wang2025ContinuousProblem, method: str) -> GroupingResult:
    return GroupingResult(
        method=method.removesuffix("-CMAES"),
        dimension=problem.dimension,
        groups=problem.groups,
        decomposition_fes=0,
        allows_overlap=True,
        origin="provided_catalog_topology_smoke",
    )


def build_grouping(
    task: WLOCBaselineTask,
    problem: Wang2025ContinuousProblem,
    config: ProtocolConfig,
) -> GroupingResult:
    if task.decomposition_mode == "provided_catalog_topology_smoke":
        return _provided_smoke_grouping(problem, task.method)
    if task.method == "DG2-CMAES":
        return dg2_grouping(problem, problem.dimension)
    if task.method == "Random-CMAES":
        return random_grouping(
            problem.dimension,
            seed=task.optimizer_seed,
            group_count=config.random_group_count,
        )
    if task.method in {"RDDSM-CMAES", "HCC-ES"}:
        design = design_matrix_from_groups(problem.dimension, problem.groups)
        return rddsm_grouping(design)
    if task.method == "RDG3-CMAES":
        return rdg3_grouping(
            problem,
            problem.dimension,
            nonseparable_threshold=config.rdg3_nonseparable_threshold,
            separable_chunk_size=config.rdg3_separable_chunk_size,
        )
    raise ValueError(f"method does not use a grouping: {task.method}")


def execute_task(
    task: WLOCBaselineTask,
    config: ProtocolConfig,
) -> tuple[Wang2025ContinuousProblem, GroupingResult | None, object]:
    if task.protocol_hash != config.protocol_hash:
        raise ValueError("task does not belong to the active protocol")
    case = get_wang2025_local_escape_case(task.case_id)
    source = case.generate()
    problem = Wang2025ContinuousProblem(source)
    if (
        task.dimension != problem.dimension
        or task.instance_seed != case.spec.seed
        or task.source_instance_hash != source.instance_hash
        or task.objective_hash != problem.objective_hash
    ):
        raise ValueError("task does not match the frozen benchmark instance")
    common = {
        "max_function_evaluations": task.optimization_fes,
        "seed": task.optimizer_seed,
        "initial_mean": config.initial_mean,
        "sigma": config.sigma,
    }
    grouping = None
    if task.method in DECOMPOSITION_METHODS:
        grouping = build_grouping(task, problem, config)
        result = run_cooperative_cmaes(
            problem,
            grouping,
            method_name=task.method,
            **common,
        )
    elif task.method in NDA_METHODS:
        result = run_pypop_baseline(
            problem,
            task.method,
            problem.dimension,
            **common,
        )
    elif task.method == "HCC-ES":
        grouping = build_grouping(task, problem, config)
        result = run_hcc_es(problem, grouping, **common)
    else:
        raise ValueError(f"unsupported WLOC baseline method: {task.method}")
    return problem, grouping, result


def _grouping_manifest(grouping: GroupingResult | None) -> dict[str, Any] | None:
    if grouping is None:
        return None
    return {
        "method": grouping.method,
        "dimension": grouping.dimension,
        "groups": [list(group) for group in grouping.groups],
        "decomposition_fes": grouping.decomposition_fes,
        "allows_overlap": grouping.allows_overlap,
        "origin": grouping.origin,
        "matrix_kind": grouping.matrix_kind,
        "grouping_hash": grouping.grouping_hash,
    }


def write_artifact(
    output_root: Path,
    task: WLOCBaselineTask,
    config: ProtocolConfig,
    problem: Wang2025ContinuousProblem,
    grouping: GroupingResult | None,
    result: object,
) -> Path:
    task_dir = output_root / task.task_hash
    task_dir.mkdir(parents=True, exist_ok=False)
    trace_path = task_dir / "best_so_far.csv"
    with trace_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("optimization_fe", "best_so_far"))
        for fe, value in enumerate(result.best_so_far_trace, start=1):
            writer.writerow((fe, format(value, ".17g")))
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_hash": config.protocol_hash,
        "synthetic_only": True,
        "real_aob_action_gate_eligible": False,
        "performance_claim_authorized": False,
        "task": task.to_dict(),
        "problem": problem.to_manifest(),
        "grouping": _grouping_manifest(grouping),
        "result_schema_version": BASELINE_RESULT_SCHEMA_VERSION,
        "result": asdict(result),
        "files": {"best_so_far.csv": _file_hash(trace_path)},
    }
    payload["artifact_hash"] = stable_hash(payload)
    manifest_path = task_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_artifact(manifest_path)
    return manifest_path


def run_task(
    task: WLOCBaselineTask,
    config: ProtocolConfig,
    output_root: Path,
) -> Path:
    problem, grouping, result = execute_task(task, config)
    return write_artifact(output_root, task, config, problem, grouping, result)


def validate_artifact(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported WLOC artifact schema")
    if payload.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("artifact protocol schema mismatch")
    if payload.get("synthetic_only") is not True:
        raise ValueError("WLOC artifact must be marked synthetic_only")
    if payload.get("real_aob_action_gate_eligible") is not False:
        raise ValueError("WLOC artifact cannot be real-AOB action-gate eligible")
    artifact_hash = payload.pop("artifact_hash", None)
    if artifact_hash != stable_hash(payload):
        raise ValueError("artifact hash mismatch")
    payload["artifact_hash"] = artifact_hash
    task = WLOCBaselineTask.from_dict(payload["task"])
    result = payload["result"]
    if result["result_hash"] != stable_hash(
        {
            "schema_version": payload["result_schema_version"],
            **{key: value for key, value in result.items() if key != "result_hash"},
        }
    ):
        raise ValueError("result hash mismatch")
    trace_path = manifest_path.parent / "best_so_far.csv"
    if _file_hash(trace_path) != payload["files"]["best_so_far.csv"]:
        raise ValueError("trace file hash mismatch")
    with trace_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != task.optimization_fes:
        raise ValueError("trace row count does not match optimization_fes")
    values = [float(row["best_so_far"]) for row in rows]
    if any(later > earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ValueError("artifact trace is not best-so-far monotonic")
    if values[-1] != result["best_y"]:
        raise ValueError("artifact final trace value does not match best_y")
    return payload


def run_mechanical_smoke(
    config: ProtocolConfig,
    output_root: Path,
    *,
    case_id: str = "WLOC01",
    methods: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    selected = methods or config.methods
    manifests = []
    for method in selected:
        task = build_task(config, case_id, method, mechanical_smoke=True)
        manifests.append(run_task(task, config, output_root))
    return tuple(manifests)


def write_task_matrix(path: Path, config: ProtocolConfig, tasks: tuple[WLOCBaselineTask, ...]) -> Path:
    task_rows = [task.to_dict() for task in tasks]
    payload = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "protocol_hash": config.protocol_hash,
        "task_count": len(task_rows),
        "tasks": task_rows,
    }
    payload["matrix_hash"] = stable_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
