"""U1 matched-host reachability for the shared_patch_v1 upgrade candidate.

Produces the per-case reachability table required by
docs/arac-oc-stepwise-upgrade-plan-v2.1.md before any S-ladder screen:

- AOB lane: the twelve mapped-host arms (S1-S6 -> ctp, R1-R6 -> gcb) are
  re-run at the reachability seed with non-invasive sweep instrumentation
  and must reproduce the frozen recovery-screen receipts bit-identically;
  A/E cases are recorded as structurally mount-absent (no CTP/GSS mount
  point exists in aor/smp by freeze contract).
- Generator lane: the six preregistered conflicting-overlap cells run the
  frozen Phase-I discovery once per cell, then both hosts are forced on
  the shared checkpoint; the host label is local to this experiment.

U1 authorizes no performance comparison; it only establishes where the
S-ladder can mount.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ACTION_NAMES, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import current_recovered_four_arm as fixed
from experiments.historical_recovery.replay import _checkpoint
from experiments.upgrade.shared_patch_v1.conflicting_generator import (
    CELL_IDS,
    GENERATOR_PROTOCOL,
    build_problem,
    relation_leverage,
    run_generator_phase1,
)
from experiments.upgrade.shared_patch_v1.host_instrumentation import SweepRecorder


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("u1_host_reachability_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-u1-arm-receipt-v1"
CHECKPOINT_SCHEMA = "arac-upgrade-u1-generator-checkpoint-v1"
FAILURE_SCHEMA = "arac-upgrade-u1-failure-v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            entries.append((path.relative_to(root).as_posix(), _sha256(path)))
    return len(entries), canonical_sha256(entries)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-u1-host-reachability-protocol-v1",
        "candidate_id": "shared_patch_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "reachability_seed": 117,
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "output_root": "artifacts/upgrade_u1_host_reachability_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"U1 protocol drifted: {key}")
    aob = protocol["aob_lane"]
    if set(aob["mapped_hosts"]) != {f"S{index}" for index in range(1, 7)} | {f"R{index}" for index in range(1, 7)}:
        raise ValueError("U1 AOB mapped hosts drifted")
    if set(aob["structural_rows"]) != {f"A{index}" for index in range(1, 7)} | {f"E{index}" for index in range(1, 7)}:
        raise ValueError("U1 AOB structural rows drifted")
    if any(action not in ACTION_NAMES for action in (*aob["mapped_hosts"].values(), *aob["structural_rows"].values())):
        raise ValueError("U1 action names drifted")
    generator = protocol["generator_lane"]
    if tuple(generator["cells"]) != CELL_IDS or sorted(generator["hosts"]) != ["ctp", "gcb"]:
        raise ValueError("U1 generator lane drifted")
    for source in protocol["sources"]:
        if not (REPOSITORY_ROOT / source).is_file():
            raise ValueError(f"U1 source is missing: {source}")
    for tree in protocol["pinned_trees"]:
        if not (REPOSITORY_ROOT / tree).is_dir():
            raise ValueError(f"U1 pinned tree is missing: {tree}")
    return protocol


@dataclass(frozen=True)
class AobArmJob:
    case_id: str
    run_seed: int
    action_name: str
    checkpoint_root: Path
    current_receipt_root: Path
    output_root: Path
    frozen_receipt_root: Path
    manifest_sha256: str

    @property
    def context(self) -> fixed.ArmContext:
        return fixed.ArmContext(
            case_id=self.case_id,
            run_seed=self.run_seed,
            action_name=self.action_name,
            checkpoint_root=self.checkpoint_root,
            current_receipt_root=self.current_receipt_root,
            output_root=self.output_root,
            manifest_sha256=self.manifest_sha256,
        )

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / "aob" / self.case_id / f"seed_{self.run_seed}" / f"{self.action_name}.json"

    @property
    def frozen_receipt_path(self) -> Path:
        return self.frozen_receipt_root / self.case_id / f"seed_{self.run_seed}" / f"{self.action_name}.json"

    @property
    def key(self) -> str:
        return f"aob:{self.case_id}:seed-{self.run_seed}:{self.action_name}"


@dataclass(frozen=True)
class GeneratorArmJob:
    cell_id: str
    run_seed: int
    host: str
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.cell_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / "generator" / self.cell_id / f"seed_{self.run_seed}" / f"{self.host}.json"

    @property
    def key(self) -> str:
        return f"generator:{self.cell_id}:seed-{self.run_seed}:{self.host}"


def _leverage_rows(checkpoint) -> dict[str, Any]:
    leverage = relation_leverage(checkpoint.blocks, checkpoint.relations)
    return {
        "block_count": len(checkpoint.blocks),
        "relation_count": len(checkpoint.relations),
        "relation_strength_max": max((relation.strength for relation in checkpoint.relations), default=0.0),
        "leverage_per_block": list(leverage),
        "leverage_positive_blocks": sum(1 for value in leverage if value > 0),
    }


def _write_failure(output_root: Path, job_key: str, exc: BaseException) -> None:
    _write_json(
        output_root / "failures" / f"{job_key.replace(':', '_')}.json",
        {
            "schema_version": FAILURE_SCHEMA,
            "key": job_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "threadpools": pools,
    }


def _terminal_contract(job_key: str, result, ledger) -> None:
    if (
        result.consumed_fes != 2_820_000
        or result.terminal_fes != 3_000_000
        or ledger.count != 3_000_000
        or result.final_error != ledger.best_error
        or not math.isfinite(result.final_error)
    ):
        raise RuntimeError(f"{job_key} terminal contract failed")


def _run_aob_arm(job: AobArmJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            checkpoint = fixed._load_verified_checkpoint(job.context)
            problem = AobBenchmark().load(job.case_id)
            registry = RecoveredActionRegistry()
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=checkpoint.total_budget_fes,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            recorder = SweepRecorder()
            recorder.install()
            try:
                result = execute_phase2_action(
                    job.action_name,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=job.run_seed,
                    registry=registry,
                )
            finally:
                recorder.uninstall()
            _terminal_contract(job.key, result, ledger)
            frozen = _load_json(job.frozen_receipt_path)
            identity = {
                field: {"rerun": value, "frozen": frozen[field], "match": value == frozen[field]}
                for field, value in (
                    ("final_error", result.final_error),
                    ("route", result.route),
                    ("action_result_hash", result.result_hash),
                    ("checkpoint_hash", checkpoint.checkpoint_hash),
                )
            }
            leverage_rows = _leverage_rows(checkpoint)
            instrumentation = recorder.summary(tuple(leverage_rows["leverage_per_block"]))
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "lane": "aob",
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "host": job.action_name,
                "forced_host": False,
                "phase1_protocol": checkpoint.protocol,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "identity": identity,
                "leverage": leverage_rows,
                "instrumentation": instrumentation,
                "segments": recorder.segments,
                "runtime": runtime,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            receipt = {**body, "receipt_hash": canonical_sha256(body)}
            _write_json(job.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


def _load_generator_checkpoint(job: GeneratorArmJob):
    wrapper = _load_json(job.checkpoint_path)
    if canonical_sha256({key: value for key, value in wrapper.items() if key != "receipt_hash"}) != wrapper.get("receipt_hash"):
        raise ValueError(f"{job.key} generator checkpoint receipt hash drifted")
    if (
        wrapper.get("schema_version") != CHECKPOINT_SCHEMA
        or wrapper.get("cell_id") != job.cell_id
        or wrapper.get("run_seed") != job.run_seed
    ):
        raise ValueError(f"{job.key} generator checkpoint identity drifted")
    checkpoint = _checkpoint(wrapper["checkpoint"])
    if checkpoint.checkpoint_hash != wrapper.get("checkpoint_hash"):
        raise ValueError(f"{job.key} generator checkpoint hash drifted")
    return checkpoint, str(wrapper["ground_truth_hash"])


def write_generator_checkpoint(job: GeneratorArmJob) -> dict[str, Any]:
    if job.checkpoint_path.is_file():
        return _load_json(job.checkpoint_path)
    checkpoint, stats = run_generator_phase1(job.cell_id, job.run_seed)
    body = {
        "schema_version": CHECKPOINT_SCHEMA,
        "cell_id": job.cell_id,
        "run_seed": job.run_seed,
        "generator_protocol": GENERATOR_PROTOCOL,
        "planted_link_count": stats["planted_link_count"],
        "planted_shared_variable_count": stats["planted_shared_variable_count"],
        "discovered_relation_count": stats["discovered_relation_count"],
        "discovered_relation_strength_max": stats["discovered_relation_strength_max"],
        "ground_truth_hash": stats["ground_truth_hash"],
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "checkpoint": stats["checkpoint"],
    }
    body["receipt_hash"] = canonical_sha256(body)
    _write_json(job.checkpoint_path, body)
    return body


def _run_generator_arm(job: GeneratorArmJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            checkpoint, recorded_truth_hash = _load_generator_checkpoint(job)
            problem, truth = build_problem(job.cell_id, job.run_seed)
            if truth.ground_truth_hash != recorded_truth_hash:
                raise ValueError(f"{job.key} rebuilt generator ground truth drifted from the frozen checkpoint receipt")
            registry = RecoveredActionRegistry()
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=checkpoint.total_budget_fes,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            recorder = SweepRecorder()
            recorder.install()
            try:
                result = execute_phase2_action(
                    job.host,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=job.run_seed,
                    registry=registry,
                )
            finally:
                recorder.uninstall()
            _terminal_contract(job.key, result, ledger)
            leverage_rows = _leverage_rows(checkpoint)
            instrumentation = recorder.summary(tuple(leverage_rows["leverage_per_block"]))
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "lane": "generator",
                "cell_id": job.cell_id,
                "run_seed": job.run_seed,
                "host": job.host,
                "forced_host": True,
                "phase1_protocol": checkpoint.protocol,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "ground_truth_hash": truth.ground_truth_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "leverage": leverage_rows,
                "instrumentation": instrumentation,
                "segments": recorder.segments,
                "runtime": runtime,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            receipt = {**body, "receipt_hash": canonical_sha256(body)}
            _write_json(job.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": "arac-upgrade-u1-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(Path(protocol_path).resolve()),
        "reachability_seed": protocol["reachability_seed"],
        "source_sha256": {source: _sha256(REPOSITORY_ROOT / source) for source in sorted(protocol["sources"])},
        "pinned_trees": {
            tree: dict(zip(("file_count", "sha256"), _tree_sha256(REPOSITORY_ROOT / tree)))
            for tree in sorted(protocol["pinned_trees"])
        },
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _aob_row(case_id: str, action: str, receipts: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any], *, structural: bool) -> dict[str, Any]:
    if structural:
        return {
            "lane": "aob",
            "case_id": case_id,
            "mapped_action": action,
            "host": None,
            "host_status": "mount_absent_by_contract",
            "mechanism_silent": None,
            "scope_visit_count": 0,
            "leverage_positive_visits": 0,
            "block_count": None,
            "relation_count": None,
            "reachable": False,
        }
    receipt = receipts[f"aob:{case_id}:seed-{protocol['reachability_seed']}:{action}"]
    leverage = receipt["leverage"]
    instrumentation = receipt["instrumentation"]
    relations_exist = leverage["relation_count"] > 0
    reachable = instrumentation["total_block_visits"] > 0
    return {
        "lane": "aob",
        "case_id": case_id,
        "mapped_action": action,
        "host": action,
        "host_status": "reachable" if reachable else "host_unreachable",
        "mechanism_silent": not relations_exist,
        "scope_visit_count": instrumentation["total_block_visits"],
        "leverage_positive_visits": instrumentation["leverage_positive_block_visits"],
        "block_count": leverage["block_count"],
        "relation_count": leverage["relation_count"],
        "leverage_positive_blocks": leverage["leverage_positive_blocks"],
        "reachable": reachable and (instrumentation["leverage_positive_block_visits"] > 0 or not relations_exist),
        "identity_with_frozen_screen": all(field["match"] for field in receipt["identity"].values()),
    }


def _generator_row(cell_id: str, host: str, receipts: Mapping[str, Mapping[str, Any]], checkpoints: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    receipt = receipts[f"generator:{cell_id}:seed-{protocol['reachability_seed']}:{host}"]
    checkpoint_stats = checkpoints[cell_id]
    leverage = receipt["leverage"]
    instrumentation = receipt["instrumentation"]
    discovered = int(checkpoint_stats["discovered_relation_count"]) > 0 and float(checkpoint_stats["discovered_relation_strength_max"]) > 0.0
    reachable = instrumentation["total_block_visits"] > 0
    return {
        "lane": "generator",
        "cell_id": cell_id,
        "host": host,
        "forced_host": True,
        "host_status": "reachable" if reachable else "host_unreachable",
        "phase1_discovery_positive": discovered,
        "scope_visit_count": instrumentation["total_block_visits"],
        "leverage_positive_visits": instrumentation["leverage_positive_block_visits"],
        "block_count": leverage["block_count"],
        "relation_count": leverage["relation_count"],
        "leverage_positive_blocks": leverage["leverage_positive_blocks"],
        "reachable": discovered and reachable and instrumentation["leverage_positive_block_visits"] > 0,
    }


def _receipt_body_for_hash(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the write-time canonical form of an as-written arm receipt.

    The first U1 execution wrote ``segments[*].per_block_visits`` with integer
    keys; ``json.dumps(sort_keys=True)`` orders integer keys numerically while
    a parsed JSON round trip orders the stringified keys lexically, so the
    hash must be recomputed over the integer-keyed form the receipt was built
    with.  Later stages build those keys as strings and need no normalization.
    """

    body = json.loads(json.dumps({key: value for key, value in receipt.items() if key != "receipt_hash"}, sort_keys=True))
    for segment in body.get("segments", ()):
        per_block = segment.get("per_block_visits")
        if isinstance(per_block, dict):
            segment["per_block_visits"] = {int(key): value for key, value in per_block.items()}
    return body


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    reachability_seed = int(protocol["reachability_seed"])
    receipts: dict[str, Mapping[str, Any]] = {}
    checkpoints: dict[str, Mapping[str, Any]] = {}
    for case_id, action in protocol["aob_lane"]["mapped_hosts"].items():
        path = output_root / "arms" / "aob" / case_id / f"seed_{reachability_seed}" / f"{action}.json"
        receipt = _load_json(path)
        if canonical_sha256(_receipt_body_for_hash(receipt)) != receipt.get("receipt_hash"):
            raise ValueError(f"U1 arm receipt hash drifted: {case_id}/{action}")
        receipts[f"aob:{case_id}:seed-{reachability_seed}:{action}"] = receipt
    for cell_id in protocol["generator_lane"]["cells"]:
        checkpoints[cell_id] = _load_json(output_root / "checkpoints" / cell_id / f"seed_{reachability_seed}" / "checkpoint.json")
        for host in protocol["generator_lane"]["hosts"]:
            path = output_root / "arms" / "generator" / cell_id / f"seed_{reachability_seed}" / f"{host}.json"
            receipt = _load_json(path)
            if canonical_sha256(_receipt_body_for_hash(receipt)) != receipt.get("receipt_hash"):
                raise ValueError(f"U1 arm receipt hash drifted: {cell_id}/{host}")
            receipts[f"generator:{cell_id}:seed-{reachability_seed}:{host}"] = receipt
    rows = []
    for case_id, action in {**protocol["aob_lane"]["structural_rows"], **protocol["aob_lane"]["mapped_hosts"]}.items():
        rows.append(_aob_row(case_id, action, receipts, protocol, structural=case_id in protocol["aob_lane"]["structural_rows"]))
    for cell_id in protocol["generator_lane"]["cells"]:
        for host in protocol["generator_lane"]["hosts"]:
            rows.append(_generator_row(cell_id, host, receipts, checkpoints, protocol))
    aob_rows = [row for row in rows if row["lane"] == "aob"]
    mapped_rows = [row for row in aob_rows if row["host"] is not None]
    structural_rows = [row for row in aob_rows if row["host"] is None]
    generator_rows = [row for row in rows if row["lane"] == "generator"]
    checks = {
        "structural_rows_complete": len(structural_rows) == 12 and all(row["host_status"] == "mount_absent_by_contract" for row in structural_rows),
        "aob_identity_all": all(row.get("identity_with_frozen_screen") is True for row in mapped_rows),
        "aob_terminal_contract_all": all(row.get("identity_with_frozen_screen") is not None for row in mapped_rows) and len(mapped_rows) == 12,
        "aob_host_reachable_all": all(row["host_status"] == "reachable" for row in mapped_rows),
        "aob_mount_scope_reachable_when_relations_exist": all(row["reachable"] for row in mapped_rows if row["relation_count"] > 0),
        "generator_discovery_all": all(row["phase1_discovery_positive"] for row in generator_rows),
        "generator_host_reachable_all": all(row["host_status"] == "reachable" for row in generator_rows),
        "generator_mount_scope_reachable_all": all(row["reachable"] for row in generator_rows),
    }
    body = {
        "schema_version": "arac-upgrade-u1-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "reachability_seed": reachability_seed,
        "reachability_table": rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "s1_screen_authorized": all(checks.values()),
        "performance_comparison_authorized": False,
        "u1_output_rule": protocol["u1_rule"],
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def _jobs(protocol: Mapping[str, Any], output_root: Path, manifest_sha256: str) -> tuple[list[AobArmJob], list[GeneratorArmJob]]:
    reachability_seed = int(protocol["reachability_seed"])
    checkpoint_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["checkpoint_root"])).resolve()
    current_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["current_e2e_receipt_root"])).resolve()
    frozen_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["frozen_screen_receipt_root"])).resolve()
    aob_jobs = [
        AobArmJob(
            case_id=case_id,
            run_seed=reachability_seed,
            action_name=action,
            checkpoint_root=checkpoint_root,
            current_receipt_root=current_root,
            output_root=output_root,
            frozen_receipt_root=frozen_root,
            manifest_sha256=manifest_sha256,
        )
        for case_id, action in sorted(protocol["aob_lane"]["mapped_hosts"].items())
    ]
    generator_jobs = [
        GeneratorArmJob(cell_id=cell, run_seed=reachability_seed, host=host, output_root=output_root, manifest_sha256=manifest_sha256)
        for cell in protocol["generator_lane"]["cells"]
        for host in protocol["generator_lane"]["hosts"]
    ]
    return aob_jobs, generator_jobs


def run_reachability(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False, workers: int | None = None, stage: str = "all") -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = _manifest(resolved, protocol)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"U1 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("U1 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    aob_jobs, generator_jobs = _jobs(protocol, output_root, str(manifest["manifest_sha256"]))
    if stage in ("all", "phase1"):
        probe_jobs = [
            GeneratorArmJob(cell_id=cell, run_seed=int(protocol["reachability_seed"]), host="ctp", output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
            for cell in protocol["generator_lane"]["cells"]
        ]
        for job in probe_jobs:
            write_generator_checkpoint(job)
    if stage == "phase1":
        return {"stage": "phase1", "checkpoints": len(probe_jobs)}
    jobs: list[AobArmJob | GeneratorArmJob] = [*aob_jobs, *generator_jobs]
    pending = [job for job in jobs if not job.receipt_path.is_file()]
    workers_value = int(protocol["max_workers"] if workers is None else workers)
    completed = len(jobs) - len(pending)
    _write_json(output_root / "progress.json", {"total": len(jobs), "completed": completed, "failed": 0, "pending": len(pending), "updated_at_utc": datetime.now(UTC).isoformat()})
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers_value) as executor:
            futures = {}
            for job in pending:
                runner = _run_aob_arm if isinstance(job, AobArmJob) else _run_generator_arm
                futures[executor.submit(runner, job)] = job
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                except BaseException as exc:
                    failures.append({"key": futures[future].key, "error": f"{type(exc).__name__}: {exc}"})
                _write_json(output_root / "progress.json", {"total": len(jobs), "completed": completed, "failed": len(failures), "pending": len(jobs) - completed - len(failures), "updated_at_utc": datetime.now(UTC).isoformat()})
    if failures:
        _write_json(output_root / "failure_summary.json", {"failures": failures})
        raise RuntimeError(f"U1 reachability campaign has {len(failures)} failed arms")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "phase1", "summarize"), default="all")
    args = parser.parse_args(argv)
    if args.stage == "summarize":
        protocol = load_protocol(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_reachability(args.protocol, resume=args.resume, workers=args.workers, stage=args.stage)
    if args.stage == "phase1":
        print(json.dumps({"stage": "phase1", "checkpoints": summary["checkpoints"]}, indent=2, sort_keys=True))
        return 0
    print(json.dumps({"stage": "u1", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "s1_screen_authorized": summary["s1_screen_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["DEFAULT_PROTOCOL", "load_protocol", "run_reachability", "summarize", "write_generator_checkpoint"]


if __name__ == "__main__":
    raise SystemExit(main())
