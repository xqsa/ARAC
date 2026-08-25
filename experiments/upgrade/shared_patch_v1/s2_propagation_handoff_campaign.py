"""S2 propagation-handoff screen for the shared_patch_v1 upgrade candidate.

Runs only the a2 arms (S1 order + slot handoff).  The paired a1 arms come
from the frozen S1 screen receipts; a0/frozen receipts remain the rollback
anchor.  Gate rules mirror the S1 screen with two additions required by
docs/arac-oc-stepwise-upgrade-plan-v2.1.md stage S2:

- the handoff trace must actually occur in each host lane;
- ``no_acceptance_event`` and ``host_unreachable`` are counted separately
  and never conflated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

from threadpoolctl import threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import current_recovered_four_arm as fixed
from experiments.upgrade.shared_patch_v1.conflicting_generator import build_problem
from experiments.upgrade.shared_patch_v1.host_instrumentation import SweepRecorder
from experiments.upgrade.shared_patch_v1.s1_leverage_sweep import MILESTONE_FES, MilestoneSampler, _lane_judgment, _load_json, _sha256, _tree_sha256, _write_json
from experiments.upgrade.shared_patch_v1.s2_propagation_handoff import S2Shim
from experiments.upgrade.shared_patch_v1.u1_host_reachability import _load_generator_checkpoint, write_generator_checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("s2_propagation_handoff_protocol_v1.json")
S1_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/upgrade_s1_leverage_sweep_v1"
ARM_SCHEMA = "arac-upgrade-s2-arm-receipt-v1"


@dataclass(frozen=True)
class S2AobJob:
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
        return self.output_root / "arms" / "aob" / self.case_id / f"seed_{self.run_seed}" / f"{self.action_name}_a2.json"

    @property
    def s1_receipt_path(self) -> Path:
        return S1_OUTPUT_ROOT / "arms" / "aob" / self.case_id / f"seed_{self.run_seed}" / f"{self.action_name}_a1.json"

    @property
    def frozen_receipt_path(self) -> Path:
        return self.frozen_receipt_root / self.case_id / f"seed_{self.run_seed}" / f"{self.action_name}.json"

    @property
    def key(self) -> str:
        return f"aob:{self.case_id}:seed-{self.run_seed}:{self.action_name}:a2"


@dataclass(frozen=True)
class S2GeneratorJob:
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
        return self.output_root / "arms" / "generator" / self.cell_id / f"seed_{self.run_seed}" / f"{self.host}_a2.json"

    @property
    def s1_receipt_path(self) -> Path:
        return S1_OUTPUT_ROOT / "arms" / "generator" / self.cell_id / f"seed_{self.run_seed}" / f"{self.host}_a1.json"

    @property
    def key(self) -> str:
        return f"generator:{self.cell_id}:seed-{self.run_seed}:{self.host}:a2"


class _CheckpointView:
    def __init__(self, cell_id: str, run_seed: int, checkpoint_path: Path) -> None:
        self.cell_id = cell_id
        self.run_seed = run_seed
        self.checkpoint_path = checkpoint_path

    @property
    def key(self) -> str:
        return f"generator:{self.cell_id}:seed-{self.run_seed}:checkpoint"


def _write_failure(output_root: Path, job_key: str, exc: BaseException) -> None:
    _write_json(
        output_root / "failures" / f"{job_key.replace(':', '_')}.json",
        {
            "schema_version": "arac-upgrade-s2-failure-v1",
            "key": job_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _execute_s2_arm(job_key: str, host: str, checkpoint, problem, *, run_seed: int, manifest_sha256: str, frozen_receipt: Path | None, output_root: Path, receipt_path: Path, ground_truth_hash: str | None) -> dict[str, Any]:
    started = datetime.now(UTC)
    registry = RecoveredActionRegistry()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=registry.allow_out_of_bounds,
    )
    sampler = MilestoneSampler(ledger)
    recorder = SweepRecorder()
    recorder.install()
    shim = S2Shim(host)
    shim.install()
    try:
        result = execute_phase2_action(host, checkpoint, problem, ledger, action_seed=run_seed, registry=registry)
    finally:
        shim.uninstall()
        recorder.uninstall()
    if (
        result.consumed_fes != 2_820_000
        or result.terminal_fes != 3_000_000
        or ledger.count != 3_000_000
        or result.final_error != ledger.best_error
        or not math.isfinite(result.final_error)
    ):
        raise RuntimeError(f"{job_key} terminal contract failed")
    identity = None
    if frozen_receipt is not None:
        frozen = _load_json(frozen_receipt)
        identity = {
            field: {"rerun": value, "frozen": frozen[field], "match": value == frozen[field]}
            for field, value in (
                ("final_error", result.final_error),
                ("route", result.route),
                ("action_result_hash", result.result_hash),
            )
        }
    body: dict[str, Any] = {
        "schema_version": ARM_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "arm": "a2",
        "host": host,
        "run_seed": run_seed,
        "phase1_protocol": checkpoint.protocol,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "ground_truth_hash": ground_truth_hash,
        "initial_error": checkpoint.incumbent_error,
        "final_error": result.final_error,
        "route": result.route,
        "action_result_hash": result.result_hash,
        "identity": identity,
        "relation_count": len(checkpoint.relations),
        "instrumentation": recorder.summary(
            tuple(sum(1 for relation in checkpoint.relations if relation.left_block == index or relation.right_block == index) for index in range(len(checkpoint.blocks)))
        ),
        "milestones": sampler.payload(),
        "s2": shim.payload(),
        "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
    }
    receipt = {**body, "receipt_hash": canonical_sha256(body)}
    _write_json(receipt_path, receipt)
    return receipt


def _run_aob_job(job: S2AobJob) -> dict[str, Any]:
    try:
        with threadpool_limits(limits=1):
            checkpoint = fixed._load_verified_checkpoint(job.context)
            problem = AobBenchmark().load(job.case_id)
            return _execute_s2_arm(
                job.key,
                job.action_name,
                checkpoint,
                problem,
                run_seed=job.run_seed,
                manifest_sha256=job.manifest_sha256,
                frozen_receipt=job.frozen_receipt_path if checkpoint.overlap_relation_count == 0 else None,
                output_root=job.output_root,
                receipt_path=job.receipt_path,
                ground_truth_hash=None,
            )
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


def _run_generator_job(job: S2GeneratorJob) -> dict[str, Any]:
    try:
        with threadpool_limits(limits=1):
            view = _CheckpointView(job.cell_id, job.run_seed, job.checkpoint_path)
            if not job.checkpoint_path.is_file():
                s1_checkpoint = S1_OUTPUT_ROOT / "checkpoints" / job.cell_id / f"seed_{job.run_seed}" / "checkpoint.json"
                if s1_checkpoint.is_file():
                    payload = _load_json(s1_checkpoint)
                    _write_json(job.checkpoint_path, payload)
                else:
                    write_generator_checkpoint(view)
            checkpoint, recorded_truth = _load_generator_checkpoint(view)
            problem, truth = build_problem(job.cell_id, job.run_seed)
            if truth.ground_truth_hash != recorded_truth:
                raise ValueError(f"{job.key} rebuilt generator ground truth drifted")
            return _execute_s2_arm(
                job.key,
                job.host,
                checkpoint,
                problem,
                run_seed=job.run_seed,
                manifest_sha256=job.manifest_sha256,
                frozen_receipt=None,
                output_root=job.output_root,
                receipt_path=job.receipt_path,
                ground_truth_hash=truth.ground_truth_hash,
            )
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-s2-propagation-handoff-protocol-v1",
        "candidate_id": "shared_patch_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "screen_seeds": [117, 123, 129, 135, 141],
        "arms": ["a2"],
        "milestone_fes": list(MILESTONE_FES),
        "s1_output_root": "artifacts/upgrade_s1_leverage_sweep_v1",
        "s1_summary": "artifacts/upgrade_s1_leverage_sweep_v1/summary.json",
        "output_root": "artifacts/upgrade_s2_propagation_handoff_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"S2 protocol drifted: {key}")
    s1_summary = _load_json(REPOSITORY_ROOT / str(protocol["s1_summary"]))
    if not bool(s1_summary.get("gate_passed")):
        raise ValueError("S2 requires a passed S1 gate first")
    s1_protocol = _load_json(REPOSITORY_ROOT / "experiments/upgrade/shared_patch_v1/s1_leverage_sweep_protocol_v1.json")
    if protocol["eps_reference"] != s1_protocol["eps_reference"] or protocol["generator_lane"] != s1_protocol["generator_lane"]:
        raise ValueError("S2 must reuse the frozen S1 eps table and generator lane")
    return protocol


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": "arac-upgrade-s2-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(Path(protocol_path).resolve()),
        "screen_seeds": list(protocol["screen_seeds"]),
        "source_sha256": {source: _sha256(REPOSITORY_ROOT / source) for source in sorted(protocol["sources"])},
        "pinned_trees": {
            tree: dict(zip(("file_count", "sha256"), _tree_sha256(REPOSITORY_ROOT / tree)))
            for tree in sorted(protocol["pinned_trees"])
        },
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    seeds = [int(seed) for seed in protocol["screen_seeds"]]
    eps_aob = {key: float(value) for key, value in protocol["eps_reference"]["aob"].items()}
    eps_generator = {key: float(value) for key, value in protocol["eps_reference"]["generator"].items()}
    aob_case_rows = []
    lane_pairs: dict[str, list[dict[str, Any]]] = {"ctp": [], "gcb": []}
    handoff_totals: dict[str, dict[str, int]] = {"ctp": {"handoff_selections": 0, "no_acceptance_events": 0}, "gcb": {"handoff_selections": 0, "no_acceptance_events": 0}}
    for case_id, action in sorted(protocol["aob_lane"]["mapped_hosts"].items()):
        case_pairs = []
        for seed in seeds:
            a1 = _load_json(S1_OUTPUT_ROOT / "arms" / "aob" / case_id / f"seed_{seed}" / f"{action}_a1.json")
            a2 = _load_json(output_root / "arms" / "aob" / case_id / f"seed_{seed}" / f"{action}_a2.json")
            handoff = a2["s2"]["handoff"]
            handoff_totals[action]["handoff_selections"] += int(handoff["handoff_selection_count"])
            handoff_totals[action]["no_acceptance_events"] += int(handoff["no_acceptance_event_count"])
            case_pairs.append(
                {
                    "key": case_id,
                    "a0_final_error": a1["final_error"],
                    "a1_final_error": a2["final_error"],
                    "a0_milestones": a1["milestones"],
                    "a1_milestones": a2["milestones"],
                    "route_equal": a1["route"] == a2["route"],
                }
            )
        lane_pairs[action].extend(case_pairs)
        zero_relation = _load_json(output_root / "arms" / "aob" / case_id / f"seed_{seeds[0]}" / f"{action}_a2.json")["relation_count"] == 0
        aob_case_rows.append(
            {
                "case_id": case_id,
                "host": action,
                "zero_relation": zero_relation,
                "ov0_exact": all(
                    _load_json(output_root / "arms" / "aob" / case_id / f"seed_{seed}" / f"{action}_a2.json")["identity"] is not None
                    and all(field["match"] for field in _load_json(output_root / "arms" / "aob" / case_id / f"seed_{seed}" / f"{action}_a2.json")["identity"].values())
                    for seed in seeds
                )
                if zero_relation
                else None,
                "judgment": _lane_judgment(case_pairs, eps_by_key=eps_aob),
            }
        )
    generator_rows = []
    generator_pairs: dict[str, list[dict[str, Any]]] = {}
    for cell_id in protocol["generator_lane"]["cells"]:
        cell_pairs = []
        for seed in seeds:
            for host in protocol["generator_lane"]["hosts"]:
                a1 = _load_json(S1_OUTPUT_ROOT / "arms" / "generator" / cell_id / f"seed_{seed}" / f"{host}_a1.json")
                a2 = _load_json(output_root / "arms" / "generator" / cell_id / f"seed_{seed}" / f"{host}_a2.json")
                cell_pairs.append(
                    {
                        "key": cell_id,
                        "a0_final_error": a1["final_error"],
                        "a1_final_error": a2["final_error"],
                        "a0_milestones": a1["milestones"],
                        "a1_milestones": a2["milestones"],
                        "route_equal": a1["route"] == a2["route"],
                    }
                )
        generator_pairs[cell_id] = cell_pairs
        generator_rows.append({"cell_id": cell_id, "judgment": _lane_judgment(cell_pairs, eps_by_key=eps_generator), "exploratory_only": True})
    lane_checks = {}
    for host in ("ctp", "gcb"):
        judgment = _lane_judgment(lane_pairs[host], eps_by_key=eps_aob)
        lane_checks[f"{host}_final_nontax"] = judgment["final_nontax"]
        lane_checks[f"{host}_anytime_nontax"] = judgment["anytime_nontax"]
    checks = {
        "coverage_complete": (
            len(aob_case_rows) == 12
            and len(generator_rows) == 6
            and all(len(pairs) == 30 for pairs in lane_pairs.values())
            and all(len(pairs) == 10 for pairs in generator_pairs.values())
        ),
        "ov0_exact_no_tax": all(row["ov0_exact"] for row in aob_case_rows if row["zero_relation"]),
        "handoff_trace_occurred_per_host": all(handoff_totals[host]["handoff_selections"] > 0 for host in ("ctp", "gcb")),
        "routes_unchanged_all_pairs": all(pair["route_equal"] for host in ("ctp", "gcb") for pair in lane_pairs[host]),
        **lane_checks,
    }
    body = {
        "schema_version": "arac-upgrade-s2-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "handoff_totals": handoff_totals,
        "aob_case_rows": aob_case_rows,
        "generator_rows": generator_rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "s3_contract_authorized": all(checks.values()),
        "performance_claim_authorized": False,
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def _jobs(protocol: Mapping[str, Any], output_root: Path, manifest_sha256: str) -> list[S2AobJob | S2GeneratorJob]:
    seeds = [int(seed) for seed in protocol["screen_seeds"]]
    checkpoint_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["checkpoint_root"])).resolve()
    current_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["current_e2e_receipt_root"])).resolve()
    frozen_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["frozen_screen_receipt_root"])).resolve()
    aob_jobs = [
        S2AobJob(
            case_id=case_id,
            run_seed=seed,
            action_name=action,
            checkpoint_root=checkpoint_root,
            current_receipt_root=current_root,
            output_root=output_root,
            frozen_receipt_root=frozen_root,
            manifest_sha256=manifest_sha256,
        )
        for seed in seeds
        for case_id, action in sorted(protocol["aob_lane"]["mapped_hosts"].items())
    ]
    generator_jobs = [
        S2GeneratorJob(cell_id=cell, run_seed=seed, host=host, output_root=output_root, manifest_sha256=manifest_sha256)
        for seed in seeds
        for cell in protocol["generator_lane"]["cells"]
        for host in protocol["generator_lane"]["hosts"]
    ]
    return [*aob_jobs, *generator_jobs]


def run_screen(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False, workers: int | None = None) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = _manifest(resolved, protocol)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"S2 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("S2 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = _jobs(protocol, output_root, str(manifest["manifest_sha256"]))
    pending = [job for job in jobs if not job.receipt_path.is_file()]
    workers_value = int(protocol["max_workers"] if workers is None else workers)
    completed = len(jobs) - len(pending)
    _write_json(output_root / "progress.json", {"total": len(jobs), "completed": completed, "failed": 0, "pending": len(pending), "updated_at_utc": datetime.now(UTC).isoformat()})
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers_value) as executor:
            futures = {}
            for job in pending:
                runner = _run_aob_job if isinstance(job, S2AobJob) else _run_generator_job
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
        raise RuntimeError(f"S2 screen has {len(failures)} failed arms")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        protocol = load_protocol(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
    else:
        summary = run_screen(args.protocol, resume=args.resume, workers=args.workers)
    print(json.dumps({"stage": "s2", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "s3_contract_authorized": summary["s3_contract_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["DEFAULT_PROTOCOL", "load_protocol", "run_screen", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
