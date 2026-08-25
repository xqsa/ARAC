"""T0 fresh matched-tail causal revalidation for hyperedge_ctp_v1.

Line A of the v5.0 plan: revalidate the known CTP long-tail causal signal on
fresh seeds before any new mechanism work.  For S3/S6 x five fresh seeds,
each pair shares one freshly generated frozen Phase-I checkpoint and one
action seed; the ``tail_20pct`` arm is the frozen CTP (whose positive-
relation 20% MMES tail reserve IS the current baseline) and the
``no_reserved_tail`` arm is a candidate-namespace copy identical except the
tail reserve is removed.  The frozen sources are never edited.
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

from arac.actions._execution import (
    BLOCK_POPULATION_SIZE,
    run_full_space,
    run_persistent_blocks,
    run_sequential_blocks,
    terminal_result,
)
from arac.actions.ctp import CtpExecutor, _relation_cover
from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.evidence.phase1 import run_phase1
from arac.runtime.contracts import ActionContext, ActionResult, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.historical_recovery.replay import _checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("t0_tail_causal_protocol_v1.json")
CHECKPOINT_SCHEMA = "arac-upgrade-hyperedge-t0-checkpoint-v1"
ARM_SCHEMA = "arac-upgrade-hyperedge-t0-arm-receipt-v1"
FAILURE_SCHEMA = "arac-upgrade-hyperedge-t0-failure-v1"
_COVERAGE_FRACTION = 0.20


class NoReservedTailCtpExecutor(CtpExecutor):
    """Frozen CTP with only the positive-relation tail reserve removed."""

    name = "ctp"

    def execute(self, context: ActionContext) -> ActionResult:
        if not isinstance(context, ActionContext) or context.action_name != self.name:
            raise TypeError("CTP requires a CTP ActionContext")
        available = context.ledger.remaining
        sweep_fes = len(context.checkpoint.blocks) * BLOCK_POPULATION_SIZE
        positive_relations = context.checkpoint.overlap_relation_count > 0
        tail_reserve = 0
        coverage_cap = max(0, available - tail_reserve)
        coverage_budget = min(
            coverage_cap,
            max(sweep_fes, int(coverage_cap * _COVERAGE_FRACTION)),
        )
        coverage_fes = run_persistent_blocks(
            context,
            requested_fes=coverage_budget,
        )
        polish_blocks = context.checkpoint.blocks if not positive_relations else _relation_cover(context)
        polish_budget = context.ledger.remaining
        polish_fes = run_sequential_blocks(
            context,
            requested_fes=polish_budget,
            blocks=polish_blocks,
        )
        tail_fes = 0
        if positive_relations and context.ledger.remaining:
            tail_fes = run_full_space(
                context,
                algorithm="mmes",
                namespace="ctp-relation-mmes-tail",
            ).consumed_fes
        elif context.ledger.remaining:
            tail_fes = run_full_space(
                context,
                algorithm="mmes",
                namespace="ctp-terminal",
            ).consumed_fes
        route = f"coverage_{coverage_fes}_then_"
        route += "relation_cover_polish_" if positive_relations else "sequential_block_polish_"
        route += f"{polish_fes}"
        if positive_relations:
            route += f"_then_mmes_tail_{tail_fes}"
        return terminal_result(
            context,
            route=route,
        )


class T0VariantRegistry(RecoveredActionRegistry):
    """Recovered registry with the ctp entry swapped for the T0 variant."""

    def __init__(self) -> None:
        super().__init__()
        self._executors["ctp"] = NoReservedTailCtpExecutor()


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


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-hyperedge-t0-protocol-v1",
        "candidate_id": "hyperedge_ctp_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "cases": ["S3", "S6"],
        "seeds": [20270111, 20270112, 20270113, 20270114, 20270115],
        "arms": ["tail_20pct", "no_reserved_tail"],
        "total_budget_fes": 3_000_000,
        "phase1_fes": 180_000,
        "phase2_fes": 2_820_000,
        "max_workers": 16,
        "output_root": "artifacts/upgrade_hyperedge_ctp_v1_t0_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"T0 protocol drifted: {key}")
    return protocol


@dataclass(frozen=True)
class T0CheckpointJob:
    case_id: str
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.case_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def key(self) -> str:
        return f"checkpoint:{self.case_id}:seed-{self.run_seed}"


@dataclass(frozen=True)
class T0ArmJob:
    case_id: str
    run_seed: int
    arm: str
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.case_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{self.arm}.json"

    @property
    def key(self) -> str:
        return f"arm:{self.case_id}:seed-{self.run_seed}:{self.arm}"


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


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


def _run_checkpoint_job(job: T0CheckpointJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            problem = AobBenchmark().load(job.case_id)
            ledger = EvaluationLedger(problem, total_budget=3_000_000)
            probe = run_phase1(problem, ledger, run_seed=job.run_seed)
            checkpoint = probe.checkpoint
            if checkpoint.phase1_fes != 180_000 or ledger.count != 180_000:
                raise RuntimeError(f"{job.key} Phase-I FE boundary drifted")
            if checkpoint.overlap_relation_count == 0:
                raise RuntimeError(f"{job.key} checkpoint has no positive relations; T0 requires the relation path")
            body = {
                "schema_version": CHECKPOINT_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "phase1_protocol": checkpoint.protocol,
                "phase1_fes": checkpoint.phase1_fes,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "relation_count": len(checkpoint.relations),
                "runtime": runtime,
                "checkpoint": checkpoint.payload(),
            }
            body["receipt_hash"] = canonical_sha256(body)
            _write_json(job.checkpoint_path, body)
            return body
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


def _load_t0_checkpoint(job: T0ArmJob):
    wrapper = _load_json(job.checkpoint_path)
    if canonical_sha256({key: value for key, value in wrapper.items() if key != "receipt_hash"}) != wrapper.get("receipt_hash"):
        raise ValueError(f"{job.key} checkpoint receipt hash drifted")
    if wrapper.get("schema_version") != CHECKPOINT_SCHEMA or wrapper.get("case_id") != job.case_id or wrapper.get("run_seed") != job.run_seed:
        raise ValueError(f"{job.key} checkpoint identity drifted")
    checkpoint = _checkpoint(wrapper["checkpoint"])
    if checkpoint.checkpoint_hash != wrapper.get("checkpoint_hash"):
        raise ValueError(f"{job.key} checkpoint hash drifted")
    return checkpoint


def _run_arm_job(job: T0ArmJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            checkpoint = _load_t0_checkpoint(job)
            problem = AobBenchmark().load(job.case_id)
            registry = RecoveredActionRegistry() if job.arm == "tail_20pct" else T0VariantRegistry()
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=checkpoint.total_budget_fes,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            result = execute_phase2_action(
                "ctp",
                checkpoint,
                problem,
                ledger,
                action_seed=job.run_seed,
                registry=registry,
            )
            if (
                result.consumed_fes != 2_820_000
                or result.terminal_fes != 3_000_000
                or ledger.count != 3_000_000
                or result.final_error != ledger.best_error
                or not math.isfinite(result.final_error)
            ):
                raise RuntimeError(f"{job.key} terminal contract failed")
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "arm": job.arm,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "relation_count": len(checkpoint.relations),
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result": result.payload(),
                "action_result_hash": result.result_hash,
                "strict_best": result.final_error <= checkpoint.incumbent_error,
                "runtime": runtime,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            receipt = {**body, "receipt_hash": canonical_sha256(body)}
            _write_json(job.receipt_path, receipt)
            return receipt
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


def _bootstrap_ci(log_ratios: Sequence[float], *, count: int = 10_000, seed: int = 20260823) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(log_ratios, dtype=float)
    samples = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    rows = []
    for case_id in protocol["cases"]:
        log_ratios = []
        contract_ok = True
        for seed in protocol["seeds"]:
            tail = _load_json(output_root / "arms" / case_id / f"seed_{seed}" / "tail_20pct.json")
            no_tail = _load_json(output_root / "arms" / case_id / f"seed_{seed}" / "no_reserved_tail.json")
            for receipt in (tail, no_tail):
                if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
                    raise ValueError(f"T0 receipt hash drifted: {case_id}/{seed}")
                if receipt["checkpoint_hash"] != tail["checkpoint_hash"] or receipt["relation_count"] == 0:
                    contract_ok = False
            # preregistered orientation: R = tail_20pct / no_reserved_tail
            log_ratios.append(
                math.log(max(tail["final_error"], 1e-300) / max(no_tail["final_error"], 1e-300))
            )
        mean_log = float(np.mean(log_ratios))
        low, high = _bootstrap_ci(log_ratios)
        rows.append(
            {
                "case_id": case_id,
                "pair_count": len(log_ratios),
                "R_tail_over_no_tail": math.exp(mean_log),
                "ci95": [math.exp(low), math.exp(high)],
                "contract_ok": contract_ok,
                "causal_signal_confirmed": contract_ok and math.exp(mean_log) < 1.0 and math.exp(high) < 1.0,
            }
        )
    checks = {
        "coverage_complete": len(rows) == 2 and all(row["pair_count"] == 5 for row in rows),
        "contract_compliance_all": all(row["contract_ok"] for row in rows),
        "both_cases_causal": all(row["causal_signal_confirmed"] for row in rows),
    }
    body = {
        "schema_version": "arac-upgrade-hyperedge-t0-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "case_rows": rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "tail_tuning_branch_closed_on_failure": not checks["both_cases_causal"],
        "line_b_unaffected": True,
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_campaign(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False, workers: int | None = None, stage: str = "all") -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen" or not all(
        verifier_report.get(key) is True for key in ("smp_smoke_green", "e1_preservation_green", "screen_contract_green")
    ):
        raise RuntimeError("T0 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-hyperedge-t0-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "cases": list(protocol["cases"]),
        "seeds": list(protocol["seeds"]),
        "verifier_report": verifier_report,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"T0 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("T0 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    checkpoint_jobs = [
        T0CheckpointJob(case_id=case, run_seed=int(seed), output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for case in protocol["cases"]
        for seed in protocol["seeds"]
    ]
    arm_jobs = [
        T0ArmJob(case_id=case, run_seed=int(seed), arm=arm, output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for case in protocol["cases"]
        for seed in protocol["seeds"]
        for arm in protocol["arms"]
    ]
    if stage == "checkpoints":
        for job in checkpoint_jobs:
            if not job.checkpoint_path.is_file():
                _run_checkpoint_job(job)
        return {"stage": "checkpoints", "count": len(checkpoint_jobs)}
    workers_value = int(protocol["max_workers"] if workers is None else workers)
    pending = [job for job in arm_jobs if not job.receipt_path.is_file()]
    for job in checkpoint_jobs:
        if not job.checkpoint_path.is_file():
            _run_checkpoint_job(job)
    completed = len(arm_jobs) - len(pending)
    _write_json(output_root / "progress.json", {"total": len(arm_jobs), "completed": completed, "failed": 0, "pending": len(pending), "updated_at_utc": datetime.now(UTC).isoformat()})
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers_value) as executor:
            futures = {executor.submit(_run_arm_job, job): job for job in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                except BaseException as exc:
                    failures.append({"key": futures[future].key, "error": f"{type(exc).__name__}: {exc}"})
                _write_json(output_root / "progress.json", {"total": len(arm_jobs), "completed": completed, "failed": len(failures), "pending": len(arm_jobs) - completed - len(failures), "updated_at_utc": datetime.now(UTC).isoformat()})
    if failures:
        _write_json(output_root / "failure_summary.json", {"failures": failures})
        raise RuntimeError(f"T0 campaign has {len(failures)} failed arms")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "checkpoints", "summarize"), default="all")
    args = parser.parse_args(argv)
    if args.stage == "summarize":
        protocol = load_protocol(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_campaign(args.protocol, resume=args.resume, workers=args.workers, stage=args.stage)
        if args.stage == "checkpoints":
            print(json.dumps({"stage": "checkpoints", "count": summary["count"]}, sort_keys=True))
            return 0
    print(json.dumps({"stage": "t0", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["DEFAULT_PROTOCOL", "NoReservedTailCtpExecutor", "T0VariantRegistry", "load_protocol", "run_campaign", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
