"""T4 AOB preservation for shared_transaction_v1 (SCST v3.0 stage T4).

Preregistered prediction (frozen in the protocol before any arm runs):

> On AOB the frozen PhaseCheckpoint contract carries no certified shared
> structure, so the A1 transaction mount is structurally silent: the kernel
> consumes zero FE, and every arm reproduces the frozen recovery-screen
> receipt bit-identically (final_error, route, action_result_hash,
> checkpoint_hash).  This is the Gate 53 zero-acceptance result restated as
> a preservation prediction - AOB is never treated as a shared-variable
> efficacy experiment.

Arms: the full 24-case x 5-seed AOB matrix with the transaction mount
installed and ENABLED (S1-S6 -> ctp, R1-R6 -> gcb, E1-E6 -> smp,
A1-A6 -> aor structural negative control).  Pass criteria: every arm
matches its frozen receipt on all four identity fields, every kernel
receipt is silent (consumed 0 FE), and every arm satisfies the exact
terminal-FE contract.  No triggering and no improvement are required or
claimed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.historical_recovery.replay import _checkpoint
from experiments.upgrade.shared_transaction_v1.transaction_kernel import TransactionMount


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("t4_aob_preservation_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-shared-transaction-t4-arm-receipt-v1"
FAILURE_SCHEMA = "arac-upgrade-shared-transaction-t4-failure-v1"
AOB_CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/historical_recovery_fixed_expert_v1/checkpoints"
FROZEN_SCREEN_ROOT = REPOSITORY_ROOT / "artifacts/recovery_first_screen_smp_topology_v3/arms"
SCREEN_SEEDS = (117, 123, 129, 135, 141)
HOST_MAPPING = {
    **{f"S{i}": "ctp" for i in range(1, 7)},
    **{f"R{i}": "gcb" for i in range(1, 7)},
    **{f"E{i}": "smp" for i in range(1, 7)},
    **{f"A{i}": "aor" for i in range(1, 7)},
}
PHASE1_FES = 180_000
TOTAL_BUDGET_FES = 3_000_000
PHASE2_FES = TOTAL_BUDGET_FES - PHASE1_FES
MAX_WORKERS = 8

_HOST_BOUNDARIES = {
    "ctp": ("run_persistent_blocks", "run_sequential_blocks"),
    "smp": ("run_stateful_block_visits", "run_stalled_block_rescue"),
    "gcb": ("run_cold_start_block_sweeps", "run_full_space"),
}


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


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-shared-transaction-t4-protocol-v1",
        "candidate_id": "shared_transaction_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "screen_seeds": list(SCREEN_SEEDS),
        "host_mapping": HOST_MAPPING,
        "phase1_fes": PHASE1_FES,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "output_root": "artifacts/upgrade_shared_transaction_v1_t4_v1",
        "prediction": (
            "AOB carries no certified shared structure, so the enabled mount is silent: "
            "zero kernel FE and bit-identical frozen receipts on every arm."
        ),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"T4 protocol drifted: {key}")
    return protocol


@dataclass(frozen=True)
class T4ArmJob:
    case_id: str
    run_seed: int
    host: str
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"{self.host}.json"

    @property
    def key(self) -> str:
        return f"t4:{self.case_id}:seed-{self.run_seed}:{self.host}"


def _run_arm(job: T4ArmJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            wrapper = _load_json(AOB_CHECKPOINT_ROOT / job.case_id / f"seed_{job.run_seed}" / "checkpoint.json")
            checkpoint = _checkpoint(wrapper["checkpoint"])
            if checkpoint.checkpoint_hash != wrapper.get("checkpoint_hash"):
                raise ValueError(f"{job.key} AOB checkpoint hash drifted")
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
            context = ActionContext(
                action_name=job.host,
                checkpoint=checkpoint,
                problem=problem,
                ledger=ledger,
                action_seed=job.run_seed,
            )
            mount = TransactionMount([], enabled=True)
            if job.host in _HOST_BOUNDARIES:
                source_phase, boundary_phase = _HOST_BOUNDARIES[job.host]
                mount.configure_boundary(source_phase=source_phase, boundary_phase=boundary_phase)
            mount.install(ledger, context)
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
                mount.uninstall()
            if (
                result.consumed_fes != PHASE2_FES
                or result.terminal_fes != TOTAL_BUDGET_FES
                or ledger.count != TOTAL_BUDGET_FES
                or result.final_error != ledger.best_error
                or not np.isfinite(result.final_error)
            ):
                raise RuntimeError(f"{job.key} terminal contract failed")
            frozen = _load_json(FROZEN_SCREEN_ROOT / job.case_id / f"seed_{job.run_seed}" / f"{job.host}.json")
            identity = {
                field: {"rerun": value, "frozen": frozen[field], "match": value == frozen[field]}
                for field, value in (
                    ("final_error", result.final_error),
                    ("route", result.route),
                    ("action_result_hash", result.result_hash),
                    ("checkpoint_hash", checkpoint.checkpoint_hash),
                )
            }
            kernel_silent = all(
                receipt["consumed_fes"] == 0 for receipt in mount.kernel_receipts
            )
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "host": job.host,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "identity": identity,
                "identity_all_match": all(row["match"] for row in identity.values()),
                "kernel_receipts": list(mount.kernel_receipts),
                "kernel_consumed_fes": sum(
                    int(receipt["consumed_fes"]) for receipt in mount.kernel_receipts
                ),
                "kernel_silent": bool(kernel_silent),
                "runtime": runtime,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            body["receipt_hash"] = canonical_sha256(body)
            _write_json(job.receipt_path, body)
            return body
    except BaseException as exc:
        _write_json(
            job.output_root / "failures" / f"{job.key.replace(':', '_')}.json",
            {
                "schema_version": FAILURE_SCHEMA,
                "key": job.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    case_rows = []
    for case_id, host in protocol["host_mapping"].items():
        seed_rows = []
        for seed in protocol["screen_seeds"]:
            receipt = _load_json(output_root / "arms" / case_id / f"seed_{seed}" / f"{host}.json")
            if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
                raise ValueError(f"T4 receipt hash drifted: {case_id}/{seed}")
            seed_rows.append(
                {
                    "seed": seed,
                    "identity_all_match": receipt["identity_all_match"],
                    "kernel_consumed_fes": receipt["kernel_consumed_fes"],
                    "kernel_silent": receipt["kernel_silent"],
                    "arm_passed": bool(receipt["identity_all_match"] and receipt["kernel_silent"]),
                }
            )
        case_rows.append(
            {
                "case_id": case_id,
                "host": host,
                "all_seeds_passed": all(row["arm_passed"] for row in seed_rows),
                "seed_rows": seed_rows,
            }
        )
    checks = {
        "coverage_complete": len(case_rows) == 24 and all(len(row["seed_rows"]) == len(SCREEN_SEEDS) for row in case_rows),
        "all_arms_bit_identical": all(row["all_seeds_passed"] for row in case_rows),
        "all_arms_kernel_silent": all(
            row["arm_passed"] for row in case_rows
        ),
    }
    body = {
        "schema_version": "arac-upgrade-shared-transaction-t4-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "prediction": protocol["prediction"],
        "case_rows": case_rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "production_parity_authorized": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("T4 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-shared-transaction-t4-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"T4 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("T4 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = [
        T4ArmJob(
            case_id=case_id,
            run_seed=int(seed),
            host=host,
            output_root=output_root,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        for case_id, host in protocol["host_mapping"].items()
        for seed in protocol["screen_seeds"]
    ]
    pending = [job for job in jobs if not job.receipt_path.is_file()]
    failures = 0
    if pending:
        with ProcessPoolExecutor(max_workers=protocol.get("max_workers", MAX_WORKERS)) as pool:
            futures = {pool.submit(_run_arm, job): job.key for job in pending}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    receipt = future.result()
                    print(f"[t4] done {key} identity={receipt['identity_all_match']}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[t4] FAILED {key}", flush=True)
    if failures:
        raise RuntimeError(f"T4 had {failures} failed arms; inspect {output_root / 'failures'}")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        protocol = load_protocol(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, resume=args.resume)
    print(json.dumps({"stage": "t4", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["load_protocol", "run_stage", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
