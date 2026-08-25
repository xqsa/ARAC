"""Screen campaign for shared_transaction_v4 (joint rescue kernel).

A0 (mount disabled) vs A1 (joint rescue kernel at the T1-qualified SMP
boundary: stateful visits -> rescue) on the T0-certified generator cells,
paired by checkpoint / ledger boundary / action seed.  Upstream evidence
(T0 certification, T1 SMP lane qualification, the v1 T3 root-cause receipt)
is consumed read-only; this candidate's own protocol froze the kernel spec
(3-point quadratic re-centering, 0.25*span probes, <=4 coordinates, <=12 FE
per boundary) and the statistics before any arm ran.

Gate: identical in form to the SCST T3 screen - contract audits green,
at least one cell with geometric-mean ratio < 0.95 and 95% paired bootstrap
CI upper < 1.0, no cell with CI lower > 1.05, reachability (non-empty
kernel receipt with at least one evaluated candidate on every A1 arm).
eps_ref is frozen from the v1 screen's patch-off arms (identical seeds and
action seed, bit-equal finals by determinism).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.runtime.contracts import ActionContext, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import GENERATOR_FREEZE
from experiments.upgrade.shared_patch_v1.conflicting_generator_v3 import build_v3_problem
from experiments.upgrade.shared_transaction_v1.t1_boundary_audit import _load_t0_receipt
from experiments.upgrade.shared_transaction_v1.t3_matched_host_attribution import (
    _paired_bootstrap,
    _terminal_contract,
)
from experiments.upgrade.shared_transaction_v4.joint_rescue_kernel import JointRescueMount


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("screen_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-shared-transaction-v4-arm-receipt-v1"
FAILURE_SCHEMA = "arac-upgrade-shared-transaction-v4-failure-v1"
CELLS = ("chain4-strong", "pairs3-strong")
SEEDS = (20270501, 20270502, 20270503, 20270504, 20270505)
CONFIRMATION_SEEDS = (20270511, 20270512, 20270513, 20270514, 20270515)
VARIANTS = ("a0", "a1")
ACTION_SEED = 314159
PHASE1_FES = 180_000
TOTAL_BUDGET_FES = 3_000_000
PHASE2_FES = TOTAL_BUDGET_FES - PHASE1_FES
MAX_WORKERS = 8
V1_T3_SCREEN_ROOT = REPOSITORY_ROOT / "artifacts/upgrade_shared_transaction_v1_t3_screen_v1"


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


@dataclass(frozen=True)
class ScreenJob:
    cell_id: str
    run_seed: int
    variant: str
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.cell_id / f"seed_{self.run_seed}" / f"smp_{self.variant}.json"

    @property
    def key(self) -> str:
        return f"v2:{self.cell_id}:seed-{self.run_seed}:{self.variant}"


def _run_arm(job: ScreenJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            receipt, checkpoint = _load_t0_receipt(job.cell_id, job.run_seed)
            problem, truth = build_v3_problem(
                job.cell_id,
                job.run_seed,
                conditioning=GENERATOR_FREEZE["conditioning"],
                shared_width=int(GENERATOR_FREEZE["shared_width"]),
                linkage_lambda=float(GENERATOR_FREEZE["linkage_lambda"]),
            )
            if truth.ground_truth_hash != receipt["sidecar"]["ground_truth_hash"]:
                raise ValueError(f"{job.key} generator truth hash drifted")
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
                action_name="smp",
                checkpoint=checkpoint,
                problem=problem,
                ledger=ledger,
                action_seed=ACTION_SEED,
            )
            links = [
                (int(item["variable"]), int(item["region_a"]) - 1, int(item["region_b"]) - 1)
                for item in receipt["sidecar"]["pairwise_certificates"]
            ]
            from experiments.upgrade.shared_transaction_v4.joint_rescue_kernel import CertifiedLink

            mount = JointRescueMount(
                [CertifiedLink(variable=variable, owner_blocks=(left, right)) for variable, left, right in links],
                enabled=job.variant == "a1",
            )
            mount.configure_boundary(
                source_phase="run_stateful_block_visits",
                boundary_phase="run_stalled_block_rescue",
            )
            mount.install(ledger, context)
            try:
                result = execute_phase2_action(
                    "smp",
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=ACTION_SEED,
                    registry=registry,
                )
            finally:
                mount.uninstall()
            _terminal_contract(job.key, result, ledger)
            receipts = list(mount.joint_receipts)
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "cell_id": job.cell_id,
                "run_seed": job.run_seed,
                "variant": job.variant,
                "action_seed": ACTION_SEED,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "kernel_receipts": receipts,
                "kernel_consumed_fes": sum(int(item["joint_fes"]) for item in receipts),
                "kernel_accepted_count": sum(int(item["accepted_links"]) for item in receipts),
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
    groups = []
    for cell_id in protocol["cells"]:
        pairs = []
        for seed in protocol["seeds"]:
            a0 = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / "smp_a0.json")
            a1 = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / "smp_a1.json")
            for arm in (a0, a1):
                if canonical_sha256({k: v for k, v in arm.items() if k != "receipt_hash"}) != arm.get("receipt_hash"):
                    raise ValueError(f"v2 receipt hash drifted: {cell_id}/{seed}")
            eps_ref = float(protocol["eps_ref"][cell_id])
            e0 = max(float(a0["final_error"]), eps_ref)
            e1 = max(float(a1["final_error"]), eps_ref)
            pairs.append(
                {
                    "seed": seed,
                    "a0_final": a0["final_error"],
                    "a1_final": a1["final_error"],
                    "log_ratio": float(np.log(e1 / e0)),
                    "a1_accepted": a1["kernel_accepted_count"],
                    "a1_consumed": a1["kernel_consumed_fes"],
                }
            )
        log_ratios = np.asarray([pair["log_ratio"] for pair in pairs], dtype=float)
        geo_r = float(np.exp(np.mean(log_ratios)))
        bootstrap = _paired_bootstrap(log_ratios)
        groups.append(
            {
                "cell_id": cell_id,
                "pairs": pairs,
                "geometric_mean_ratio": geo_r,
                "ci95_exp_lower": float(np.exp(bootstrap["ci95_lower"])),
                "ci95_exp_upper": float(np.exp(bootstrap["ci95_upper"])),
                "win_tie_loss": [
                    int(sum(1 for pair in pairs if pair["log_ratio"] < 0.0)),
                    int(sum(1 for pair in pairs if pair["log_ratio"] == 0.0)),
                    int(sum(1 for pair in pairs if pair["log_ratio"] > 0.0)),
                ],
                "total_accepted": sum(pair["a1_accepted"] for pair in pairs),
                "total_consumed_fes": sum(pair["a1_consumed"] for pair in pairs),
                "strict_win": bool(
                    geo_r < float(protocol["strict_win_ratio"])
                    and bootstrap["ci95_upper"] < float(protocol["strict_win_ci_upper"])
                ),
                "breaks_non_inferiority": bool(
                    bootstrap["ci95_lower"] > float(protocol["non_inferior_ci_lower"])
                ),
            }
        )
    reachability_ok = True
    for cell_id in protocol["cells"]:
        for seed in protocol["seeds"]:
            a1 = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / "smp_a1.json")
            joint_fes = sum(
                int(receipt["joint_fes"]) for receipt in a1["kernel_receipts"]
            )
            scopes = sum(
                len(receipt["link_scopes"]) for receipt in a1["kernel_receipts"]
            )
            if not a1["kernel_receipts"] or joint_fes < 1 or scopes < 1:
                reachability_ok = False
    checks = {
        "at_least_one_strict_win": any(group["strict_win"] for group in groups),
        "no_cell_breaks_non_inferiority": not any(group["breaks_non_inferiority"] for group in groups),
        "reachability_all_arms": reachability_ok,
    }
    body = {
        "schema_version": "arac-upgrade-shared-transaction-v4-summary-v1",
        "candidate_id": "shared_transaction_v4",
        "freeze_anchor": protocol["freeze_anchor"],
        "stage": protocol.get("stage", "screen"),
        "groups": groups,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def _frozen_eps_ref() -> dict[str, float]:
    """eps_ref from the v1 screen patch-off arms (same seeds/action seed)."""

    eps: dict[str, float] = {}
    for cell in CELLS:
        finals = []
        for seed in SEEDS:
            arm = _load_json(V1_T3_SCREEN_ROOT / "arms" / cell / f"seed_{seed}" / "smp_a0.json")
            finals.append(float(arm["final_error"]))
        positive = [value for value in finals if value > 0.0]
        eps[cell] = (min(positive) / 10.0) if positive else 1e-12
    return eps


def run_stage(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    stage: str = "screen",
    resume: bool = False,
) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = dict(_load_json(resolved))
    protocol["stage"] = stage
    if stage == "confirmation":
        if not protocol.get("confirmation_seeds"):
            raise ValueError("confirmation seeds missing")
        protocol["seeds"] = list(protocol["confirmation_seeds"])
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("v2 screen refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (
        REPOSITORY_ROOT / str(protocol["output_root"] if stage == "screen" else protocol["confirmation_output_root"])
    ).resolve()
    manifest = {
        "schema_version": "arac-upgrade-shared-transaction-v4-manifest-v1",
        "candidate_id": "shared_transaction_v4",
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
        "stage": stage,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"v2 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("v2 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = [
        ScreenJob(
            cell_id=cell,
            run_seed=int(seed),
            variant=variant,
            output_root=output_root,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        for cell in protocol["cells"]
        for seed in protocol["seeds"]
        for variant in VARIANTS
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
                    print(f"[v2:{stage}] done {key} final={receipt['final_error']}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[v2:{stage}] FAILED {key}", flush=True)
    if failures:
        raise RuntimeError(f"v2 had {failures} failed arms; inspect {output_root / 'failures'}")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        protocol = dict(_load_json(args.protocol))
        protocol["stage"] = args.stage
        if args.stage == "confirmation":
            protocol["seeds"] = list(protocol["confirmation_seeds"])
        root = (
            REPOSITORY_ROOT / str(protocol["output_root"] if args.stage == "screen" else protocol["confirmation_output_root"])
        ).resolve()
        summary = summarize(protocol, root)
        _write_json(root / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, stage=args.stage, resume=args.resume)
    print(json.dumps({"stage": f"v2-{summary['stage']}", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["run_stage", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
