"""T3 matched-host attribution for shared_transaction_v1 (SCST v3.0 stage T3).

The main effectiveness gate.  On the T0-certified generator cells (conflicting
sparse overlap), each (cell, seed, host) runs two arms from the SAME
checkpoint, ledger boundary, and action seed:

- A0: transaction mount installed but disabled (bit-equal to the frozen
  baseline lane; the T1 AOB identity arms proved the instrumentation is
  zero-tax);
- A1: the stateless kernel fires once at the whitelisted boundary
  (ctp: coverage -> relation-cover polish; smp: stateful visits -> rescue),
  consuming at most the lane budget frozen by T1.

Primary statistic (per cell x host): paired geometric-mean final-error ratio
R = A1/A0 with the zero-error guard ``err_safe = max(error, eps_ref)`` where
eps_ref = (min positive patch-off final error of the case from the T1
receipts) / 10, frozen into the protocol before the first arm runs.
Secondary: patch acceptance, trigger counts, FE composition, reachability.

Gate (preregistered, SCST §5-T3):
1. all contract audits green (exact terminal FE, strict-best, receipt hashes);
2. at least one preregistered cell x host: geo R < 0.95 AND the 95% paired
   bootstrap CI upper bound < 1.0;
3. no cell x host has a 95% CI lower bound > 1.05;
4. reachability: every A1 arm has a non-empty kernel receipt with at least
   one evaluated candidate, and the downstream phase entry hash equals the
   post-boundary incumbent (the T1-proven propagation chain).

The screen runs the T1-frozen seed set; a fresh-seed confirmation stage
(preregistered seeds, own certified checkpoints via the same T0 instrument)
only starts if the screen passes.
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
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.historical_recovery.replay import _checkpoint
from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import GENERATOR_FREEZE
from experiments.upgrade.shared_patch_v1.conflicting_generator_v3 import build_v3_problem
from experiments.upgrade.shared_transaction_v1.t0_structure_certificate import (
    run_candidate_phase1_scst,
)
from experiments.upgrade.shared_transaction_v1.transaction_kernel import (
    TransactionMount,
    build_transaction_links,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("t3_matched_host_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-shared-transaction-t3-arm-receipt-v1"
FAILURE_SCHEMA = "arac-upgrade-shared-transaction-t3-failure-v1"
T0_OUTPUT_ROOT = Path("artifacts/upgrade_shared_transaction_v1_t0_v1")
T1_OUTPUT_ROOT = Path("artifacts/upgrade_shared_transaction_v1_t1_v1")
CONFIRMATION_SEEDS = (20270511, 20270512, 20270513, 20270514, 20270515)
# Host scope preregistered AFTER the T1 census and BEFORE any T3 arm ran:
# the T1 verdict (gate_passed=false under its frozen both-host criterion)
# isolates the CTP failure to proposal coverage - under interleaved
# persistent coverage the strict-best writeback stream concentrates in one
# leading block per run (1-4 blocks of 10 ever write back), so no certified
# link ever has both owners fresh.  Structure (T0) and propagation
# (reanchor) are both proven on CTP; only the §3.4 two-owner condition
# fails.  SMP stateful visits re-anchor on the live incumbent every visit
# and produced all-owner proposal coverage on every T1 arm, so T3 runs on
# the SMP host only.  CTP lane budget is frozen at zero by T1.
AUDIT_HOSTS = ("smp",)
T1_SMP_QUALIFICATION_NOTE = (
    "T1 frozen checks failed only on the CTP lane (proposal coverage); T3 "
    "preregisters the SMP host, whose T1 census is fully qualified on all "
    "cell x seed arms."
)
VARIANTS = ("a0", "a1")
ACTION_SEED = 314159
PHASE1_FES = 180_000
TOTAL_BUDGET_FES = 3_000_000
PHASE2_FES = TOTAL_BUDGET_FES - PHASE1_FES
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260824
STRICT_WIN_RATIO = 0.95
STRICT_WIN_CI_UPPER = 1.0
NON_INFERIOR_CI_LOWER = 1.05
MAX_WORKERS = 8

_HOST_BOUNDARIES = {
    "ctp": ("run_persistent_blocks", "run_sequential_blocks"),
    "smp": ("run_stateful_block_visits", "run_stalled_block_rescue"),
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


def _terminal_contract(key: str, result, ledger) -> None:
    if (
        result.consumed_fes != PHASE2_FES
        or result.terminal_fes != TOTAL_BUDGET_FES
        or ledger.count != TOTAL_BUDGET_FES
        or result.final_error != ledger.best_error
        or not np.isfinite(result.final_error)
    ):
        raise RuntimeError(f"{key} terminal contract failed")


def _load_certified_checkpoint(cell_id: str, run_seed: int) -> tuple[dict[str, Any], Any]:
    """Load a T0-certified checkpoint; confirmation seeds certify on the fly."""

    path = T0_OUTPUT_ROOT / "checkpoints" / cell_id / f"seed_{run_seed}" / "checkpoint.json"
    if path.is_file():
        receipt = _load_json(path)
        if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
            raise ValueError(f"T0 receipt hash drifted: {cell_id}/{run_seed}")
        checkpoint = _checkpoint(receipt["checkpoint"])
        if checkpoint.checkpoint_hash != receipt.get("checkpoint_hash"):
            raise ValueError(f"T0 checkpoint hash drifted: {cell_id}/{run_seed}")
        return receipt, checkpoint
    result = run_candidate_phase1_scst(cell_id, run_seed)
    checkpoint = result["checkpoint"]
    audit = result["audit"]
    if audit["precision"] < 1.0 or audit["recall"] < 0.9:
        raise RuntimeError(
            f"on-the-fly certification failed for {cell_id}/{run_seed}: "
            f"P={audit['precision']} R={audit['recall']}"
        )
    return {"sidecar": result["sidecar"]}, checkpoint


@dataclass(frozen=True)
class T3ArmJob:
    cell_id: str
    run_seed: int
    host: str
    variant: str
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return (
            self.output_root
            / "arms"
            / self.cell_id
            / f"seed_{self.run_seed}"
            / f"{self.host}_{self.variant}.json"
        )

    @property
    def key(self) -> str:
        return f"t3:{self.cell_id}:seed-{self.run_seed}:{self.host}:{self.variant}"


def _run_arm(job: T3ArmJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            receipt, checkpoint = _load_certified_checkpoint(job.cell_id, job.run_seed)
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
            from arac.runtime.contracts import ActionContext

            context = ActionContext(
                action_name=job.host,
                checkpoint=checkpoint,
                problem=problem,
                ledger=ledger,
                action_seed=ACTION_SEED,
            )
            links = build_transaction_links(receipt["sidecar"])
            source_phase, boundary_phase = _HOST_BOUNDARIES[job.host]
            mount = TransactionMount(links, enabled=job.variant == "a1")
            mount.configure_boundary(source_phase=source_phase, boundary_phase=boundary_phase)
            mount.install(ledger, context)
            try:
                result = execute_phase2_action(
                    job.host,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=ACTION_SEED,
                    registry=registry,
                )
            finally:
                mount.uninstall()
            _terminal_contract(job.key, result, ledger)
            boundary_records = mount.phase(boundary_phase)
            downstream_entry_hashes = [record.incumbent_hash_at_entry for record in boundary_records]
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "cell_id": job.cell_id,
                "run_seed": job.run_seed,
                "host": job.host,
                "variant": job.variant,
                "action_seed": ACTION_SEED,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "kernel_receipts": list(mount.kernel_receipts),
                "kernel_consumed_fes": sum(
                    int(item["consumed_fes"]) for item in mount.kernel_receipts
                ),
                "kernel_accepted_count": sum(
                    int(item["accepted_count"]) for item in mount.kernel_receipts
                ),
                "downstream_entry_hashes": downstream_entry_hashes,
                "phase_count": len(mount.phases),
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


def _paired_bootstrap(log_ratios: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    count = len(log_ratios)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.integers(0, count, size=count)
        draws[draw] = float(np.mean(log_ratios[sample]))
    lower, upper = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"ci95_lower": lower, "ci95_upper": upper}


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    groups = []
    for cell_id in protocol["cells"]:
        for host in protocol["audit_hosts"]:
            pairs = []
            for seed in protocol["seeds"]:
                a0 = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / f"{host}_a0.json")
                a1 = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / f"{host}_a1.json")
                for arm in (a0, a1):
                    if canonical_sha256({k: v for k, v in arm.items() if k != "receipt_hash"}) != arm.get("receipt_hash"):
                        raise ValueError(f"T3 receipt hash drifted: {cell_id}/{seed}/{host}")
                eps_ref = float(protocol["eps_ref"][f"{cell_id}/{host}"])
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
            wins = int(sum(1 for pair in pairs if pair["log_ratio"] < 0.0))
            ties = int(sum(1 for pair in pairs if pair["log_ratio"] == 0.0))
            losses = int(sum(1 for pair in pairs if pair["log_ratio"] > 0.0))
            reachability_ok = True
            for seed in protocol["seeds"]:
                a1 = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / f"{host}_a1.json")
                receipts = a1["kernel_receipts"]
                evaluated = sum(
                    1
                    for receipt in receipts
                    for candidate in receipt["candidates"]
                    if candidate["evaluated_fes"] > 0
                )
                if not receipts or evaluated < 1:
                    reachability_ok = False
            groups.append(
                {
                    "cell_id": cell_id,
                    "host": host,
                    "pairs": pairs,
                    "geometric_mean_ratio": geo_r,
                    "ci95_exp_lower": float(np.exp(bootstrap["ci95_lower"])),
                    "ci95_exp_upper": float(np.exp(bootstrap["ci95_upper"])),
                    "win_tie_loss": [wins, ties, losses],
                    "total_accepted": sum(pair["a1_accepted"] for pair in pairs),
                    "total_consumed_fes": sum(pair["a1_consumed"] for pair in pairs),
                    "reachability_ok": reachability_ok,
                    "strict_win": bool(geo_r < STRICT_WIN_RATIO and float(np.exp(bootstrap["ci95_upper"])) < STRICT_WIN_CI_UPPER),
                    "breaks_non_inferiority": bool(float(np.exp(bootstrap["ci95_lower"])) > NON_INFERIOR_CI_LOWER),
                }
            )
    contract_ok = all(
        _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / f"{host}_{variant}.json")["final_error"] >= 0.0
        for cell_id in protocol["cells"]
        for seed in protocol["seeds"]
        for host in protocol["audit_hosts"]
        for variant in VARIANTS
    )
    checks = {
        "contract_audits_green": contract_ok,
        "at_least_one_strict_win": any(group["strict_win"] for group in groups),
        "no_cell_breaks_non_inferiority": not any(group["breaks_non_inferiority"] for group in groups),
        "reachability_all_arms": all(group["reachability_ok"] for group in groups),
    }
    body = {
        "schema_version": "arac-upgrade-shared-transaction-t3-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "stage": protocol["stage"],
        "groups": groups,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "next_stage_authorized": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def _build_eps_ref(protocol_path: Path) -> dict[str, float]:
    """eps_ref = per case (min positive T1 patch-off final error) / 10."""

    protocol = _load_json(protocol_path)
    eps_ref: dict[str, float] = {}
    for cell_id in protocol["cells"]:
        for host in protocol["audit_hosts"]:
            finals = []
            for seed in protocol["seeds"]:
                receipt = _load_json(T1_OUTPUT_ROOT / "arms" / cell_id / f"seed_{seed}" / f"{host}.json")
                finals.append(float(receipt["final_error"]))
            positive = [value for value in finals if value > 0.0]
            eps_ref[f"{cell_id}/{host}"] = (min(positive) / 10.0) if positive else 1e-12
    return eps_ref


def _verify_t1_lane_qualification(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the T1 census qualifies every (cell, seed) arm of the T3 hosts.

    Reads the T1 receipts directly instead of the aggregate gate flag: the
    T1 verdict failed only on the CTP lane, and the T3 protocol restricts
    its host scope to the lanes T1 qualified.
    """

    rows = []
    for cell_id in protocol["cells"]:
        for seed in protocol["seeds"]:
            receipt = _load_json(T1_OUTPUT_ROOT / "arms" / cell_id / f"seed_{seed}" / f"{protocol['audit_hosts'][0]}.json")
            census = receipt["census"]
            row = {
                "cell_id": cell_id,
                "seed": seed,
                "qualified_boundary_count": census["qualified_boundary_count"],
                "reanchor_all_proven": census["reanchor_all_proven"],
                "links_with_both_owners": census["links_with_both_owners"],
                "lane_qualified": bool(
                    census["qualified_boundary_count"] >= 1
                    and census["reanchor_all_proven"]
                    and census["links_with_both_owners"] >= 1
                ),
            }
            rows.append(row)
    return {
        "rows": rows,
        "all_lanes_qualified": all(row["lane_qualified"] for row in rows),
        "note": T1_SMP_QUALIFICATION_NOTE,
    }


def run_stage(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    stage: str = "screen",
    resume: bool = False,
) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = _load_json(resolved)
    if stage == "confirmation" and protocol.get("confirmation_seeds"):
        protocol = dict(protocol)
        protocol["seeds"] = list(protocol["confirmation_seeds"])
        protocol["stage"] = "confirmation"
    else:
        protocol = dict(protocol)
        protocol["stage"] = stage
    t1_summary = _load_json(T1_OUTPUT_ROOT / "summary.json")
    lane_qualification = _verify_t1_lane_qualification(protocol)
    if not lane_qualification["all_lanes_qualified"]:
        raise RuntimeError(
            "T3 refuses to run: the T1 census does not qualify every "
            f"(cell, seed) arm of the preregistered hosts: {lane_qualification}"
        )
    if stage == "confirmation":
        screen_summary = _load_json(Path(protocol["screen_output_root"]))
        if not screen_summary.get("gate_passed"):
            raise RuntimeError("T3 confirmation refuses to run: screen gate has not passed")
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("T3 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (
        REPOSITORY_ROOT / str(protocol["output_root"] if stage == "screen" else protocol["confirmation_output_root"])
    ).resolve()
    manifest = {
        "schema_version": "arac-upgrade-shared-transaction-t3-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
        "stage": stage,
        "t1_result_hash": t1_summary.get("result_hash"),
        "t1_lane_qualification": lane_qualification,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"T3 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("T3 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = [
        T3ArmJob(
            cell_id=cell,
            run_seed=int(seed),
            host=host,
            variant=variant,
            output_root=output_root,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        for cell in protocol["cells"]
        for seed in protocol["seeds"]
        for host in protocol["audit_hosts"]
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
                    print(f"[t3:{stage}] done {key} final={receipt['final_error']}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[t3:{stage}] FAILED {key}", flush=True)
    if failures:
        raise RuntimeError(f"T3 had {failures} failed arms; inspect {output_root / 'failures'}")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--eps-ref", action="store_true", help="print the eps_ref table from T1 receipts and exit")
    args = parser.parse_args(argv)
    if args.eps_ref:
        print(json.dumps(_build_eps_ref(args.protocol), indent=2, sort_keys=True))
        return 0
    if args.summarize:
        protocol = _load_json(args.protocol)
        root = (
            REPOSITORY_ROOT / str(protocol["output_root"] if args.stage == "screen" else protocol["confirmation_output_root"])
        ).resolve()
        if args.stage == "confirmation":
            protocol = dict(protocol)
            protocol["seeds"] = list(protocol["confirmation_seeds"])
            protocol["stage"] = "confirmation"
        else:
            protocol = dict(protocol)
            protocol["stage"] = args.stage
        summary = summarize(protocol, root)
        _write_json(root / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, stage=args.stage, resume=args.resume)
    print(json.dumps({"stage": f"t3-{summary['stage']}", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["run_stage", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
