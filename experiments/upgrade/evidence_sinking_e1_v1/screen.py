"""Evidence-sinking E1: CTP tail reserve conditioned on checkpoint evidence.

Mechanism (preregistered in docs/arac-oc-evidence-sinking-plan-v4.0.md §E1,
frozen here before any arm ran): the frozen CTP gives every positive-relation
run a fixed 20% MMES tail.  E1 conditions the tail share on the checkpoint's
static relation evidence - the only lever class the project's meta-law has
ever rewarded (budget ownership, static evidence, no online feedback):

```text
mean_strength = sum(relation.strength) / max(1, relation_count)
norm          = min(1.0, mean_strength / 0.08)     # 0.08 = observed upper
                                                     # end of the S-family
                                                     # mean-strength range
tail_share    = 0.20 * (1.0 + norm)                 # monotone, bounded [0.20, 0.40]
```

The share is swapped into ``ctp_module._POSITIVE_RELATION_MMES_TAIL_FRACTION``
before the action runs and restored after; nothing else changes - session
behavior, ordering, and the zero-relation path are untouched (the constant is
only read when ``overlap_relation_count > 0``, so S1 is bitwise identical by
construction).  A0 leaves the frozen constant alone.

Gate (frozen): S6 paired geometric-mean ratio <= 0.98 on fresh seeds;
S2-S5 non-inferior (95% paired bootstrap CI upper < 1.05, exp scale);
S1 bitwise identity between arms; every arm satisfies the exact terminal-FE
contract; A1 arms record the applied share and a route whose tail FE reflects
it (mechanism-fired evidence).
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
from threadpoolctl import threadpool_limits, threadpool_info

import arac.actions.ctp as ctp_module
from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.evidence.phase1 import run_phase1
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("screen_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-evidence-sinking-e1-arm-receipt-v1"
CHECKPOINT_SCHEMA = "arac-upgrade-evidence-sinking-e1-checkpoint-v1"
FAILURE_SCHEMA = "arac-upgrade-evidence-sinking-e1-failure-v1"
CASES = ("S1", "S2", "S3", "S4", "S5", "S6")
SEEDS = (20270601, 20270602, 20270603, 20270604, 20270605)
VARIANTS = ("a0", "a1")
FROZEN_TAIL_FRACTION = 0.20
STRENGTH_REF = 0.08
PHASE1_FES = 180_000
TOTAL_BUDGET_FES = 3_000_000
PHASE2_FES = TOTAL_BUDGET_FES - PHASE1_FES
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260824
S6_WIN_RATIO = 0.98
NON_INFERIOR_CI_UPPER = 1.05
MAX_WORKERS = 8


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
    return {"python_executable": __import__("sys").executable, "numpy_version": np.__version__, "threadpools": pools}


def conditioned_tail_share(checkpoint) -> tuple[float, float]:
    """(share, norm) from the frozen checkpoint's static relation evidence."""

    relations = checkpoint.relations
    if not relations:
        return FROZEN_TAIL_FRACTION, 0.0
    mean_strength = sum(relation.strength for relation in relations) / len(relations)
    norm = min(1.0, mean_strength / STRENGTH_REF)
    return FROZEN_TAIL_FRACTION * (1.0 + norm), norm


def _terminal_contract(key: str, result, ledger) -> None:
    if (
        result.consumed_fes != PHASE2_FES
        or result.terminal_fes != TOTAL_BUDGET_FES
        or ledger.count != TOTAL_BUDGET_FES
        or result.final_error != ledger.best_error
        or not np.isfinite(result.final_error)
    ):
        raise RuntimeError(f"{key} terminal contract failed")


@dataclass(frozen=True)
class Phase1Job:
    case_id: str
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.case_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def key(self) -> str:
        return f"e1-p1:{self.case_id}:seed-{self.run_seed}"


@dataclass(frozen=True)
class ArmJob:
    case_id: str
    run_seed: int
    variant: str
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.case_id / f"seed_{self.run_seed}" / f"ctp_{self.variant}.json"

    @property
    def key(self) -> str:
        return f"e1:{self.case_id}:seed-{self.run_seed}:{self.variant}"


def _run_phase1_job(job: Phase1Job) -> dict[str, Any]:
    if job.checkpoint_path.is_file():
        return _load_json(job.checkpoint_path)
    started = datetime.now(UTC)
    with threadpool_limits(limits=1):
        runtime = _runtime_block()
        problem = AobBenchmark().load(job.case_id)
        ledger = EvaluationLedger(problem, TOTAL_BUDGET_FES)
        probe = run_phase1(problem, ledger, run_seed=job.run_seed)
        checkpoint = probe.checkpoint
        if ledger.count != PHASE1_FES or checkpoint.phase1_fes != PHASE1_FES:
            raise RuntimeError(f"{job.key} Phase-I boundary drifted: {ledger.count}")
        body = {
            "schema_version": CHECKPOINT_SCHEMA,
            "manifest_sha256": job.manifest_sha256,
            "case_id": job.case_id,
            "run_seed": job.run_seed,
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "block_count": len(checkpoint.blocks),
            "relation_count": len(checkpoint.relations),
            "checkpoint": checkpoint.payload(),
            "runtime": runtime,
            "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
        }
        body["receipt_hash"] = canonical_sha256(body)
        _write_json(job.checkpoint_path, body)
        return body


def _load_checkpoint(job: ArmJob):
    wrapper = _load_json(job.output_root / "checkpoints" / job.case_id / f"seed_{job.run_seed}" / "checkpoint.json")
    if canonical_sha256({k: v for k, v in wrapper.items() if k != "receipt_hash"}) != wrapper.get("receipt_hash"):
        raise ValueError(f"{job.key} checkpoint receipt hash drifted")
    from experiments.historical_recovery.replay import _checkpoint

    checkpoint = _checkpoint(wrapper["checkpoint"])
    if checkpoint.checkpoint_hash != wrapper.get("checkpoint_hash"):
        raise ValueError(f"{job.key} checkpoint hash drifted")
    return checkpoint


def _run_arm(job: ArmJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            checkpoint = _load_checkpoint(job)
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
            share, norm = conditioned_tail_share(checkpoint)
            applied_share = FROZEN_TAIL_FRACTION
            if job.variant == "a1":
                applied_share = share
                setattr(ctp_module, "_POSITIVE_RELATION_MMES_TAIL_FRACTION", applied_share)
            try:
                result = execute_phase2_action(
                    "ctp",
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=job.run_seed,
                    registry=registry,
                )
            finally:
                setattr(ctp_module, "_POSITIVE_RELATION_MMES_TAIL_FRACTION", FROZEN_TAIL_FRACTION)
            _terminal_contract(job.key, result, ledger)
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "variant": job.variant,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "frozen_tail_fraction": FROZEN_TAIL_FRACTION,
                "conditioned_tail_fraction": applied_share,
                "conditioning_norm": norm,
                "relation_count": len(checkpoint.relations),
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


def _paired_bootstrap_ci(log_ratios: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(log_ratios)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.integers(0, count, size=count)
        draws[draw] = float(np.mean(log_ratios[sample]))
    return float(np.exp(np.percentile(draws, 2.5))), float(np.exp(np.percentile(draws, 97.5)))


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    case_rows = []
    for case_id in protocol["cases"]:
        pairs = []
        for seed in protocol["seeds"]:
            a0 = _load_json(output_root / "arms" / case_id / f"seed_{seed}" / "ctp_a0.json")
            a1 = _load_json(output_root / "arms" / case_id / f"seed_{seed}" / "ctp_a1.json")
            for arm in (a0, a1):
                if canonical_sha256({k: v for k, v in arm.items() if k != "receipt_hash"}) != arm.get("receipt_hash"):
                    raise ValueError(f"E1 receipt hash drifted: {case_id}/{seed}")
            if a0["checkpoint_hash"] != a1["checkpoint_hash"]:
                raise ValueError(f"E1 checkpoint pairing drifted: {case_id}/{seed}")
            pairs.append(
                {
                    "seed": seed,
                    "a0_final": a0["final_error"],
                    "a1_final": a1["final_error"],
                    "log_ratio": float(np.log(max(a1["final_error"], 1e-9) / max(a0["final_error"], 1e-9))),
                    "applied_share": a1["conditioned_tail_fraction"],
                    "route_a0": a0["route"],
                    "route_a1": a1["route"],
                    "bitwise_identical": bool(
                        a0["final_error"] == a1["final_error"]
                        and a0["route"] == a1["route"]
                        and a0["action_result_hash"] == a1["action_result_hash"]
                    ),
                }
            )
        log_ratios = np.asarray([pair["log_ratio"] for pair in pairs], dtype=float)
        geo = float(np.exp(np.mean(log_ratios)))
        ci_lower, ci_upper = _paired_bootstrap_ci(log_ratios)
        case_rows.append(
            {
                "case_id": case_id,
                "pairs": pairs,
                "geometric_mean_ratio": geo,
                "ci95_exp": [ci_lower, ci_upper],
                "all_pairs_bitwise_identical": all(pair["bitwise_identical"] for pair in pairs),
            }
        )
    rows = {row["case_id"]: row for row in case_rows}
    checks = {
        "s1_bitwise_identical": rows["S1"]["all_pairs_bitwise_identical"],
        "s6_strict_win": bool(rows["S6"]["geometric_mean_ratio"] <= S6_WIN_RATIO),
        "s2_s5_non_inferior": all(
            rows[case]["ci95_exp"][1] < NON_INFERIOR_CI_UPPER for case in ("S2", "S3", "S4", "S5")
        ),
    }
    body = {
        "schema_version": "arac-upgrade-evidence-sinking-e1-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "case_rows": case_rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = _load_json(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("E1 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-evidence-sinking-e1-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"E1 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("E1 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)

    phase1_jobs = [
        Phase1Job(case_id=case, run_seed=int(seed), output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for case in protocol["cases"]
        for seed in protocol["seeds"]
    ]
    arm_jobs = [
        ArmJob(
            case_id=case,
            run_seed=int(seed),
            variant=variant,
            output_root=output_root,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        for case in protocol["cases"]
        for seed in protocol["seeds"]
        for variant in VARIANTS
    ]
    failures = 0
    pending_p1 = [job for job in phase1_jobs if not job.checkpoint_path.is_file()]
    if pending_p1:
        with ProcessPoolExecutor(max_workers=protocol.get("max_workers", MAX_WORKERS)) as pool:
            futures = {pool.submit(_run_phase1_job, job): job.key for job in pending_p1}
            for future in as_completed(futures):
                try:
                    future.result()
                    print(f"[e1] phase1 done {futures[future]}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[e1] phase1 FAILED {futures[future]}", flush=True)
    pending_arms = [job for job in arm_jobs if not job.receipt_path.is_file()]
    if pending_arms:
        with ProcessPoolExecutor(max_workers=protocol.get("max_workers", MAX_WORKERS)) as pool:
            futures = {pool.submit(_run_arm, job): job.key for job in pending_arms}
            for future in as_completed(futures):
                try:
                    receipt = future.result()
                    print(f"[e1] arm done {futures[future]} final={receipt['final_error']}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[e1] arm FAILED {futures[future]}", flush=True)
    if failures:
        raise RuntimeError(f"E1 had {failures} failed jobs; inspect {output_root / 'failures'}")
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
        protocol = _load_json(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, resume=args.resume)
    print(json.dumps({"stage": "e1", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["conditioned_tail_share", "run_stage", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
