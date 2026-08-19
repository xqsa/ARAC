"""Gate 41b: online 25-seed AOB-24 run of the evidence-dispatched method."""

# Thread caps must be set before NumPy and optimizer imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import statistics
import traceback
from pathlib import Path

from threadpoolctl import threadpool_limits

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.dispatch_policy import dispatch_action
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery.replay import _checkpoint

CASES = tuple(f"{family}{index}" for family in "AERS" for index in range(1, 7))
SEEDS = tuple(range(117, 142))
CHECKPOINT_ROOT = Path("artifacts/historical_recovery_fixed_expert_v1/checkpoints")
E2E_ROOT = Path("artifacts/current_arac_aob24_recovery_v1/runs")
FOUR_ARM_ROOT = Path("artifacts/current_recovered_four_arm_matrix_v2/arms")
REFERENCE_CSV = Path("output/pdf/aob_arac_method_comparison_corrected.csv")
RECEIPT_SCHEMA = "arac-overlap-action-dispatch-gate41b-receipt-v1"
OUTPUT_SCHEMA = "arac-overlap-action-dispatch-gate41b-v1"


@dataclass(frozen=True)
class RunContext:
    case_id: str
    run_seed: int

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_ROOT / self.case_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def e2e_receipt_path(self) -> Path:
        return E2E_ROOT / self.case_id / f"seed_{self.run_seed}" / "receipt.json"

    @property
    def four_arm_path(self) -> Path:
        return FOUR_ARM_ROOT / self.case_id / f"seed_{self.run_seed}"


def _four_arm_reuse(context: RunContext, action: str) -> dict[str, object] | None:
    path = context.four_arm_path / f"{action}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_id") != context.case_id or payload.get("run_seed") != context.run_seed:
        raise RuntimeError(f"four-arm receipt identity drifted: {path}")
    return {
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action": action,
        "final_error": float(payload["final_error"]),
        "terminal_fes": int(payload["terminal_fes"]),
        "checkpoint_hash": payload["checkpoint_hash"],
        "source": "four_arm_reuse",
    }


def run_context(context: RunContext) -> dict[str, object]:
    wrapper = json.loads(context.checkpoint_path.read_text(encoding="utf-8"))
    checkpoint = _checkpoint(wrapper["checkpoint"])
    if wrapper.get("checkpoint_hash") != checkpoint.checkpoint_hash:
        raise RuntimeError(f"{context.case_id}/{context.run_seed} checkpoint hash drifted")
    e2e = json.loads(context.e2e_receipt_path.read_text(encoding="utf-8"))
    if e2e["phase1_checkpoint_hash"] != checkpoint.checkpoint_hash:
        raise RuntimeError(f"{context.case_id}/{context.run_seed} checkpoint identity drifted")
    features = dict(zip(checkpoint.feature_names, checkpoint.feature_values))
    action = dispatch_action(
        features["tail_log10_gain"],
        features["structural_relation_density"],
    )
    reuse = _four_arm_reuse(context, action)
    if reuse is not None:
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "case_id": context.case_id,
            "run_seed": context.run_seed,
            "action": action,
            "dispatch_features": {
                "tail_log10_gain": features["tail_log10_gain"],
                "structural_relation_density": features["structural_relation_density"],
            },
            "final_error": reuse["final_error"],
            "terminal_fes": reuse["terminal_fes"],
            "checkpoint_hash": reuse["checkpoint_hash"],
            "source": "four_arm_reuse",
        }
        body["receipt_hash"] = canonical_sha256(
            {k: v for k, v in body.items() if k != "receipt_hash"}
        )
        return body
    with threadpool_limits(limits=1):
        problem = AobBenchmark().load(context.case_id)
        registry = RecoveredActionRegistry()
        ledger = EvaluationLedger.from_checkpoint(
            problem,
            total_budget=checkpoint.total_budget_fes,
            phase1_fes=checkpoint.phase1_fes,
            incumbent=checkpoint.incumbent,
            incumbent_error=checkpoint.incumbent_error,
            allow_out_of_bounds=registry.allow_out_of_bounds,
        )
        result = execute_phase2_action(
            action,
            checkpoint,
            problem,
            ledger,
            action_seed=context.run_seed,
            registry=registry,
        )
        if (
            result.consumed_fes != 2_820_000
            or result.terminal_fes != 3_000_000
            or ledger.count != 3_000_000
            or not math.isfinite(result.final_error)
            or result.final_error != ledger.best_error
        ):
            raise RuntimeError(f"{context.case_id}/{context.run_seed} terminal contract failed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "action": action,
        "dispatch_features": {
            "tail_log10_gain": features["tail_log10_gain"],
            "structural_relation_density": features["structural_relation_density"],
        },
        "final_error": float(result.final_error),
        "terminal_fes": int(result.terminal_fes),
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "source": "online_run",
    }
    body["receipt_hash"] = canonical_sha256(
        {k: v for k, v in body.items() if k != "receipt_hash"}
    )
    return body


def _receipt_path(root: Path, row: dict[str, object]) -> Path:
    return root / str(row["case_id"]) / f"seed_{row['run_seed']}.json"


def _load_reference() -> dict[str, dict[str, float]]:
    import csv
    import re

    references: dict[str, dict[str, float]] = {}
    valid = {f"{family}{index}" for family in "AERS" for index in range(1, 7)}
    with REFERENCE_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case"] not in valid:
                continue
            references[row["case"]] = {
                "arac": float(re.match(r"\s*([0-9.eE+-]+)", row["ARAC Mean +/- Std"]).group(1)),
                "hcc": float(re.match(r"\s*([0-9.eE+-]+)", row["HCC-ES Mean +/- Std"]).group(1)),
            }
    return references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--receipt-root",
        type=Path,
        default=Path("artifacts/overlap_action_dispatch_gate41_online/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_action_dispatch_gate41_online/confirmation.json"),
    )
    args = parser.parse_args()
    jobs = [RunContext(case, seed) for case in CASES for seed in SEEDS]
    rows: list[dict[str, object]] = []
    pending: list[RunContext] = []
    for context in jobs:
        path = _receipt_path(args.receipt_root, {
            "case_id": context.case_id,
            "run_seed": context.run_seed,
        })
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != RECEIPT_SCHEMA:
                raise RuntimeError(f"receipt schema drifted: {path}")
            rows.append(payload)
        else:
            pending.append(context)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_context, context): context for context in pending}
            for future in as_completed(futures):
                context = futures[future]
                try:
                    row = future.result()
                except BaseException:
                    print(f"FAILED {context.case_id}/{context.run_seed}")
                    traceback.print_exc()
                    raise
                _receipt_path(args.receipt_root, row).parent.mkdir(parents=True, exist_ok=True)
                _receipt_path(args.receipt_root, row).write_text(
                    json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
                )
                rows.append(row)
                print(f"completed {row['case_id']}/{row['run_seed']} -> {row['action']}", flush=True)

    references = _load_reference()
    per_case: dict[str, list[float]] = {case: [] for case in CASES}
    for row in rows:
        per_case[str(row["case_id"])].append(float(row["final_error"]))
    case_rows = []
    for case in CASES:
        mean = statistics.mean(per_case[case])
        historical = references[case]["arac"]
        hcc = references[case]["hcc"]
        case_rows.append(
            {
                "case": case,
                "seed_count": len(per_case[case]),
                "mean": mean,
                "historical_arac_mean": historical,
                "hcc_mean": hcc,
                "ratio_vs_historical": mean / historical,
                "better_than_hcc": mean < hcc,
            }
        )
    ratios = [row["ratio_vs_historical"] for row in case_rows]
    hcc_wins = sum(1 for row in case_rows if row["better_than_hcc"])
    protocol_checks = {
        "context_count_600": len(rows) == 600,
        "seed_count_25_each": all(row["seed_count"] == 25 for row in case_rows),
        "terminal_exact": all(row["terminal_fes"] == 3_000_000 for row in rows),
    }
    screening_checks = {
        "every_case_within_1_10x_of_historical": all(ratio <= 1.10 for ratio in ratios),
        "geometric_mean_ratio_le_1_0": statistics.geometric_mean(ratios) <= 1.0,
        "hcc_win_count_ge_18": hcc_wins >= 18,
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "seeds": list(SEEDS),
            "checkpoint_source": str(CHECKPOINT_ROOT),
            "e2e_source": str(E2E_ROOT),
            "four_arm_reuse_source": str(FOUR_ARM_ROOT),
            "reference_source": str(REFERENCE_CSV),
        },
        "summary": {
            "hcc_win_count": hcc_wins,
            "hcc_loss_count": len(case_rows) - hcc_wins,
            "geometric_mean_ratio_vs_historical": statistics.geometric_mean(ratios),
            "max_ratio_vs_historical": max(ratios),
            "cases_better_than_historical_column": sum(
                1 for row in case_rows if row["mean"] < row["historical_arac_mean"]
            ),
            "online_run_count": sum(1 for row in rows if row["source"] == "online_run"),
            "four_arm_reuse_count": sum(1 for row in rows if row["source"] == "four_arm_reuse"),
        },
        "protocol_checks": protocol_checks,
        "screening_checks": screening_checks,
        "gate_passed": all(protocol_checks.values()) and all(screening_checks.values()),
        "cases": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{'case':>4} {'mean':>12} {'histARAC':>12} {'ratio':>7} {'vsHCC':>6}"
    )
    for row in case_rows:
        print(
            f"{row['case']:>4} {row['mean']:>12.3g} {row['historical_arac_mean']:>12.3g} "
            f"{row['ratio_vs_historical']:>7.3f} {'WIN' if row['better_than_hcc'] else 'loss':>6}"
        )
    print(json.dumps({"summary": payload["summary"], "checks": {**protocol_checks, **screening_checks}}, indent=1))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
