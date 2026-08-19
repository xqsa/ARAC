"""Run a paired CTP tail ablation on fresh Phase-I checkpoints."""

# Thread caps must be set before NumPy, PyPop7, or ARAC imports.
# ruff: noqa: E402

from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

from threadpoolctl import threadpool_info, threadpool_limits

import experiments.final.run as final_run
from arac.actions._execution import (
    BLOCK_POPULATION_SIZE,
    run_full_space,
    run_persistent_blocks,
    run_sequential_blocks,
    terminal_result,
)
from arac.actions.ctp import CtpExecutor, _relation_cover
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import ActionContext, ActionResult
from arac.runtime.ledger import EvaluationLedger


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("ctp_tail_ablation_protocol.json")
CASES = ("S3", "S6")
SEEDS = (31_001, 31_002, 31_003)
VARIANTS = ("ctp_no_reserved_tail", "ctp_mmes_tail_20pct")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
TAIL_FRACTION = 0.20
RECEIPT_SCHEMA = "arac-ctp-tail-ablation-receipt-v1"
SUMMARY_SCHEMA = "arac-ctp-tail-ablation-summary-v1"
MANIFEST_SCHEMA = "arac-ctp-tail-ablation-manifest-v1"
SOURCE_PATHS = (
    "experiments/historical_recovery/ctp_tail_ablation.py",
    "experiments/historical_recovery/ctp_tail_ablation_protocol.json",
    "experiments/final/run.py",
    "src/arac/benchmarks/aob.py",
    "src/arac/evidence/phase1.py",
    "src/arac/evidence/structural.py",
    "src/arac/evidence/mechanism_features.py",
    "src/arac/runtime/contracts.py",
    "src/arac/runtime/ledger.py",
    "src/arac/runtime/optimizers.py",
    "src/arac/actions/_execution.py",
    "src/arac/actions/ctp.py",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    final_run._atomic_json(path, payload)


def _resolved(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def _production_hcc_imports() -> list[str]:
    matches = []
    for path in (REPOSITORY_ROOT / "src" / "arac").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(
            token in source
            for token in ("from hcc", "import hcc", "vendor.hcc", "vendor/hcc")
        ):
            matches.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return matches


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = _load_json(protocol_path)
    expected_keys = {
        "schema_version",
        "status",
        "purpose",
        "cases",
        "seeds",
        "variants",
        "total_budget_fes",
        "expected_phase1_fes",
        "positive_relation_tail_fraction",
        "native_threads",
        "max_workers",
        "output_root",
        "legacy_source",
        "legacy_source_sha256",
        "production_hcc_runtime_imports_allowed",
        "selector_execution_allowed",
        "reference_thresholds_used_for_decision",
        "acceptance_gates",
    }
    if set(protocol) != expected_keys:
        raise ValueError("CTP tail-ablation protocol keys drifted")
    expected = {
        "schema_version": "arac-ctp-tail-ablation-protocol-v1",
        "status": "frozen_paired_development_gate",
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "total_budget_fes": TOTAL_BUDGET_FES,
        "expected_phase1_fes": PHASE1_FES,
        "positive_relation_tail_fraction": TAIL_FRACTION,
        "native_threads": 1,
        "max_workers": 12,
        "production_hcc_runtime_imports_allowed": False,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"CTP tail-ablation protocol drifted: {key}")
    expected_gates = {
        "all_checkpoints_positive_relation": True,
        "same_checkpoint_per_pair": True,
        "exact_terminal_fes": True,
        "all_final_errors_finite": True,
        "candidate_reserved_tail_executed": True,
        "per_case_geometric_mean_ratio_lt": 1.0,
        "overall_candidate_win_count_gte": 4,
        "maximum_pair_ratio_lte": 10.0,
        "all_runtime_warnings_known": True,
    }
    if protocol.get("acceptance_gates") != expected_gates:
        raise ValueError("CTP tail-ablation acceptance gates drifted")
    legacy_path = _resolved(str(protocol["legacy_source"]))
    if _sha256(legacy_path) != protocol["legacy_source_sha256"]:
        raise ValueError("frozen no-reserved-tail CTP source drifted")
    return protocol


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(_resolved(relative)) for relative in SOURCE_PATHS}


def _preflight(path: Path, *, resume: bool) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    if output_root.exists() and not resume:
        raise ValueError(f"fresh CTP ablation output already exists: {output_root}")
    missing = [relative for relative in SOURCE_PATHS if not _resolved(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"CTP tail-ablation inputs are missing: {missing}")
    imports = _production_hcc_imports()
    if imports:
        raise ValueError(f"production HCC imports remain: {imports}")
    vendor_trees = final_run._vendor_tree_hashes()
    return {
        "schema_version": "arac-ctp-tail-ablation-preflight-v1",
        "protocol_sha256": _sha256(protocol_path),
        "output_root": str(output_root),
        "source_sha256": _source_hashes(),
        "legacy_source_sha256": protocol["legacy_source_sha256"],
        "vendor_trees": vendor_trees,
        "checkpoint_count": len(CASES) * len(SEEDS),
        "arm_count": len(CASES) * len(SEEDS) * len(VARIANTS),
        "production_hcc_runtime_imports": imports,
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
        "passed": True,
    }


def preflight(
    path: Path = DEFAULT_PROTOCOL,
    *,
    resume: bool = False,
) -> dict[str, object]:
    return _preflight(Path(path).resolve(), resume=resume)


def _manifest(protocol_path: Path, preflight_payload: Mapping[str, object]) -> dict[str, object]:
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(protocol_path),
        "source_sha256": preflight_payload["source_sha256"],
        "legacy_source_sha256": preflight_payload["legacy_source_sha256"],
        "vendor_trees": preflight_payload["vendor_trees"],
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    return {**body, "manifest_sha256": _canonical(body)}


def _load_manifest(output_root: Path, protocol_path: Path) -> dict[str, object]:
    manifest = _load_json(output_root / "manifest.json")
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical(manifest):
        raise ValueError("CTP tail-ablation manifest hash drifted")
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("CTP tail-ablation protocol hash drifted")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("CTP tail-ablation source hashes drifted")
    if manifest.get("vendor_trees") != final_run._vendor_tree_hashes():
        raise ValueError("CTP tail-ablation AOB vendor tree drifted")
    manifest["manifest_sha256"] = claimed
    return manifest


@dataclass(frozen=True)
class ArmContext:
    case_id: str
    run_seed: int
    variant: str
    output_root: Path
    manifest_sha256: str

    @property
    def key(self) -> str:
        return f"{self.case_id}:seed-{self.run_seed}:{self.variant}"

    @property
    def receipt_path(self) -> Path:
        return (
            self.output_root
            / "arms"
            / self.case_id
            / f"seed_{self.run_seed}"
            / f"{self.variant}.json"
        )


def _execute_no_reserved_tail(
    context: ActionContext,
) -> tuple[ActionResult, dict[str, int]]:
    """Reproduce the frozen CTP allocation while using the current runtime."""

    sweep_fes = len(context.checkpoint.blocks) * BLOCK_POPULATION_SIZE
    coverage_budget = min(
        context.ledger.remaining,
        max(sweep_fes, int(context.ledger.remaining * TAIL_FRACTION)),
    )
    coverage_fes = run_persistent_blocks(context, requested_fes=coverage_budget)
    blocks = (
        context.checkpoint.blocks
        if context.checkpoint.overlap_relation_count == 0
        else _relation_cover(context)
    )
    polish_fes = run_sequential_blocks(
        context,
        requested_fes=context.ledger.remaining,
        blocks=blocks,
    )
    residual_tail_fes = 0
    if context.ledger.remaining:
        residual_tail_fes = run_full_space(
            context,
            algorithm="mmes",
            namespace="ctp-terminal",
        ).consumed_fes
    route = (
        f"coverage_{coverage_fes}_then_sequential_block_polish_{polish_fes}"
        if context.checkpoint.overlap_relation_count == 0
        else f"coverage_{coverage_fes}_then_relation_cover_polish_{polish_fes}"
    )
    return terminal_result(context, route=route), {
        "coverage_fes": coverage_fes,
        "polish_fes": polish_fes,
        "tail_fes": residual_tail_fes,
    }


def _candidate_components(result: ActionResult) -> dict[str, int]:
    route = result.route
    try:
        coverage_raw, remainder = route.removeprefix("coverage_").split("_then_", 1)
        polish_raw, tail_raw = remainder.rsplit("_then_mmes_tail_", 1)
        polish_fes = int(polish_raw.rsplit("_", 1)[1])
        return {
            "coverage_fes": int(coverage_raw),
            "polish_fes": polish_fes,
            "tail_fes": int(tail_raw),
        }
    except (ValueError, IndexError) as error:
        raise ValueError(f"candidate CTP route is invalid: {route}") from error


def _execute_variant(
    variant: str,
    context: ActionContext,
) -> tuple[ActionResult, dict[str, int]]:
    if variant == "ctp_no_reserved_tail":
        return _execute_no_reserved_tail(context)
    if variant == "ctp_mmes_tail_20pct":
        result = CtpExecutor().execute(context)
        return result, _candidate_components(result)
    raise ValueError(f"unsupported CTP tail-ablation variant: {variant}")


def _threadpools() -> list[dict[str, object]]:
    return [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
            "user_api": item.get("user_api"),
        }
        for item in threadpool_info()
    ]


def _run_arm(context: ArmContext) -> dict[str, object]:
    checkpoint_context = final_run.CheckpointContext(
        context.case_id,
        context.run_seed,
        TOTAL_BUDGET_FES,
        context.output_root,
    )
    checkpoint_receipt = final_run._validate_checkpoint(
        checkpoint_context.receipt_path,
        checkpoint_context,
    )
    checkpoint_payload = checkpoint_receipt.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(f"{context.key} shared checkpoint is invalid")
    checkpoint = final_run._checkpoint_from_payload(checkpoint_payload)
    problem = AobBenchmark().load(
        context.case_id,
        output_directory=context.receipt_path.parent / "benchmark",
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    action_context = ActionContext(
        "ctp",
        checkpoint,
        problem,
        ledger,
        action_seed=context.run_seed,
    )
    with threadpool_limits(limits=1):
        pools = _threadpools()
        if not pools or any(item["num_threads"] != 1 for item in pools):
            raise RuntimeError(f"{context.key} native thread limit is not one: {pools}")
        (result, components), runtime_warnings = final_run._call_with_warning_capture(
            _execute_variant,
            context.variant,
            action_context,
        )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "variant": context.variant,
        "action_seed": context.run_seed,
        "phase1_fes": checkpoint.phase1_fes,
        "phase1_relation_count": checkpoint.overlap_relation_count,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "action_checkpoint_hash": result.checkpoint_hash,
        "phase2_consumed_fes": result.consumed_fes,
        "terminal_fes": result.terminal_fes,
        "final_error": result.final_error,
        "terminal_state_finite": math.isfinite(result.final_error),
        "route": result.route,
        "coverage_fes": components["coverage_fes"],
        "polish_fes": components["polish_fes"],
        "tail_fes": components["tail_fes"],
        "action_result_hash": result.result_hash,
        "runtime_warnings": runtime_warnings,
        "native_thread_limit_verified": True,
        "threadpools": pools,
        "production_hcc_runtime_imports": _production_hcc_imports(),
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    receipt = {**body, "receipt_sha256": _canonical(body)}
    _write_json(context.receipt_path, receipt)
    return receipt


def _validate_arm(path: Path, context: ArmContext) -> dict[str, object]:
    receipt = _load_json(path)
    claimed = receipt.pop("receipt_sha256", None)
    if claimed != _canonical(receipt):
        raise ValueError(f"{context.key} receipt hash drifted")
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "case_id": context.case_id,
        "run_seed": context.run_seed,
        "variant": context.variant,
        "action_seed": context.run_seed,
        "phase1_fes": PHASE1_FES,
        "phase2_consumed_fes": TOTAL_BUDGET_FES - PHASE1_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "terminal_state_finite": True,
        "native_thread_limit_verified": True,
        "production_hcc_runtime_imports": [],
        "selector_execution_allowed": False,
        "reference_thresholds_used_for_decision": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{context.key} receipt field drifted: {key}")
    if receipt.get("checkpoint_hash") != receipt.get("action_checkpoint_hash"):
        raise ValueError(f"{context.key} action did not use its shared checkpoint")
    if int(receipt.get("phase1_relation_count", 0)) <= 0:
        raise ValueError(f"{context.key} is not a positive-relation checkpoint")
    if not math.isfinite(float(receipt.get("final_error", math.nan))):
        raise ValueError(f"{context.key} final error is not finite")
    if sum(int(receipt[name]) for name in ("coverage_fes", "polish_fes", "tail_fes")) != (
        TOTAL_BUDGET_FES - PHASE1_FES
    ):
        raise ValueError(f"{context.key} phase-II component ledger drifted")
    final_run._validate_runtime_warnings(receipt.get("runtime_warnings"), context.key)
    receipt["receipt_sha256"] = claimed
    return receipt


def _paired_ratio(candidate: float, baseline: float) -> float | None:
    if baseline > 0.0:
        return candidate / baseline
    return 1.0 if candidate == 0.0 else None


def _summarize(
    rows: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    by_key = {
        (str(row["case_id"]), int(row["run_seed"]), str(row["variant"])): row
        for row in rows
    }
    pairs = []
    for case_id in CASES:
        for seed in SEEDS:
            baseline = by_key[(case_id, seed, "ctp_no_reserved_tail")]
            candidate = by_key[(case_id, seed, "ctp_mmes_tail_20pct")]
            baseline_error = float(baseline["final_error"])
            candidate_error = float(candidate["final_error"])
            ratio = _paired_ratio(candidate_error, baseline_error)
            pairs.append(
                {
                    "case_id": case_id,
                    "run_seed": seed,
                    "checkpoint_hash": candidate["checkpoint_hash"],
                    "same_checkpoint": (
                        candidate["checkpoint_hash"] == baseline["checkpoint_hash"]
                    ),
                    "baseline_final_error": baseline_error,
                    "candidate_final_error": candidate_error,
                    "candidate_to_baseline_ratio": ratio,
                    "candidate_won_or_tied": candidate_error <= baseline_error,
                    "baseline_tail_fes": int(baseline["tail_fes"]),
                    "candidate_tail_fes": int(candidate["tail_fes"]),
                }
            )
    case_summaries = []
    for case_id in CASES:
        active = [pair for pair in pairs if pair["case_id"] == case_id]
        ratios = [pair["candidate_to_baseline_ratio"] for pair in active]
        geometric_mean = (
            math.exp(sum(math.log(float(value)) for value in ratios) / len(ratios))
            if all(value is not None and float(value) > 0.0 for value in ratios)
            else 0.0 if all(value == 0.0 for value in ratios) else None
        )
        case_summaries.append(
            {
                "case_id": case_id,
                "pair_count": len(active),
                "candidate_win_or_tie_count": sum(
                    bool(pair["candidate_won_or_tied"]) for pair in active
                ),
                "candidate_to_baseline_geometric_mean_ratio": geometric_mean,
                "paired_improvement_passed": (
                    geometric_mean is not None
                    and geometric_mean
                    < float(
                        protocol["acceptance_gates"][  # type: ignore[index]
                            "per_case_geometric_mean_ratio_lt"
                        ]
                    )
                ),
            }
        )
    warning_summary = final_run._runtime_warning_summary(rows)
    ratios = [pair["candidate_to_baseline_ratio"] for pair in pairs]
    exact_terminal = all(int(row["terminal_fes"]) == TOTAL_BUDGET_FES for row in rows)
    same_checkpoints = all(bool(pair["same_checkpoint"]) for pair in pairs)
    finite = all(math.isfinite(float(row["final_error"])) for row in rows)
    positive_relations = all(int(row["phase1_relation_count"]) > 0 for row in rows)
    required_tail = int((TOTAL_BUDGET_FES - PHASE1_FES) * TAIL_FRACTION)
    candidate_tail = all(
        int(row["tail_fes"]) >= required_tail
        for row in rows
        if row["variant"] == "ctp_mmes_tail_20pct"
    )
    win_count = sum(bool(pair["candidate_won_or_tied"]) for pair in pairs)
    maximum_ratio = max(float(value) for value in ratios if value is not None)
    gates = protocol["acceptance_gates"]
    gate_passed = (
        positive_relations
        and same_checkpoints
        and exact_terminal
        and finite
        and candidate_tail
        and all(bool(row["paired_improvement_passed"]) for row in case_summaries)
        and win_count >= int(gates["overall_candidate_win_count_gte"])  # type: ignore[index]
        and all(value is not None for value in ratios)
        and maximum_ratio <= float(gates["maximum_pair_ratio_lte"])  # type: ignore[index]
        and warning_summary["all_runtime_warnings_known"] is True
    )
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_count": len(CASES) * len(SEEDS),
        "arm_count": len(rows),
        "pair_count": len(pairs),
        "phase1_fes": PHASE1_FES,
        "terminal_fes": TOTAL_BUDGET_FES,
        "all_checkpoints_positive_relation": positive_relations,
        "same_checkpoint_per_pair": same_checkpoints,
        "exact_terminal_fes": exact_terminal,
        "all_final_errors_finite": finite,
        "candidate_reserved_tail_executed": candidate_tail,
        "candidate_win_or_tie_count": win_count,
        "maximum_candidate_to_baseline_ratio": maximum_ratio,
        "case_summaries": case_summaries,
        "pairs": pairs,
        **warning_summary,
        "paired_development_gate_passed": gate_passed,
        "final_25_seed_recovery_evaluated": False,
        "selector_or_routing_authorized": False,
    }
    return {**body, "summary_sha256": _canonical(body)}


def _contexts(output_root: Path, manifest_sha256: str) -> tuple[ArmContext, ...]:
    return tuple(
        ArmContext(case_id, seed, variant, output_root, manifest_sha256)
        for seed in SEEDS
        for case_id in CASES
        for variant in VARIANTS
    )


def run(
    path: Path = DEFAULT_PROTOCOL,
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    gate = _preflight(protocol_path, resume=resume)
    output_root = _resolved(str(protocol["output_root"]))
    max_workers = int(protocol["max_workers"] if workers is None else workers)
    if not 1 <= max_workers <= len(CASES) * len(SEEDS) * len(VARIANTS):
        raise ValueError("workers must be in 1..12")
    if resume:
        manifest = _load_manifest(output_root, protocol_path)
    else:
        output_root.mkdir(parents=True)
        _write_json(output_root / "protocol.json", protocol)
        _write_json(output_root / "preflight.json", gate)
        manifest = _manifest(protocol_path, gate)
        _write_json(output_root / "manifest.json", manifest)
    checkpoint_contexts = tuple(
        final_run.CheckpointContext(case_id, seed, TOTAL_BUDGET_FES, output_root)
        for seed in SEEDS
        for case_id in CASES
    )
    final_run._run_parallel(
        checkpoint_contexts,
        final_run._run_checkpoint,
        max_workers=min(max_workers, len(checkpoint_contexts)),
        progress_path=output_root / "checkpoint_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=final_run._validate_checkpoint,
        resume=resume,
    )
    contexts = _contexts(output_root, str(manifest["manifest_sha256"]))
    rows = final_run._run_parallel(
        contexts,
        _run_arm,
        max_workers=max_workers,
        progress_path=output_root / "parallel_progress.json",
        receipt_path=lambda context: context.receipt_path,
        validator=_validate_arm,
        resume=resume,
    )
    summary = _summarize(rows, protocol)
    _write_json(output_root / "summary.json", summary)
    final_run._require_known_runtime_warnings(summary, stage="CTP paired development gate")
    return summary


def verify(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _resolved(str(protocol["output_root"]))
    manifest = _load_manifest(output_root, protocol_path)
    checkpoint_contexts = tuple(
        final_run.CheckpointContext(case_id, seed, TOTAL_BUDGET_FES, output_root)
        for seed in SEEDS
        for case_id in CASES
    )
    for context in checkpoint_contexts:
        final_run._validate_checkpoint(context.receipt_path, context)
    contexts = _contexts(output_root, str(manifest["manifest_sha256"]))
    rows = [_validate_arm(context.receipt_path, context) for context in contexts]
    expected = _summarize(rows, protocol)
    stored = _load_json(output_root / "summary.json")
    stored_claimed = stored.pop("summary_sha256", None)
    if stored_claimed != _canonical(stored):
        raise ValueError("CTP tail-ablation summary hash drifted")
    expected.pop("summary_sha256", None)
    expected.pop("generated_at_utc", None)
    stored.pop("generated_at_utc", None)
    if stored != expected:
        raise ValueError("CTP tail-ablation summary content drifted")
    stored["summary_sha256"] = stored_claimed
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.protocol, resume=args.resume)
    elif args.command == "run":
        result = run(args.protocol, resume=args.resume, workers=args.workers)
    else:
        result = verify(args.protocol)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
