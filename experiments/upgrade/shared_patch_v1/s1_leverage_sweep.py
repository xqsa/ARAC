"""S1 leverage-priority sweep for the shared_patch_v1 upgrade candidate.

Mechanism (docs/arac-oc-stepwise-upgrade-plan-v2.1.md, stage S1):

- static scope priority ``(-leverage(scope), original_scope_rank)`` where
  leverage is the executor-lane incident-relation count preregistered in
  the U1 protocol;
- the host's FIRST block sweep puts the top ``ceil(20% * n_blocks)`` scopes
  (by that priority) into its head slots and keeps every other scope in
  the frozen baseline order; all later sweeps restore the baseline order;
- zero additional FE: the shim only rewrites the ``block_order`` argument
  of the frozen sweep functions; per-scope budgets, seeds and routes are
  untouched.

Host granularity (preregistered):

- ctp host: the coverage segment (one interleaved ``run_persistent_blocks``
  call) is the first block sweep; the frozen API exposes one static order
  per call, so the head-tail order applies to the whole coverage call.
- gcb host: the first source sweep (``namespace == "gcb-source"`` with
  ``start_sweep_index == 0``) is the first block sweep.

Zero-relation checkpoints make leverage identically zero, so the S1 order
equals the baseline order bit-for-bit; those arms are the built-in exact
no-tax control and must reproduce the frozen receipts exactly.
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
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

import arac.actions._execution as execution_module
import arac.actions.ctp as ctp_module
import arac.actions.gcb as gcb_module
from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import current_recovered_four_arm as fixed
from experiments.upgrade.shared_patch_v1.conflicting_generator import build_problem, relation_leverage
from experiments.upgrade.shared_patch_v1.host_instrumentation import SweepRecorder
from experiments.upgrade.shared_patch_v1.u1_host_reachability import (
    _load_generator_checkpoint,
    write_generator_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("s1_leverage_sweep_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-s1-arm-receipt-v1"
MILESTONE_FES = (600_000, 1_000_000, 2_000_000, 3_000_000)
ARMS = ("a0", "a1")


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


def head_slot_count(block_count: int) -> int:
    return max(1, math.ceil(0.20 * block_count))


def s1_order(blocks_count: int, leverage: Sequence[int], baseline_order: Sequence[int]) -> tuple[int, ...]:
    baseline = tuple(int(index) for index in baseline_order)
    if sorted(baseline) != list(range(blocks_count)):
        raise ValueError("baseline order must be a complete block permutation")
    priority = sorted(range(blocks_count), key=lambda index: (-leverage[index], baseline.index(index)))
    head = set(priority[: head_slot_count(blocks_count)])
    order = tuple(index for index in priority if index in head) + tuple(index for index in baseline if index not in head)
    if sorted(order) != list(range(blocks_count)):
        raise RuntimeError("S1 reorder lost block coverage")
    return order


class MilestoneSampler:
    """Record strict-best error at fixed FE milestones without perturbing runs."""

    def __init__(self, ledger: EvaluationLedger, milestones: Sequence[int] = MILESTONE_FES) -> None:
        self._ledger = ledger
        self._milestones = tuple(sorted(int(value) for value in milestones))
        self._next = 0
        self.samples: list[dict[str, Any]] = []
        try:
            self.samples.append({"fes": ledger.count, "best_error": float(ledger.best_error)})
        except RuntimeError:
            pass
        self._original = ledger.evaluate

        def wrapper(candidate):  # noqa: ANN001, ANN202
            result = self._original(candidate)
            self._record()
            return result

        ledger.evaluate = wrapper

    def _record(self) -> None:
        while self._next < len(self._milestones) and self._ledger.count >= self._milestones[self._next]:
            self.samples.append(
                {"fes": self._milestones[self._next], "best_error": float(self._ledger.best_error)}
            )
            self._next += 1

    def payload(self) -> list[dict[str, Any]]:
        return [dict(sample) for sample in self.samples]


class LeverageOrderShim:
    """Rewrite the first block-sweep order of one host; record both orders."""

    def __init__(self, host: str) -> None:
        if host not in ("ctp", "gcb"):
            raise ValueError(f"unsupported S1 host: {host}")
        self.host = host
        self.applied = False
        self.records: list[dict[str, Any]] = []

    def _maybe_rewrite(self, context, kwargs, *, namespace: str, start_sweep_index: int | None) -> None:
        if self.applied:
            return
        self.applied = True
        blocks = kwargs.get("blocks") or context.checkpoint.blocks
        baseline = tuple(kwargs.get("block_order") or range(len(blocks)))
        leverage = relation_leverage(blocks, context.checkpoint.relations)
        applied_order = s1_order(len(blocks), leverage, baseline)
        kwargs["block_order"] = applied_order
        self.records.append(
            {
                "host": self.host,
                "namespace": namespace,
                "start_sweep_index": start_sweep_index,
                "block_count": len(blocks),
                "head_slots": head_slot_count(len(blocks)),
                "leverage_per_block": list(leverage),
                "baseline_order": [int(index) for index in baseline],
                "applied_order": [int(index) for index in applied_order],
                "order_changed": tuple(baseline) != applied_order,
                "relation_count": len(context.checkpoint.relations),
            }
        )

    def install(self) -> None:
        if self.host == "ctp":
            original = ctp_module.run_persistent_blocks

            def wrapper(context, **kwargs):
                self._maybe_rewrite(context, kwargs, namespace="ctp-coverage", start_sweep_index=None)
                return original(context, **kwargs)

            ctp_module.run_persistent_blocks = wrapper
        else:
            original = gcb_module.run_cold_start_block_sweeps

            def wrapper(context, **kwargs):
                self._maybe_rewrite(
                    context,
                    kwargs,
                    namespace=str(kwargs.get("namespace", "")),
                    start_sweep_index=kwargs.get("start_sweep_index"),
                )
                return original(context, **kwargs)

            gcb_module.run_cold_start_block_sweeps = wrapper

    def uninstall(self) -> None:
        if self.host == "ctp":
            ctp_module.run_persistent_blocks = execution_module.run_persistent_blocks
        else:
            gcb_module.run_cold_start_block_sweeps = execution_module.run_cold_start_block_sweeps


@dataclass(frozen=True)
class S1AobJob:
    case_id: str
    run_seed: int
    action_name: str
    arm: str
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
        return (
            self.output_root
            / "arms"
            / "aob"
            / self.case_id
            / f"seed_{self.run_seed}"
            / f"{self.action_name}_{self.arm}.json"
        )

    @property
    def frozen_receipt_path(self) -> Path:
        return self.frozen_receipt_root / self.case_id / f"seed_{self.run_seed}" / f"{self.action_name}.json"

    @property
    def key(self) -> str:
        return f"aob:{self.case_id}:seed-{self.run_seed}:{self.action_name}:{self.arm}"


@dataclass(frozen=True)
class S1GeneratorJob:
    cell_id: str
    run_seed: int
    host: str
    arm: str
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.cell_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / "generator" / self.cell_id / f"seed_{self.run_seed}" / f"{self.host}_{self.arm}.json"

    @property
    def key(self) -> str:
        return f"generator:{self.cell_id}:seed-{self.run_seed}:{self.host}:{self.arm}"


def _leverage_rows(checkpoint) -> dict[str, Any]:
    leverage = relation_leverage(checkpoint.blocks, checkpoint.relations)
    return {
        "block_count": len(checkpoint.blocks),
        "relation_count": len(checkpoint.relations),
        "leverage_per_block": list(leverage),
        "leverage_positive_blocks": sum(1 for value in leverage if value > 0),
    }


def _write_failure(output_root: Path, job_key: str, exc: BaseException) -> None:
    _write_json(
        output_root / "failures" / f"{job_key.replace(':', '_')}.json",
        {
            "schema_version": "arac-upgrade-s1-failure-v1",
            "key": job_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _execute_arm(job_key: str, host: str, arm: str, checkpoint, problem, *, case_id: str | None, cell_id: str | None, run_seed: int, manifest_sha256: str, frozen_receipt: Path | None, output_root: Path, receipt_path: Path, ground_truth_hash: str | None = None, expected_phase2_fes: int = 2_820_000, expected_terminal_fes: int = 3_000_000) -> dict[str, Any]:
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
    order_shim: LeverageOrderShim | None = LeverageOrderShim(host) if arm == "a1" else None
    if order_shim is not None:
        order_shim.install()
    try:
        result = execute_phase2_action(host, checkpoint, problem, ledger, action_seed=run_seed, registry=registry)
    finally:
        if order_shim is not None:
            order_shim.uninstall()
        recorder.uninstall()
    if (
        result.consumed_fes != expected_phase2_fes
        or result.terminal_fes != expected_terminal_fes
        or ledger.count != expected_terminal_fes
        or result.final_error != ledger.best_error
        or not math.isfinite(result.final_error)
    ):
        raise RuntimeError(f"{job_key} terminal contract failed")
    leverage_rows = _leverage_rows(checkpoint)
    instrumentation = recorder.summary(tuple(leverage_rows["leverage_per_block"]))
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
        "arm": arm,
        "host": host,
        "case_id": case_id,
        "cell_id": cell_id,
        "run_seed": run_seed,
        "forced_host": cell_id is not None,
        "phase1_protocol": checkpoint.protocol,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "initial_error": checkpoint.incumbent_error,
        "final_error": result.final_error,
        "route": result.route,
        "action_result_hash": result.result_hash,
        "identity": identity,
        "leverage": leverage_rows,
        "instrumentation": instrumentation,
        "milestones": sampler.payload(),
        "order_records": order_shim.records if order_shim is not None else [],
        "ground_truth_hash": ground_truth_hash,
        "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
    }
    receipt = {**body, "receipt_hash": canonical_sha256(body)}
    _write_json(receipt_path, receipt)
    return receipt


def _run_aob_job(job: S1AobJob) -> dict[str, Any]:
    try:
        with threadpool_limits(limits=1):
            checkpoint = fixed._load_verified_checkpoint(job.context)
            problem = AobBenchmark().load(job.case_id)
            return _execute_arm(
                job.key,
                job.action_name,
                job.arm,
                checkpoint,
                problem,
                case_id=job.case_id,
                cell_id=None,
                run_seed=job.run_seed,
                manifest_sha256=job.manifest_sha256,
                frozen_receipt=job.frozen_receipt_path if job.arm == "a0" or checkpoint.overlap_relation_count == 0 else None,
                output_root=job.output_root,
                receipt_path=job.receipt_path,
            )
    except BaseException as exc:
        _write_failure(job.output_root, job.key, exc)
        raise


class _CheckpointView:
    """Adapter so the U1 checkpoint helpers can serve S1 generator jobs."""

    def __init__(self, cell_id: str, run_seed: int, checkpoint_path: Path) -> None:
        self.cell_id = cell_id
        self.run_seed = run_seed
        self.checkpoint_path = checkpoint_path

    @property
    def key(self) -> str:
        return f"generator:{self.cell_id}:seed-{self.run_seed}:checkpoint"


def _run_generator_job(job: S1GeneratorJob) -> dict[str, Any]:
    try:
        with threadpool_limits(limits=1):
            view = _CheckpointView(job.cell_id, job.run_seed, job.checkpoint_path)
            if not job.checkpoint_path.is_file():
                write_generator_checkpoint(view)
            checkpoint, _recorded_truth = _load_generator_checkpoint(view)
            problem, truth = build_problem(job.cell_id, job.run_seed)
            if truth.ground_truth_hash != _recorded_truth:
                raise ValueError(f"{job.key} rebuilt generator ground truth drifted")
            return _execute_arm(
                job.key,
                job.host,
                job.arm,
                checkpoint,
                problem,
                case_id=None,
                cell_id=job.cell_id,
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
        "schema_version": "arac-upgrade-s1-leverage-sweep-protocol-v1",
        "candidate_id": "shared_patch_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "screen_seeds": [117, 123, 129, 135, 141],
        "arms": list(ARMS),
        "head_fraction": 0.20,
        "milestone_fes": list(MILESTONE_FES),
        "total_budget_fes": 3_000_000,
        "phase2_fes": 2_820_000,
        "u1_summary": "artifacts/upgrade_u1_host_reachability_v1/summary.json",
        "output_root": "artifacts/upgrade_s1_leverage_sweep_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"S1 protocol drifted: {key}")
    aob = protocol["aob_lane"]
    if set(aob["mapped_hosts"]) != {f"S{index}" for index in range(1, 7)} | {f"R{index}" for index in range(1, 7)}:
        raise ValueError("S1 AOB mapped hosts drifted")
    generator = protocol["generator_lane"]
    if tuple(generator["cells"]) != tuple(
        f"{topology}-{conflict}" for topology in ("chain", "hub", "pairs") for conflict in ("mild", "strong")
    ):
        raise ValueError("S1 generator cells drifted")
    eps = protocol["eps_reference"]
    if set(eps["aob"]) != set(aob["mapped_hosts"]) or set(eps["generator"]) != set(generator["cells"]):
        raise ValueError("S1 eps reference table does not cover every cell")
    if any(float(value) <= 0.0 for value in (*eps["aob"].values(), *eps["generator"].values())):
        raise ValueError("S1 eps reference values must be positive")
    for source in protocol["sources"]:
        if not (REPOSITORY_ROOT / source).is_file():
            raise ValueError(f"S1 source is missing: {source}")
    u1_summary = _load_json(REPOSITORY_ROOT / str(protocol["u1_summary"]))
    if not bool(u1_summary.get("gate_passed")):
        raise ValueError("S1 requires a passed U1 gate first")
    return protocol


def _manifest(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    pinned = {
        tree: dict(zip(("file_count", "sha256"), _tree_sha256(REPOSITORY_ROOT / tree)))
        for tree in sorted(protocol["pinned_trees"])
    }
    body = {
        "schema_version": "arac-upgrade-s1-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(Path(protocol_path).resolve()),
        "screen_seeds": list(protocol["screen_seeds"]),
        "source_sha256": {source: _sha256(REPOSITORY_ROOT / source) for source in sorted(protocol["sources"])},
        "pinned_trees": pinned,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _anytime_score(milestones: Sequence[Mapping[str, Any]], eps_ref: float) -> float:
    values = [
        math.log10(max(float(sample["best_error"]), eps_ref))
        for sample in milestones
        if int(sample["fes"]) in MILESTONE_FES
    ]
    if len(values) != len(MILESTONE_FES):
        raise ValueError("milestone samples are incomplete")
    return float(np.mean(values))


def _paired_bootstrap_ci(differences: Sequence[float], *, count: int = 10_000, seed: int = 20260823) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    samples = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _lane_judgment(pairs: Sequence[Mapping[str, Any]], *, eps_by_key: Mapping[str, float]) -> dict[str, Any]:
    ratios = []
    score_diffs = []
    rel_score_diffs = []
    for pair in pairs:
        eps = float(eps_by_key[pair["key"]])
        baseline = max(pair["a0_final_error"], eps)
        candidate = max(pair["a1_final_error"], eps)
        ratios.append(math.log(candidate / baseline))
        a0_score = _anytime_score(pair["a0_milestones"], eps)
        a1_score = _anytime_score(pair["a1_milestones"], eps)
        score_diffs.append(a1_score - a0_score)
        rel_score_diffs.append(a1_score / a0_score - 1.0)
    log_ratio_mean = float(np.mean(ratios))
    ratio_low, ratio_high = _paired_bootstrap_ci(ratios)
    rel_low, rel_high = _paired_bootstrap_ci(rel_score_diffs)
    score_low, score_high = _paired_bootstrap_ci(score_diffs)
    return {
        "pair_count": len(pairs),
        "final_error_geometric_mean_ratio": math.exp(log_ratio_mean),
        "final_error_ratio_ci95": [math.exp(ratio_low), math.exp(ratio_high)],
        "anytime_relative_mean_diff": float(np.mean(rel_score_diffs)),
        "anytime_relative_ci95": [rel_low, rel_high],
        "anytime_absolute_mean_diff": float(np.mean(score_diffs)),
        "anytime_absolute_ci95": [score_low, score_high],
        "final_nontax": math.exp(log_ratio_mean) <= 1.05 and math.exp(ratio_high) <= 1.05,
        "anytime_nontax": rel_low >= -0.05,
    }


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    seeds = [int(seed) for seed in protocol["screen_seeds"]]
    eps_aob = {key: float(value) for key, value in protocol["eps_reference"]["aob"].items()}
    eps_generator = {key: float(value) for key, value in protocol["eps_reference"]["generator"].items()}
    receipts: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for case_id, action in protocol["aob_lane"]["mapped_hosts"].items():
        for seed in seeds:
            for arm in ARMS:
                path = output_root / "arms" / "aob" / case_id / f"seed_{seed}" / f"{action}_{arm}.json"
                receipt = _load_json(path)
                if canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_hash"}) != receipt.get("receipt_hash"):
                    raise ValueError(f"S1 receipt hash drifted: {case_id}/{seed}/{arm}")
                receipts[("aob", case_id, seed, action, arm)] = receipt
    for cell_id in protocol["generator_lane"]["cells"]:
        for seed in seeds:
            for host in protocol["generator_lane"]["hosts"]:
                for arm in ARMS:
                    path = output_root / "arms" / "generator" / cell_id / f"seed_{seed}" / f"{host}_{arm}.json"
                    receipt = _load_json(path)
                    if canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_hash"}) != receipt.get("receipt_hash"):
                        raise ValueError(f"S1 receipt hash drifted: {cell_id}/{seed}/{host}/{arm}")
                    receipts[("generator", cell_id, seed, host, arm)] = receipt

    def pairs(lane: str, key: str, seed: int, host: str) -> dict[str, Any]:
        a0 = receipts[(lane, key, seed, host, "a0")]
        a1 = receipts[(lane, key, seed, host, "a1")]
        return {
            "key": key,
            "seed": seed,
            "host": host,
            "a0_final_error": a0["final_error"],
            "a1_final_error": a1["final_error"],
            "a0_route": a0["route"],
            "a1_route": a1["route"],
            "a0_milestones": a0["milestones"],
            "a1_milestones": a1["milestones"],
            "route_equal": a0["route"] == a1["route"],
            "a0_order_changed": bool(a0["order_records"]) and any(record["order_changed"] for record in a0["order_records"]),
            "a1_order_changed": bool(a1["order_records"]) and any(record["order_changed"] for record in a1["order_records"]),
            "relation_count": a0["leverage"]["relation_count"],
            "a1_bit_identical_to_a0": a1["final_error"] == a0["final_error"] and a1["action_result_hash"] == a0["action_result_hash"],
        }

    aob_pairs_by_host: dict[str, list[dict[str, Any]]] = {"ctp": [], "gcb": []}
    aob_case_rows = []
    for case_id, action in sorted(protocol["aob_lane"]["mapped_hosts"].items()):
        case_pairs = [pairs("aob", case_id, seed, action) for seed in seeds]
        aob_pairs_by_host[action].extend(case_pairs)
        aob_case_rows.append(
            {
                "case_id": case_id,
                "host": action,
                "relation_count": case_pairs[0]["relation_count"],
                "mechanism_silent": case_pairs[0]["relation_count"] == 0,
                "a0_identity_with_frozen": all(
                    all(field["match"] for field in receipts[("aob", case_id, seed, action, arm)]["identity"].values())
                    for seed in seeds
                    for arm in ARMS
                    if receipts[("aob", case_id, seed, action, arm)]["identity"] is not None
                ),
                "ov0_exact": all(pair["a1_bit_identical_to_a0"] for pair in case_pairs) if case_pairs[0]["relation_count"] == 0 else None,
                "judgment": _lane_judgment(case_pairs, eps_by_key=eps_aob),
            }
        )
    generator_rows = []
    generator_pairs_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell_id in protocol["generator_lane"]["cells"]:
        cell_pairs = [pairs("generator", cell_id, seed, host) for seed in seeds for host in protocol["generator_lane"]["hosts"]]
        generator_pairs_by_cell[cell_id] = cell_pairs
        generator_rows.append(
            {
                "cell_id": cell_id,
                "judgment": _lane_judgment(cell_pairs, eps_by_key=eps_generator),
                "exploratory_only": True,
            }
        )
    reachable_cases = [row for row in aob_case_rows if row["relation_count"] > 0]
    order_trace_checks = {
        "a0_orders_never_changed": not any(
            any(record["order_changed"] for record in receipts[key]["order_records"])
            for key in receipts
            if receipts[key]["order_records"]
        ),
        "a1_order_changed_where_leverage_exists": all(
            any(record["order_changed"] for record in receipts[("aob", case_id, seed, action, "a1")]["order_records"])
            for case_id, action in protocol["aob_lane"]["mapped_hosts"].items()
            for seed in seeds
            if receipts[("aob", case_id, seed, action, "a1")]["leverage"]["relation_count"] > 0
        ),
        "a1_order_unchanged_where_silent": all(
            not any(record["order_changed"] for record in receipts[("aob", case_id, seed, action, "a1")]["order_records"])
            for case_id, action in protocol["aob_lane"]["mapped_hosts"].items()
            for seed in seeds
            if receipts[("aob", case_id, seed, action, "a1")]["leverage"]["relation_count"] == 0
        ),
    }
    lane_checks = {}
    for host in ("ctp", "gcb"):
        host_pairs = aob_pairs_by_host[host]
        lane_checks[f"{host}_final_nontax"] = _lane_judgment(host_pairs, eps_by_key=eps_aob)["final_nontax"]
        lane_checks[f"{host}_anytime_nontax"] = _lane_judgment(host_pairs, eps_by_key=eps_aob)["anytime_nontax"]
    checks = {
        "coverage_complete": len(receipts) == 240,
        "a0_identity_with_frozen_all": all(row["a0_identity_with_frozen"] for row in aob_case_rows),
        "ov0_exact_no_tax": all(row["ov0_exact"] for row in aob_case_rows if row["mechanism_silent"]),
        "routes_unchanged_all_pairs": all(pair["route_equal"] for host in ("ctp", "gcb") for pair in aob_pairs_by_host[host]),
        "order_trace_rules": all(order_trace_checks.values()),
        **lane_checks,
    }
    body = {
        "schema_version": "arac-upgrade-s1-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "head_fraction": protocol["head_fraction"],
        "milestone_fes": list(MILESTONE_FES),
        "aob_case_rows": aob_case_rows,
        "generator_rows": generator_rows,
        "order_trace_checks": order_trace_checks,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "s2_screen_authorized": all(checks.values()),
        "performance_claim_authorized": False,
        "screen_note": "screen exposes reachability and regression only; directional generator results are exploratory",
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def _jobs(protocol: Mapping[str, Any], output_root: Path, manifest_sha256: str) -> list[S1AobJob | S1GeneratorJob]:
    seeds = [int(seed) for seed in protocol["screen_seeds"]]
    checkpoint_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["checkpoint_root"])).resolve()
    current_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["current_e2e_receipt_root"])).resolve()
    frozen_root = (REPOSITORY_ROOT / str(protocol["aob_lane"]["frozen_screen_receipt_root"])).resolve()
    aob_jobs = [
        S1AobJob(
            case_id=case_id,
            run_seed=seed,
            action_name=action,
            arm=arm,
            checkpoint_root=checkpoint_root,
            current_receipt_root=current_root,
            output_root=output_root,
            frozen_receipt_root=frozen_root,
            manifest_sha256=manifest_sha256,
        )
        for seed in seeds
        for case_id, action in sorted(protocol["aob_lane"]["mapped_hosts"].items())
        for arm in ARMS
    ]
    generator_jobs = [
        S1GeneratorJob(
            cell_id=cell,
            run_seed=seed,
            host=host,
            arm=arm,
            output_root=output_root,
            manifest_sha256=manifest_sha256,
        )
        for seed in seeds
        for cell in protocol["generator_lane"]["cells"]
        for host in protocol["generator_lane"]["hosts"]
        for arm in ARMS
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
            raise FileExistsError(f"S1 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("S1 manifest does not match the frozen protocol")
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
                runner = _run_aob_job if isinstance(job, S1AobJob) else _run_generator_job
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
        raise RuntimeError(f"S1 screen has {len(failures)} failed arms")
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
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_screen(args.protocol, resume=args.resume, workers=args.workers)
    print(json.dumps({"stage": "s1", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "s2_screen_authorized": summary["s2_screen_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["DEFAULT_PROTOCOL", "LeverageOrderShim", "MilestoneSampler", "head_slot_count", "load_protocol", "run_screen", "s1_order", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
