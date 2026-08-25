"""T0 structure certificate for shared_transaction_v1 (SCST v3.0 stage T0).

Runs the split-repair soft-RDDSM discovery (frozen SoftDsmConfig values,
``dsm_budget`` capped exactly like pairwise_edge_ctp_v1 P0) on the six
preregistered generator-v3 cells with five discovery seeds, tops the Phase-I
boundary up to exactly 180,000 FE with MMES, and certifies:

- the Stage-1 region-merge artifact is repaired (``merged_region_count == 0``
  and ``hyperedges_with_more_than_two_regions == 0`` on every activation
  cell/seed; the v5.1 P0 failure signature was 2/5 seeds at recall 16/24 with
  a ~200-variable merged region);
- pairwise shared-edge certificates achieve precision 1.0 and recall >= 0.9
  per seed (audit consumes generator truth offline only);
- the certificate region graph is a forest with maximum degree <= 2 and the
  block-level leverage vector has strictly positive variance (generator v3
  sparse-heterogeneity hard criteria);
- exact FE accounting (discovery + top-up == 180,000) and bit-level
  determinism (each job replays the discovery on a second ledger and must
  reproduce identical hashes).

T0 never evaluates optimization performance and never reads generator truth
at runtime; activation is preregistered for ``chain4-strong`` and
``pairs3-strong`` only, with hub3 and mild cells recorded but never executed
(the same activation policy as hyperedge_ctp_v1 H0).
"""

from __future__ import annotations

import argparse
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

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.soft_rddsm import SoftDsmConfig
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.upgrade.pairwise_edge_ctp_v1.p0_pairwise_evidence import (
    _offline_audit,
    _region_merge_rate,
    pairwise_certificates,
)
from experiments.upgrade.shared_patch_v1.conflicting_generator import relation_leverage
from experiments.upgrade.shared_patch_v1.conflicting_generator_v3 import (
    V3_CELL_IDS,
    build_v3_problem,
)
from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import GENERATOR_FREEZE
from experiments.upgrade.shared_transaction_v1.split_repair_discovery import (
    SplitRepairRecord,
    discover_hierarchical_soft_split_repair,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("t0_structure_certificate_protocol_v1.json")
SIDECAR_SCHEMA = "arac-upgrade-shared-transaction-t0-sidecar-v1"
CHECKPOINT_SCHEMA = "arac-upgrade-shared-transaction-t0-checkpoint-v1"
CANDIDATE_PHASE1_PROTOCOL = "arac-shared-transaction-phase1-v1"
DISCOVERY_SEEDS = (20270501, 20270502, 20270503, 20270504, 20270505)
ACTIVATION_CELLS = ("chain4-strong", "pairs3-strong")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
DSM_BUDGET_CAP = 110_000
PRECISION_REQUIRED = 1.0
RECALL_REQUIRED = 0.9
_TAIL_SEED_NAMESPACE = 0xA17E5D  # mirrors the frozen Phase-I tail namespace


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
        "schema_version": "arac-upgrade-shared-transaction-t0-protocol-v1",
        "candidate_id": "shared_transaction_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "cells": list(V3_CELL_IDS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "activation_cells": list(ACTIVATION_CELLS),
        "dsm_budget_cap": DSM_BUDGET_CAP,
        "precision_required": PRECISION_REQUIRED,
        "recall_required": RECALL_REQUIRED,
        "phase1_fes": PHASE1_FES,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "output_root": "artifacts/upgrade_shared_transaction_v1_t0_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"T0 protocol drifted: {key}")
    if protocol.get("generator_freeze") != GENERATOR_FREEZE:
        raise ValueError("T0 generator freeze drifted")
    if protocol.get("soft_config") != json.loads(json.dumps(SoftDsmConfig(dsm_budget=DSM_BUDGET_CAP).__dict__)):
        raise ValueError("T0 soft-RDDSM config drifted from the effective plan values")
    return protocol


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


def _repair_record_payload(record: SplitRepairRecord) -> dict[str, Any]:
    return {
        "block_index": record.block_index,
        "block_size": record.block_size,
        "first_group_inside": list(record.first_group_inside),
        "last_group_inside": list(record.last_group_inside),
        "third_group_inside": list(record.third_group_inside),
        "pieces": [list(piece) for piece in record.pieces],
        "adopted": record.adopted,
        "reason": record.reason,
    }


def _structure_hard_criteria(blocks: tuple[tuple[int, ...], ...], relations: tuple[RelationEvidence, ...]) -> dict[str, Any]:
    leverage = relation_leverage(blocks, relations)
    positive = [value for value in leverage if value > 0]
    block_count = len(blocks)
    edge_count = len(relations)
    degree: dict[int, set[int]] = {index: set() for index in range(block_count)}
    for relation in relations:
        degree[relation.left_block].add(relation.right_block)
        degree[relation.right_block].add(relation.left_block)
    max_degree = max((len(nodes) for nodes in degree.values()), default=0)
    connected = block_count > 1 and sum(1 for nodes in degree.values() if nodes) >= block_count - 1
    return {
        "block_count": block_count,
        "relation_count": edge_count,
        "leverage": list(leverage),
        "leverage_variance_positive": bool(np.var(np.asarray(leverage, dtype=float)) > 0.0),
        "positive_leverage_blocks": len(positive),
        "complete_graph": bool(block_count > 1 and edge_count == block_count * (block_count - 1) // 2),
        "max_relation_degree": max_degree,
        "max_relation_degree_within_bound": bool(max_degree <= 3),
        "relation_graph_connected": bool(connected),
    }


def _discovery_pass(
    problem: OptimizationProblem,
    run_seed: int,
    *,
    config: SoftDsmConfig,
    ledger: EvaluationLedger | None = None,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...], tuple[RelationEvidence, ...], Any, EvaluationLedger]:
    """One split-repair discovery, optionally on a caller-owned ledger."""

    if ledger is None:
        ledger = EvaluationLedger(problem, total_budget=TOTAL_BUDGET_FES)
    result = discover_hierarchical_soft_split_repair(
        problem, ledger, run_seed=run_seed, config=config
    )
    discovery = result.discovery
    evidence = discovery.evidence
    leaf_variables = [tuple(leaf.variables) for leaf in evidence.region_tree.leaves]
    leaf_variables_by_id = {leaf.node_id: tuple(leaf.variables) for leaf in evidence.region_tree.leaves}
    leaf_index = {leaf.node_id: index for index, leaf in enumerate(evidence.region_tree.leaves)}
    blocks = tuple(tuple(int(variable) for variable in members) for members in leaf_variables)
    relations = []
    for relation in evidence.region_relations:
        left = leaf_index.get(relation.left)
        right = leaf_index.get(relation.right)
        if left is None or right is None or left == right:
            continue
        relations.append(
            RelationEvidence(
                left_block=min(left, right),
                right_block=max(left, right),
                strength=float(relation.score),
                disagreement=0.0,
            )
        )
    certificates, rejected = pairwise_certificates(evidence)
    payload = {
        "blocks": [list(block) for block in blocks],
        "relations": [
            {
                "left_block": relation.left_block,
                "right_block": relation.right_block,
                "strength": relation.strength,
                "disagreement": relation.disagreement,
            }
            for relation in relations
        ],
        "certificates": certificates,
        "rejected": rejected,
        "hyperedges": [
            {"variable": hyperedge.variable, "regions": list(hyperedge.regions)}
            for hyperedge in evidence.resolved_hyperedges
        ],
        "shared_candidates": [int(variable) for variable in discovery.shared_candidates],
        "repair_records": [_repair_record_payload(record) for record in result.repair_records],
        "last_endpoint_refinement_fes": int(result.last_endpoint_refinement_fes),
        "target_validation_fes": int(result.target_validation_fes),
        "target_validation": [
            [int(variable), list(raw), list(validated)]
            for variable, raw, validated in result.target_validation
        ],
        "discovery_fes": int(ledger.count),
    }
    return payload, blocks, tuple(relations), leaf_variables_by_id, ledger


def run_candidate_phase1_scst(cell_id: str, run_seed: int) -> dict[str, Any]:
    """Split-repair discovery + exact 180k MMES top-up + checkpoint + sidecar."""

    problem, truth = build_v3_problem(
        cell_id,
        run_seed,
        conditioning=GENERATOR_FREEZE["conditioning"],
        shared_width=int(GENERATOR_FREEZE["shared_width"]),
        linkage_lambda=float(GENERATOR_FREEZE["linkage_lambda"]),
    )
    config = SoftDsmConfig(dsm_budget=DSM_BUDGET_CAP)
    ledger = EvaluationLedger(problem, total_budget=TOTAL_BUDGET_FES)
    primary, blocks, relations, leaf_variables_by_id, ledger = _discovery_pass(
        problem, run_seed, config=config, ledger=ledger
    )

    replay, _replay_blocks, _replay_relations, _replay_leaves, _replay_ledger = _discovery_pass(
        problem, run_seed, config=config
    )
    replay_match = {
        "blocks": primary["blocks"] == replay["blocks"],
        "relations": primary["relations"] == replay["relations"],
        "certificates": primary["certificates"] == replay["certificates"],
        "rejected": primary["rejected"] == replay["rejected"],
        "shared_candidates": primary["shared_candidates"] == replay["shared_candidates"],
        "repair_records": primary["repair_records"] == replay["repair_records"],
        "discovery_fes": primary["discovery_fes"] == replay["discovery_fes"],
    }

    discovery_fes = ledger.count
    if discovery_fes > PHASE1_FES:
        raise RuntimeError(f"{cell_id}/{run_seed} discovery consumed {discovery_fes} FE, exceeding the Phase-I boundary")
    if PHASE1_FES - discovery_fes > 0:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=int(run_seed) ^ _TAIL_SEED_NAMESPACE,
            budget_fes=PHASE1_FES - discovery_fes,
            population_size=24,
            restart=False,
        )
    if ledger.count != PHASE1_FES:
        raise RuntimeError(f"{cell_id}/{run_seed} candidate Phase-I did not stop at the frozen boundary")

    checkpoint = PhaseCheckpoint(
        protocol=CANDIDATE_PHASE1_PROTOCOL,
        run_seed=int(run_seed),
        total_budget_fes=TOTAL_BUDGET_FES,
        phase1_fes=PHASE1_FES,
        incumbent=tuple(float(value) for value in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=(
            "split_repair_discovery_fes",
            "split_repair_topup_fes",
            "split_repair_certificate_count",
            "split_repair_shared_candidate_count",
            "split_repair_relation_count",
        ),
        feature_values=(
            float(discovery_fes),
            float(PHASE1_FES - discovery_fes),
            float(len(primary["certificates"])),
            float(len(primary["shared_candidates"])),
            float(len(relations)),
        ),
        blocks=blocks,
        relations=relations,
    )

    certificates = primary["certificates"]
    audit = _offline_audit(certificates, leaf_variables_by_id, truth)
    region_merge = _region_merge_rate(blocks)
    hard = _structure_hard_criteria(blocks, relations)
    certificate_graph: dict[int, set[int]] = {}
    for certificate in certificates:
        left, right = certificate["region_a"], certificate["region_b"]
        certificate_graph.setdefault(left, set()).add(right)
        certificate_graph.setdefault(right, set()).add(left)

    def _acyclic(graph: dict[int, set[int]]) -> bool:
        seen: set[int] = set()
        for start in graph:
            if start in seen:
                continue
            seen.add(start)
            stack = [(start, None)]
            while stack:
                node, parent = stack.pop()
                for neighbor in graph[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in seen:
                        return False
                    seen.add(neighbor)
                    stack.append((neighbor, node))
        return True

    sidecar = {
        "sidecar_schema": SIDECAR_SCHEMA,
        "candidate_id": "shared_transaction_v1",
        "cell_id": cell_id,
        "run_seed": int(run_seed),
        "soft_config": json.loads(json.dumps(config.__dict__)),
        "signature_config": {"probe_count": 12, "probe_size": 16, "step": 0.25},
        "phase1_budget_breakdown": {
            "discovery_fes": discovery_fes,
            "topup_fes": PHASE1_FES - discovery_fes,
            "phase1_fes": PHASE1_FES,
        },
        "pairwise_certificates": certificates,
        "rejected_hyperedges": primary["rejected"],
        "hyperedges": primary["hyperedges"],
        "shared_candidates": primary["shared_candidates"],
        "repair_records": primary["repair_records"],
        "last_endpoint_refinement_fes": primary["last_endpoint_refinement_fes"],
        "target_validation_fes": primary["target_validation_fes"],
        "target_validation": primary["target_validation"],
        "replay_match": replay_match,
        "region_merge": region_merge,
        "structure_hard_criteria": hard,
        "certificate_graph": {
            "node_count": len(certificate_graph),
            "max_degree": max((len(nodes) for nodes in certificate_graph.values()), default=0),
            "forest": _acyclic(certificate_graph),
        },
        "offline_audit": audit,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "ground_truth_hash": truth.ground_truth_hash,
        "truth_used_for_audit_only": True,
    }
    sidecar["evidence_hash"] = canonical_sha256(
        {key: value for key, value in sidecar.items() if key != "evidence_hash"}
    )
    return {
        "checkpoint": checkpoint,
        "sidecar": sidecar,
        "audit": audit,
        "truth": truth,
    }


@dataclass(frozen=True)
class T0Job:
    cell_id: str
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.cell_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def key(self) -> str:
        return f"t0:{self.cell_id}:seed-{self.run_seed}"


def _run_t0_job(job: T0Job) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            result = run_candidate_phase1_scst(job.cell_id, job.run_seed)
            checkpoint: PhaseCheckpoint = result["checkpoint"]
            sidecar = result["sidecar"]
            audit = sidecar["offline_audit"]
            body = {
                "schema_version": CHECKPOINT_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "cell_id": job.cell_id,
                "run_seed": job.run_seed,
                "phase1_protocol": checkpoint.protocol,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "block_count": len(checkpoint.blocks),
                "relation_count": len(checkpoint.relations),
                "audit_summary": {
                    "certificate_count": audit["certificate_count"],
                    "true_positive_count": audit["true_positive_count"],
                    "precision": audit["precision"],
                    "recall": audit["recall"],
                    "rejected_hyperedge_count": len(sidecar["rejected_hyperedges"]),
                    "hyperedges_with_more_than_two_regions": sum(
                        1 for item in sidecar["rejected_hyperedges"]
                        if item["reason"] == "not_exactly_two_regions"
                    ),
                    "merged_region_count": sidecar["region_merge"]["merged_region_count"],
                    "certificate_graph_max_degree": sidecar["certificate_graph"]["max_degree"],
                    "certificate_graph_forest": sidecar["certificate_graph"]["forest"],
                    "target_validation_fes": sidecar["target_validation_fes"],
                    "adopted_split_count": sum(
                        1 for record in sidecar["repair_records"] if record["adopted"]
                    ),
                },
                "runtime": runtime,
                "sidecar": sidecar,
                "checkpoint": checkpoint.payload(),
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            body["receipt_hash"] = canonical_sha256(body)
            _write_json(job.checkpoint_path, body)
            return body
    except BaseException as exc:
        _write_json(
            job.output_root / "failures" / f"{job.key.replace(':', '_')}.json",
            {
                "schema_version": "arac-upgrade-shared-transaction-t0-failure-v1",
                "key": job.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def _seed_passed(summary: Mapping[str, Any], replay_match: Mapping[str, bool], hard: Mapping[str, Any]) -> bool:
    return bool(
        summary["precision"] >= PRECISION_REQUIRED
        and summary["recall"] >= RECALL_REQUIRED
        and summary["certificate_count"] >= 1
        and summary["rejected_hyperedge_count"] == 0
        and summary["hyperedges_with_more_than_two_regions"] == 0
        and summary["merged_region_count"] == 0
        and summary["certificate_graph_max_degree"] <= 2
        and summary["certificate_graph_forest"]
        and all(replay_match.values())
        and hard["leverage_variance_positive"]
        and not hard["complete_graph"]
        and hard["max_relation_degree_within_bound"]
    )


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cell_rows = []
    for cell_id in protocol["cells"]:
        seed_rows = []
        for seed in protocol["discovery_seeds"]:
            receipt = _load_json(output_root / "checkpoints" / cell_id / f"seed_{seed}" / "checkpoint.json")
            if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
                raise ValueError(f"T0 receipt hash drifted: {cell_id}/{seed}")
            summary = receipt["audit_summary"]
            sidecar = receipt["sidecar"]
            breakdown = sidecar["phase1_budget_breakdown"]
            seed_rows.append(
                {
                    "seed": seed,
                    "phase1_exact": bool(
                        breakdown["discovery_fes"] + breakdown["topup_fes"] == PHASE1_FES
                        and breakdown["phase1_fes"] == PHASE1_FES
                    ),
                    "precision": summary["precision"],
                    "recall": summary["recall"],
                    "certificate_count": summary["certificate_count"],
                    "rejected_hyperedge_count": summary["rejected_hyperedge_count"],
                    "hyperedges_with_more_than_two_regions": summary["hyperedges_with_more_than_two_regions"],
                    "merged_region_count": summary["merged_region_count"],
                    "certificate_graph_max_degree": summary["certificate_graph_max_degree"],
                    "certificate_graph_forest": summary["certificate_graph_forest"],
                    "adopted_split_count": summary["adopted_split_count"],
                    "target_validation_fes": summary["target_validation_fes"],
                    "discovery_fes": sidecar["phase1_budget_breakdown"]["discovery_fes"],
                    "seed_passed": _seed_passed(summary, sidecar["replay_match"], sidecar["structure_hard_criteria"]),
                }
            )
        cell_rows.append(
            {
                "cell_id": cell_id,
                "preregistered_activation": cell_id in protocol["activation_cells"],
                "all_seeds_passed": all(row["seed_passed"] for row in seed_rows),
                "seed_rows": seed_rows,
                "activated": bool(cell_id in protocol["activation_cells"] and all(row["seed_passed"] for row in seed_rows)),
            }
        )
    checks = {
        "coverage_complete": len(cell_rows) == 6 and all(len(row["seed_rows"]) == len(DISCOVERY_SEEDS) for row in cell_rows),
        "activation_cells_certified": all(
            row["all_seeds_passed"] for row in cell_rows if row["preregistered_activation"]
        ),
        "no_uncertified_activation": all(not row["activated"] for row in cell_rows if not row["preregistered_activation"]),
        "exact_phase1_boundary": all(
            row["phase1_exact"] for cell_row in cell_rows for row in cell_row["seed_rows"]
        ),
    }
    body = {
        "schema_version": "arac-upgrade-shared-transaction-t0-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "generator_freeze": GENERATOR_FREEZE,
        "cell_rows": cell_rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "t1_authorized": all(checks.values()),
        "truth_usage_note": "generator truth was consumed offline for T0 certification only; runtime paths never read it",
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("T0 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-shared-transaction-t0-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
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
    jobs = [
        T0Job(cell_id=cell, run_seed=int(seed), output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for cell in protocol["cells"]
        for seed in protocol["discovery_seeds"]
    ]
    for job in jobs:
        if not job.checkpoint_path.is_file():
            _run_t0_job(job)
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
    print(json.dumps({"stage": "t0", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "t1_authorized": summary["t1_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = [
    "ACTIVATION_CELLS",
    "DISCOVERY_SEEDS",
    "load_protocol",
    "run_candidate_phase1_scst",
    "run_stage",
    "summarize",
]


if __name__ == "__main__":
    raise SystemExit(main())
