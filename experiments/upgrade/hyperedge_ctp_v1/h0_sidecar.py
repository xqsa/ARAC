"""H0 explicit-evidence sidecar for hyperedge_ctp_v1 (line B, stage H0).

Runs the soft-RDDSM discovery branch (frozen configuration, identical to
``SoftDsmConfig`` defaults) on the six preregistered generator-v3 cells with
three discovery seeds, tops the Phase-I boundary up to exactly 180,000 FE
with MMES from the discovery incumbent, and produces:

- a candidate ``PhaseCheckpoint`` that is still a plain partition (primary
  blocks + region relations mapped onto block indices); the frozen
  ``PhaseCheckpoint`` schema is never extended;
- a read-only ``OverlapEvidenceSidecar`` artifact carrying the true
  multi-membership evidence (``resolved_hyperedges``) and its hashes;
- an offline truth audit (precision/recall against the planted links) used
  only for H0 certification - the runtime never reads generator truth.

Activation is preregistered for ``chain4-strong`` and ``pairs3-strong``
only; hub3 and mild cells are recorded but never executed.  A cell
activates only when every discovery seed satisfies precision >= 0.95,
recall >= 0.75, at least one resolved hyperedge, every certified hyperedge
spans exactly two regions, and the region graph induced by certified
hyperedges is a forest with maximum degree <= 2.
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

from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence, canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.upgrade.shared_patch_v1.conflicting_generator_v3 import (
    V3_CELL_IDS,
    build_v3_problem,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("h0_sidecar_protocol_v1.json")
SIDECAR_SCHEMA = "arac-upgrade-hyperedge-overlap-sidecar-v1"
CHECKPOINT_SCHEMA = "arac-upgrade-hyperedge-h0-checkpoint-v1"
CANDIDATE_PHASE1_PROTOCOL = "arac-hyperedge-sidecar-phase1-v1"
GENERATOR_FREEZE = {
    "generator_protocol": "arac-upgrade-conflicting-generator-v3",
    "conditioning": "linked-elliptic",
    "shared_width": 8,
    "linkage_lambda": 2.0,
    "note": "selected as the best configuration of the preregistered v3 grid (26/30 preflight passes; zero-optimum convention enforced) and frozen here before any H0 run",
}
DISCOVERY_SEEDS = (20270101, 20270102, 20270103)
ACTIVATION_CELLS = ("chain4-strong", "pairs3-strong")
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
_TAIL_SEED_NAMESPACE = 0xA17E5D  # mirrors the frozen Phase-I tail namespace
PRECISION_MIN = 0.95
RECALL_MIN = 0.75


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
        "schema_version": "arac-upgrade-hyperedge-h0-protocol-v1",
        "candidate_id": "hyperedge_ctp_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "cells": list(V3_CELL_IDS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "activation_cells": list(ACTIVATION_CELLS),
        "phase1_fes": PHASE1_FES,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "output_root": "artifacts/upgrade_hyperedge_ctp_v1_h0_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"H0 protocol drifted: {key}")
    if protocol.get("generator_freeze") != GENERATOR_FREEZE:
        raise ValueError("H0 generator freeze drifted")
    if protocol.get("soft_config") != json.loads(json.dumps(SoftDsmConfig().__dict__)):
        raise ValueError("H0 soft-RDDSM config drifted from the plan values")
    return protocol


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


def _planted_owner_of(variable: int) -> int:
    return variable // 100


def _region_planted_block(region_variables: Sequence[int]) -> int:
    owners = [_planted_owner_of(variable) for variable in region_variables]
    values, counts = np.unique(np.asarray(owners), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _hyperedge_audit(evidence, leaf_variables_by_id: Mapping[int, Sequence[int]], truth) -> dict[str, Any]:
    """Certify two-region hyperedges against the planted owner pairs.

    A certified hyperedge is a true positive iff its variable is planted
    shared and the two spanned regions map exactly to the variable's two
    planted owner blocks.  Hyperedges spanning more than two regions are
    counted separately and never certified.
    """

    planted_owners = {variable: (left, right) for variable, left, right in truth.shared_owner_pairs}
    certified = []
    three_region = 0
    for hyperedge in evidence.resolved_hyperedges:
        regions = hyperedge.regions
        if len(regions) != 2:
            three_region += 1
            continue
        left_block = _region_planted_block(leaf_variables_by_id[regions[0]])
        right_block = _region_planted_block(leaf_variables_by_id[regions[1]])
        variable = hyperedge.variable
        expected = planted_owners.get(variable)
        certified.append(
            {
                "variable": int(variable),
                "regions": [int(region) for region in regions],
                "planted_blocks": [left_block, right_block],
                "variable_planted_shared": expected is not None,
                "correct": bool(
                    expected is not None
                    and left_block != right_block
                    and {left_block, right_block} == set(expected)
                ),
            }
        )
    true_positive = sum(1 for row in certified if row["correct"])
    precision = (true_positive / len(certified)) if certified else 0.0
    recall = (true_positive / len(truth.shared_variables)) if truth.shared_variables else 0.0
    region_graph: dict[int, set[int]] = {}
    for row in certified:
        left, right = row["regions"]
        region_graph.setdefault(left, set()).add(right)
        region_graph.setdefault(right, set()).add(left)

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

    max_degree = max((len(nodes) for nodes in region_graph.values()), default=0)
    return {
        "certified_hyperedge_count": len(certified),
        "hyperedges_with_more_than_two_regions": three_region,
        "true_positive_count": true_positive,
        "precision": precision,
        "recall": recall,
        "region_graph": {str(node): sorted(nodes) for node, nodes in sorted(region_graph.items())},
        "region_graph_max_degree": max_degree,
        "region_graph_forest": _acyclic(region_graph),
        "certified": certified,
    }


def run_candidate_phase1(cell_id: str, run_seed: int, *, config: SoftDsmConfig | None = None) -> dict[str, Any]:
    """Soft-RDDSM discovery + exact 180k MMES top-up + sidecar + checkpoint."""

    problem, truth = build_v3_problem(
        cell_id,
        run_seed,
        conditioning=GENERATOR_FREEZE["conditioning"],
        shared_width=int(GENERATOR_FREEZE["shared_width"]),
        linkage_lambda=float(GENERATOR_FREEZE["linkage_lambda"]),
    )
    effective_config = SoftDsmConfig() if config is None else config
    ledger = EvaluationLedger(problem, total_budget=TOTAL_BUDGET_FES)
    discovery = discover_hierarchical_soft(problem, ledger, run_seed=run_seed, config=effective_config)
    discovery_fes = ledger.count
    if discovery_fes > PHASE1_FES:
        raise RuntimeError(f"{cell_id}/{run_seed} discovery consumed {discovery_fes} FE, exceeding the Phase-I boundary")
    if ledger.remaining < PHASE1_FES - discovery_fes:
        raise RuntimeError("ledger budget arithmetic drifted")
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
    checkpoint = PhaseCheckpoint(
        protocol=CANDIDATE_PHASE1_PROTOCOL,
        run_seed=int(run_seed),
        total_budget_fes=TOTAL_BUDGET_FES,
        phase1_fes=PHASE1_FES,
        incumbent=tuple(float(value) for value in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=(
            "sidecar_discovery_fes",
            "sidecar_topup_fes",
            "sidecar_hyperedge_count",
            "sidecar_shared_candidate_count",
            "sidecar_relation_count",
        ),
        feature_values=(
            float(discovery_fes),
            float(PHASE1_FES - discovery_fes),
            float(len(evidence.resolved_hyperedges)),
            float(len(discovery.shared_candidates)),
            float(len(relations)),
        ),
        blocks=blocks,
        relations=tuple(relations),
    )
    audit = _hyperedge_audit(evidence, leaf_variables_by_id, truth)
    sidecar = {
        "sidecar_schema": SIDECAR_SCHEMA,
        "candidate_id": "hyperedge_ctp_v1",
        "cell_id": cell_id,
        "run_seed": int(run_seed),
        "soft_config": json.loads(json.dumps(effective_config.__dict__)),
        "signature_config": {"probe_count": 12, "probe_size": 16, "step": 0.25},
        "phase1_budget_breakdown": {"discovery_fes": discovery_fes, "topup_fes": PHASE1_FES - discovery_fes, "phase1_fes": PHASE1_FES},
        "primary_blocks": [list(block) for block in blocks],
        "region_relation_map": [
            {
                "leaf_left": relation.left,
                "leaf_right": relation.right,
                "block_left": leaf_index[relation.left],
                "block_right": leaf_index[relation.right],
                "score": relation.score,
                "stability": relation.stability,
            }
            for relation in evidence.region_relations
        ],
        "resolved_hyperedges": [
            {
                "variable": hyperedge.variable,
                "regions": list(hyperedge.regions),
                "evidence_count": len(hyperedge.evidence),
            }
            for hyperedge in evidence.resolved_hyperedges
        ],
        "active_hyperedge_ids": [
            index
            for index, hyperedge in enumerate(evidence.resolved_hyperedges)
            if len(hyperedge.regions) == 2
        ],
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "truth_audit": audit,
        "ground_truth_hash": truth.ground_truth_hash,
        "truth_used_for_audit_only": True,
    }
    sidecar["evidence_hash"] = canonical_sha256(
        {key: value for key, value in sidecar.items() if key != "evidence_hash"}
    )
    return {
        "checkpoint": checkpoint,
        "sidecar": sidecar,
        "discovery": discovery,
        "audit": audit,
        "truth": truth,
    }


@dataclass(frozen=True)
class H0Job:
    cell_id: str
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / self.cell_id / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def key(self) -> str:
        return f"h0:{self.cell_id}:seed-{self.run_seed}"


def _run_h0_job(job: H0Job) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            result = run_candidate_phase1(job.cell_id, job.run_seed)
            checkpoint: PhaseCheckpoint = result["checkpoint"]
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
                    key: result["audit"][key]
                    for key in (
                        "certified_hyperedge_count",
                        "hyperedges_with_more_than_two_regions",
                        "true_positive_count",
                        "precision",
                        "recall",
                        "region_graph_max_degree",
                        "region_graph_forest",
                    )
                },
                "runtime": runtime,
                "sidecar": result["sidecar"],
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
                "schema_version": "arac-upgrade-hyperedge-h0-failure-v1",
                "key": job.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cell_rows = []
    for cell_id in protocol["cells"]:
        seed_rows = []
        for seed in protocol["discovery_seeds"]:
            receipt = _load_json(output_root / "checkpoints" / cell_id / f"seed_{seed}" / "checkpoint.json")
            if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
                raise ValueError(f"H0 receipt hash drifted: {cell_id}/{seed}")
            summary = receipt["audit_summary"]
            seed_rows.append(
                {
                    "seed": seed,
                    "precision": summary["precision"],
                    "recall": summary["recall"],
                    "certified_hyperedge_count": summary["certified_hyperedge_count"],
                    "hyperedges_with_more_than_two_regions": summary["hyperedges_with_more_than_two_regions"],
                    "region_graph_max_degree": summary["region_graph_max_degree"],
                    "region_graph_forest": summary["region_graph_forest"],
                    "seed_passed": bool(
                        summary["precision"] >= PRECISION_MIN
                        and summary["recall"] >= RECALL_MIN
                        and summary["certified_hyperedge_count"] >= 1
                        and summary["hyperedges_with_more_than_two_regions"] == 0
                        and summary["region_graph_max_degree"] <= 2
                        and summary["region_graph_forest"]
                    ),
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
        "coverage_complete": len(cell_rows) == 6 and all(len(row["seed_rows"]) == 3 for row in cell_rows),
        "activation_cells_certified": all(
            row["all_seeds_passed"] for row in cell_rows if row["preregistered_activation"]
        ),
        "no_uncertified_activation": all(not row["activated"] for row in cell_rows if not row["preregistered_activation"]),
    }
    body = {
        "schema_version": "arac-upgrade-hyperedge-h0-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "generator_freeze": GENERATOR_FREEZE,
        "cell_rows": cell_rows,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "h1_authorized": all(checks.values()),
        "truth_usage_note": "generator truth was consumed offline for H0 certification only; runtime paths never read it",
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("H0 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-hyperedge-h0-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"H0 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("H0 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = [
        H0Job(cell_id=cell, run_seed=int(seed), output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for cell in protocol["cells"]
        for seed in protocol["discovery_seeds"]
    ]
    for job in jobs:
        if not job.checkpoint_path.is_file():
            _run_h0_job(job)
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
    print(json.dumps({"stage": "h0", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "h1_authorized": summary["h1_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = [
    "ACTIVATION_CELLS",
    "DISCOVERY_SEEDS",
    "GENERATOR_FREEZE",
    "load_protocol",
    "run_candidate_phase1",
    "run_stage",
    "summarize",
]


if __name__ == "__main__":
    raise SystemExit(main())
