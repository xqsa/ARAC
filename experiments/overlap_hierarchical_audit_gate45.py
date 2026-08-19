"""Gate 45: identifiability audit of the five-stage hierarchical discovery.

Runs discovery-only (no incumbent optimization) on the 1000-D sparse-overlap
grid and on AOB cases, reporting structure-recovery quality against the truth
group memberships (offline audit use only):
- region relations spanning distinct truth groups;
- conditional interaction precision (does an interacting variable truly
  co-membership the target leaf's block?);
- per-component mode assignment;
- stage budget reconciliation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.evidence.hierarchical_discovery import (
    HierarchicalDiscoveryConfig,
    discover_hierarchical,
)
from arac.runtime.ledger import EvaluationLedger
from experiments.overlap_arac_gate29_screening import (
    ACTIVE_DIMENSION,
    BOUNDS,
    MAX_GROUP_SIZE,
    MIN_GROUP_SIZE,
)

AUDIT_SEED = 20260841
DISCOVERY_BUDGET = 80_000
OUTPUT_SCHEMA = "arac-overlap-hierarchical-audit-gate45-v1"


def _grid_cells() -> list[dict[str, object]]:
    return [
        {"mode": mode, "topology": topology, "overlap_budget": overlap}
        for mode in ("conflicting",)
        for topology in ("chain", "star", "random")
        for overlap in (3, 6)
    ]


def _grid_problem(cell: dict[str, object]):
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        active = np.asarray(truth.evaluate(batch[:, :ACTIVE_DIMENSION]), dtype=float)
        tail = np.sum((batch[:, ACTIVE_DIMENSION:] / BOUNDS) ** 2, axis=1)
        result = active + tail
        return float(result[0]) if rows.ndim == 1 else result

    _, truth = build_overlap_problem(
        ACTIVE_DIMENSION,
        overlap_budget=int(cell["overlap_budget"]),
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        num_groups=4,
        base_function="rastrigin",
        conflict_mode=str(cell["mode"]),
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        seed=AUDIT_SEED,
        topology=str(cell["topology"]),
        interaction_strength=0.25,
    )
    problem = OptimizationProblem(
        objective=objective,
        dimension=1000,
        lower_bounds=(-BOUNDS,) * 1000,
        upper_bounds=(BOUNDS,) * 1000,
    )
    return problem, truth


def _audit_case(kind: str, label: str, problem, truth_groups) -> dict[str, object]:
    config = HierarchicalDiscoveryConfig(incumbent_min=0)
    ledger = EvaluationLedger(problem, DISCOVERY_BUDGET)
    result = discover_hierarchical(problem, ledger, run_seed=AUDIT_SEED, config=config)
    evidence = result.evidence
    tree = evidence.region_tree
    truth_of = {}
    for index, group in enumerate(truth_groups):
        for variable in group:
            truth_of[variable] = index

    relations = evidence.region_relations
    cross_group = 0
    for relation in relations:
        left_originals = {
            truth_of[v] for v in tree.require_leaf(relation.left).variables if v in truth_of
        }
        right_originals = {
            truth_of[v] for v in tree.require_leaf(relation.right).variables if v in truth_of
        }
        if left_originals and right_originals and not (left_originals & right_originals):
            cross_group += 1

    interactions = evidence.variable_region_interactions
    correct = 0
    for interaction in interactions:
        target_group = {
            truth_of[v] for v in tree.require_leaf(interaction.target_region).variables if v in truth_of
        }
        if truth_of.get(interaction.variable) in target_group or truth_of.get(interaction.variable) is None:
            correct += 1
    precision = correct / len(interactions) if interactions else None

    modes = [mode for _, mode in evidence.per_component_mode]
    truth_shared = {
        variable
        for variable, groups in (
            (v, [g for g in truth_groups if v in g]) for g in [] for v in []
        ) for variable, groups in []
    } if False else {
        variable
        for variable in range(1000)
        if sum(1 for group in truth_groups if variable in group) > 1
    }
    recovered_shared = {h.variable for h in evidence.resolved_hyperedges}
    shared_recall = (
        len(recovered_shared & truth_shared) / len(truth_shared) if truth_shared else None
    )
    shared_precision = (
        len(recovered_shared & truth_shared) / len(recovered_shared) if recovered_shared else None
    )
    return {
        "kind": kind,
        "label": label,
        "relation_count": len(relations),
        "cross_group_relations": cross_group,
        "interaction_count": len(interactions),
        "interaction_precision": precision,
        "hyperedge_count": len(evidence.resolved_hyperedges),
        "shared_recall": shared_recall,
        "shared_precision": shared_precision,
        "modes": modes,
        "leaf_count": len(tree.leaves),
        "level_budgets": dict(result.level_budgets),
        "ledger_count": ledger.count,
        "budget_reconciled": sum(dict(result.level_budgets).values()) == ledger.count,
    }


def _run_grid_cell(cell: dict[str, object]) -> dict[str, object]:
    problem, truth = _grid_problem(cell)
    groups = [set(group) for group in truth.structure.groups]
    return _audit_case("grid", f"{cell['topology']}/ov{cell['overlap_budget']}", problem, groups)


def _run_aob_case(case_id: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    # AOB construction files are offline audits only; here we pass an empty
    # truth group list and observe the runtime behaviour of the pipeline.
    return _audit_case("aob", case_id, problem, [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_hierarchical_audit_gate45/audit.json"),
    )
    args = parser.parse_args()
    jobs = [("grid", cell) for cell in _grid_cells()] + [("aob", case) for case in ("R1", "S1", "E1")]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for kind, payload in jobs:
            future = executor.submit(
                _run_grid_cell if kind == "grid" else _run_aob_case,
                payload if kind == "grid" else payload,
            )
            futures[future] = (kind, payload)
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['kind']}/{row['label']}: relations={row['relation_count']} "
                f"cross={row['cross_group_relations']} interactions={row['interaction_count']} "
                f"precision={row['interaction_precision']} modes={row['modes']} "
                f"budget_ok={row['budget_reconciled']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["kind"], row["label"]))
    grid_rows = [row for row in rows if row["kind"] == "grid"]
    aob_rows = [row for row in rows if row["kind"] == "aob"]
    checks = {
        "budget_reconciled_all": all(row["budget_reconciled"] for row in rows),
        "grid_relations_present": all(row["relation_count"] > 0 for row in grid_rows),
        "grid_cross_group_relations_any": sum(row["cross_group_relations"] for row in grid_rows) > 0,
        "aob_completed_without_failure": len(aob_rows) == 3
        and all(row["budget_reconciled"] for row in aob_rows),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "audit_seed": AUDIT_SEED,
            "discovery_budget": DISCOVERY_BUDGET,
            "config": asdict(HierarchicalDiscoveryConfig(incumbent_min=0)),
        },
        "checks": checks,
        "audit_passed": all(checks.values()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"checks": checks}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
