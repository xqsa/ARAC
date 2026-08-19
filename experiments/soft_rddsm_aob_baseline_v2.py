"""AOB baseline re-audit of the soft-RDDSM branch after the dataflow fix.

2026-08-15: the zero-recall result of the six shared-detection rounds was
traced to a structural dataflow bug (mutually exclusive blocks + a
membership-count criterion that no run could satisfy).  The v3 protocol adds
complete-candidate coarse grouping, small-region refinement, complete
intersection removal, and two-sided residual confirmation.  This script
re-measures the branch on AOB against the construction truth (offline audit
use only, same policy as the SOTA oracle audit):

- shared-variable recall/precision vs truth overlap members;
- block structure quality (fragmentation, coverage, contamination);
- exact stage-level FE reconciliation within the 180k Phase-I contract.
"""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import statistics
from pathlib import Path

import numpy as np
import yaml

from arac.benchmarks.aob import AobBenchmark
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.ledger import EvaluationLedger

AUDIT_SEED = 20260845
PHASE1_BUDGET_FES = 180_000
OUTPUT_SCHEMA = "arac-soft-rddsm-aob-baseline-v3"
DEFAULT_CASES = ("R1", "R2", "R6", "E1", "S5", "A3")


def _truth_groups(data_root: Path, function_id: int) -> list[frozenset[int]]:
    """Rebuild the AOB chain-of-overlapping-groups truth from metadata."""

    stem = f"F{function_id}"
    with (data_root / f"{stem}-info.txt").open(encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    sizes = np.loadtxt(data_root / f"{stem}-s.txt").astype(int).tolist()
    permutation = np.loadtxt(data_root / f"{stem}-p.txt", delimiter=",").astype(int).tolist()
    overlap = int(info["overlap_degree"])
    groups: list[frozenset[int]] = []
    begin = 0
    for index, size in enumerate(sizes):
        end = begin + int(size)
        groups.append(frozenset(int(v) - 1 for v in permutation[begin:end]))
        if index != len(sizes) - 1:
            begin = end - overlap
    return groups


def audit_case(case_id: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    truth_groups = _truth_groups(AobBenchmark().data_root, int(case_id[1]))
    membership = np.zeros(problem.dimension, dtype=int)
    for group in truth_groups:
        membership[sorted(group)] += 1
    truth_shared = frozenset(int(v) for v in np.nonzero(membership > 1)[0])

    ledger = EvaluationLedger(problem, PHASE1_BUDGET_FES)
    result = discover_hierarchical_soft(problem, ledger, run_seed=AUDIT_SEED)
    recovered = frozenset(result.shared_candidates)
    evidence = result.evidence

    blocks = [frozenset(block) for block in result.blocks]
    covered = frozenset().union(*blocks) if blocks else frozenset()
    fragments_per_group = [
        sum(1 for block in blocks if block & group) for group in truth_groups
    ]
    return {
        "case_id": case_id,
        "truth_group_count": len(truth_groups),
        "truth_group_sizes": sorted(len(g) for g in truth_groups),
        "truth_shared_count": len(truth_shared),
        "block_count": len(blocks),
        "block_sizes": sorted(len(b) for b in blocks),
        "block_min_size": min((len(b) for b in blocks), default=0),
        "block_max_size": max((len(b) for b in blocks), default=0),
        "variables_in_blocks": len(covered),
        "singleton_count": problem.dimension - len(covered),
        "mean_fragments_per_truth_group": statistics.mean(fragments_per_group),
        "interaction_count": len(evidence.variable_region_interactions),
        "hyperedge_count": len(evidence.resolved_hyperedges),
        "recovered_shared_count": len(recovered),
        "shared_recall": len(recovered & truth_shared) / len(truth_shared) if truth_shared else None,
        "shared_precision": (
            len(recovered & truth_shared) / len(recovered) if recovered else None
        ),
        "level_budgets": dict(result.level_budgets),
        "ledger_count": ledger.count,
        "budget_reconciled": sum(dict(result.level_budgets).values()) == ledger.count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/soft_rddsm_aob_baseline_v3/audit.json"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_case, case): case for case in args.cases}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['case_id']}: blocks={row['block_count']} "
                f"(sizes {row['block_min_size']}-{row['block_max_size']}, "
                f"covered {row['variables_in_blocks']}/1000, "
                f"frag/group={row['mean_fragments_per_truth_group']:.1f}) "
                f"shared {row['recovered_shared_count']}/{row['truth_shared_count']} "
                f"recall={row['shared_recall']} precision={row['shared_precision']} "
                f"FE={row['ledger_count']} budget_ok={row['budget_reconciled']}",
                flush=True,
            )
    rows.sort(key=lambda row: row["case_id"])
    recalls = [row["shared_recall"] for row in rows if row["shared_recall"] is not None]
    precisions = [row["shared_precision"] for row in rows if row["shared_precision"] is not None]
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "audit_seed": AUDIT_SEED,
            "phase1_budget_fes": PHASE1_BUDGET_FES,
            "soft_config": {
                "dsm_budget": SoftDsmConfig().dsm_budget,
                "edge_threshold": SoftDsmConfig().edge_threshold,
                "k_mutual": SoftDsmConfig().k_mutual,
                "block_separability_probes": SoftDsmConfig().block_separability_probes,
                "rdg_region_size": SoftDsmConfig().rdg_region_size,
                "min_residual_size": SoftDsmConfig().min_residual_size,
            },
            "cases": list(args.cases),
            "truth_source": "vendor/aob/AOB/AOBG/datafile F*-info/s/p (offline audit only)",
        },
        "summary": {
            "case_count": len(rows),
            "mean_shared_recall": statistics.mean(recalls) if recalls else None,
            "mean_shared_precision": statistics.mean(precisions) if precisions else None,
            "nonzero_recall_case_count": sum(
                1 for value in recalls if value is not None and value > 0
            ),
            "budget_reconciled_all": all(row["budget_reconciled"] for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
