"""Topology-signal transfer calibration for Gate 40 base-function families.

Gate 38 froze the relative-hub threshold (0.9) on Rastrigin inferred
structures.  Before running the coordinator on ackley/elliptic/schwefel
base functions, this calibration verifies on fresh seeds that the same
signal still separates chain from star topologies, or reports that it
does not transfer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.benchmarks.overlap_objective import build_overlap_problem
from experiments.overlap_arac_gate29_screening import (
    ACTIVE_DIMENSION,
    ACTIVE_GROUP_COUNT,
    BOUNDS,
    MAX_GROUP_SIZE,
    MIN_GROUP_SIZE,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
)
from experiments.overlap_topology_calibration_gate38 import _hub_degrees

FAMILIES = ("ackley", "elliptic", "schwefel")
TOPOLOGIES = ("chain", "star", "random")
OVERLAPS = (3, 6)
CALIBRATION_SEEDS = (20260835, 20260836)
OUTPUT_SCHEMA = "arac-overlap-family-calibration-gate40-v1"


@dataclass(frozen=True)
class FamilyCell:
    base_function: str
    topology: str
    overlap_budget: int
    seed: int


def build_family_cell(cell: FamilyCell):
    _, truth = build_overlap_problem(
        ACTIVE_DIMENSION,
        overlap_budget=cell.overlap_budget,
        min_group_size=MIN_GROUP_SIZE,
        max_group_size=MAX_GROUP_SIZE,
        num_groups=ACTIVE_GROUP_COUNT,
        base_function=cell.base_function,
        conflict_mode="conflicting",
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        seed=cell.seed,
        topology=cell.topology,
        interaction_strength=0.25,
    )

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        active = np.asarray(truth.evaluate(batch[:, :ACTIVE_DIMENSION]), dtype=float)
        tail = np.sum((batch[:, ACTIVE_DIMENSION:] / BOUNDS) ** 2, axis=1)
        result = active + tail
        return float(result[0]) if rows.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=1000,
        lower_bounds=(-BOUNDS,) * 1000,
        upper_bounds=(BOUNDS,) * 1000,
        optimum=0.0,
    )
    return problem, truth


def run_family_cell(cell: FamilyCell) -> dict[str, object]:
    from arac.evidence import run_phase1_overlap_pilot

    problem, truth = build_family_cell(cell)
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **PHASE1_KWARGS,
    )
    structure = pilot.adaptation.structure
    ready = pilot.adaptation.ready and structure is not None
    row: dict[str, object] = {
        "cell": asdict(cell),
        "adapter_ready": bool(ready),
        "adapter_reason": pilot.adaptation.reason,
        "discovery_reason": pilot.discovery.complete_reason,
        "candidate_pair_count": pilot.discovery.candidate_pair_count,
    }
    if ready:
        components = tuple(
            component
            for component in structure.connected_components()
            if any(
                set(structure.owners(variable)).issubset(component)
                for variable in structure.shared_variables
            )
        )
        stats = [_hub_degrees(structure, component) for component in components]
        row["component_stats"] = stats
        row["largest_component"] = max(stats, key=lambda item: item["absolute_hub"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_family_calibration_gate40/calibration.json"),
    )
    args = parser.parse_args()
    jobs = tuple(
        FamilyCell(base, topology, overlap, seed)
        for base in FAMILIES
        for topology in TOPOLOGIES
        for overlap in OVERLAPS
        for seed in CALIBRATION_SEEDS
    )
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_family_cell, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            cell = row["cell"]
            print(
                f"calibrated {cell['base_function']}/{cell['topology']}/"
                f"ov={cell['overlap_budget']}/seed={cell['seed']}",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            row["cell"]["base_function"],
            row["cell"]["topology"],
            row["cell"]["overlap_budget"],
            row["cell"]["seed"],
        )
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "families": FAMILIES,
            "calibration_seeds": CALIBRATION_SEEDS,
            "phase1_kwargs": PHASE1_KWARGS,
            "total_budget_fes": TOTAL_BUDGET_FES,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    from collections import defaultdict

    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["adapter_ready"]:
            by_group[(row["cell"]["base_function"], row["cell"]["topology"])].append(
                row["largest_component"]["relative_hub"]
            )
    for key in sorted(by_group):
        values = by_group[key]
        print(f"{key[0]:>9}/{key[1]:>7}: rel_hub {min(values):.3f}..{max(values):.3f} n={len(values)}")
    not_ready = [row for row in rows if not row["adapter_ready"]]
    print(f"adapter_not_ready: {len(not_ready)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
