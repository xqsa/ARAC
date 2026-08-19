"""Offline calibration of topology signals on inferred overlap structures.

Gate 37 showed that the absolute hub degree computed on Phase-I inferred
structures (10-20) never separates star from chain/random topologies (truth
hub degree 2-3), so the pre-registered coordinate-CTP branch never fired.
This calibration runs Phase-I only (no 3M-FE optimization) on fresh seeds and
dumps the inferred-structure statistics needed to pre-register a normalized
topology signal for Gate 38.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from experiments.overlap_arac_gate29_screening import (
    Cell,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    build_cell,
    cells,
)

CALIBRATION_SEEDS = (20260815, 20260816, 20260817)
CONFIDENCE_CUTOFFS = (0.5, 0.6, 0.7, 0.8, 0.9)
OUTPUT_SCHEMA = "arac-overlap-topology-calibration-gate38-v1"


@dataclass(frozen=True)
class CalibrationCell:
    mode: str
    topology: str
    overlap_budget: int
    seed: int


def _truth_hub(truth_structure) -> float:
    """Absolute hub degree of the constructed truth topology."""

    partners: dict[int, set[int]] = {}
    for variable in truth_structure.shared_variables:
        owners = sorted(truth_structure.membership[variable])
        for left in range(len(owners)):
            for right in range(left + 1, len(owners)):
                partners.setdefault(owners[left], set()).add(owners[right])
                partners.setdefault(owners[right], set()).add(owners[left])
    return float(max((len(value) for value in partners.values()), default=0))


def _overlap_pairs_by_confidence(structure) -> dict[tuple[int, int], list[float]]:
    """Map each group pair to the confidences of the variables they share."""

    pairs: dict[tuple[int, int], list[float]] = {}
    for variable in structure.shared_variables:
        owners = structure.owners(variable)
        pair_confidence = sum(
            structure.confidence(variable, owner) for owner in owners
        ) / len(owners)
        sorted_owners = sorted(owners)
        for left in range(len(sorted_owners)):
            for right in range(left + 1, len(sorted_owners)):
                pairs.setdefault((sorted_owners[left], sorted_owners[right]), []).append(
                    pair_confidence
                )
    return pairs


def _hub_degrees(structure, component: tuple[int, ...]) -> dict[str, float]:
    component_set = set(component)
    pairs = _overlap_pairs_by_confidence(structure)
    partners: dict[int, set[int]] = {group: set() for group in component}
    shared_in_component = tuple(
        variable
        for variable in structure.shared_variables
        if set(structure.owners(variable)).issubset(component_set)
    )
    owner_counts = [len(structure.owners(variable)) for variable in shared_in_component]

    def register(pairs_in_component):
        for (left, right) in pairs_in_component:
            if left in partners:
                partners[left].add(right)
            if right in partners:
                partners[right].add(left)

    in_component = {
        pair: values for pair, values in pairs.items() if set(pair).issubset(component_set)
    }
    register(in_component)
    absolute = max((len(value) for value in partners.values()), default=0)
    denominator = max(1, len(component) - 1)
    result = {
        "component_groups": len(component),
        "shared_variable_count": len(shared_in_component),
        "max_shared_owner_count": max(owner_counts, default=0),
        "mean_shared_owner_count": (
            sum(owner_counts) / len(owner_counts) if owner_counts else 0.0
        ),
        "overlap_pair_count": len(in_component),
        "absolute_hub": float(absolute),
        "relative_hub": absolute / denominator,
    }
    for cutoff in CONFIDENCE_CUTOFFS:
        filtered = {
            pair: values
            for pair, values in in_component.items()
            if any(value >= cutoff for value in values)
        }
        for group in partners:
            partners[group].clear()
        register(filtered)
        hub = max((len(value) for value in partners.values()), default=0)
        result[f"absolute_hub_conf{cutoff:.1f}"] = float(hub)
        result[f"relative_hub_conf{cutoff:.1f}"] = hub / denominator
    return result


def run_calibration_cell(cell: CalibrationCell) -> dict[str, object]:
    from arac.evidence import run_phase1_overlap_pilot

    problem, truth = build_cell(
        Cell(cell.mode, cell.topology, cell.overlap_budget, cell.seed)
    )
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=cell.seed,
        **PHASE1_KWARGS,
    )
    structure = pilot.adaptation.structure
    components = tuple(
        component
        for component in structure.connected_components()
        if any(
            set(structure.owners(variable)).issubset(component)
            for variable in structure.shared_variables
        )
    )
    component_stats = [_hub_degrees(structure, component) for component in components]
    shared_confidences = []
    for variable in structure.shared_variables:
        owners = structure.owners(variable)
        shared_confidences.append(
            sum(structure.confidence(variable, owner) for owner in owners) / len(owners)
        )
    return {
        "cell": asdict(cell),
        "truth_hub": _truth_hub(truth.structure),
        "inferred_groups": len(structure.groups),
        "inferred_shared_count": len(structure.shared_variables),
        "component_count": len(components),
        "shared_confidence_mean": (
            sum(shared_confidences) / len(shared_confidences) if shared_confidences else 0.0
        ),
        "shared_confidence_min": min(shared_confidences) if shared_confidences else 0.0,
        "phase1_fes": pilot.consumed_fes,
        "components": component_stats,
        "largest_component": max(component_stats, key=lambda item: item["absolute_hub"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_topology_calibration_gate38/calibration.json"),
    )
    args = parser.parse_args()
    jobs = tuple(
        CalibrationCell(cell.mode, cell.topology, cell.overlap_budget, seed)
        for cell in cells()
        for seed in CALIBRATION_SEEDS
    )
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_calibration_cell, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"calibrated {job.mode}/{job.topology}/ov={job.overlap_budget}/seed={job.seed}",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            row["cell"]["mode"],
            row["cell"]["topology"],
            row["cell"]["overlap_budget"],
            row["cell"]["seed"],
        )
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "calibration_seeds": CALIBRATION_SEEDS,
            "confidence_cutoffs": CONFIDENCE_CUTOFFS,
            "phase1_kwargs": PHASE1_KWARGS,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "gate_seeds_excluded": [20260829],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
