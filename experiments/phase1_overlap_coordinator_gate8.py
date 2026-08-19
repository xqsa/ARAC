"""Gate 8: consume real Phase-I overlap evidence in the coordinator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.coordination import LocalProposal, OverlapCoordinator
from arac.evidence import Phase1OverlapAdapter, run_phase1_overlap_pilot
from arac.runtime.ledger import EvaluationLedger
if __package__:
    from experiments.phase1_overlap_integration_gate7 import (
        BUCKET_SIZE,
        DIMENSION,
        GROUPS,
        ROUNDS,
        SEED,
        TOTAL_BUDGET_FES,
        _problem,
    )
else:
    from phase1_overlap_integration_gate7 import (  # type: ignore[no-redef]
        BUCKET_SIZE,
        DIMENSION,
        GROUPS,
        ROUNDS,
        SEED,
        TOTAL_BUDGET_FES,
        _problem,
    )


def _proposals(structure, base: np.ndarray, component: tuple[int, ...]) -> tuple[LocalProposal, ...]:
    proposals = []
    for group in component:
        values = []
        uncertainty = []
        for variable in structure.groups[group]:
            value = 0.0 if variable in structure.shared_variables else float(base[variable])
            values.append((variable, value))
            uncertainty.append((variable, 0.1))
        proposals.append(
            LocalProposal(
                group=group,
                values=tuple(values),
                improvement=1.0,
                uncertainty=tuple(uncertainty),
            )
        )
    return tuple(proposals)


def run_gate() -> dict[str, object]:
    problem = _problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        anchor_count=5,
        step=0.25,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=128,
    )
    adaptation = Phase1OverlapAdapter().adapt(pilot.checkpoint, pilot.evidence)
    if not adaptation.ready or adaptation.structure is None:
        raise RuntimeError("Gate 8 requires complete Phase-I overlap adaptation")
    structure = adaptation.structure
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
    )
    component_results = []
    for component in structure.connected_components():
        if len(component) <= 1:
            continue
        proposals = _proposals(structure, ledger.best_x, component)
        result = OverlapCoordinator(structure, ledger).coordinate(component, proposals)
        component_results.append(
            {
                "component": component,
                "shared_variables": tuple(
                    variable
                    for variable in structure.shared_variables
                    if set(structure.owners(variable)).issubset(set(component))
                ),
                "residual_variables": tuple(item.variable for item in result.residuals),
                "candidate_count": len(result.candidates),
                "accepted": result.accepted,
                "best_error_before": result.best_error_before,
                "best_error_after": result.best_error_after,
                "consumed_fes": len(result.candidates),
            }
        )
    expected_components = 2
    all_shared_consumed = all(
        set(item["shared_variables"]) == set(item["residual_variables"])
        for item in component_results
    )
    payload = {
        "schema_version": "arac-phase1-overlap-coordinator-gate8-v1",
        "dimension": DIMENSION,
        "groups": GROUPS,
        "phase1_fes": pilot.checkpoint.phase1_fes,
        "phase2_budget": TOTAL_BUDGET_FES - pilot.checkpoint.phase1_fes,
        "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
        "adapter_ready": adaptation.ready,
        "shared_variables": structure.shared_variables,
        "components": component_results,
        "phase2_ledger_count_after_coordination": ledger.count,
        "gate_checks": {
            "adapter_ready": adaptation.ready,
            "two_overlap_components_consumed": len(component_results) == expected_components,
            "all_shared_variables_consumed": all_shared_consumed,
            "strict_best_preserved": all(
                item["best_error_after"] <= item["best_error_before"]
                for item in component_results
            ),
            "phase2_budget_not_exhausted": ledger.count < TOTAL_BUDGET_FES,
        },
    }
    payload["gate_passed"] = all(payload["gate_checks"].values())
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase1_overlap_coordinator_gate8/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    payload["gate_passed"] = all(payload["gate_checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
