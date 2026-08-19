"""Single-cell Gate 31 probe for persistent shared-core CTP."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from arac.overlap_core import PERSISTENT_CTP_MODE, run_overlap_from_pilot
from arac.evidence import run_phase1_overlap_pilot
from experiments.overlap_arac_gate29_screening import (
    DEFAULT_NEIGHBORHOOD_FES,
    DEFAULT_REFRESH_CYCLES,
    PHASE1_KWARGS,
    TOTAL_BUDGET_FES,
    Cell,
    build_cell,
)


CELL = Cell("conflicting", "chain", 3, 20260829)
OUTPUT = Path("artifacts/overlap_joint_core_ctp_gate32/real_cell.json")
BASELINE = Path("artifacts/overlap_arac_gate29_screening/confirmation_fresh.json")


def _summary(result):
    return {
        "mode": result.coordination_mode,
        "final_error": result.final_error,
        "checkpoint_hash": result.phase1.checkpoint.checkpoint_hash,
        "phase1_fes": result.phase1.consumed_fes,
        "proposal_budget_fes": result.proposal_budget_fes,
        "phase2_consumed_fes": result.phase2_consumed_fes,
        "terminal_fes": result.terminal_fes,
        "strict_best": all(item.best_error_after <= item.best_error_before for item in result.cycles),
        "ctp_fes": [item.ctp_fes for item in result.cycles],
        "ctp_triggered_components": [item.ctp_triggered_components for item in result.cycles],
        "max_conflict_streak": [item.max_conflict_streak for item in result.cycles],
        "coordination_gain": [item.coordination_gain for item in result.cycles],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    problem, truth = build_cell(CELL)
    baseline_payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_row = next(
        row
        for row in baseline_payload["cells"]
        if row["cell"] == asdict(CELL)
    )
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=CELL.seed,
        **PHASE1_KWARGS,
    )
    if pilot.checkpoint.checkpoint_hash != baseline_row["checkpoint_hash"]:
        raise RuntimeError("fresh Phase-I checkpoint does not match Gate 29 baseline")
    result = run_overlap_from_pilot(
        problem,
        pilot,
        coordination_mode=PERSISTENT_CTP_MODE,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
    )
    payload = {
        "schema_version": f"arac-{args.output.parent.name}-real-cell-v1",
        "cell": asdict(CELL),
        "truth_shared_count": len(truth.structure.shared_variables),
        "baseline_source": str(BASELINE),
        "pilot": {
            "checkpoint_hash": pilot.checkpoint.checkpoint_hash,
            "phase1_fes": pilot.consumed_fes,
            "checkpoint_error": pilot.checkpoint.incumbent_error,
        },
        "arms": [
            arm
            for arm in baseline_row["arms"]
            if arm["mode"] in {"proposal_neighborhood", "full_context"}
        ]
        + [_summary(result)],
        "checkpoint_parity": all(
            arm["checkpoint_hash"] == pilot.checkpoint.checkpoint_hash
            for arm in baseline_row["arms"]
        ),
        "proposal_budget_parity": len(
            {
                arm["proposal_budget_fes"]
                for arm in baseline_row["arms"]
                if arm["mode"] in {"proposal_neighborhood", "full_context"}
            }
            | {result.proposal_budget_fes}
        )
        == 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
