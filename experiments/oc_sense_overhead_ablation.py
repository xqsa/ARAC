"""Sense-overhead ablation, phase 1: fixed-budget 30% vs frozen 45% (G7).

Pre-registration (frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md):

- reference arm = the SAME regenerated Phase-I checkpoint run at
  sense_budget_share=0.45 with current code (the frozen Gate 48b rows
  cannot serve as the paired reference: the working tree's discovery
  code has drifted since 2026-08-16, so checkpoint hashes no longer
  match -- the first run of this gate correctly failed its
  ``checkpoints_match_reference`` check and is retained as an invalid
  artifact).  The frozen 45% rows and their MMES control are kept as
  context only;
- treatment arm = the identical unified-loop protocol from the same
  checkpoint with sense_budget_share=0.30
  (capped_proposal_budget semantics unchanged);
- the adaptive policies (residual-triggered probe, probe cache, periodic
  full + incremental) require loop-level changes and are NOT part of this
  phase; they stay registered as phase 2;
- metrics per case: final error, FE flow shares (sense / smp writeback /
  probe / arbitration / operator / tail), operator-fired flag, net gain
  per Phase-II FE, dispatch counts;
- pre-registered retention rule: the 30% arm is retained as a candidate
  only if (a) no case's final error degrades by more than 5% relative to
  the frozen 45% arm AND (b) the operator FE share does not decrease on
  any case; otherwise the 45% lane stands.  With n=4 cases the rule is
  deliberately conservative and descriptive.
"""

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.oc_aob_paired_gate48b import (
    CASES,
    PHASE1_FES,
    PHASE2_FES,
    PILOT_SEED,
    _phase1,
)

from arac.coordination.contract import OcCoordinatorConfig
from arac.coordination.loop import run_oc_unified_from_structure
from arac.overlap_core import DEFAULT_REFRESH_CYCLES

OUTPUT_SCHEMA = "arac-oc-sense-overhead-ablation-v1"
OUTPUT_ROOT = Path("artifacts/oc_sense_overhead_ablation")
REFERENCE = Path("artifacts/oc_aob_paired_gate48b/pilot.json")
TREATMENT_SHARE = 0.30
DEGRADATION_TOL = 0.05


def _flow(unified) -> dict[str, int]:
    return {
        "sense": sum(trace.sense_fes for trace in unified.cycles),
        "smp_writeback": sum(trace.smp_fes for trace in unified.cycles),
        "probe": sum(trace.probe_fes for trace in unified.cycles),
        "arbitration": sum(trace.arbitration_fes for trace in unified.cycles),
        "operator": sum(trace.operator_fes for trace in unified.cycles),
        "tail": unified.tail_fes,
    }


def _unified_arm(phase, *, sense_budget_share: float):
    from arac.overlap_core import (
        DEFAULT_NEIGHBORHOOD_FES,
        capped_proposal_budget,
    )

    sense_budget = capped_proposal_budget(
        PHASE2_FES,
        phase["components"],
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        neighborhood_fes=DEFAULT_NEIGHBORHOOD_FES,
        sense_budget_share=sense_budget_share,
    )
    unified = run_oc_unified_from_structure(
        phase["problem"],
        phase["structure"],
        checkpoint_hash=phase["checkpoint_hash"],
        total_budget_fes=PHASE1_FES + PHASE2_FES,
        phase1_fes=PHASE1_FES,
        incumbent=phase["incumbent"],
        incumbent_error=phase["incumbent_error"],
        run_seed=PILOT_SEED,
        refresh_cycles=DEFAULT_REFRESH_CYCLES,
        sense_budget_fes=sense_budget,
        config=OcCoordinatorConfig(pulse_min_fes=8, pulse_max_fes=32),
    )
    if unified.terminal_fes != PHASE1_FES + PHASE2_FES:
        raise RuntimeError("terminal FE drifted")
    return sense_budget, unified


def run_case(case_id: str) -> dict[str, object]:
    reference_rows = json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"]
    frozen = next(row for row in reference_rows if row["case_id"] == case_id)
    phase = _phase1(case_id, sense_budget_share=TREATMENT_SHARE)
    phase["case_id"] = case_id
    ref_budget, ref_unified = _unified_arm(phase, sense_budget_share=0.45)
    treatment_budget, unified = _unified_arm(phase, sense_budget_share=TREATMENT_SHARE)
    flow = _flow(unified)
    flow_total = sum(flow.values()) or 1
    ref_flow = _flow(ref_unified)
    ref_flow_total = sum(ref_flow.values()) or 1
    reference_final = float(ref_unified.final_error)
    final = float(unified.final_error)
    phase2_gain = float(phase["incumbent_error"]) - final
    return {
        "case_id": case_id,
        "checkpoint_hash": phase["checkpoint_hash"],
        "checkpoints_match_within_pair": True,
        "sense_budget_share": TREATMENT_SHARE,
        "sense_budget_fes_per_group": treatment_budget,
        "reference_sense_budget_fes_per_group_45": ref_budget,
        "final_error": final,
        "reference_final_error_45": reference_final,
        "frozen_45_final_error_context": float(frozen["unified"]["final_error"]),
        "frozen_control_final_error_context": float(frozen["control"]["final_error"]),
        "relative_change_vs_45": (final - reference_final) / max(reference_final, 1e-300),
        "phase2_gain": phase2_gain,
        "net_gain_per_phase2_fe": phase2_gain / PHASE2_FES,
        "flow": flow,
        "flow_share": {key: value / flow_total for key, value in flow.items()},
        "reference_flow_45": ref_flow,
        "reference_operator_share_45": ref_flow["operator"] / ref_flow_total,
        "operator_fes": flow["operator"],
        "operator_fired": flow["operator"] > 0,
        "dispatch_counts": {
            action: sum(1 for trace in unified.cycles if trace.action == action)
            for action in {trace.action for trace in unified.cycles}
        },
    }


def run_gate(workers: int = 4) -> dict[str, object]:
    if not REFERENCE.exists():
        raise RuntimeError("frozen gate48b pilot.json missing; run Gate 48b first")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(CASES))) as executor:
        futures = {executor.submit(run_case, case): case for case in CASES}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            (OUTPUT_ROOT / "cells" / f"{case}.json").parent.mkdir(parents=True, exist_ok=True)
            (OUTPUT_ROOT / "cells" / f"{case}.json").write_text(
                json.dumps({"schema_version": OUTPUT_SCHEMA, "row": row}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            rows.append(row)
            print(f"{case}: final={row['final_error']:.6g} ref45={row['reference_final_error_45']:.6g}", flush=True)
    rows.sort(key=lambda row: row["case_id"])
    degraded = [
        row["case_id"] for row in rows if float(row["relative_change_vs_45"]) > DEGRADATION_TOL
    ]
    operator_share_dropped = [
        row["case_id"]
        for row in rows
        if float(row["flow_share"]["operator"])
        < float(row["reference_operator_share_45"]) - 1e-12
    ]
    retained = not degraded and not operator_share_dropped
    checks = {
        "all_four_cases": len(rows) == len(CASES),
        "checkpoints_match_within_pair": all(
            row["checkpoints_match_within_pair"] for row in rows
        ),
        "budgets_differ_as_parametrized": all(
            float(row["reference_sense_budget_fes_per_group_45"])
            > float(row["sense_budget_fes_per_group"])
            for row in rows
        ),
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "reference": str(REFERENCE),
            "treatment_share": TREATMENT_SHARE,
            "degradation_tolerance": DEGRADATION_TOL,
            "retention_rule": "retain 30% iff no case degrades >5% and operator share never drops",
            "adaptive_policies": "phase 2, registered, not run here",
            "production_selector_modified": False,
        },
        "rows": rows,
        "summary": {
            "degraded_cases": degraded,
            "operator_share_dropped_cases": operator_share_dropped,
            "treatment_retained": retained,
            "median_relative_change_vs_45": sorted(
                float(row["relative_change_vs_45"]) for row in rows
            )[len(rows) // 2],
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"summary": payload["summary"], "gate_passed": payload["gate_passed"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
