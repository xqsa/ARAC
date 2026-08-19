"""Gate 50c: 3M six-arm pairing -- OC(handoff on/off) vs four standalone episodes."""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.episodes import run_oc_episode_schedule
from arac.evidence.hierarchical import to_overlap_structure
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger

CASES = ("R2", "A3", "S5", "R6")
PILOT_SEED = 20260845
ACTION_SEED = 20260845
PHASE1_FES = 180_000
TOTAL_FES = 3_000_000
SEGMENT_FES = 300_000
PROBE_MIN = 20_000
GATE50_ROOT = Path("artifacts/oc_action_episode_gate50/cells")
CELL_SCHEMA = "arac-oc-gate50c-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate50c-v1"
OUTPUT_ROOT = Path("artifacts/oc_action_episode_gate50c")
STANDALONE_ARMS = ("ctp", "gcb", "smp", "aor")


def _cached_checkpoint(case_id: str) -> PhaseCheckpoint:
    payload = json.loads((GATE50_ROOT / f"{case_id}_phase1.json").read_text(encoding="utf-8"))
    body = payload["checkpoint"]
    return PhaseCheckpoint(
        protocol=body["protocol"],
        run_seed=body["run_seed"],
        total_budget_fes=body["total_budget_fes"],
        phase1_fes=body["phase1_fes"],
        incumbent=tuple(body["incumbent"]),
        incumbent_error=body["incumbent_error"],
        feature_names=tuple(body["feature_names"]),
        feature_values=tuple(body["feature_values"]),
        blocks=tuple(tuple(block) for block in body["blocks"]),
        relations=tuple(
            RelationEvidence(
                left_block=r["left_block"],
                right_block=r["right_block"],
                strength=r["strength"],
                disagreement=r["disagreement"],
            )
            for r in body["relations"]
        ),
    )


def _discovery_structure(problem):
    ledger = EvaluationLedger(problem, PHASE1_FES)
    discovery = discover_hierarchical_soft(
        problem, ledger, run_seed=PILOT_SEED, config=SoftDsmConfig()
    )
    return to_overlap_structure(discovery.evidence)


def run_cell(case_id: str, arm: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _cached_checkpoint(case_id)
    structure = _discovery_structure(problem)
    schedule = run_oc_episode_schedule(
        problem,
        checkpoint,
        action_seed=ACTION_SEED,
        structure=structure,
        segment_fes=SEGMENT_FES,
        probe_min_fes=PROBE_MIN,
        handoff_enabled=(arm == "on"),
    )
    return {
        "case_id": case_id,
        "arm": arm,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "result": {
            "final_error": schedule.final_error,
            "terminal_fes": schedule.terminal_fes,
            "funded_fes": schedule.funded_fes,
            "switches": schedule.switches,
            "sensing": schedule.sensing,
            "probe_order": list(schedule.probe_order),
            "probe_tax_fes": schedule.probe_tax_fes,
            "exploitation_fes": schedule.exploitation_fes,
            "scoped_checkpoint_hash": schedule.scoped_checkpoint_hash,
            "probes": [p.__dict__ for p in schedule.probes],
            "handoffs": [h.__dict__ for h in schedule.handoffs],
            "segments": [r.__dict__ for r in schedule.receipts],
            "magnitude_repairs": schedule.magnitude_repairs,
            "schedule_hash": schedule.schedule_hash,
        },
    }


def _audit_oc(result: dict[str, object]) -> bool:
    segments = result["segments"]
    if [s["segment_index"] for s in segments] != list(range(len(segments))):
        return False
    for previous, current in zip(segments, segments[1:]):
        if current["global_error_before"] > previous["global_error_after"] + 1e-9:
            return False
    sensing_fes = int((result.get("sensing") or {}).get("probe_fes") or 0)
    if PHASE1_FES + sensing_fes + sum(result["funded_fes"].values()) != result["terminal_fes"]:
        return False
    for s in segments:
        if len(s.get("snapshot_hash", "")) != 64 or len(s.get("state_hash", "")) != 64:
            return False
    for h in result["handoffs"]:
        if h["handoff_mode"] not in {
            "reanchor_next_segment", "reanchor_next_visit", "fresh_by_design", "disabled",
        }:
            return False
        if h["refusal"] not in {"none", "not_better", "oob_incumbent", "disabled"}:
            return False
    payload = {
        "schema_version": "arac-oc-episode-schedule-v3",
        "dispatcher": "gcb_coordinator",
        "sensing": result["sensing"],
        "probe_order": result["probe_order"],
        "probe_tax_fes": result["probe_tax_fes"],
        "exploitation_fes": result["exploitation_fes"],
        "scoped_checkpoint_hash": result["scoped_checkpoint_hash"],
        "probes": result["probes"],
        "handoffs": result["handoffs"],
        "receipts": result["segments"],
        "funded_fes": result["funded_fes"],
        "switches": result["switches"],
        **({"magnitude_repairs": result["magnitude_repairs"]} if "magnitude_repairs" in result else {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() == result["schedule_hash"]


def _material_episodes(result: dict[str, object]) -> set[str]:
    material = set()
    handoff_first_index = None
    for h in result["handoffs"]:
        if h["adopted"]:
            handoff_first_index = h["segment_index"]
            break
    for s in result["segments"]:
        if s["material"] and s["global_gain"] > 0.0:
            if handoff_first_index is None or s["segment_index"] >= handoff_first_index:
                material.add(s["episode"])
    return material


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    standalone: dict[str, dict[str, dict[str, object]]] = {case: {} for case in CASES}
    for case in CASES:
        for arm in STANDALONE_ARMS:
            cell = json.loads((GATE50_ROOT / f"{case}_{arm}.json").read_text(encoding="utf-8"))["result"]
            if cell["checkpoint_hash"] != _cached_checkpoint(case).checkpoint_hash:
                raise RuntimeError(f"standalone checkpoint drift: {case}/{arm}")
            standalone[case][arm] = cell["result"]

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    pending = [
        (case, arm)
        for case in CASES
        for arm in ("on", "off")
        if not (cells_dir / f"{case}_{arm}.json").exists()
    ]
    for case in CASES:
        for arm in ("on", "off"):
            path = cells_dir / f"{case}_{arm}.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_cell, case, arm): (case, arm) for case, arm in pending}
            for future in as_completed(futures):
                case, arm = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    import traceback
                    print(f"{case}/{arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    failures.append({"case": case, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                (cells_dir / f"{case}_{arm}.json").write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case}/{arm}: final={row['result']['final_error']:.6g} "
                    f"switches={row['result']['switches']} "
                    f"funded={ {k: v for k, v in row['result']['funded_fes'].items() if v} }",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["case_id"], row["arm"]))
    by_case: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["arm"]] = row["result"]

    protocol_checks: dict[str, bool] = {}
    for case in CASES:
        for arm in STANDALONE_ARMS:
            protocol_checks[f"{case}_{arm}_terminal_exact"] = (
                standalone[case][arm]["terminal_fes"] == TOTAL_FES
            )
        for arm in ("on", "off"):
            if case in by_case and arm in by_case[case]:
                result = by_case[case][arm]
                protocol_checks[f"{case}_{arm}_terminal_exact"] = result["terminal_fes"] == TOTAL_FES
                protocol_checks[f"{case}_{arm}_receipts_valid"] = _audit_oc(result)

    performance: dict[str, object] = {}
    strictly_better: list[str] = []
    for case in CASES:
        if case not in by_case or "on" not in by_case[case]:
            continue
        on_final = by_case[case]["on"]["final_error"]
        best_standalone = min(standalone[case][arm]["final_error"] for arm in STANDALONE_ARMS)
        performance[f"{case}_ratio_vs_best"] = on_final / best_standalone
        if on_final > best_standalone * 1.05 + 1e-9:
            performance[f"{case}_not_worse"] = False
        else:
            performance[f"{case}_not_worse"] = True
        if on_final < best_standalone - 1e-9:
            strictly_better.append(case)

    complementarity: dict[str, object] = {}
    for case in strictly_better:
        on_result = by_case[case]["on"]
        off_final = by_case[case].get("off", {}).get("final_error")
        material = _material_episodes(on_result)
        complementarity[f"{case}_material_episode_count"] = len(material)
        complementarity[f"{case}_material_episodes"] = sorted(material)
        complementarity[f"{case}_two_episodes_material"] = len(material) >= 2
        complementarity[f"{case}_on_beats_off"] = (
            off_final is not None and on_result["final_error"] < off_final - 1e-9
        )

    gate_passed = (
        not failures
        and all(protocol_checks.values())
        and all(bool(v) for k, v in performance.items() if k.endswith("_not_worse"))
        and len(strictly_better) >= 1
        and len(complementarity) >= 2  # at least count + episodes entries per case
        and all(
            v for k, v in complementarity.items()
            if k.endswith("_two_episodes_material") or k.endswith("_on_beats_off")
        )
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "total_fes": TOTAL_FES,
            "segment_fes": SEGMENT_FES,
            "probe_share": 0.10,
            "probe_min_fes": PROBE_MIN,
            "action_seed": ACTION_SEED,
            "standalone_source": str(GATE50_ROOT),
            "protocol_doc": "docs/arac-oc-gate50c-protocol.md",
        },
        "protocol_checks": protocol_checks,
        "performance": performance,
        "strictly_better_cases": strictly_better,
        "complementarity": complementarity,
        "failures": failures,
        "gate_passed": gate_passed,
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "failures": failures,
        "performance": performance,
        "strictly_better": strictly_better,
        "complementarity": complementarity,
        "gate_passed": gate_passed,
    }, indent=1))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
