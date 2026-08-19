"""Gate 51b-v5: frozen 3M HPR-GCB ON/OFF cells.

Standalone values are read from the already revalidated Gate 51b standalone
tree; this script only executes the new coordinator and never overwrites the
v4.4 cells.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5,
    run_oc_episode_schedule_v5,
)
from arac.runtime.manifest import implementation_manifest_hash
from experiments.oc_action_episode_gate50 import (
    ACTION_SEED,
    PHASE1_FES,
    TOTAL_FES,
)
from experiments.oc_phase_aware_gate51a import _load_cached_phase_one, load_config as load_v4_config

CASES = ("R2", "A3", "S5", "R6")
ARMS = ("on", "off")
OUTPUT_ROOT = Path("artifacts/oc_phase_aware_gate51b_v5")
STANDALONE_ROOT = Path("artifacts/oc_phase_aware_gate51b_standalone_v1/cells")
HCC_REFERENCE = Path("references/hcc_aob24_reference.json")
CELL_SCHEMA = "arac-oc-gate51b-v5-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate51b-v5"
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("src/arac/actions/phase2_v2.py"),
    Path("src/arac/runtime/phase2.py"),
    Path("experiments/oc_phase_aware_gate51b_v5.py"),
]


def _manifest() -> str:
    return implementation_manifest_hash(Path(__file__).resolve().parents[1], MANIFEST_FILES)


def load_config():
    config = load_v4_config()[0]
    return dataclasses.replace(
        config,
        scheduler_version=DEFAULT_SCHEDULER_VERSION_V5,
        horizon_protected=True,
    )


def run_cell(case_id: str, arm: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _load_cached_phase_one(case_id)
    if checkpoint is None:
        raise RuntimeError(f"missing cached Phase-I checkpoint for {case_id}")
    checkpoint = dataclasses.replace(checkpoint, total_budget_fes=TOTAL_FES)
    config = dataclasses.replace(load_config(), handoff_enabled=(arm == "on"))
    result = run_oc_episode_schedule_v5(
        problem, checkpoint, action_seed=ACTION_SEED, config=config
    )
    return {
        "case_id": case_id,
        "arm": arm,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "manifest_hash": _manifest(),
        "result": {
            "schema_version": result.schema_version,
            "scheduler_policy": result.scheduler_policy,
            "scheduler_version": result.scheduler_version,
            "calibration_ref": result.calibration_ref,
            "final_error": result.final_error,
            "terminal_fes": result.terminal_fes,
            "schedule_hash": result.schedule_hash,
            "audit": result.audit,
            "funded_fes": result.funded_fes,
            "cold_start_probe_tax_fes": result.cold_start_probe_tax_fes,
            "development_fes": result.development_fes,
            "exploitation_fes": result.exploitation_fes,
            "switches": result.switches,
            "probe_order": list(result.probe_order),
            "tickets": [t.__dict__ for t in result.tickets],
            "probes": [p.__dict__ for p in result.probes],
            "handoffs": [h.__dict__ for h in result.handoffs],
            "receipts": [r.__dict__ for r in result.receipts],
            "horizon_reservation_count": sum(
                r.reservation_kind == "horizon" for r in result.receipts
            ),
            "plateau_release_count": sum(
                r.plateau_release for r in result.receipts
            ),
            "magnitude_repairs": result.magnitude_repairs,
        },
    }


def _valid(path: Path, case_id: str, arm: str, manifest: str) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    result = row.get("result", {})
    return (
        row.get("schema_version") == CELL_SCHEMA
        and result.get("case_id") == case_id
        and result.get("arm") == arm
        and result.get("manifest_hash") == manifest
        and result.get("result", {}).get("schema_version") == "arac-oc-episode-schedule-v5"
        and result.get("result", {}).get("terminal_fes") == TOTAL_FES
        and all(result.get("result", {}).get("audit", {}).values())
    )


def _final_standalone(case_id: str, arm: str) -> float:
    path = STANDALONE_ROOT / f"{case_id}_{arm}.json"
    if not path.exists():
        raise RuntimeError(f"missing standalone reference: {path}")
    return float(json.loads(path.read_text(encoding="utf-8"))["result"]["result"]["final_error"])


def _hcc(case_id: str) -> float:
    payload = json.loads(HCC_REFERENCE.read_text(encoding="utf-8"))
    return float(payload["functions"][case_id]["HCC-ES"]["mean"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = _manifest()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(case, arm) for case in CASES for arm in ARMS]
    rows: list[dict[str, object]] = []
    pending: list[tuple[str, str]] = []
    for case_id, arm in jobs:
        path = cells_dir / f"{case_id}_{arm}.json"
        if _valid(path, case_id, arm, manifest):
            rows.append(json.loads(path.read_text(encoding="utf-8"))["result"])
        else:
            pending.append((case_id, arm))
    failures: list[dict[str, str]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_cell, *job): job for job in pending}
            for future in as_completed(futures):
                case_id, arm = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    failures.append({"case": case_id, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"{case_id}/{arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    continue
                (cells_dir / f"{case_id}_{arm}.json").write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case_id}/{arm}: final={row['result']['final_error']:.6g} "
                    f"reservations={row['result']['horizon_reservation_count']}",
                    flush=True,
                )

    by_case = {case: {} for case in CASES}
    for row in rows:
        by_case[row["case_id"]][row["arm"]] = row["result"]
    ratios: dict[str, dict[str, float]] = {}
    checks: dict[str, bool] = {}
    for case in CASES:
        if not all(arm in by_case[case] for arm in ARMS):
            checks[f"{case}_complete"] = False
            continue
        on = float(by_case[case]["on"]["final_error"])
        off = float(by_case[case]["off"]["final_error"])
        standalone = {arm: _final_standalone(case, arm) for arm in ("ctp", "gcb", "smp", "aor")}
        best_standalone = min(standalone.values())
        hcc = _hcc(case)
        ratios[case] = {
            "on": on,
            "off": off,
            "on_vs_off": on / off,
            "best_standalone": best_standalone,
            "on_vs_best_standalone": on / best_standalone,
            "hcc": hcc,
            "on_vs_hcc": on / hcc,
            **{f"standalone_{arm}": value for arm, value in standalone.items()},
        }
        checks[f"{case}_on_complete"] = (
            by_case[case]["on"]["terminal_fes"] == TOTAL_FES
            and all(by_case[case]["on"]["audit"].values())
        )
        checks[f"{case}_off_complete"] = (
            by_case[case]["off"]["terminal_fes"] == TOTAL_FES
            and all(by_case[case]["off"]["audit"].values())
        )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "scheduler_version": DEFAULT_SCHEDULER_VERSION_V5,
        "manifest_hash": manifest,
        "protocol": {"cases": list(CASES), "arms": list(ARMS), "total_fes": TOTAL_FES, "phase1_fes": PHASE1_FES},
        "ratios": ratios,
        "checks": checks,
        "failures": failures,
        "passed": not failures and len(rows) == len(jobs) and all(checks.values()),
        "interpretation": {
            "primary_comparison": "ON vs best standalone and ON vs HCC are reported separately",
            "v4_4_baseline": "artifacts/oc_phase_aware_gate51b_v4_4",
        },
    }
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
