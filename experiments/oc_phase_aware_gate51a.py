"""Gate 51a: 600k mechanism screening for the phase-aware v4 scheduler.

Mechanism layer only -- no performance claims (plan section 7).  The
scheduler configuration is loaded from the Gate 51-0 calibration table;
a missing or incomplete table aborts loudly (parameters are frozen by
measurement, never guessed).
"""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import hashlib
import json
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V4,
    OC_EPISODE_SCHEMA_V4,
    PhaseAwareSchedulerConfig,
    run_oc_episode_schedule_v4,
)
from arac.runtime.manifest import implementation_manifest_hash
from experiments.oc_action_episode_gate50 import (
    _load_cached_phase_one,
)

CASES = ("R2", "A3", "S5", "R6")
TOTAL_FES = 600_000
ACTION_SEED = 20260845
ARMS = ("on", "off")
# Version-scoped output (P0 review): a screening directory only ever
# holds cells of one scheduler version and one calibration state.
OUTPUT_ROOT = Path(
    f"artifacts/oc_phase_aware_gate51a_{DEFAULT_SCHEDULER_VERSION_V4.replace('.', '_')}"
)
CALIBRATION_PATH = Path("artifacts/oc_horizon_gate51_0/calibration.json")
CELL_SCHEMA = "arac-oc-gate51a-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate51a-v1"
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("src/arac/actions/phase2_v2.py"),
    Path("src/arac/runtime/phase2.py"),
    Path("experiments/oc_phase_aware_gate51a.py"),
    Path("docs/arac-oc-v4-upgrade-plan.md"),
]


def _manifest() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return implementation_manifest_hash(repo_root, MANIFEST_FILES)


def load_config() -> tuple[PhaseAwareSchedulerConfig, str]:
    if not CALIBRATION_PATH.exists():
        raise RuntimeError(
            f"Gate 51-0 calibration table missing: {CALIBRATION_PATH}. "
            "Freeze the calibration before running 51a (plan section 4: "
            "placeholder window parameters are forbidden)."
        )
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise RuntimeError("Gate 51-0 calibration table is not frozen")
    config = PhaseAwareSchedulerConfig(
        maturity_window_fes=int(payload["maturity_window_fes"]),
        revelation_horizon_fes=int(payload["revelation_horizon_fes"]),
        exploration_and_development_cap=float(payload["exploration_and_development_cap"]),
        exploitation_reserve_ratio=float(payload["exploitation_reserve_ratio"]),
        cold_start_probe_cap=float(payload.get("cold_start_probe_cap", 0.25)),
        probe_min_fes=int(payload.get("probe_min_fes", 20_000)),
        escalation_factor=int(payload.get("escalation_factor", 2)),
        escalation_grants_k=int(payload.get("escalation_grants_k", 3)),
        calibration_ref=str(payload.get("calibration_ref", "")),
    )
    if not config.calibration_ref:
        raise RuntimeError("calibration_ref is required in the frozen table")
    calibration_hash = hashlib.sha256(
        CALIBRATION_PATH.read_bytes()
    ).hexdigest()
    return config, calibration_hash


def run_cell(case_id: str, arm: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _load_cached_phase_one(case_id)
    if checkpoint is None:
        raise RuntimeError(f"missing cached Phase-I checkpoint for {case_id}")
    checkpoint = dataclasses.replace(checkpoint, total_budget_fes=TOTAL_FES)
    config, calibration_hash = load_config()
    config = dataclasses.replace(
        config, handoff_enabled=(arm == "on")
    )
    result = run_oc_episode_schedule_v4(
        problem, checkpoint, action_seed=ACTION_SEED, config=config
    )
    return {
        "case_id": case_id,
        "arm": arm,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "manifest_hash": _manifest(),
        "scheduler_version": result.scheduler_version,
        "calibration_hash": calibration_hash,
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
            "magnitude_repairs": result.magnitude_repairs,
        },
    }


def _cell_reusable(case_id: str, arm: str, manifest: str, calibration_hash: str) -> bool:
    """A cached cell counts only against the same scheduler version,
    implementation manifest and frozen calibration (P0 review)."""

    path = OUTPUT_ROOT / "cells" / f"{case_id}_{arm}.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))["result"]
    return (
        row.get("scheduler_version") == DEFAULT_SCHEDULER_VERSION_V4
        and row.get("manifest_hash") == manifest
        and row.get("calibration_hash") == calibration_hash
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config, calibration_hash = load_config()
    manifest = _manifest()
    cells_dir = OUTPUT_ROOT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    pending = [
        job for job in ((c, a) for c in CASES for a in ARMS)
        if not _cell_reusable(job[0], job[1], manifest, calibration_hash)
    ]
    for case_id, arm in ((c, a) for c in CASES for a in ARMS):
        if _cell_reusable(case_id, arm, manifest, calibration_hash):
            rows.append(
                json.loads(
                    (cells_dir / f"{case_id}_{arm}.json").read_text(encoding="utf-8")
                )["result"]
            )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell, case_id, arm): (case_id, arm)
                for case_id, arm in pending
            }
            for future in as_completed(futures):
                case_id, arm = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    import traceback

                    print(f"{case_id}/{arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    failures.append({"case": case_id, "arm": arm,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    continue
                (cells_dir / f"{case_id}_{arm}.json").write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{case_id}/{arm}: final={row['result']['final_error']:.6g} "
                    f"switches={row['result']['switches']} "
                    f"cold={row['result']['cold_start_probe_tax_fes']} "
                    f"dev={row['result']['development_fes']}",
                    flush=True,
                )

    mechanism: dict[str, bool] = {}
    for row in rows:
        key = f"{row['case_id']}_{row['arm']}"
        result = row["result"]
        mechanism[f"{key}_schema_v4"] = result["schema_version"] == OC_EPISODE_SCHEMA_V4
        mechanism[f"{key}_terminal_exact"] = result["terminal_fes"] == TOTAL_FES
        mechanism[f"{key}_audit_all"] = all(result["audit"].values())
        mechanism[f"{key}_manifest_stamped"] = row["manifest_hash"] == manifest
        mechanism[f"{key}_probes_four_first"] = (
            len([r for r in result["receipts"] if r["grant_kind"] == "probe"]) == 4
            and all(r["grant_kind"] == "probe" for r in result["receipts"][:4])
        )
        mechanism[f"{key}_tickets_or_unaffordable"] = bool(result["tickets"])
        mechanism[f"{key}_progress_contract_surface"] = all(
            r["progress_after"]["contract"] for r in result["receipts"]
        )
        # Private credit promotion-only: every exploit grant went to the
        # recorded leader.
        mechanism[f"{key}_exploit_grants_to_leader"] = all(
            r["leader"] == r["episode"] for r in result["receipts"] if r["grant_kind"] == "exploit"
        )
        # CTP never matures inside coverage.
        mechanism[f"{key}_ctp_not_mature_in_coverage"] = all(
            r["progress_after"]["phase"] != "coverage" or not r["progress_after"]["protocol_mature"]
            for r in result["receipts"] if r["episode"] == "ctp"
        )
        # Handoff receipts keep the refusal semantics legal.
        mechanism[f"{key}_handoff_semantics"] = all(
            (h["adopted"] and h["refusal"] == "none")
            or (not h["adopted"] and h["refusal"] in ("not_better", "oob_incumbent", "disabled"))
            for h in result["handoffs"]
        )
    across = {
        "escalation_exercised": any(
            any(r["grant_kind"] == "escalation" for r in row["result"]["receipts"]) for row in rows
        ),
        "unaffordable_ticket_recorded": any(
            any(not t["affordable"] for t in row["result"]["tickets"]) for row in rows
        ),
        "forced_challenger_exercised": any(
            any(r["grant_kind"] == "challenger" for r in row["result"]["receipts"]) for row in rows
        ),
        # Informational at 600k: the plan only guarantees minimal probes at
        # this scale; AOR's two-window ticket is exercised in the unit
        # tests and must be affordable at 3M (Gate 51b budget arithmetic).
        "aor_two_window_ticket": any(
            any(
                t["episode"] == "aor" and t["affordable"] and t["protocol_mature_after"]
                for t in row["result"]["tickets"]
            )
            for row in rows
        ),
    }
    gate_passed = (
        not failures
        and len(rows) == len(CASES) * len(ARMS)
        and all(mechanism.values())
        and across["forced_challenger_exercised"]
        and across["unaffordable_ticket_recorded"]
        and across["escalation_exercised"]
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "cases": list(CASES),
            "arms": list(ARMS),
            "total_fes": TOTAL_FES,
            "action_seed": ACTION_SEED,
            "mechanism_screening_only": True,
        },
        "implementation_manifest_hash": manifest,
        "calibration": {
            "calibration_ref": config.calibration_ref,
            "maturity_window_fes": config.maturity_window_fes,
            "revelation_horizon_fes": config.revelation_horizon_fes,
            "exploration_and_development_cap": config.exploration_and_development_cap,
            "exploitation_reserve_ratio": config.exploitation_reserve_ratio,
        },
        "mechanism_checks": mechanism,
        "across_case_flags": across,
        "gate_passed": gate_passed,
        "failures": failures,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"passed": gate_passed, "across": across}, indent=1))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
