"""Gate 51c (v5.2): fresh-seed re-judgment of the bounded-runway scheduler.

OC arms rerun under scheduler v5.2 in a version-isolated directory.  The
standalone arms and the Phase-I checkpoints are reused verbatim from the
frozen v5.1 gate51c run: neither is on the scheduler's execution path, and
each reused cell is anchored by the v5.1 confirmation's implementation
manifest plus the checkpoint hash of the Phase-I cache both arms share.
Judgment math and protocol constants are imported unchanged from the v5.1
entry, so the two judgments are comparable by construction.

Pre-registered levers for v5.2 (docs/arac-oc-gate51c-protocol.md appendix):
the protected-runway verification window is bounded by one calibrated
maturity window (plateau release after w1 FE instead of a full segment)
and the released state is a receipt/audit field.
"""

# Thread caps must be set before NumPy imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import ctypes
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
import math
from pathlib import Path

from arac.benchmarks.aob import AobBenchmark
from arac.coordination.episodes import (
    DEFAULT_SCHEDULER_VERSION_V5_2,
    PhaseAwareSchedulerConfig,
    run_oc_episode_schedule_v5_2,
)
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from arac.runtime.manifest import implementation_manifest_hash
from experiments.oc_phase_aware_gate51a import load_config as load_v4_config
from experiments.oc_phase_aware_gate51c import (
    ANYTIME_CHECKPOINTS,
    ARMS,
    CASES,
    CELL_SCHEMA,
    FRESH_SEEDS,
    NONINFERIOR_TOL,
    OUTPUT_SCHEMA,
    PHASE1_FES,
    STRICT_WIN_TOL,
    STANDALONE_ARMS,
    TOTAL_FES,
    WORST_SEED_TOL,
    _anytime_points,
    _log_error_auc,
    _sample_anytime,
)

SCHEDULER_VERSION = DEFAULT_SCHEDULER_VERSION_V5_2
OUTPUT_ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_2")
PREDECESSOR_ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_1")
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("src/arac/actions/phase2_v2.py"),
    Path("src/arac/runtime/phase2.py"),
    Path("src/arac/runtime/contracts.py"),
    Path("experiments/oc_phase_aware_gate51c.py"),
    Path("experiments/oc_phase_aware_gate51c_v5_2.py"),
    Path("docs/arac-oc-gate51c-protocol.md"),
]


def _manifest() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return implementation_manifest_hash(repo_root, MANIFEST_FILES)


def _predecessor_manifest() -> str:
    """Manifest anchor of the frozen v5.1 run that standalones come from."""

    path = PREDECESSOR_ROOT / "confirmation.json"
    if not path.exists():
        raise RuntimeError(
            f"frozen v5.1 confirmation missing: {path}. The standalone reuse "
            "of this gate is anchored by that run's implementation manifest."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    if (
        list(protocol.get("cases", [])) != list(CASES)
        or list(protocol.get("fresh_seeds", [])) != list(FRESH_SEEDS)
        or list(protocol.get("arms", [])) != list(ARMS)
    ):
        raise RuntimeError("frozen v5.1 protocol grid does not match this gate's grid")
    return str(payload["implementation_manifest_hash"])


def _load_scheduler_config() -> PhaseAwareSchedulerConfig:
    return dataclasses.replace(
        load_v4_config()[0],
        adaptive_exploration=True,
        scheduler_version=SCHEDULER_VERSION,
        horizon_protected=True,
    )


def _pin_p_cores() -> None:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_ulonglong)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 0xFFFF):
        raise RuntimeError("SetProcessAffinityMask failed")


def _parse_phase_one(payload: dict) -> PhaseCheckpoint:
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


def _phase_one(problem, case_id: str, seed: int) -> PhaseCheckpoint:
    """Load the Phase-I checkpoint from the frozen v5.1 cache.

    Phase-I discovery is scheduler-independent (soft-RDDSM + mmes burn-in),
    so the v5.1 caches are authoritative for this grid; this re-judgment
    never regenerates them.  A cache miss means the grid changed, which is
    a protocol decision, not something to paper over here.
    """

    own = OUTPUT_ROOT / "phase1" / f"{case_id}_{seed}.json"
    if own.exists():
        return _parse_phase_one(json.loads(own.read_text(encoding="utf-8")))
    source = PREDECESSOR_ROOT / "phase1" / f"{case_id}_{seed}.json"
    if not source.exists():
        raise RuntimeError(
            f"Phase-I cache missing for {case_id}/{seed}: expected {source}. "
            "This gate reuses the frozen v5.1 checkpoints; a new grid needs "
            "a new protocol decision, not a silent rediscovery."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CELL_SCHEMA:
        raise RuntimeError(f"Phase-I cache schema mismatch: {source}")
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return _parse_phase_one(payload)


def _cell_path(case_id: str, seed: int, arm: str) -> Path:
    return OUTPUT_ROOT / "cells" / f"{case_id}_{seed}_{arm}.json"


def run_oc_cell(case_id: str, seed: int, arm: str, manifest: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _phase_one(problem, case_id, seed)
    config = dataclasses.replace(
        _load_scheduler_config(), handoff_enabled=(arm == "on")
    )
    schedule = run_oc_episode_schedule_v5_2(
        problem, checkpoint, action_seed=seed, config=config
    )
    result: dict[str, object] = {
        "final_error": schedule.final_error,
        "terminal_fes": schedule.terminal_fes,
        "audit": schedule.audit,
        "funded_fes": schedule.funded_fes,
        "receipts": [r.__dict__ for r in schedule.receipts],
        "handoffs": [h.__dict__ for h in schedule.handoffs],
        "cold_start_probe_tax_fes": schedule.cold_start_probe_tax_fes,
        "development_fes": schedule.development_fes,
        "exploitation_fes": schedule.exploitation_fes,
        "schedule_hash": schedule.schedule_hash,
        "scheduler_version": schedule.scheduler_version,
    }
    points = _anytime_points(result, float(checkpoint.incumbent_error))
    result["anytime_points"] = points
    result["anytime"] = _sample_anytime(points)
    result["log_error_auc"] = _log_error_auc(points)
    return {
        "case_id": case_id,
        "seed": seed,
        "arm": arm,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "checkpoint_error": float(checkpoint.incumbent_error),
        "manifest_hash": manifest,
        "result": result,
    }


def _standalone_reusable(
    case_id: str, seed: int, arm: str, predecessor_manifest: str
) -> dict[str, object] | None:
    """Validate a frozen v5.1 standalone cell and adopt it for judgment.

    Scheduler code is not on the standalone execution path, so the cell is
    anchored by the v5.1 confirmation manifest, the structural completeness
    checks of the original gate, and the checkpoint hash shared with the
    Phase-I cache this gate's OC arms run from.  The frozen cells predate
    the anytime analysis layer, so their anytime trace is re-derived here
    with the v5.1 gate's own helpers from the recorded segments -- a
    deterministic enrichment of the copy, never of the frozen source.
    """

    own = _cell_path(case_id, seed, arm)
    source = PREDECESSOR_ROOT / "cells" / f"{case_id}_{seed}_{arm}.json"
    path = own if own.exists() else source
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    row = envelope["result"]
    if row.get("manifest_hash") != predecessor_manifest:
        return None
    result = row["result"]
    if result.get("terminal_fes") != TOTAL_FES:
        return None
    if sum(int(s["consumed_fes"]) for s in result["segments"]) != TOTAL_FES - PHASE1_FES:
        return None
    cache = PREDECESSOR_ROOT / "phase1" / f"{case_id}_{seed}.json"
    if not cache.exists():
        return None
    expected = _parse_phase_one(json.loads(cache.read_text(encoding="utf-8")))
    if row.get("checkpoint_hash") != expected.checkpoint_hash:
        return None
    if "anytime" not in result:
        # The frozen cells also predate the checkpoint_error field; the
        # Phase-I cache carries the same incumbent error.
        points = _anytime_points(result, float(expected.incumbent_error))
        result["anytime_points"] = points
        result["anytime"] = _sample_anytime(points)
        result["log_error_auc"] = _log_error_auc(points)
    return row


def _oc_cell_reusable(case_id: str, seed: int, arm: str, manifest: str) -> bool:
    path = _cell_path(case_id, seed, arm)
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))["result"]
    if row.get("manifest_hash") != manifest:
        return False
    result = row["result"]
    return bool(
        result.get("terminal_fes") == TOTAL_FES
        and all(result.get("audit", {}).values())
        and result.get("scheduler_version") == SCHEDULER_VERSION
        and set(result.get("anytime", {})) == {str(point) for point in ANYTIME_CHECKPOINTS}
        and math.isfinite(float(result.get("log_error_auc", math.nan)))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pin-p-cores", action="store_true")
    args = parser.parse_args()
    if args.pin_p_cores:
        _pin_p_cores()
        print("affinity pinned to P-cores", flush=True)
    manifest = _manifest()
    predecessor_manifest = _predecessor_manifest()
    (OUTPUT_ROOT / "cells").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    oc_jobs = [
        (case, seed, arm)
        for seed in FRESH_SEEDS
        for case in CASES
        for arm in ("on", "off")
    ]
    for case in CASES:
        for seed in FRESH_SEEDS:
            for arm in STANDALONE_ARMS:
                row = _standalone_reusable(case, seed, arm, predecessor_manifest)
                if row is None:
                    raise RuntimeError(
                        f"frozen v5.1 standalone cell not reusable: "
                        f"{case}/{seed}/{arm}"
                    )
                target = _cell_path(case, seed, arm)
                if not target.exists():
                    target.write_text(
                        json.dumps(
                            {"schema_version": CELL_SCHEMA, "result": row},
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                rows.append(row)
    pending = [job for job in oc_jobs if not _oc_cell_reusable(*job, manifest)]
    for job in oc_jobs:
        if _oc_cell_reusable(*job, manifest):
            rows.append(
                json.loads(_cell_path(*job).read_text(encoding="utf-8"))["result"]
            )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_oc_cell, *job, manifest): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    import traceback

                    print(f"{job}: FAILED {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    failures.append(
                        {"job": "_".join(map(str, job)), "error": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                _cell_path(*job).write_text(
                    json.dumps({"schema_version": CELL_SCHEMA, "result": row}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"{row['case_id']}/{row['seed']}/{row['arm']}: "
                    f"final={row['result']['final_error']:.6g}",
                    flush=True,
                )

    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault((row["case_id"], row["seed"]), {})[row["arm"]] = row["result"]
    per_case: dict[str, dict[str, object]] = {}
    for case in CASES:
        ratios: list[float] = []
        on_off: list[float] = []
        strict = 0
        on_better = 0
        for seed in FRESH_SEEDS:
            arms = by_key.get((case, seed), {})
            if not all(a in arms for a in ARMS):
                continue
            best = min(arms[a]["final_error"] for a in STANDALONE_ARMS)
            on = arms["on"]["final_error"]
            off = arms["off"]["final_error"]
            ratios.append(on / best)
            on_off.append(on / off)
            if on <= best * STRICT_WIN_TOL:
                strict += 1
            if on < off:
                on_better += 1
        if not ratios:
            continue
        ordered = sorted(ratios)
        median = ordered[len(ordered) // 2] if len(ordered) % 2 else (
            (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        )
        anytime_ratio: dict[str, list[float]] = {str(point): [] for point in ANYTIME_CHECKPOINTS}
        auc_ratios: list[float] = []
        for seed in FRESH_SEEDS:
            arms = by_key.get((case, seed), {})
            if not all(a in arms for a in ARMS):
                continue
            best_arm = min(
                STANDALONE_ARMS,
                key=lambda arm: float(arms[arm].get("final_error", math.inf)),
            )
            on_anytime = arms["on"].get("anytime", {})
            best_anytime = arms[best_arm].get("anytime", {})
            for point in anytime_ratio:
                if point in on_anytime and point in best_anytime:
                    anytime_ratio[point].append(
                        float(on_anytime[point]) / max(float(best_anytime[point]), 1e-300)
                    )
            if "log_error_auc" in arms["on"] and "log_error_auc" in arms[best_arm]:
                auc_ratios.append(
                    float(arms["on"]["log_error_auc"])
                    / max(float(arms[best_arm]["log_error_auc"]), 1e-300)
                )
        per_case[case] = {
            "ratios": ratios,
            "median_ratio": median,
            "worst_ratio": max(ratios),
            "strict_win_seeds": strict,
            "on_beats_off_seeds": on_better,
            "on_off": on_off,
            "anytime_ratio_on_vs_best": anytime_ratio,
            "log_error_auc_on_vs_best": auc_ratios,
        }
    checks: dict[str, bool] = {}
    for case, stats in per_case.items():
        checks[f"{case}_median_le_1.05"] = stats["median_ratio"] <= NONINFERIOR_TOL + 1e-9
        checks[f"{case}_worst_le_1.10"] = stats["worst_ratio"] <= WORST_SEED_TOL + 1e-9
    strict_cases = [
        c for c, s in per_case.items() if s["median_ratio"] <= STRICT_WIN_TOL + 1e-9
    ]
    handoff_confirm = all(
        per_case[c]["on_beats_off_seeds"] >= 2 for c in strict_cases
    ) if strict_cases else True
    checks["all_cells_complete"] = len(rows) + len(failures) == len(
        [(c, s, a) for s in FRESH_SEEDS for c in CASES for a in ARMS]
    ) and not failures
    checks["protocol_audits_all"] = all(
        all(arms[a].get("audit", {"ok": True}).values())
        for arms in by_key.values()
        for a in ("on", "off")
        if a in arms
    )
    gate_passed = (
        all(v for k, v in checks.items() if k.endswith("_median_le_1.05"))
        and all(v for k, v in checks.items() if k.endswith("_worst_le_1.10"))
        and handoff_confirm
        and checks["all_cells_complete"]
        and checks["protocol_audits_all"]
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "scheduler_version": SCHEDULER_VERSION,
        "standalone_reuse": {
            "root": str(PREDECESSOR_ROOT),
            "implementation_manifest_hash": predecessor_manifest,
        },
        "protocol": {
            "cases": list(CASES),
            "fresh_seeds": list(FRESH_SEEDS),
            "arms": list(ARMS),
            "total_fes": TOTAL_FES,
            "noninferior_tol": NONINFERIOR_TOL,
            "worst_seed_tol": WORST_SEED_TOL,
            "strict_win_tol": STRICT_WIN_TOL,
        },
        "implementation_manifest_hash": manifest,
        "per_case": per_case,
        "strict_win_cases": strict_cases,
        "checks": checks,
        "gate_passed": gate_passed,
        "failures": failures,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "median_ratios": {c: round(s["median_ratio"], 4) for c, s in per_case.items()},
        "worst_ratios": {c: round(s["worst_ratio"], 4) for c, s in per_case.items()},
        "strict_win_cases": strict_cases,
        "on_beats_off_seeds": {c: s["on_beats_off_seeds"] for c, s in per_case.items()},
        "passed": gate_passed,
    }
    print(json.dumps(summary, indent=1))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
