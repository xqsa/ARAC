"""Gate 51c: fresh-seed confirmation of the v5.1 frozen candidate.

Three seeds that never touched any design decision each get a full
pipeline per case: fresh Phase-I checkpoint, four standalones (the
same-seed reference), and the OC handoff ON/OFF arms.  Judgment follows
the pre-registered protocol (docs/arac-oc-gate51c-protocol.md).
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
    PhaseAwareSchedulerConfig,
    run_oc_episode_schedule_v5_1,
)
from arac.evidence.hierarchical import Phase1Evidence
from arac.evidence.soft_rddsm import SoftDsmConfig, discover_hierarchical_soft
from arac.runtime.contracts import (
    ActionContext,
    PhaseCheckpoint,
    RelationEvidence,
)
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.manifest import implementation_manifest_hash
from arac.runtime.phase2 import Phase2StateError
from experiments.oc_action_episode_gate50 import EXECUTORS

CASES = ("R2", "A3", "S5", "R6")
STANDALONE_ARMS = ("ctp", "gcb", "smp", "aor")
ARMS = (*STANDALONE_ARMS, "on", "off")
FRESH_SEEDS = (20260901, 20260902, 20260903)
PHASE1_FES = 180_000
TOTAL_FES = 3_000_000
SEGMENT_FES = 150_000
OUTPUT_ROOT = Path("artifacts/oc_phase_aware_gate51c_v5_1")
CELL_SCHEMA = "arac-oc-gate51c-cell-v1"
OUTPUT_SCHEMA = "arac-oc-gate51c-v1"
NONINFERIOR_TOL = 1.05
WORST_SEED_TOL = 1.10
STRICT_WIN_TOL = 0.98
ANYTIME_CHECKPOINTS = (600_000, 1_000_000, 2_000_000, 3_000_000)
MANIFEST_FILES = [
    Path("src/arac/coordination/episodes.py"),
    Path("src/arac/actions/phase2_v2.py"),
    Path("src/arac/runtime/phase2.py"),
    Path("src/arac/runtime/contracts.py"),
    Path("experiments/oc_phase_aware_gate51c.py"),
    Path("docs/arac-oc-gate51c-protocol.md"),
]


def _manifest() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return implementation_manifest_hash(repo_root, MANIFEST_FILES)


def _load_v5_1_config() -> PhaseAwareSchedulerConfig:
    from experiments.oc_phase_aware_gate51b_v5_1 import load_config

    return load_config()


def _pin_p_cores() -> None:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_ulonglong)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 0xFFFF):
        raise RuntimeError("SetProcessAffinityMask failed")


def _phase_one(problem, case_id: str, seed: int) -> PhaseCheckpoint:
    """Fresh Phase-I discovery for one (case, seed); cached on disk."""

    cache = OUTPUT_ROOT / "phase1" / f"{case_id}_{seed}.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("schema_version") == CELL_SCHEMA:
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
    from arac.runtime.optimizers import PypopOptimizerPort

    ledger = EvaluationLedger(problem, PHASE1_FES)
    discovery = discover_hierarchical_soft(
        problem, ledger, run_seed=seed, config=SoftDsmConfig()
    )
    if ledger.remaining:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(v) for v in ledger.best_x),
            sigma=0.5,
            seed=seed ^ 0x1D_E71D,
            budget_fes=ledger.remaining,
            population_size=24,
            restart=False,
        )
    if ledger.count != PHASE1_FES:
        raise RuntimeError("Phase-I did not land on the exact boundary")
    evidence: Phase1Evidence = discovery.evidence
    leaves = sorted(evidence.region_tree.leaves, key=lambda leaf: leaf.node_id)
    leaf_index = {leaf.node_id: position for position, leaf in enumerate(leaves)}
    blocks = tuple(tuple(leaf.variables) for leaf in leaves)
    relations = tuple(
        RelationEvidence(
            left_block=min(leaf_index[relation.left], leaf_index[relation.right]),
            right_block=max(leaf_index[relation.left], leaf_index[relation.right]),
            strength=float(relation.score),
            disagreement=0.1,
        )
        for relation in evidence.region_relations
        if relation.left in leaf_index and relation.right in leaf_index
    )
    checkpoint = PhaseCheckpoint(
        protocol="gate51c-fresh-v1",
        run_seed=seed,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=tuple(float(v) for v in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=("log10_center_error", "line_high_frequency_fraction_median"),
        feature_values=(math.log10(max(float(ledger.best_error), 1.0)), 0.4),
        blocks=blocks,
        relations=relations,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "schema_version": CELL_SCHEMA,
                "checkpoint": {
                    "protocol": checkpoint.protocol,
                    "run_seed": checkpoint.run_seed,
                    "total_budget_fes": checkpoint.total_budget_fes,
                    "phase1_fes": checkpoint.phase1_fes,
                    "incumbent": list(checkpoint.incumbent),
                    "incumbent_error": checkpoint.incumbent_error,
                    "feature_names": list(checkpoint.feature_names),
                    "feature_values": list(checkpoint.feature_values),
                    "blocks": [list(block) for block in checkpoint.blocks],
                    "relations": [
                        {
                            "left_block": r.left_block,
                            "right_block": r.right_block,
                            "strength": r.strength,
                            "disagreement": r.disagreement,
                        }
                        for r in checkpoint.relations
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint


def _cell_path(case_id: str, seed: int, arm: str) -> Path:
    return OUTPUT_ROOT / "cells" / f"{case_id}_{seed}_{arm}.json"


def _anytime_points(result: dict[str, object], checkpoint_error: float) -> list[dict[str, float]]:
    """Return a monotone strict-best trace in total-FE coordinates.

    Receipts are segment boundaries, so the trace is piecewise constant between
    boundaries.  Keeping the Phase-I point makes curves comparable even when a
    standalone episode and OC spend different Phase-II FE on their first
    executable unit.
    """

    points: list[dict[str, float]] = [
        {"total_fes": float(PHASE1_FES), "error": float(checkpoint_error)}
    ]
    if "segments" in result:
        for segment in result["segments"]:
            points.append(
                {
                    "total_fes": float(PHASE1_FES + int(segment["cumulative_phase2_fes"])),
                    "error": float(segment["error_after"]),
                }
            )
    else:
        consumed = 0
        for receipt in result.get("receipts", []):
            consumed += int(receipt["consumed_fes"])
            points.append(
                {
                    "total_fes": float(PHASE1_FES + consumed),
                    "error": float(receipt["global_error_after"]),
                }
            )
    points.sort(key=lambda point: point["total_fes"])
    compact: list[dict[str, float]] = []
    for point in points:
        if compact and point["total_fes"] == compact[-1]["total_fes"]:
            compact[-1] = point
        elif not compact or point["error"] <= compact[-1]["error"]:
            compact.append(point)
        else:
            # A strict-best archive must never worsen.  Keep the lower archive
            # value visible if an imported legacy segment violates the claim.
            compact.append({"total_fes": point["total_fes"], "error": compact[-1]["error"]})
    return compact


def _sample_anytime(points: list[dict[str, float]], checkpoints: tuple[int, ...] = ANYTIME_CHECKPOINTS) -> dict[str, float]:
    """Sample a piecewise-constant strict-best curve at fixed total FE."""

    if not points:
        raise ValueError("anytime points must be non-empty")
    sampled: dict[str, float] = {}
    index = 0
    for checkpoint in checkpoints:
        while index + 1 < len(points) and points[index + 1]["total_fes"] <= checkpoint:
            index += 1
        sampled[str(checkpoint)] = float(points[index]["error"])
    return sampled


def _log_error_auc(points: list[dict[str, float]], *, end_fes: int = TOTAL_FES) -> float:
    """Trapezoidal AUC of log10 strict-best error over the Phase-II budget."""

    if not points:
        raise ValueError("anytime points must be non-empty")
    ordered = sorted(points, key=lambda point: point["total_fes"])
    if ordered[-1]["total_fes"] < end_fes:
        ordered.append({"total_fes": float(end_fes), "error": ordered[-1]["error"]})
    area = 0.0
    start = float(PHASE1_FES)
    previous = ordered[0]
    for current in ordered[1:]:
        left = max(float(previous["total_fes"]), start)
        right = min(float(current["total_fes"]), float(end_fes))
        if right > left:
            y0 = math.log10(max(float(previous["error"]), 1e-300))
            y1 = math.log10(max(float(current["error"]), 1e-300))
            area += (right - left) * (y0 + y1) / 2.0
        previous = current
    return area / max(float(end_fes - PHASE1_FES), 1.0)


def _run_standalone(problem, checkpoint: PhaseCheckpoint, arm: str, seed: int) -> dict[str, object]:
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_FES,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=True,
    )
    context = ActionContext(
        arm, checkpoint, problem, ledger, action_seed=seed, retain_trajectory=False
    )
    state = EXECUTORS[arm]().initialize(context)
    segments: list[dict[str, object]] = []
    index = 0
    while ledger.remaining > 0 and not state.complete:
        request = min(SEGMENT_FES, ledger.remaining)
        error_before = float(ledger.best_error)
        request_work = request
        consumed = 0
        while consumed == 0:
            try:
                step = state.step(request_work)
                consumed = step.step_fes
            except Phase2StateError:
                if request_work >= ledger.remaining:
                    raise
                request_work = min(request_work * 2, ledger.remaining)
        segments.append(
            {
                "segment_index": index,
                "requested_fes": request,
                "consumed_fes": consumed,
                "cumulative_phase2_fes": ledger.count - checkpoint.phase1_fes,
                "error_before": error_before,
                "error_after": float(ledger.best_error),
                "state_hash": state.snapshot().state_hash,
            }
        )
        index += 1
    return {
        "final_error": float(ledger.best_error),
        "terminal_fes": ledger.count,
        "route": state.route,
        "segments": segments,
    }


def run_cell(case_id: str, seed: int, arm: str, manifest: str) -> dict[str, object]:
    problem = AobBenchmark().load(case_id)
    checkpoint = _phase_one(problem, case_id, seed)
    if arm in STANDALONE_ARMS:
        result = _run_standalone(problem, checkpoint, arm, seed)
    else:
        config = dataclasses.replace(
            _load_v5_1_config(), handoff_enabled=(arm == "on")
        )
        schedule = run_oc_episode_schedule_v5_1(
            problem, checkpoint, action_seed=seed, config=config
        )
        result = {
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


def _cell_reusable(case_id: str, seed: int, arm: str, manifest: str) -> bool:
    path = _cell_path(case_id, seed, arm)
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))["result"]
    if row.get("manifest_hash") != manifest:
        return False
    result = row["result"]
    if result.get("terminal_fes") != TOTAL_FES:
        return False
    if arm in STANDALONE_ARMS:
        return (
            sum(int(s["consumed_fes"]) for s in result["segments"]) == TOTAL_FES - PHASE1_FES
            and set(result.get("anytime", {})) == {str(point) for point in ANYTIME_CHECKPOINTS}
            and math.isfinite(float(result.get("log_error_auc", math.nan)))
        )
    return (
        all(result.get("audit", {}).values())
        and result.get("scheduler_version") == "v5.1"
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
    (OUTPUT_ROOT / "cells").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    jobs = [
        (case, seed, arm)
        for seed in FRESH_SEEDS
        for case in CASES
        for arm in ARMS
    ]
    pending = [job for job in jobs if not _cell_reusable(*job, manifest)]
    for job in jobs:
        if _cell_reusable(*job, manifest):
            rows.append(json.loads(_cell_path(*job).read_text(encoding="utf-8"))["result"])
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell, *job, manifest): job for job in pending
            }
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
    checks["all_cells_complete"] = len(rows) + len(failures) == len(jobs) and not failures
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
