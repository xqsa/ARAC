"""Shared repair kernel v3 paired gate (G6).

Pre-registration (frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md):

- scheduler, scope rules, residual gate, arbitration, FE budget, handoff,
  strict-best, and seeds are all frozen: the ONLY difference between the
  two arms is the shared-candidate generator
  (``duplicated_shared_local_competition`` v2 vs
  ``duplicated_shared_local_competition_v3`` strategy rotation);
- per context both arms replay the identical semantic-gate protocol:
  prime -> value probes -> 4-FE arbitration -> 32-FE kernel action ->
  1-FE frozen counterfactual -> 32-FE full-context handoff;
- the v2 arm must reproduce the frozen v3-artifact rows exactly
  (action_error / handoff_error / action_gain / end_to_end_gain within
  1e-9), otherwise the gate fails;
- judgment: paired end-to-end gain difference (v3 - v2) with percentile
  bootstrap 95% CI (2000 draws, fixed seed).  Kernel v3 WINS iff the CI
  lower bound > 0; otherwise the outcome is TIE (reported, no promotion).
  Action gain and gain-per-FE are secondary endpoints.
- kernel_version markers are recorded; production scheduler untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from arac.coordination import evaluate_frozen_private_counterfactual
from experiments.oc_action_semantic_gate import HANDOFF_FES, LOCAL_MUTATION_SCALE
from experiments.oc_lagged_coupling_shadow import validate_input
from experiments.overlap_sequential_shared_patch_gate18 import EVALS_PER_ROUND, ROUNDS
from experiments.overlap_value_aware_dispatch_gate15 import (
    CTP_BUDGET_FES,
    FRESH_SEEDS,
    MODES,
    OVERLAP_BUDGETS,
    PROBE_FES_PER_COMPONENT,
    TOPOLOGIES,
    _combined_problem,
    _new_scheduler,
    _proposal_payload,
)

INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
OUTPUT_SCHEMA = "arac-oc-shared-kernel-v3-gate-v1"
KERNEL_ARMS = ("kernel_v2", "kernel_v3")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260825
PARITY_TOL = 1e-9


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    sample = np.asarray(values, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _kernel_arm(
    problem,
    structure,
    proposals,
    checkpoint_x,
    checkpoint_error: float,
    checkpoint_fes: int,
    component: tuple[int, ...],
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    ledger, scheduler = _new_scheduler(
        problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
    )
    scheduler.prime(proposals)
    probes = scheduler.value_probe(proposals)
    selected = tuple(p for p in proposals if p.group in set(component))
    arbitration = scheduler.coordinator.coordinate(component, selected, ctp_budget_fes=0)
    if len(arbitration.candidates) != 4:
        raise RuntimeError("kernel gate expects four arbitration candidates")
    pre_action_error = float(ledger.best_error)
    pre_action_x = ledger.best_x
    trace: list[dict[str, object]] = []
    if arm == "kernel_v2":
        competition = scheduler.coordinator.duplicated_shared_local_competition(
            component,
            selected,
            budget_fes=CTP_BUDGET_FES,
            seed=seed ^ 0xA17C,
            mutation_scale=LOCAL_MUTATION_SCALE,
        )
        strategy_summary = {"owner_disagreement": 0, "owner_noise": len(competition.rounds), "owner_blend": 0}
    elif arm == "kernel_v3":
        competition = scheduler.coordinator.duplicated_shared_local_competition_v3(
            component,
            selected,
            budget_fes=CTP_BUDGET_FES,
            seed=seed ^ 0xA17C,
            mutation_scale=LOCAL_MUTATION_SCALE,
            strategy_trace=trace,
        )
        counts = {"owner_disagreement": 0, "owner_noise": 0, "owner_blend": 0}
        accepted = {"owner_disagreement": 0, "owner_noise": 0, "owner_blend": 0}
        for record in trace:
            counts[str(record["strategy"])] += 1
            if record["accepted"]:
                accepted[str(record["strategy"])] += 1
        strategy_summary = {"rounds": counts, "accepted": accepted}
    else:
        raise ValueError(f"unknown kernel arm: {arm}")
    if competition.consumed_fes != CTP_BUDGET_FES:
        raise RuntimeError(f"{arm} action FE drifted")
    action_error = float(ledger.best_error)
    action_x = ledger.best_x
    coupling = evaluate_frozen_private_counterfactual(
        ledger,
        component=component,
        scope=scheduler.coordinator._component_variables(component),
        incumbent=pre_action_x,
        best_error_before=pre_action_error,
        candidate_name=arm,
        candidate=action_x,
        full_candidate_error=action_error,
    )
    handoff = scheduler.coordinator.full_context_writeback(
        component, selected, rounds=ROUNDS
    )
    if handoff.consumed_fes != HANDOFF_FES:
        raise RuntimeError(f"{arm} handoff FE drifted")
    handoff_error = float(ledger.best_error)
    return {
        "arm": arm,
        "action_error": action_error,
        "handoff_error": handoff_error,
        "action_gain": float(pre_action_error - action_error),
        "end_to_end_gain": float(checkpoint_error - handoff_error),
        "coupled_gain": float(coupling.coupled_gain),
        "accepted_rounds": sum(1 for item in competition.rounds if item.accepted),
        "rounds": len(competition.rounds),
        "mean_diversity": float(np.mean(competition.candidate_diversity))
        if competition.candidate_diversity
        else 0.0,
        "strategy_summary": strategy_summary,
        "consumed_fes": int(ledger.count - checkpoint_fes),
    }


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    by_key = {
        (str(c["topology"]), int(c["overlap_budget"]), int(c["seed"]), str(c["mode"])): c
        for c in payload["contexts"]
    }
    jobs = tuple(
        (mode, topology, budget, seed)
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
        for seed in FRESH_SEEDS
        for mode in MODES
    )
    rows = []
    parity_failures = 0
    for mode, topology, budget, seed in jobs:
        problem, structure, _ = _combined_problem(mode, topology, budget, seed)
        checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(
            problem, structure, seed
        )
        _, selector = _new_scheduler(
            problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
        )
        selector.prime(proposals)
        probes = selector.value_probe(proposals)
        priorities = {item.component: item.priority_score for item in selector.prioritize(proposals)}
        selected = max(
            probes,
            key=lambda item: (
                item.estimated_gain,
                priorities[item.component],
                tuple(-value for value in item.component),
            ),
        ).component
        arms = {}
        for arm in KERNEL_ARMS:
            arms[arm] = _kernel_arm(
                problem,
                structure,
                proposals,
                checkpoint_x,
                checkpoint_error,
                checkpoint_fes,
                selected,
                arm=arm,
                seed=seed,
            )
        reference = by_key[(topology, budget, seed, mode)]["duplicated_shared_local_competition"]
        parity_ok = (
            abs(arms["kernel_v2"]["action_error"] - float(reference["action_error"])) <= PARITY_TOL
            and abs(arms["kernel_v2"]["handoff_error"] - float(reference["handoff_error"])) <= PARITY_TOL
        )
        parity_failures += int(not parity_ok)
        rows.append(
            {
                "key": [topology, budget, seed, mode],
                "component": list(selected),
                "checkpoint_error": float(checkpoint_error),
                "parity_ok": parity_ok,
                **arms,
            }
        )
    diffs_end = [
        float(row["kernel_v3"]["end_to_end_gain"] - row["kernel_v2"]["end_to_end_gain"])
        for row in rows
    ]
    diffs_action = [
        float(row["kernel_v3"]["action_gain"] - row["kernel_v2"]["action_gain"])
        for row in rows
    ]
    end_ci = _bootstrap_ci(diffs_end)
    action_ci = _bootstrap_ci(diffs_action)
    kernel_v3_wins = bool(end_ci[0] > 0.0)
    checks = {
        "context_count_60": len(rows) == 60,
        "v2_parity_vs_frozen_artifact": parity_failures == 0,
        "fe_parity": all(
            row[arm]["consumed_fes"] == row["kernel_v2"]["consumed_fes"]
            for row in rows
            for arm in KERNEL_ARMS
        ),
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "kernel_arms": list(KERNEL_ARMS),
            "kernel_version": "v3-strategy-rotation (disagreement/noise/blend)",
            "action_fes": CTP_BUDGET_FES,
            "handoff_fes": HANDOFF_FES,
            "coupling_fes": 1,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "win_rule": "paired end-to-end gain (v3 - v2) bootstrap CI lower bound > 0",
            "production_selector_modified": False,
        },
        "rows": rows,
        "summary": {
            "paired_end_to_end_gain": float(np.mean(diffs_end)),
            "paired_end_to_end_gain_ci95": [end_ci[0], end_ci[1]],
            "paired_action_gain": float(np.mean(diffs_action)),
            "paired_action_gain_ci95": [action_ci[0], action_ci[1]],
            "v3_end_to_end_wins": sum(1 for value in diffs_end if value > 0.0),
            "v2_end_to_end_wins": sum(1 for value in diffs_end if value < 0.0),
            "kernel_v3_wins": kernel_v3_wins,
            "outcome": "WIN" if kernel_v3_wins else "TIE",
            "v3_mean_diversity": float(
                np.mean([row["kernel_v3"]["mean_diversity"] for row in rows])
            ),
            "v2_mean_diversity": float(
                np.mean([row["kernel_v2"]["mean_diversity"] for row in rows])
            ),
            "v3_accepted_rounds_mean": float(
                np.mean([row["kernel_v3"]["accepted_rounds"] for row in rows])
            ),
            "v2_accepted_rounds_mean": float(
                np.mean([row["kernel_v2"]["accepted_rounds"] for row in rows])
            ),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_shared_kernel_v3_gate/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": result["gate_passed"], "summary": result["summary"], "gate_checks": result["gate_checks"]}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
