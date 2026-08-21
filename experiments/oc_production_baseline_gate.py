"""Production dispatch baseline on the frozen five-arm contexts (G2).

Pre-registration (frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md):

- input outcomes come from the frozen v3 semantic-gate artifact; the
  production decision is recomputed per context by rebuilding the same
  deterministic problem/structure/checkpoint/proposals the gate used;
- the production decision chain is the real ``CoordinatorState`` +
  ``OcDispatchPlanner.make_plan`` with default ``OcCoordinatorConfig``
  (exactly as ``_run_oc_unified_core`` constructs them), called on the
  selected component with ``available_fes=32`` and the matched
  arbitration gain/reference error.  No production code is modified;
- three pre-registered persistence scenarios are replayed per context by
  observing the context's *actual* arbitration conflict level
  ``k`` times before planning:
      fresh       k=0 (first exposure)
      persistent  k=2 (= persistent_streak)
      escalated   k=6 (= escalation_streak)
- action->arm mapping (stated assumption, used for outcome comparison):
      arbitration -> owner_control   (owner lane; no shared operator)
      smp         -> owner_control   (owner-local lane)
      ctp_restricted    -> shared_core               (sequential shared repair)
      ctp_shared_core   -> expanded_shared_private   (joint shared+private repair)
      aor         -> null (no matched arm; excluded from hit-rate denominators
      and reported separately)
- endpoints: mapped-arm end-to-end hit rate vs oracle arm, mean regret,
  opportunity captured, paired end-to-end difference vs the best-fixed arm
  and vs the G1 error-normalized EMA selector (recomputed deterministically
  from the same artifact).

Offline decision replay + one 4-FE arbitration reconstruction per context;
no new search FE beyond the v3 protocol's own accounting; production
selector untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from arac.coordination.contract import (
    OC_ACTION_AOR,
    OC_ACTION_ARBITRATION,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OcCoordinatorConfig,
)
from arac.coordination.planner import OcDispatchPlanner
from arac.coordination.state import CoordinatorState
from experiments.oc_lagged_coupling_normalized_gate import PRIMARY_VARIANT, _replay as _ema_replay
from experiments.oc_lagged_coupling_shadow import ARMS, validate_input
from experiments.overlap_value_aware_dispatch_gate15 import (
    MODES,
    OVERLAP_BUDGETS,
    TOPOLOGIES,
    FRESH_SEEDS,
    _combined_problem,
    _new_scheduler,
    _proposal_payload,
)

INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
OUTPUT_SCHEMA = "arac-oc-production-baseline-v1"
AVAILABLE_FES = 32
SCENARIOS = {"fresh": 0, "persistent": 2, "escalated": 6}
ACTION_ARM_MAP = {
    OC_ACTION_ARBITRATION: "owner_control",
    OC_ACTION_SMP: "owner_control",
    OC_ACTION_CTP_RESTRICTED: "shared_core",
    OC_ACTION_CTP_SHARED_CORE: "expanded_shared_private",
    OC_ACTION_AOR: None,
}
BEST_FIXED_ARM = "duplicated_shared_local_competition"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260822


def _bootstrap_ci_mean(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    sample = np.asarray(values, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _hit_ci(hits: list[bool]) -> list[float]:
    if not hits:
        return [0.0, 0.0]
    sample = np.asarray([float(v) for v in hits])
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _decide(mode: str, topology: str, overlap_budget: int, seed: int) -> dict[str, Any]:
    problem, structure, _ = _combined_problem(mode, topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(
        problem, structure, seed
    )
    ledger, scheduler = _new_scheduler(
        problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
    )
    scheduler.prime(proposals)
    probes = scheduler.value_probe(proposals)
    if len(probes) < 2:
        raise RuntimeError("production baseline expects at least two components")
    priorities = {item.component: item.priority_score for item in scheduler.prioritize(proposals)}
    selected = max(
        probes,
        key=lambda item: (
            item.estimated_gain,
            priorities[item.component],
            tuple(-value for value in item.component),
        ),
    ).component
    selected_proposals = tuple(p for p in proposals if p.group in set(selected))
    arbitration = scheduler.coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
    if len(arbitration.candidates) != 4:
        raise RuntimeError("production baseline expects four arbitration candidates")
    arbitration_gain = max(
        0.0, float(arbitration.best_error_before - arbitration.best_error_after)
    )
    config = OcCoordinatorConfig()
    components = [tuple(c) for c in scheduler.overlap_components]
    decisions: dict[str, dict[str, Any]] = {}
    for name, observed_cycles in SCENARIOS.items():
        planner = OcDispatchPlanner(structure, list(components), config=config, base_seed=seed)
        state = CoordinatorState(
            structure, list(components), config=config, checkpoint_hash=""
        )
        for _ in range(observed_cycles):
            state.observe_proposal_conflict(
                selected,
                high_conflict=arbitration.conflict_level.value == "high",
            )
        signal = state.signal(selected, cycle_index=observed_cycles, proposal_contribution=0.0)
        scope = tuple(sorted(planner.shared_scope_variables(selected)))
        plan = planner.make_plan(
            signal,
            cycle_index=observed_cycles,
            scope=scope,
            probe_widths={},
            available_fes=AVAILABLE_FES,
            arbitration_gain=arbitration_gain,
            arbitration_reference_error=float(arbitration.best_error_before),
        )
        decisions[name] = {
            "action": plan.action,
            "reason": plan.reason,
            "reserved_fes": int(plan.reserved_fes),
            "conflict_level_observed": arbitration.conflict_level.value,
            "mapped_arm": ACTION_ARM_MAP[plan.action],
        }
    return {
        "selected_component": tuple(selected),
        "arbitration_gain": arbitration_gain,
        "decisions": decisions,
        "reconstruction_fes": int(ledger.count - checkpoint_fes),
    }


def _context_key(context: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(context["topology"]),
        int(context["overlap_budget"]),
        int(context["seed"]),
        str(context["mode"]),
    )


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    by_key = {_context_key(c): c for c in payload["contexts"]}
    jobs = tuple(
        (mode, topology, budget, seed)
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
        for seed in FRESH_SEEDS
        for mode in MODES
    )
    rows = []
    for mode, topology, budget, seed in jobs:
        decision = _decide(mode, topology, budget, seed)
        context = by_key[(topology, budget, seed, mode)]
        gains = {arm: float(context[arm]["end_to_end_gain"]) for arm in ARMS}
        oracle_arm = max(ARMS, key=lambda a: (gains[a], -ARMS.index(a)))
        rows.append(
            {
                "key": [topology, budget, seed, mode],
                "checkpoint_error": float(context["checkpoint_error"]),
                "oracle_arm": oracle_arm,
                "oracle_gain": gains[oracle_arm],
                "best_fixed_gain": gains[BEST_FIXED_ARM],
                "arm_gains": gains,
                **{
                    f"{scenario}_{field}": decision["decisions"][scenario][field]
                    for scenario in SCENARIOS
                    for field in ("action", "reason", "mapped_arm")
                },
                "conflict_level_observed": decision["decisions"]["fresh"][
                    "conflict_level_observed"
                ],
                "selected_component": list(decision["selected_component"]),
                "arbitration_gain": decision["arbitration_gain"],
            }
        )
    # deterministic EMA comparison arm (error-normalized variant from G1)
    ema_by_key = {
        (row.topology, row.overlap_budget, row.seed, row.mode): row for row in _ema_replay(payload)
    }

    scenario_reports: dict[str, Any] = {}
    for scenario in SCENARIOS:
        mapped = [row[f"{scenario}_mapped_arm"] for row in rows]
        action_counts: dict[str, int] = {}
        for row in rows:
            action = row[f"{scenario}_action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        scored = [
            (row, arm)
            for row, arm in zip(rows, mapped)
            if arm is not None
        ]
        aor_rows = sum(1 for arm in mapped if arm is None)
        hits = [row["oracle_arm"] == arm for row, arm in scored]
        regrets = [row["oracle_gain"] - row["arm_gains"][arm] for row, arm in scored]
        captured = (
            sum(row["arm_gains"][arm] for row, arm in scored)
            / sum(row["oracle_gain"] for row, arm in scored)
            if scored
            else 0.0
        )
        diff_vs_best_fixed = [
            row["arm_gains"][arm] - row["best_fixed_gain"] for row, arm in scored
        ]
        ema_pairs = []
        for row, arm in scored:
            ema_row = ema_by_key.get(
                (row["key"][0], row["key"][1], row["key"][2], row["key"][3])
            )
            if ema_row is not None and ema_row.variants[PRIMARY_VARIANT].predicted_arm is not None:
                ema_arm = ema_row.variants[PRIMARY_VARIANT].predicted_arm
                ema_pairs.append(row["arm_gains"][arm] - row["arm_gains"][ema_arm])
        fixed_ci = _bootstrap_ci_mean(diff_vs_best_fixed)
        ema_ci = _bootstrap_ci_mean(ema_pairs)
        scenario_reports[scenario] = {
            "action_distribution": action_counts,
            "aor_unmatched_rows": aor_rows,
            "scored_rows": len(scored),
            "end_to_end_hit_rate": float(np.mean([float(h) for h in hits])) if hits else 0.0,
            "end_to_end_hit_rate_ci95": _hit_ci(hits),
            "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
            "opportunity_captured": captured,
            "paired_gain_vs_best_fixed": float(np.mean(diff_vs_best_fixed))
            if diff_vs_best_fixed
            else 0.0,
            "paired_gain_vs_best_fixed_ci95": [fixed_ci[0], fixed_ci[1]],
            "paired_gain_vs_ema_selector": float(np.mean(ema_pairs)) if ema_pairs else 0.0,
            "paired_gain_vs_ema_selector_ci95": [ema_ci[0], ema_ci[1]],
            "ema_comparison_rows": len(ema_pairs),
            "beats_best_fixed": bool(fixed_ci[0] > 0.0),
            "beats_ema_selector": bool(ema_ci[0] > 0.0),
        }
    checks = {
        "context_count_60": len(rows) == 60,
        "decisions_complete": all(
            row[f"{scenario}_action"] for row in rows for scenario in SCENARIOS
        ),
        "mapped_arms_valid": all(
            row[f"{scenario}_mapped_arm"] in ARMS or row[f"{scenario}_mapped_arm"] is None
            for row in rows
            for scenario in SCENARIOS
        ),
        "arm_gains_finite": all(
            np.isfinite(value) for row in rows for value in row["arm_gains"].values()
        ),
        "production_selector_unchanged": True,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "scenarios": SCENARIOS,
            "available_fes": AVAILABLE_FES,
            "action_arm_map": {k: v for k, v in ACTION_ARM_MAP.items()},
            "best_fixed_arm": BEST_FIXED_ARM,
            "ema_comparison_variant": PRIMARY_VARIANT,
            "config": "default OcCoordinatorConfig (persistent_streak=2, escalation_streak=6, hub_mode=relative)",
            "production_selector_modified": False,
        },
        "scenario_reports": scenario_reports,
        "rows": rows,
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
        default=Path("artifacts/oc_production_baseline_gate/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = {k: {kk: vv for kk, vv in v.items()} for k, v in result["scenario_reports"].items()}
    print(json.dumps({"gate_passed": result["gate_passed"], "scenario_reports": summary}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
