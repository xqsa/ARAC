"""Multi-window maturity/revelation horizon gate for overlap actions.

This is an offline experiment.  Each matched arm receives the same common
prefix, then three sequential arbitration/action/handoff windows.  The only
state crossing a window boundary is the strict-best archive.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from experiments.oc_action_value_gate import _owner_control_after_arbitration
from experiments.overlap_joint_patch_gate20 import _repair_joint
from experiments.overlap_sequential_shared_patch_gate18 import (
    EVALS_PER_ROUND,
    ROUNDS,
    _repair_sequential,
)
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


OUTPUT_SCHEMA = "arac-oc-multi-window-action-horizon-gate-v1"
WINDOW_COUNT = 3
ARBITRATION_FES = 4
ACTION_FES = CTP_BUDGET_FES
HANDOFF_FES = ROUNDS * EVALS_PER_ROUND
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2026082021
AUTHORITY_THRESHOLD = 0.30


@dataclass(frozen=True)
class WindowResult:
    window_index: int
    pre_window_error: float
    pre_action_error: float
    action_error: float
    handoff_error: float
    arbitration_gain: float
    action_gain: float
    handoff_gain: float
    window_end_to_end_gain: float
    arbitration_fes: int
    action_fes: int
    handoff_fes: int
    accepted_handoff_rounds: int
    strict_best: bool


@dataclass(frozen=True)
class ArmResult:
    arm: str
    selected_component: tuple[int, ...]
    boundary_error: float
    final_error: float
    horizon_end_to_end_gain: float
    consumed_fes: int
    common_prefix_fes: int
    strict_best: bool
    windows: tuple[WindowResult, ...]


@dataclass(frozen=True)
class ContextResult:
    mode: str
    topology: str
    overlap_budget: int
    seed: int
    component_count: int
    selected_component: tuple[int, ...]
    probes_identical: bool
    proposals_identical: bool
    boundary_parity: bool
    fe_parity: bool
    strict_best: bool
    horizon_trace_complete: bool
    owner_control: ArmResult
    shared_sequential: ArmResult
    shared_joint: ArmResult
    rung1_selected: str
    rung2_selected: str
    horizon_oracle: str
    rung1_selection_hit: bool
    rung2_selection_hit: bool
    rung1_selection_regret: float
    rung2_selection_regret: float
    shared_sequential_vs_owner_horizon_gain: float
    shared_joint_vs_owner_horizon_gain: float


def _probe_map(probes) -> tuple[tuple[tuple[int, ...], float], ...]:
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _arm(
    problem,
    structure,
    proposals,
    checkpoint_x,
    checkpoint_error,
    checkpoint_fes,
    component: tuple[int, ...],
    *,
    arm: str,
    seed: int,
) -> tuple[ArmResult, tuple[tuple[tuple[int, ...], float], ...]]:
    ledger, scheduler = _new_scheduler(
        problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
    )
    scheduler.prime(proposals)
    priming_fes = ledger.count - checkpoint_fes
    probes = scheduler.value_probe(proposals)
    if len(probes) < 2:
        raise RuntimeError("horizon gate expects at least two overlap components")
    selected = tuple(component)
    selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)
    boundary_error = float(ledger.best_error)
    windows: list[WindowResult] = []
    for window_index in range(WINDOW_COUNT):
        pre_window_error = float(ledger.best_error)
        arbitration = scheduler.coordinator.coordinate(
            selected,
            selected_proposals,
            ctp_budget_fes=0,
            ctp_seed=seed ^ (0xC7A5 + window_index * 0x9E37),
        )
        if len(arbitration.candidates) != 4:
            raise RuntimeError("expected four arbitration candidates per horizon window")
        pre_action_error = float(ledger.best_error)
        if arm == "owner_control":
            action_fes, action_error, action_ok, _ = _owner_control_after_arbitration(
                scheduler,
                structure,
                proposals,
                selected,
                budget_fes=ACTION_FES,
                seed=seed ^ 0x51ED ^ (window_index * 0x9E37),
            )
        elif arm == "shared_sequential":
            action_fes, _trace, _accepted = _repair_sequential(
                scheduler,
                selected,
                proposals,
                seed=seed ^ 0x18A7 ^ (window_index * 0x9E37),
            )
            action_error = float(ledger.best_error)
            action_ok = action_error <= pre_action_error
        elif arm == "shared_joint":
            action_fes, _trace, _accepted, _shared_count, _joint_count = _repair_joint(
                scheduler, selected, proposals
            )
            action_error = float(ledger.best_error)
            action_ok = action_error <= pre_action_error
        else:
            raise ValueError(f"unknown arm: {arm}")
        if action_fes != ACTION_FES:
            raise RuntimeError(f"{arm} action FE drifted: {action_fes} != {ACTION_FES}")
        handoff_before = float(ledger.best_error)
        handoff = scheduler.coordinator.full_context_writeback(
            selected,
            selected_proposals,
            rounds=ROUNDS,
        )
        if handoff.consumed_fes != HANDOFF_FES:
            raise RuntimeError(f"{arm} handoff FE drifted: {handoff.consumed_fes} != {HANDOFF_FES}")
        handoff_error = float(ledger.best_error)
        windows.append(
            WindowResult(
                window_index=window_index,
                pre_window_error=pre_window_error,
                pre_action_error=pre_action_error,
                action_error=action_error,
                handoff_error=handoff_error,
                arbitration_gain=pre_window_error - pre_action_error,
                action_gain=pre_action_error - action_error,
                handoff_gain=handoff_before - handoff_error,
                window_end_to_end_gain=pre_window_error - handoff_error,
                arbitration_fes=ARBITRATION_FES,
                action_fes=action_fes,
                handoff_fes=handoff.consumed_fes,
                accepted_handoff_rounds=sum(item.accepted for item in handoff.rounds),
                strict_best=bool(
                    pre_window_error >= pre_action_error >= action_error >= handoff_error
                    and action_ok
                ),
            )
        )
    consumed = ledger.count - checkpoint_fes
    common_prefix_fes = priming_fes + len(probes) * PROBE_FES_PER_COMPONENT
    expected = common_prefix_fes + WINDOW_COUNT * (ARBITRATION_FES + ACTION_FES + HANDOFF_FES)
    if consumed != expected:
        raise RuntimeError(f"{arm} FE mismatch: {consumed} != {expected}")
    final_error = float(ledger.best_error)
    return (
        ArmResult(
            arm=arm,
            selected_component=selected,
            boundary_error=boundary_error,
            final_error=final_error,
            horizon_end_to_end_gain=boundary_error - final_error,
            consumed_fes=consumed,
            common_prefix_fes=common_prefix_fes,
            strict_best=all(item.strict_best for item in windows),
            windows=tuple(windows),
        ),
        _probe_map(probes),
    )


def _best_arm(arms: tuple[ArmResult, ...], gain: str) -> str:
    return max(arms, key=lambda item: (float(getattr(item, gain)), item.arm)).arm


def _cumulative_gain(arm: ArmResult, count: int) -> float:
    return float(sum(item.window_end_to_end_gain for item in arm.windows[:count]))


def _context(mode: str, topology: str, overlap_budget: int, seed: int) -> ContextResult:
    problem, structure, _ = _combined_problem(mode, topology, overlap_budget, seed)
    checkpoint_x, checkpoint_error, proposals, checkpoint_fes = _proposal_payload(
        problem, structure, seed
    )
    _, selector = _new_scheduler(problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes)
    selector.prime(proposals)
    probes = selector.value_probe(proposals)
    components = selector.overlap_components
    if len(components) < 2:
        raise RuntimeError(f"expected at least two overlap components, got {components}")
    priorities = {item.component: item.priority_score for item in selector.prioritize(proposals)}
    selected = max(
        probes,
        key=lambda item: (
            item.estimated_gain,
            priorities[item.component],
            tuple(-value for value in item.component),
        ),
    ).component
    arm_runs = tuple(
        _arm(
            problem,
            structure,
            proposals,
            checkpoint_x,
            checkpoint_error,
            checkpoint_fes,
            selected,
            arm=arm_name,
            seed=seed,
        )
        for arm_name in ("owner_control", "shared_sequential", "shared_joint")
    )
    arms = tuple(item[0] for item in arm_runs)
    probe_maps = tuple(item[1] for item in arm_runs)
    by_name = {item.arm: item for item in arms}
    arm_tuple = tuple(arms)
    rung1 = _best_arm(arm_tuple, "final_error")
    # Select by the cumulative gain observed after the first or two windows;
    # the final horizon oracle is only used to score the selection offline.
    rung1 = max(arms, key=lambda item: (_cumulative_gain(item, 1), item.arm)).arm
    rung2 = max(arms, key=lambda item: (_cumulative_gain(item, 2), item.arm)).arm
    oracle = _best_arm(arms, "horizon_end_to_end_gain")
    oracle_gain = by_name[oracle].horizon_end_to_end_gain
    scale = max(abs(float(by_name["owner_control"].boundary_error)), np.finfo(float).eps)
    return ContextResult(
        mode=mode,
        topology=topology,
        overlap_budget=overlap_budget,
        seed=seed,
        component_count=len(components),
        selected_component=selected,
        probes_identical=probe_maps[0] == probe_maps[1] == probe_maps[2],
        proposals_identical=True,
        boundary_parity=len({item.boundary_error for item in arms}) == 1,
        fe_parity=len({item.consumed_fes for item in arms}) == 1,
        strict_best=all(item.strict_best for item in arms),
        horizon_trace_complete=all(
            len(item.windows) == WINDOW_COUNT
            and all(
                window.arbitration_fes == ARBITRATION_FES
                and window.action_fes == ACTION_FES
                and window.handoff_fes == HANDOFF_FES
                for window in item.windows
            )
            for item in arms
        ),
        owner_control=by_name["owner_control"],
        shared_sequential=by_name["shared_sequential"],
        shared_joint=by_name["shared_joint"],
        rung1_selected=rung1,
        rung2_selected=rung2,
        horizon_oracle=oracle,
        rung1_selection_hit=rung1 == oracle,
        rung2_selection_hit=rung2 == oracle,
        rung1_selection_regret=float(
            (oracle_gain - by_name[rung1].horizon_end_to_end_gain) / scale
        ),
        rung2_selection_regret=float(
            (oracle_gain - by_name[rung2].horizon_end_to_end_gain) / scale
        ),
        shared_sequential_vs_owner_horizon_gain=float(
            by_name["shared_sequential"].horizon_end_to_end_gain
            - by_name["owner_control"].horizon_end_to_end_gain
        ),
        shared_joint_vs_owner_horizon_gain=float(
            by_name["shared_joint"].horizon_end_to_end_gain
            - by_name["owner_control"].horizon_end_to_end_gain
        ),
    )


def _correlation_interval(
    left: list[float], right: list[float], *, rank: bool, seed_offset: int
) -> tuple[float, float, float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_REPLICATES, len(x)))
    values = []
    for sample in indices:
        sx, sy = x[sample], y[sample]
        if rank:
            sx = np.argsort(np.argsort(sx, kind="stable"), kind="stable").astype(float)
            sy = np.argsort(np.argsort(sy, kind="stable"), kind="stable").astype(float)
        values.append(float(np.corrcoef(sx, sy)[0, 1]) if np.std(sx) and np.std(sy) else 0.0)
    point = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_gate(*, workers: int = 1) -> dict[str, object]:
    jobs = tuple(
        (mode, topology, budget, seed)
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
        for seed in FRESH_SEEDS
        for mode in MODES
    )
    if workers == 1:
        contexts = tuple(_context(*job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            contexts = tuple(executor.map(lambda job: _context(*job), jobs))
    contexts = tuple(sorted(contexts, key=lambda row: (row.topology, row.overlap_budget, row.seed, row.mode)))
    early_action: list[float] = []
    rung1_signal: list[float] = []
    future1: list[float] = []
    rung2_signal: list[float] = []
    future2: list[float] = []
    direct_records: list[dict[str, float | str]] = []
    horizon_records: list[dict[str, float | str]] = []
    horizon_excess: list[float] = []
    for row in contexts:
        owner = row.owner_control
        for arm in (row.shared_sequential, row.shared_joint):
            early_action.append(arm.windows[0].action_gain - owner.windows[0].action_gain)
            rung1_signal.append(
                arm.windows[0].window_end_to_end_gain
                - owner.windows[0].window_end_to_end_gain
            )
            future1.append(
                sum(item.window_end_to_end_gain for item in arm.windows[1:])
                - sum(item.window_end_to_end_gain for item in owner.windows[1:])
            )
            rung2_signal.append(
                _cumulative_gain(arm, 2) - _cumulative_gain(owner, 2)
            )
            future2.append(arm.windows[2].window_end_to_end_gain - owner.windows[2].window_end_to_end_gain)
            horizon_excess.append(arm.horizon_end_to_end_gain - owner.horizon_end_to_end_gain)
        for arm in (owner, row.shared_sequential, row.shared_joint):
            direct_records.append(
                {
                    "arm": arm.arm,
                    "window1_action_gain": arm.windows[0].action_gain,
                    "window1_end_to_end_gain": arm.windows[0].window_end_to_end_gain,
                    "horizon_end_to_end_gain": arm.horizon_end_to_end_gain,
                }
            )
        horizon_records.append(
            {
                "rung1_selected": row.rung1_selected,
                "rung2_selected": row.rung2_selected,
                "horizon_oracle": row.horizon_oracle,
                "rung1_regret": row.rung1_selection_regret,
                "rung2_regret": row.rung2_selection_regret,
            }
        )
    direct_to_future = _correlation_interval(early_action, future1, rank=True, seed_offset=0)
    rung1_to_future = _correlation_interval(rung1_signal, future1, rank=True, seed_offset=1)
    rung2_to_future = _correlation_interval(rung2_signal, future2, rank=True, seed_offset=2)
    cells = tuple(
        {
            "topology": topology,
            "overlap_budget": budget,
            "context_count": sum(
                row.topology == topology and row.overlap_budget == budget for row in contexts
            ),
            "complete": sum(
                row.topology == topology and row.overlap_budget == budget for row in contexts
            ) == 2 * len(FRESH_SEEDS),
        }
        for topology in TOPOLOGIES
        for budget in OVERLAP_BUDGETS
    )
    integrity = {
        "context_count_60": len(contexts) == 60,
        "cells_complete": all(item["complete"] for item in cells),
        "at_least_two_components": all(row.component_count >= 2 for row in contexts),
        "probes_identical": all(row.probes_identical for row in contexts),
        "proposals_identical": all(row.proposals_identical for row in contexts),
        "boundary_parity": all(row.boundary_parity for row in contexts),
        "fe_parity": all(row.fe_parity for row in contexts),
        "strict_best": all(row.strict_best for row in contexts),
        "horizon_trace_complete": all(row.horizon_trace_complete for row in contexts),
    }
    promotion = {
        "authority_threshold": AUTHORITY_THRESHOLD,
        "direct_action_to_future_spearman_ci": direct_to_future,
        "rung1_to_future_spearman_ci": rung1_to_future,
        "rung2_to_future_spearman_ci": rung2_to_future,
        "shared_horizon_win_or_tie_rate": float(np.mean(np.asarray(horizon_excess) >= 0.0)),
        "shared_horizon_median_excess_gain": float(np.median(horizon_excess)),
        "rung1_selection_hit_rate": float(np.mean([row.rung1_selection_hit for row in contexts])),
        "rung2_selection_hit_rate": float(np.mean([row.rung2_selection_hit for row in contexts])),
        "rung1_selection_regret_median": float(np.median([row.rung1_selection_regret for row in contexts])),
        "rung2_selection_regret_median": float(np.median([row.rung2_selection_regret for row in contexts])),
        "promotion_recommended": bool(
            rung2_to_future[1] >= AUTHORITY_THRESHOLD
            and float(np.mean(np.asarray(horizon_excess) >= 0.0)) >= 0.60
            and float(np.median(horizon_excess)) >= 0.0
        ),
        "reason": "promote only when rung-2 value predicts later value and shared actions beat owner control",
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "contexts": len(jobs),
            "actions": ("owner_control", "shared_sequential", "shared_joint"),
            "window_count": WINDOW_COUNT,
            "arbitration_fes_per_window": ARBITRATION_FES,
            "action_fes_per_window": ACTION_FES,
            "handoff_fes_per_window": HANDOFF_FES,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "authority_threshold": AUTHORITY_THRESHOLD,
            "production_scheduler_modified": False,
        },
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "direct_records": direct_records,
        "horizon_records": horizon_records,
        "cell_summary": cells,
        "summary": {
            "direct_action_to_future_spearman_ci": direct_to_future,
            "rung1_to_future_spearman_ci": rung1_to_future,
            "rung2_to_future_spearman_ci": rung2_to_future,
            "shared_horizon_win_or_tie_rate": float(np.mean(np.asarray(horizon_excess) >= 0.0)),
            "shared_horizon_median_excess_gain": float(np.median(horizon_excess)),
        },
        "promotion": promotion,
        "gate_checks": integrity,
        "gate_passed": all(integrity.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_action_horizon_gate/confirmation.json"),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "gate_checks": payload["gate_checks"],
                "promotion": payload["promotion"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
