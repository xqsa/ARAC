"""Direct action-value gate for overlap coordination.

The gate compares matched owner and shared-variable actions at the same
checkpoint, then gives every arm the same complete-context handoff episode.
It is an offline diagnostic and deliberately does not modify production GCB.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler
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


OUTPUT_SCHEMA = "arac-oc-direct-action-value-gate-v1"
HANDOFF_ROUNDS = ROUNDS
HANDOFF_FES = HANDOFF_ROUNDS * EVALS_PER_ROUND
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2026082017
AUTHORITY_THRESHOLD = 0.30


@dataclass(frozen=True)
class ArmResult:
    arm: str
    selected_component: tuple[int, ...]
    checkpoint_error: float
    pre_action_error: float
    action_error: float
    handoff_error: float
    action_gain: float
    continuation_gain: float
    end_to_end_gain: float
    consumed_fes: int
    probe_fes: int
    arbitration_fes: int
    action_fes: int
    handoff_fes: int
    action_strict_best: bool
    handoff_strict_best: bool
    strict_best: bool
    handoff_rounds: int
    accepted_handoff_rounds: int


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
    fe_parity: bool
    strict_best: bool
    handoff_trace_complete: bool
    owner_control: ArmResult
    shared_sequential: ArmResult
    shared_joint: ArmResult
    action_value_selected: str
    end_to_end_oracle: str
    value_selection_hit: bool
    value_selection_regret: float
    shared_sequential_vs_owner_gain: float
    shared_joint_vs_owner_gain: float
    best_shared_vs_owner_gain: float


def _probe_map(probes) -> tuple[tuple[tuple[int, ...], float], ...]:
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _owner_control_after_arbitration(
    scheduler: GraphCoordinationScheduler,
    structure,
    proposals,
    component: tuple[int, ...],
    *,
    budget_fes: int,
    seed: int,
) -> tuple[int, float, bool, float]:
    """Run the matched full-proposal owner control after one arbitration."""

    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    base_before = float(scheduler.coordinator.ledger.best_error)
    rng = np.random.default_rng(seed)
    base = scheduler.coordinator.ledger.best_x
    candidates = np.repeat(base[np.newaxis, :], budget_fes, axis=0)
    for index in range(budget_fes):
        proposal = selected[index % len(selected)]
        for variable, value in proposal.values:
            candidates[index, variable] = value + float(
                rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable)))
            )
    np.clip(
        candidates,
        scheduler.coordinator.ledger.problem.lower_array,
        scheduler.coordinator.ledger.problem.upper_array,
        out=candidates,
    )
    scheduler.coordinator.ledger.evaluate(candidates)
    return budget_fes, float(scheduler.coordinator.ledger.best_error), scheduler.coordinator.ledger.best_error <= base_before, base_before


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
        raise RuntimeError("direct action-value gate expects at least two components")
    selected = tuple(component)
    selected_proposals = tuple(proposal for proposal in proposals if proposal.group in selected)
    arbitration = scheduler.coordinator.coordinate(selected, selected_proposals, ctp_budget_fes=0)
    if len(arbitration.candidates) != 4:
        raise RuntimeError("expected four arbitration candidates")
    pre_action_error = float(ledger.best_error)

    if arm == "owner_control":
        action_fes, action_error, action_ok, _ = _owner_control_after_arbitration(
            scheduler,
            structure,
            proposals,
            selected,
            budget_fes=CTP_BUDGET_FES,
            seed=seed ^ 0x51ED,
        )
    elif arm == "shared_sequential":
        action_fes, _trace, _accepted = _repair_sequential(
            scheduler, selected, proposals, seed=seed ^ 0x18A7
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
    if action_fes != CTP_BUDGET_FES:
        raise RuntimeError(f"{arm} action FE drifted: {action_fes} != {CTP_BUDGET_FES}")

    action_gain = pre_action_error - action_error
    handoff_before = float(ledger.best_error)
    handoff = scheduler.coordinator.full_context_writeback(
        selected, selected_proposals, rounds=HANDOFF_ROUNDS
    )
    if handoff.consumed_fes != HANDOFF_FES:
        raise RuntimeError(f"{arm} handoff FE drifted: {handoff.consumed_fes} != {HANDOFF_FES}")
    handoff_error = float(ledger.best_error)
    consumed = ledger.count - checkpoint_fes
    expected = (
        priming_fes
        + len(probes) * PROBE_FES_PER_COMPONENT
        + 4
        + CTP_BUDGET_FES
        + HANDOFF_FES
    )
    if consumed != expected:
        raise RuntimeError(f"{arm} FE mismatch: {consumed} != {expected}")
    accepted_handoff = sum(item.accepted for item in handoff.rounds)
    return (
        ArmResult(
            arm=arm,
            selected_component=selected,
            checkpoint_error=float(checkpoint_error),
            pre_action_error=pre_action_error,
            action_error=action_error,
            handoff_error=handoff_error,
            action_gain=float(action_gain),
            continuation_gain=float(handoff_before - handoff_error),
            end_to_end_gain=float(checkpoint_error - handoff_error),
            consumed_fes=consumed,
            probe_fes=len(probes) * PROBE_FES_PER_COMPONENT,
            arbitration_fes=4,
            action_fes=action_fes,
            handoff_fes=handoff.consumed_fes,
            action_strict_best=bool(action_ok and action_error <= pre_action_error),
            handoff_strict_best=bool(handoff_error <= handoff_before),
            strict_best=bool(
                action_ok
                and action_error <= pre_action_error
                and handoff_error <= handoff_before
                and handoff_error <= checkpoint_error
            ),
            handoff_rounds=len(handoff.rounds),
            accepted_handoff_rounds=accepted_handoff,
        ),
        _probe_map(probes),
    )


def _select_arm(arms: tuple[ArmResult, ...], field: str, *, reverse: bool) -> str:
    return min(
        arms,
        key=lambda item: (
            -float(getattr(item, field)) if reverse else float(getattr(item, field)),
            item.arm,
        ),
    ).arm


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
    arms_with_probes = tuple(
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
    arms = tuple(item[0] for item in arms_with_probes)
    probe_maps = tuple(item[1] for item in arms_with_probes)
    by_name = {item.arm: item for item in arms}
    action_selected = _select_arm(arms, "action_gain", reverse=True)
    end_to_end_oracle = _select_arm(arms, "end_to_end_gain", reverse=True)
    scale = max(abs(float(checkpoint_error)), np.finfo(float).eps)
    regret = (
        by_name[action_selected].end_to_end_gain - by_name[end_to_end_oracle].end_to_end_gain
    ) / scale
    seq_excess = by_name["shared_sequential"].end_to_end_gain - by_name["owner_control"].end_to_end_gain
    joint_excess = by_name["shared_joint"].end_to_end_gain - by_name["owner_control"].end_to_end_gain
    return ContextResult(
        mode=mode,
        topology=topology,
        overlap_budget=overlap_budget,
        seed=seed,
        component_count=len(components),
        selected_component=selected,
        probes_identical=probe_maps[0] == probe_maps[1] == probe_maps[2],
        proposals_identical=True,
        fe_parity=len({item.consumed_fes for item in arms}) == 1,
        strict_best=all(item.strict_best for item in arms),
        handoff_trace_complete=all(
            item.handoff_rounds == HANDOFF_ROUNDS for item in arms
        ),
        owner_control=by_name["owner_control"],
        shared_sequential=by_name["shared_sequential"],
        shared_joint=by_name["shared_joint"],
        action_value_selected=action_selected,
        end_to_end_oracle=end_to_end_oracle,
        value_selection_hit=action_selected == end_to_end_oracle,
        value_selection_regret=float(regret),
        shared_sequential_vs_owner_gain=float(seq_excess),
        shared_joint_vs_owner_gain=float(joint_excess),
        best_shared_vs_owner_gain=float(max(seq_excess, joint_excess)),
    )


def _correlation_interval(left: list[float], right: list[float], *, rank: bool) -> tuple[float, float, float]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(rank))
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_REPLICATES, len(x)))
    values = []
    for sample in indices:
        sx = x[sample]
        sy = y[sample]
        if rank:
            sx = np.argsort(np.argsort(sx, kind="stable"), kind="stable").astype(float)
            sy = np.argsort(np.argsort(sy, kind="stable"), kind="stable").astype(float)
        values.append(float(np.corrcoef(sx, sy)[0, 1]) if np.std(sx) and np.std(sy) else 0.0)
    point = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _median_interval(values: list[float], *, seed: int) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.median(data)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


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
    paired_action_excess: list[float] = []
    paired_continuation_excess: list[float] = []
    paired_end_to_end_excess: list[float] = []
    action_records: list[dict[str, float | str]] = []
    for row in contexts:
        owner = row.owner_control
        for arm in (row.shared_sequential, row.shared_joint):
            paired_action_excess.append(arm.action_gain - owner.action_gain)
            paired_continuation_excess.append(arm.continuation_gain - owner.continuation_gain)
            paired_end_to_end_excess.append(arm.end_to_end_gain - owner.end_to_end_gain)
        for arm in (row.owner_control, row.shared_sequential, row.shared_joint):
            action_records.append(
                {
                    "arm": arm.arm,
                    "action_gain": arm.action_gain,
                    "continuation_gain": arm.continuation_gain,
                    "end_to_end_gain": arm.end_to_end_gain,
                }
            )
    action_to_continuation = _correlation_interval(
        paired_action_excess, paired_continuation_excess, rank=False
    )
    action_to_continuation_rank = _correlation_interval(
        paired_action_excess, paired_continuation_excess, rank=True
    )
    action_to_end_to_end_rank = _correlation_interval(
        paired_action_excess, paired_end_to_end_excess, rank=True
    )
    best_shared = np.asarray([row.best_shared_vs_owner_gain for row in contexts], dtype=float)
    regret = [row.value_selection_regret for row in contexts]
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
        "fe_parity": all(row.fe_parity for row in contexts),
        "strict_best": all(row.strict_best for row in contexts),
        "handoff_trace_complete": all(row.handoff_trace_complete for row in contexts),
    }
    promotion = {
        "authority_threshold": AUTHORITY_THRESHOLD,
        "action_value_spearman_ci_lower": action_to_continuation_rank[1],
        "shared_win_or_tie_rate": float(np.mean(best_shared >= 0.0)),
        "shared_median_excess_gain": float(np.median(best_shared)),
        "selection_hit_rate": float(np.mean([row.value_selection_hit for row in contexts])),
        "selection_regret_median_ci": _median_interval(regret, seed=BOOTSTRAP_SEED + 2),
        "promotion_recommended": bool(
            action_to_continuation_rank[1] >= AUTHORITY_THRESHOLD
            and float(np.mean(best_shared >= 0.0)) >= 0.60
            and float(np.median(best_shared)) >= 0.0
        ),
        "reason": "promote only when direct action value has stable predictive rank and shared action beats owner control",
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "contexts": len(jobs),
            "actions": ("owner_control", "shared_sequential", "shared_joint"),
            "action_budget_fes": CTP_BUDGET_FES,
            "handoff_rounds": HANDOFF_ROUNDS,
            "handoff_fes": HANDOFF_FES,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "authority_threshold": AUTHORITY_THRESHOLD,
            "production_scheduler_modified": False,
        },
        "context_count": len(contexts),
        "contexts": [asdict(row) for row in contexts],
        "action_records": action_records,
        "cell_summary": cells,
        "summary": {
            "action_excess_to_continuation_pearson_ci": action_to_continuation,
            "action_excess_to_continuation_spearman_ci": action_to_continuation_rank,
            "action_excess_to_end_to_end_spearman_ci": action_to_end_to_end_rank,
            "best_shared_vs_owner_win_or_tie_rate": float(np.mean(best_shared >= 0.0)),
            "best_shared_vs_owner_median_excess_gain": float(np.median(best_shared)),
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
        default=Path("artifacts/oc_action_value_gate/confirmation.json"),
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
