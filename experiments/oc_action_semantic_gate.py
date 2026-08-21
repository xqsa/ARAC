"""Matched four-arm semantic gate for the first overlap-specific ARAC-OC action.

The gate isolates action semantics at a shared Phase-II checkpoint.  It keeps
the production GCB selector unchanged and reports the best arm only as an
offline diagnostic.  Every arm receives the same proposal set, probes,
arbitration, action budget, and complete-context handoff.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import GraphCoordinationScheduler, evaluate_frozen_private_counterfactual
from experiments.oc_action_value_gate import _owner_control_after_arbitration
from experiments.overlap_joint_patch_gate20 import _repair_joint
from experiments.overlap_sequential_shared_patch_gate18 import (
    EVALS_PER_ROUND,
    ROUNDS,
    _repair_sequential,
)
from experiments.overlap_value_aware_dispatch_gate15 import (
    ARM_TOTAL_BUDGET_FES,
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


OUTPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
HANDOFF_ROUNDS = ROUNDS
HANDOFF_FES = HANDOFF_ROUNDS * EVALS_PER_ROUND
LOCAL_MUTATION_SCALE = 1.0
ARMS = (
    "owner_control",
    "shared_core",
    "expanded_shared_private",
    "duplicated_shared_competition",
    "duplicated_shared_local_competition",
)


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
    competition_rounds: int
    competition_diversity: tuple[int, ...]
    competition_mutation_scale: float
    competition_perturbation_norms: tuple[float, ...]
    coupled_gain: float
    coupling_fes: int
    coupling_archive_preserved: bool


@dataclass(frozen=True)
class ContextResult:
    mode: str
    topology: str
    overlap_budget: int
    seed: int
    component_count: int
    selected_component: tuple[int, ...]
    checkpoint_fes: int
    checkpoint_error: float
    probes_identical: bool
    proposals_identical: bool
    checkpoint_parity: bool
    arbitration_parity: bool
    action_parity: bool
    handoff_parity: bool
    fe_parity: bool
    strict_best: bool
    handoff_trace_complete: bool
    promotion_applied: bool
    diagnostic_action_winner: str
    diagnostic_end_to_end_winner: str
    owner_control: ArmResult
    shared_core: ArmResult
    expanded_shared_private: ArmResult
    duplicated_shared_competition: ArmResult
    duplicated_shared_local_competition: ArmResult
    coupling_receipt_parity: bool


def _probe_map(probes) -> tuple[tuple[tuple[int, ...], float], ...]:
    return tuple((item.component, float(item.estimated_gain)) for item in probes)


def _proposal_signature(proposals) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            proposal.group,
            tuple(proposal.values),
            float(proposal.improvement),
            tuple(proposal.uncertainty),
        )
        for proposal in proposals
    )


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
) -> tuple[ArmResult, tuple[tuple[tuple[int, ...], float], ...], tuple[object, ...]]:
    ledger, scheduler = _new_scheduler(
        problem, structure, checkpoint_x, checkpoint_error, checkpoint_fes
    )
    scheduler.prime(proposals)
    priming_fes = ledger.count - checkpoint_fes
    probes = scheduler.value_probe(proposals)
    if len(probes) < 2:
        raise RuntimeError("semantic gate expects at least two overlap components")
    selected_proposals = tuple(proposal for proposal in proposals if proposal.group in component)
    arbitration = scheduler.coordinator.coordinate(component, selected_proposals, ctp_budget_fes=0)
    if len(arbitration.candidates) != 4:
        raise RuntimeError("semantic gate expects four arbitration candidates")
    pre_action_error = float(ledger.best_error)
    pre_action_x = ledger.best_x
    coupling_scope = scheduler.coordinator._component_variables(component)

    competition_rounds = 0
    competition_diversity: tuple[int, ...] = ()
    competition_mutation_scale = 0.0
    competition_perturbation_norms: tuple[float, ...] = ()
    if arm == "owner_control":
        action_fes, action_error, action_ok, _ = _owner_control_after_arbitration(
            scheduler,
            structure,
            proposals,
            component,
            budget_fes=CTP_BUDGET_FES,
            seed=seed ^ 0x51ED,
        )
    elif arm == "shared_core":
        action_fes, _trace, _accepted = _repair_sequential(
            scheduler, component, proposals, seed=seed ^ 0x18A7
        )
        action_error = float(ledger.best_error)
        action_ok = action_error <= pre_action_error
    elif arm == "expanded_shared_private":
        action_fes, _trace, _accepted, _shared_count, _joint_count = _repair_joint(
            scheduler, component, proposals
        )
        action_error = float(ledger.best_error)
        action_ok = action_error <= pre_action_error
    elif arm == "duplicated_shared_competition":
        competition = scheduler.coordinator.duplicated_shared_competition(
            component,
            selected_proposals,
            budget_fes=CTP_BUDGET_FES,
        )
        action_fes = competition.consumed_fes
        competition_rounds = len(competition.rounds)
        competition_diversity = competition.candidate_diversity
        action_error = float(ledger.best_error)
        action_ok = action_error <= pre_action_error
    elif arm == "duplicated_shared_local_competition":
        competition = scheduler.coordinator.duplicated_shared_local_competition(
            component,
            selected_proposals,
            budget_fes=CTP_BUDGET_FES,
            seed=seed ^ 0xA17C,
            mutation_scale=LOCAL_MUTATION_SCALE,
        )
        action_fes = competition.consumed_fes
        competition_rounds = len(competition.rounds)
        competition_diversity = competition.candidate_diversity
        competition_mutation_scale = competition.mutation_scale
        competition_perturbation_norms = tuple(
            norm
            for round_item in competition.rounds
            for norm in round_item.perturbation_norms
        )
        action_error = float(ledger.best_error)
        action_ok = action_error <= pre_action_error
    else:
        raise ValueError(f"unknown arm: {arm}")
    if action_fes != CTP_BUDGET_FES:
        raise RuntimeError(f"{arm} action FE drifted: {action_fes} != {CTP_BUDGET_FES}")

    action_error = float(ledger.best_error)
    action_x = ledger.best_x
    coupling = evaluate_frozen_private_counterfactual(
        ledger,
        component=component,
        scope=coupling_scope,
        incumbent=pre_action_x,
        best_error_before=pre_action_error,
        candidate_name=arm,
        candidate=action_x,
        full_candidate_error=action_error,
    )
    handoff_before = action_error
    handoff = scheduler.coordinator.full_context_writeback(
        component, selected_proposals, rounds=HANDOFF_ROUNDS
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
        + coupling.consumed_fes
        + HANDOFF_FES
    )
    if consumed != expected:
        raise RuntimeError(f"{arm} FE mismatch: {consumed} != {expected}")
    accepted_handoff = sum(item.accepted for item in handoff.rounds)
    return (
        ArmResult(
            arm=arm,
            selected_component=tuple(component),
            checkpoint_error=float(checkpoint_error),
            pre_action_error=pre_action_error,
            action_error=action_error,
            handoff_error=handoff_error,
            action_gain=float(pre_action_error - action_error),
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
            competition_rounds=competition_rounds,
            competition_diversity=competition_diversity,
            competition_mutation_scale=competition_mutation_scale,
            competition_perturbation_norms=competition_perturbation_norms,
            coupled_gain=coupling.coupled_gain,
            coupling_fes=coupling.consumed_fes,
            coupling_archive_preserved=coupling.archive_preserved,
        ),
        _probe_map(probes),
        _proposal_signature(proposals),
    )


def _winner(arms: tuple[ArmResult, ...], field: str) -> str:
    return min(arms, key=lambda item: (float(getattr(item, field)), item.arm)).arm


def _correlation(left: list[float], right: list[float], *, rank: bool) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) != len(y) or len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    if rank:
        x = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        y = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(x, y)[0, 1])


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
    records = tuple(
        _arm(
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
        for arm in ARMS
    )
    arms = tuple(record[0] for record in records)
    probe_maps = tuple(record[1] for record in records)
    proposal_signatures = tuple(record[2] for record in records)
    by_name = {item.arm: item for item in arms}
    return ContextResult(
        mode=mode,
        topology=topology,
        overlap_budget=overlap_budget,
        seed=seed,
        component_count=len(components),
        selected_component=selected,
        checkpoint_fes=checkpoint_fes,
        checkpoint_error=float(checkpoint_error),
        probes_identical=all(item == probe_maps[0] for item in probe_maps[1:]),
        proposals_identical=all(item == proposal_signatures[0] for item in proposal_signatures[1:]),
        checkpoint_parity=all(
            item.checkpoint_error == checkpoint_error and item.selected_component == selected
            for item in arms
        ),
        arbitration_parity=all(item.arbitration_fes == 4 for item in arms),
        action_parity=all(item.action_fes == CTP_BUDGET_FES for item in arms),
        handoff_parity=all(item.handoff_fes == HANDOFF_FES for item in arms),
        fe_parity=len({item.consumed_fes for item in arms}) == 1,
        strict_best=all(item.strict_best for item in arms),
        handoff_trace_complete=all(item.handoff_rounds == HANDOFF_ROUNDS for item in arms),
        promotion_applied=False,
        diagnostic_action_winner=_winner(arms, "action_error"),
        diagnostic_end_to_end_winner=_winner(arms, "handoff_error"),
        owner_control=by_name["owner_control"],
        shared_core=by_name["shared_core"],
        expanded_shared_private=by_name["expanded_shared_private"],
        duplicated_shared_competition=by_name["duplicated_shared_competition"],
        duplicated_shared_local_competition=by_name["duplicated_shared_local_competition"],
        coupling_receipt_parity=all(
            item.coupling_fes == 1 and item.coupling_archive_preserved for item in arms
        ),
    )


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
    contexts = tuple(
        sorted(
            contexts,
            key=lambda item: (item.topology, item.overlap_budget, item.seed, item.mode),
        )
    )
    checks = {
        "context_count_60": len(contexts) == 60,
        "at_least_two_components_all": all(item.component_count >= 2 for item in contexts),
        "proposals_identical": all(item.proposals_identical for item in contexts),
        "probes_identical": all(item.probes_identical for item in contexts),
        "checkpoint_parity": all(item.checkpoint_parity for item in contexts),
        "arbitration_parity": all(item.arbitration_parity for item in contexts),
        "action_parity": all(item.action_parity for item in contexts),
        "handoff_parity": all(item.handoff_parity for item in contexts),
        "fe_parity": all(item.fe_parity for item in contexts),
        "coupling_receipt_contracts": all(
            item.coupling_receipt_parity for item in contexts
        ),
        "coupling_signals_finite": all(
            np.isfinite(float(arm.coupled_gain))
            for item in contexts
            for arm in (
                item.owner_control,
                item.shared_core,
                item.expanded_shared_private,
                item.duplicated_shared_competition,
                item.duplicated_shared_local_competition,
            )
        ),
        "strict_best": all(item.strict_best for item in contexts),
        "handoff_trace_complete": all(item.handoff_trace_complete for item in contexts),
        "promotion_not_applied": all(not item.promotion_applied for item in contexts),
        "competition_receipts_present": all(
            item.duplicated_shared_competition.competition_rounds > 0
            and item.duplicated_shared_local_competition.competition_rounds > 0
            for item in contexts
        ),
        "local_perturbations_present": all(
            item.duplicated_shared_local_competition.competition_mutation_scale > 0.0
            and any(
                norm > 0.0
                for norm in item.duplicated_shared_local_competition.competition_perturbation_norms
            )
            for item in contexts
        ),
    }
    action_wins = {
        arm: sum(item.diagnostic_action_winner == arm for item in contexts) for arm in ARMS
    }
    end_to_end_wins = {
        arm: sum(item.diagnostic_end_to_end_winner == arm for item in contexts) for arm in ARMS
    }
    local_rows = [item.duplicated_shared_local_competition for item in contexts]
    local_coupled = [float(item.coupled_gain) for item in local_rows]
    local_action_gain = [float(item.action_gain) for item in local_rows]
    local_handoff_gain = [float(item.end_to_end_gain) for item in local_rows]
    coupling_correlations = {
        "local_action_gain_pearson": _correlation(local_coupled, local_action_gain, rank=False),
        "local_action_gain_spearman": _correlation(local_coupled, local_action_gain, rank=True),
        "local_end_to_end_gain_pearson": _correlation(
            local_coupled, local_handoff_gain, rank=False
        ),
        "local_end_to_end_gain_spearman": _correlation(
            local_coupled, local_handoff_gain, rank=True
        ),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "contexts": 60,
            "arms": ARMS,
            "probe_fes_per_component": PROBE_FES_PER_COMPONENT,
            "arbitration_fes": 4,
            "action_fes": CTP_BUDGET_FES,
            "coupling_fes": 1,
            "arm_tail_fes": CTP_BUDGET_FES + 1 + HANDOFF_FES,
            "handoff_fes": HANDOFF_FES,
            "handoff_rounds": HANDOFF_ROUNDS,
            "local_mutation_scale": LOCAL_MUTATION_SCALE,
            "production_selector_modified": False,
        },
        "context_count": len(contexts),
        "contexts": [asdict(item) for item in contexts],
        "summary": {
            "diagnostic_action_wins": action_wins,
            "diagnostic_end_to_end_wins": end_to_end_wins,
            "coupling_correlations": coupling_correlations,
            "local_coupled_gain_median": float(np.median(local_coupled)),
            "interpretation": (
                "post_action_shadow_only; coupling is measured after the action "
                "and cannot select the same action window"
            ),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json"
        ),
    )
    args = parser.parse_args()
    payload = run_gate(workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "gate_checks": payload["gate_checks"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
