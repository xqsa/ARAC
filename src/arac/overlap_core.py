"""Complete evidence-guided ARAC path for overlapping black-box optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import (
    DISPATCH_NEIGHBORHOOD,
    DISPATCH_STRATEGIES,
    DispatchReceipt,
    GcbDispatchConfig,
    GcbDispatchPlanner,
    LocalProposal,
    OverlapCoordinator,
    OverlapStructure,
    produce_local_proposal,
)
from arac.coordination.episodes import (
    PhaseAwareSchedulerConfig,
    V4ScheduleResult,
    run_oc_episode_schedule_v5_1,
    run_oc_episode_schedule_v5_2,
    run_oc_episode_schedule_v5_3,
)
from arac.coordination.loop import OcLoopResult, run_oc_unified
from arac.evidence import Phase1OverlapPilotResult, run_phase1_overlap_pilot
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort


DEFAULT_REFRESH_CYCLES = 16
DEFAULT_NEIGHBORHOOD_FES = 32
MIN_PROPOSAL_FES = 8
PROPOSAL_POPULATION_SIZE = 8
COORDINATION_MODES = (
    "proposal_neighborhood",
    "proposal_only",
    "full_context",
)
PERSISTENT_CTP_MODE = "persistent_ctp"
GCB_COORDINATED_MODE = "gcb_coordinated"
_SUPPORTED_COORDINATION_MODES = COORDINATION_MODES + (
    PERSISTENT_CTP_MODE,
    GCB_COORDINATED_MODE,
)
ARAC_OC_SCHEDULER_MODES = ("legacy_unified", "v5_1", "v5_2", "v5_3")


@dataclass(frozen=True)
class OverlapCycleResult:
    cycle_index: int
    proposal_fes: int
    arbitration_fes: int
    endpoint_fes: int
    neighborhood_fes: int
    best_error_before: float
    best_error_after_proposals: float
    best_error_after: float
    proposal_gain: float
    coordination_gain: float
    accepted_coordination_steps: int
    ctp_fes: int = 0
    ctp_triggered_components: int = 0
    max_conflict_streak: int = 0
    dispatch_receipts: tuple[DispatchReceipt, ...] = ()


@dataclass(frozen=True)
class OverlapAracResult:
    phase1: Phase1OverlapPilotResult
    coordination_mode: str
    cycles: tuple[OverlapCycleResult, ...]
    overlap_groups: tuple[int, ...]
    overlap_components: tuple[tuple[int, ...], ...]
    proposal_budget_fes: int
    phase2_consumed_fes: int
    tail_fes: int
    final_error: float
    terminal_fes: int


def _overlap_components(structure: OverlapStructure) -> tuple[tuple[int, ...], ...]:
    shared = set(structure.shared_variables)
    return tuple(
        component
        for component in structure.connected_components()
        if any(shared.intersection(structure.groups[group]) for group in component)
    )


def _proposal_budget(
    phase2_fes: int,
    components: tuple[tuple[int, ...], ...],
    *,
    refresh_cycles: int,
    neighborhood_fes: int,
) -> int:
    groups = sum(len(component) for component in components)
    overhead_per_cycle = sum(
        4 + 2 * len(component) + neighborhood_fes for component in components
    )
    available = phase2_fes - refresh_cycles * overhead_per_cycle
    budget = available // (refresh_cycles * groups)
    if budget < MIN_PROPOSAL_FES:
        raise ValueError("Phase-II budget cannot fund the frozen overlap refresh protocol")
    return int(budget)


def capped_proposal_budget(
    phase2_fes: int,
    components: tuple[tuple[int, ...], ...],
    *,
    refresh_cycles: int,
    neighborhood_fes: int,
    sense_budget_share: float,
) -> int:
    """Return a per-group sense budget under an explicit total-share cap.

    ``_proposal_budget`` is retained for the frozen sparse-domain protocol.
    Dense overlap components need a separate, pre-registered cap because
    sensing every owner in every cycle can otherwise consume the whole
    Phase-II budget before GCB can dispatch an operator.  The returned value
    is still per group and per cycle, so the caller's existing exact-FE loop
    remains unchanged.
    """

    if isinstance(phase2_fes, bool) or not isinstance(phase2_fes, int) or phase2_fes <= 0:
        raise ValueError("phase2_fes must be a positive integer")
    if isinstance(refresh_cycles, bool) or not isinstance(refresh_cycles, int) or refresh_cycles <= 0:
        raise ValueError("refresh_cycles must be a positive integer")
    if isinstance(neighborhood_fes, bool) or not isinstance(neighborhood_fes, int) or neighborhood_fes <= 0:
        raise ValueError("neighborhood_fes must be a positive integer")
    if not components:
        raise ValueError("components must be non-empty")
    groups = sum(len(component) for component in components)
    if groups <= 0:
        raise ValueError("components must contain at least one group")
    if not isinstance(sense_budget_share, (int, float)) or isinstance(sense_budget_share, bool):
        raise ValueError("sense_budget_share must be a finite number in (0, 1]")
    share = float(sense_budget_share)
    if not 0.0 < share <= 1.0:
        raise ValueError("sense_budget_share must be in (0, 1]")

    overhead_per_cycle = sum(
        4 + 2 * len(component) + neighborhood_fes for component in components
    )
    available_after_overhead = phase2_fes - refresh_cycles * overhead_per_cycle
    if available_after_overhead <= 0:
        raise ValueError("Phase-II budget cannot fund the overlap refresh overhead")

    capped_total = min(
        available_after_overhead,
        int(phase2_fes * share),
    )
    budget = capped_total // (refresh_cycles * groups)
    if budget < MIN_PROPOSAL_FES:
        raise ValueError("sense budget cap cannot fund the minimum proposal session")
    return int(budget)


def run_overlap_arac(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    refresh_cycles: int = DEFAULT_REFRESH_CYCLES,
    neighborhood_fes: int = DEFAULT_NEIGHBORHOOD_FES,
    phase1_kwargs: dict[str, object] | None = None,
) -> OverlapAracResult:
    """Run the canonical ARAC-OC path.

    The historical phase-II arms remain available only through
    :func:`run_overlap_from_pilot`; this public convenience entry point now
    delegates to the unified coordinator loop.
    """

    if isinstance(refresh_cycles, bool) or not isinstance(refresh_cycles, int) or refresh_cycles <= 0:
        raise ValueError("refresh_cycles must be a positive integer")
    if isinstance(neighborhood_fes, bool) or not isinstance(neighborhood_fes, int) or neighborhood_fes <= 0:
        raise ValueError("neighborhood_fes must be a positive integer")
    unified = run_arac_oc(
        problem,
        total_budget_fes=total_budget_fes,
        run_seed=run_seed,
        refresh_cycles=refresh_cycles,
        sense_budget_fes=neighborhood_fes,
        phase1_kwargs=phase1_kwargs,
    )
    structure = unified.phase1.adaptation.structure
    if structure is None:
        raise RuntimeError("ARAC-OC Phase-I did not produce an overlap structure")
    components = _overlap_components(structure)
    overlap_groups = tuple(group for component in components for group in component)
    cycles = tuple(
        OverlapCycleResult(
            cycle_index=trace.cycle_index,
            proposal_fes=trace.sense_fes,
            arbitration_fes=trace.arbitration_fes,
            endpoint_fes=trace.probe_fes + trace.arbitration_fes,
            neighborhood_fes=trace.smp_fes + trace.operator_fes,
            best_error_before=trace.best_error_before,
            best_error_after_proposals=trace.best_error_after_sense,
            best_error_after=trace.best_error_after,
            proposal_gain=max(0.0, trace.best_error_before - trace.best_error_after_sense),
            coordination_gain=max(0.0, trace.best_error_after_sense - trace.best_error_after),
            accepted_coordination_steps=int(
                trace.best_error_after < trace.best_error_after_arbitration
            ),
            ctp_fes=trace.operator_fes
            if trace.action in {"ctp_restricted", "ctp_shared_core"}
            else 0,
            ctp_triggered_components=int(
                trace.action in {"ctp_restricted", "ctp_shared_core"}
            ),
        )
        for trace in unified.cycles
    )
    return OverlapAracResult(
        phase1=unified.phase1,
        coordination_mode=unified.coordination_mode,
        cycles=cycles,
        overlap_groups=overlap_groups,
        overlap_components=components,
        proposal_budget_fes=neighborhood_fes,
        phase2_consumed_fes=unified.phase2_consumed_fes,
        tail_fes=unified.tail_fes,
        final_error=unified.final_error,
        terminal_fes=unified.terminal_fes,
    )


def run_arac_oc(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    refresh_cycles: int = DEFAULT_REFRESH_CYCLES,
    sense_budget_fes: int = DEFAULT_NEIGHBORHOOD_FES,
    phase1_kwargs: dict[str, object] | None = None,
    config=None,
    scheduler_mode: Literal["legacy_unified", "v5_1", "v5_2", "v5_3"] = "legacy_unified",
    scheduler_config: PhaseAwareSchedulerConfig | None = None,
) -> OcLoopResult | V4ScheduleResult:
    """Run the canonical ARAC-OC Phase-I -> coordinator pipeline.

    ``run_overlap_arac`` and ``run_overlap_from_pilot`` remain explicit
    historical/control arms for frozen comparisons.  ``scheduler_mode`` is
    deliberately explicit: ``legacy_unified`` preserves the original
    coordinator contract, while ``v5_1`` enters the four-episode production
    scheduler and returns its audited schedule receipt.  The explicit switch
    prevents a cached legacy result from being mistaken for a v5.1 result.
    """

    if scheduler_mode not in ARAC_OC_SCHEDULER_MODES:
        raise ValueError(
            f"unsupported scheduler_mode: {scheduler_mode}; "
            f"expected one of {ARAC_OC_SCHEDULER_MODES}"
        )
    if scheduler_mode in ("v5_1", "v5_2", "v5_3"):
        if scheduler_config is None and isinstance(config, PhaseAwareSchedulerConfig):
            scheduler_config = config
        if scheduler_config is None:
            raise ValueError(
                f"scheduler_config is required for scheduler_mode={scheduler_mode!r}; "
                "pass the frozen PhaseAwareSchedulerConfig explicitly"
            )
        if not isinstance(scheduler_config, PhaseAwareSchedulerConfig):
            raise TypeError("scheduler_config must be PhaseAwareSchedulerConfig")
    elif scheduler_config is not None:
        raise ValueError(
            "scheduler_config is only valid with a v5 scheduler_mode"
        )

    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=total_budget_fes,
        run_seed=run_seed,
        **({} if phase1_kwargs is None else phase1_kwargs),
    )
    if scheduler_mode == "v5_1":
        # The retired scheduler entry raises here: a v5.1-labelled schedule
        # can no longer be produced (runway bounded in v5.2).
        structure = pilot.adaptation.structure
        return run_oc_episode_schedule_v5_1(
            problem,
            pilot.checkpoint,
            action_seed=run_seed,
            config=scheduler_config,
            structure=structure,
        )
    if scheduler_mode == "v5_2":
        # The retired scheduler entry raises here: a v5.2-labelled schedule
        # can no longer be produced (verification windows laddered in v5.3).
        structure = pilot.adaptation.structure
        return run_oc_episode_schedule_v5_2(
            problem,
            pilot.checkpoint,
            action_seed=run_seed,
            config=scheduler_config,
            structure=structure,
        )
    if scheduler_mode == "v5_3":
        structure = pilot.adaptation.structure
        return run_oc_episode_schedule_v5_3(
            problem,
            pilot.checkpoint,
            action_seed=run_seed,
            config=scheduler_config,
            structure=structure,
        )
    return run_oc_unified(
        problem,
        pilot,
        refresh_cycles=refresh_cycles,
        sense_budget_fes=sense_budget_fes,
        config=config,
    )


def run_arac_oc_v5_3(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    scheduler_config: PhaseAwareSchedulerConfig,
    phase1_kwargs: dict[str, object] | None = None,
) -> V4ScheduleResult:
    """Named production entry point for the audited v5.3 episode scheduler."""

    result = run_arac_oc(
        problem,
        total_budget_fes=total_budget_fes,
        run_seed=run_seed,
        phase1_kwargs=phase1_kwargs,
        scheduler_mode="v5_3",
        scheduler_config=scheduler_config,
    )
    if not isinstance(result, V4ScheduleResult):
        raise RuntimeError("v5.3 production entry point returned a legacy result")
    return result


def run_arac_oc_v5_2(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    scheduler_config: PhaseAwareSchedulerConfig,
    phase1_kwargs: dict[str, object] | None = None,
) -> V4ScheduleResult:
    """Named production entry point for the audited v5.2 episode scheduler."""

    result = run_arac_oc(
        problem,
        total_budget_fes=total_budget_fes,
        run_seed=run_seed,
        phase1_kwargs=phase1_kwargs,
        scheduler_mode="v5_2",
        scheduler_config=scheduler_config,
    )
    if not isinstance(result, V4ScheduleResult):
        raise RuntimeError("v5.2 production entry point returned a legacy result")
    return result


def run_arac_oc_v5_1(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    scheduler_config: PhaseAwareSchedulerConfig,
    phase1_kwargs: dict[str, object] | None = None,
) -> V4ScheduleResult:
    """Named production entry point for the audited v5.1 episode scheduler."""

    result = run_arac_oc(
        problem,
        total_budget_fes=total_budget_fes,
        run_seed=run_seed,
        phase1_kwargs=phase1_kwargs,
        scheduler_mode="v5_1",
        scheduler_config=scheduler_config,
    )
    if not isinstance(result, V4ScheduleResult):
        raise RuntimeError("v5.1 production entry point returned a legacy result")
    return result


def run_overlap_from_pilot(
    problem: OptimizationProblem,
    pilot: Phase1OverlapPilotResult,
    *,
    coordination_mode: str,
    refresh_cycles: int = DEFAULT_REFRESH_CYCLES,
    neighborhood_fes: int = DEFAULT_NEIGHBORHOOD_FES,
    dispatch_config: GcbDispatchConfig | None = None,
) -> OverlapAracResult:
    """Run one exact-budget Phase-II arm from a frozen Phase-I pilot."""

    if coordination_mode not in _SUPPORTED_COORDINATION_MODES:
        raise ValueError(f"unsupported coordination_mode: {coordination_mode}")
    if isinstance(refresh_cycles, bool) or not isinstance(refresh_cycles, int) or refresh_cycles <= 0:
        raise ValueError("refresh_cycles must be a positive integer")
    if isinstance(neighborhood_fes, bool) or not isinstance(neighborhood_fes, int) or neighborhood_fes <= 0:
        raise ValueError("neighborhood_fes must be a positive integer")
    if coordination_mode == "full_context" and neighborhood_fes % 2:
        raise ValueError("full_context requires an even neighborhood_fes budget")
    if not isinstance(pilot, Phase1OverlapPilotResult):
        raise TypeError("pilot must be Phase1OverlapPilotResult")
    if len(pilot.checkpoint.incumbent) != problem.dimension:
        raise ValueError("pilot checkpoint and problem dimensions disagree")
    if not pilot.adaptation.ready or pilot.adaptation.structure is None:
        raise RuntimeError(f"Phase-I overlap evidence is incomplete: {pilot.adaptation.reason}")
    structure = pilot.adaptation.structure
    components = _overlap_components(structure)
    if not components:
        raise RuntimeError("Phase-I evidence contains no shared-variable component")
    overlap_groups = tuple(group for component in components for group in component)
    total_budget_fes = pilot.checkpoint.total_budget_fes
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=total_budget_fes,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
    )
    proposal_budget = _proposal_budget(
        ledger.remaining,
        components,
        refresh_cycles=refresh_cycles,
        neighborhood_fes=neighborhood_fes,
    )
    persistent_coordinators = (
        {
            component: OverlapCoordinator(structure, ledger)
            for component in components
        }
        if coordination_mode in (PERSISTENT_CTP_MODE, GCB_COORDINATED_MODE)
        else {}
    )
    dispatch_planner = (
        GcbDispatchPlanner(
            structure,
            components,
            envelope_fes=neighborhood_fes,
            config=dispatch_config,
        )
        if coordination_mode == GCB_COORDINATED_MODE
        else None
    )
    traces = []
    for cycle_index in range(refresh_cycles):
        cycle_start = ledger.count
        error_before = float(ledger.best_error)
        anchor = tuple(float(value) for value in ledger.best_x)
        anchor_error = float(ledger.best_error)
        proposals: list[LocalProposal] = []
        for group in overlap_groups:
            run = produce_local_proposal(
                structure,
                group,
                problem=problem,
                global_ledger=ledger,
                anchor=anchor,
                anchor_error=anchor_error,
                budget_fes=proposal_budget,
                seed=pilot.checkpoint.run_seed
                ^ (0x9E37 * (group + 1))
                ^ (0x51ED * (cycle_index + 1)),
                algorithm="sepcmaes",
                population_size=PROPOSAL_POPULATION_SIZE,
                sigma=0.5,
            )
            proposals.append(run.proposal)
        proposal_fes = ledger.count - cycle_start
        error_after_proposals = float(ledger.best_error)
        arbitration_fes = 0
        endpoint_fes = 0
        coordination_fes = 0
        accepted = 0
        ctp_fes = 0
        ctp_triggered_components = 0
        max_conflict_streak = 0
        dispatch_receipts: list[DispatchReceipt] = []
        if coordination_mode != "proposal_only":
            for component_index, component in enumerate(components):
                coordinator = (
                    persistent_coordinators[component]
                    if coordination_mode in (PERSISTENT_CTP_MODE, GCB_COORDINATED_MODE)
                    else OverlapCoordinator(structure, ledger)
                )
                selected = tuple(proposal for proposal in proposals if proposal.group in component)
                before = ledger.count
                arbitration = coordinator.coordinate(
                    component,
                    selected,
                    ctp_budget_fes=(
                        neighborhood_fes
                        if coordination_mode == PERSISTENT_CTP_MODE
                        else 0
                    ),
                    ctp_seed=pilot.checkpoint.run_seed
                    ^ (0xD1CE * (cycle_index + 1))
                    ^ (0xBEEF * (component_index + 1)),
                    ctp_strategy=(
                        "sequential_coordinate_patch"
                        if coordination_mode == PERSISTENT_CTP_MODE
                        else "random"
                    ),
                )
                arbitration_fes += ledger.count - before
                ctp_fes += arbitration.ctp_consumed_fes
                ctp_triggered_components += int(arbitration.ctp_triggered)
                max_conflict_streak = max(max_conflict_streak, arbitration.conflict_streak)
                if len(arbitration.candidates) not in (3, 4):
                    raise RuntimeError("unexpected overlap arbitration candidate count")
                if coordination_mode == "proposal_neighborhood":
                    endpoints = coordinator.full_context_writeback(
                        component, selected, rounds=len(component)
                    )
                    endpoint_fes += endpoints.consumed_fes
                    accepted += sum(item.accepted for item in endpoints.rounds)
                    neighborhood = coordinator.proposal_neighborhood_writeback(
                        component,
                        selected,
                        budget_fes=neighborhood_fes,
                        seed=pilot.checkpoint.run_seed
                        ^ (0xA0B0 * (cycle_index + 1))
                        ^ (0xC7A5 * (component_index + 1)),
                    )
                    coordination_fes += neighborhood.consumed_fes
                    accepted += sum(item.accepted for item in neighborhood.rounds)
                elif coordination_mode == PERSISTENT_CTP_MODE:
                    endpoints = coordinator.full_context_writeback(
                        component, selected, rounds=len(component)
                    )
                    endpoint_fes += endpoints.consumed_fes
                    accepted += sum(item.accepted for item in endpoints.rounds)
                    fallback_fes = neighborhood_fes - arbitration.ctp_consumed_fes
                    if fallback_fes > 0:
                        neighborhood = coordinator.proposal_neighborhood_writeback(
                            component,
                            selected,
                            budget_fes=fallback_fes,
                            seed=pilot.checkpoint.run_seed
                            ^ (0xA0B0 * (cycle_index + 1))
                            ^ (0xC7A5 * (component_index + 1)),
                        )
                        coordination_fes += neighborhood.consumed_fes
                        accepted += sum(item.accepted for item in neighborhood.rounds)
                elif coordination_mode == GCB_COORDINATED_MODE:
                    if dispatch_planner is None:
                        raise RuntimeError("gcb_coordinated mode requires a dispatch planner")
                    plan = dispatch_planner.plan(
                        component,
                        cycle_index=cycle_index,
                        conflict_streak=arbitration.conflict_streak,
                    )
                    remainder = neighborhood_fes
                    if plan.action != DISPATCH_NEIGHBORHOOD:
                        ctp_triggered_components += 1
                        dispatch_before = float(ledger.best_error)
                        consumed = coordinator.dispatch_repair(
                            component,
                            selected,
                            budget_fes=plan.reserved_fes,
                            seed=pilot.checkpoint.run_seed
                            ^ (0xD1CE * (cycle_index + 1))
                            ^ (0xBEEF * (component_index + 1)),
                            strategy=DISPATCH_STRATEGIES[plan.action],
                        )
                        dispatch_after = float(ledger.best_error)
                        gained = dispatch_after < dispatch_before
                        ctp_fes += consumed
                        dispatch_planner.record_outcome(
                            component,
                            cycle_index=cycle_index,
                            action=plan.action,
                            gained=gained,
                        )
                        dispatch_receipts.append(
                            DispatchReceipt(
                                cycle_index=cycle_index,
                                component=component,
                                action=plan.action,
                                reason=plan.reason,
                                conflict_streak=arbitration.conflict_streak,
                                hub_degree=plan.hub_degree,
                                relative_hub=plan.relative_hub,
                                reserved_fes=plan.reserved_fes,
                                consumed_fes=consumed,
                                gained=gained,
                                best_error_before=dispatch_before,
                                best_error_after=dispatch_after,
                            )
                        )
                        remainder = neighborhood_fes - consumed
                    endpoints = coordinator.full_context_writeback(
                        component, selected, rounds=len(component)
                    )
                    endpoint_fes += endpoints.consumed_fes
                    accepted += sum(item.accepted for item in endpoints.rounds)
                    if remainder > 0:
                        neighborhood = coordinator.proposal_neighborhood_writeback(
                            component,
                            selected,
                            budget_fes=remainder,
                            seed=pilot.checkpoint.run_seed
                            ^ (0xA0B0 * (cycle_index + 1))
                            ^ (0xC7A5 * (component_index + 1)),
                        )
                        coordination_fes += neighborhood.consumed_fes
                        accepted += sum(item.accepted for item in neighborhood.rounds)
                else:
                    writeback = coordinator.full_context_writeback(
                        component,
                        selected,
                        rounds=len(component) + neighborhood_fes // 2,
                    )
                    endpoint_fes += writeback.consumed_fes
                    accepted += sum(item.accepted for item in writeback.rounds)
        traces.append(
            OverlapCycleResult(
                cycle_index=cycle_index,
                proposal_fes=proposal_fes,
                arbitration_fes=arbitration_fes,
                endpoint_fes=endpoint_fes,
                neighborhood_fes=coordination_fes,
                best_error_before=error_before,
                best_error_after_proposals=error_after_proposals,
                best_error_after=float(ledger.best_error),
                proposal_gain=error_before - error_after_proposals,
                coordination_gain=error_after_proposals - float(ledger.best_error),
                accepted_coordination_steps=accepted,
                ctp_fes=ctp_fes,
                ctp_triggered_components=ctp_triggered_components,
                max_conflict_streak=max_conflict_streak,
                dispatch_receipts=tuple(dispatch_receipts),
            )
        )
    tail_fes = ledger.remaining
    if tail_fes:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=pilot.checkpoint.run_seed ^ 0xE71D_3A26,
            budget_fes=tail_fes,
            population_size=24,
            restart=False,
        )
    if ledger.count != total_budget_fes:
        raise RuntimeError("overlap ARAC did not stop at the declared terminal FE")
    if any(item.best_error_after > item.best_error_before for item in traces):
        raise RuntimeError("overlap ARAC strict-best trace regressed")
    return OverlapAracResult(
        phase1=pilot,
        coordination_mode=coordination_mode,
        cycles=tuple(traces),
        overlap_groups=overlap_groups,
        overlap_components=components,
        proposal_budget_fes=proposal_budget,
        phase2_consumed_fes=ledger.count - pilot.checkpoint.phase1_fes,
        tail_fes=tail_fes,
        final_error=float(ledger.best_error),
        terminal_fes=ledger.count,
    )


__all__ = [
    "COORDINATION_MODES",
    "capped_proposal_budget",
    "DEFAULT_NEIGHBORHOOD_FES",
    "DEFAULT_REFRESH_CYCLES",
    "GCB_COORDINATED_MODE",
    "run_arac_oc",
    "OverlapAracResult",
    "OverlapCycleResult",
    "PERSISTENT_CTP_MODE",
    "run_overlap_arac",
    "run_overlap_from_pilot",
]
