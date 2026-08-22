"""The unified ARAC-OC coordination loop (design section 8).

One cycle executes the canonical pipeline

```
SMP.sense -> GCB.select_scope -> counted_probe -> B/W/C -> arbitration
  -> GCB.make_plan -> operator.execute(plan) -> strict-best
  -> CoordinatorState.update (diagnostic EMA / residual streak / cooldown / pulse / qhat)
```

for the top-priority active component only, mirroring the design's
priority-queue semantics (section 7).  Operator exceptions fail closed:
the failed receipt is attached to the raised :class:`OperatorFailure`,
no retry or silent hand-off happens, and no terminal result is claimed.

Budget architecture (exact terminal FE): each cycle reserves the sense
budget for every remaining cycle plus a minimal arbitration window; the
operator pool may only spend what is left beyond that reservation, and
the pre-registered MMES terminal tail drains the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.contract import (
    OC_ACTION_ARBITRATION,
    OC_ACTION_AOR,
    OC_ACTION_CTP_RESTRICTED,
    OC_ACTION_CTP_SHARED_CORE,
    OC_ACTION_SMP,
    OC_PROBE_FES_PER_VARIABLE,
    OcCoordinatorConfig,
    OperatorPlan,
    OperatorReceipt,
    receipt_from_plan,
)
from arac.coordination.counted_probe import counted_probe
from arac.coordination.counterfactual import (
    CounterfactualCouplingReceipt,
    evaluate_frozen_private_counterfactual,
)
from arac.coordination.operators import (
    AorOperator,
    CtpRestrictedOperator,
    CtpSharedCoreOperator,
    OperatorExecution,
    SmpOperator,
    SmpSenseOperator,
)
from arac.coordination.overlap import OverlapCoordinator, OverlapStructure
from arac.coordination.planner import OcDispatchPlanner
from arac.coordination.shared_patch import (
    K_PATCH_FES,
    PATCH_MODES,
    SharedPatchKernel,
    patch_stable_hash,
)
from arac.coordination.state import CoordinatorState
from arac.evidence import Phase1OverlapPilotResult
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

OC_UNIFIED_MODE = "oc_unified"
OC_LOOP_SCHEMA = "arac-oc-unified-loop-v1"
# Three newly assembled candidates plus one frozen private counterfactual.
# The incumbent is already represented by the strict-best archive and is not
# charged again in the v6 production path.
_ARBITRATION_WINDOW_FES = 4
_SMP_WRITEBACK_FES = 32


def _smp_writeback_budget(component: tuple[int, ...]) -> int:
    return 2 * len(component) + _SMP_WRITEBACK_FES


def _counterfactual_candidate(arbitration):
    """Choose one evaluated complete candidate for the shadow receipt."""

    errors = dict(arbitration.candidate_errors)
    candidates = {candidate.name: candidate for candidate in arbitration.candidates}
    candidates.pop("incumbent", None)
    if not candidates:
        raise RuntimeError("arbitration produced no newly assembled candidates")
    preferred = arbitration.accepted_candidate
    if preferred in candidates and preferred in errors:
        name = preferred
    else:
        available = [(name, error) for name, error in errors.items() if name in candidates]
        if not available:
            raise RuntimeError("arbitration candidate errors have no matching vectors")
        name = min(available, key=lambda item: (item[1], item[0]))[0]
    return name, candidates[name], float(errors[name])


class OperatorFailure(RuntimeError):
    """Fail-closed operator exception; carries the failed receipt."""

    def __init__(self, receipt: OperatorReceipt) -> None:
        super().__init__(
            f"operator {receipt.action!r} failed after {receipt.actual_fes} FE "
            f"({receipt.exception_name}); remaining {receipt.remaining_fes}"
        )
        self.receipt = receipt


@dataclass(frozen=True)
class OcCycleTrace:
    cycle_index: int
    component: tuple[int, ...]
    sense_fes: int
    smp_fes: int
    probe_fes: int
    arbitration_fes: int
    operator_fes: int
    tail_fes: int
    action: str
    reason: str
    reserved_fes: int
    probe_budget_unavailable: bool
    operator_pool_exhausted: bool
    best_error_before: float
    best_error_after_sense: float
    best_error_after_arbitration: float
    best_error_after: float
    state_hash: str
    probe_max_c: float = 0.0
    probe_mean_c: float = 0.0
    arbitration_gain: float = 0.0
    arbitration_value_ratio: float = 0.0
    operator_value_ratio: float = 0.0
    operator_value_gated: bool = False
    counterfactual: CounterfactualCouplingReceipt | None = None
    counterfactual_unavailable: bool = False


@dataclass(frozen=True)
class OcLoopResult:
    schema_version: str
    coordination_mode: str
    phase1: Phase1OverlapPilotResult | None
    cycles: tuple[OcCycleTrace, ...]
    receipts: tuple[OperatorReceipt, ...]
    phase2_consumed_fes: int
    tail_fes: int
    final_error: float
    terminal_fes: int
    final_state_hash: str


def _overlap_components(structure: OverlapStructure) -> tuple[tuple[int, ...], ...]:
    shared = set(structure.shared_variables)
    return tuple(
        component
        for component in structure.connected_components()
        if any(shared.intersection(structure.groups[group]) for group in component)
    )


def run_oc_unified(
    problem: OptimizationProblem,
    pilot: Phase1OverlapPilotResult,
    *,
    refresh_cycles: int,
    sense_budget_fes: int,
    config: OcCoordinatorConfig | None = None,
    patch_config: dict | None = None,
) -> OcLoopResult:
    """Run the unified loop from a frozen Phase-I pilot, exact terminal FE."""

    if not isinstance(pilot, Phase1OverlapPilotResult):
        raise TypeError("pilot must be Phase1OverlapPilotResult")
    if not pilot.adaptation.ready or pilot.adaptation.structure is None:
        raise RuntimeError(f"Phase-I overlap evidence is incomplete: {pilot.adaptation.reason}")
    return _run_oc_unified_core(
        problem,
        pilot.adaptation.structure,
        checkpoint_hash=pilot.checkpoint.checkpoint_hash,
        total_budget_fes=pilot.checkpoint.total_budget_fes,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
        run_seed=pilot.checkpoint.run_seed,
        refresh_cycles=refresh_cycles,
        sense_budget_fes=sense_budget_fes,
        config=config,
        phase1=pilot,
        patch_config=patch_config,
    )


def run_oc_unified_from_structure(
    problem: OptimizationProblem,
    structure: OverlapStructure,
    *,
    checkpoint_hash: str,
    total_budget_fes: int,
    phase1_fes: int,
    incumbent: tuple[float, ...],
    incumbent_error: float,
    run_seed: int,
    refresh_cycles: int,
    sense_budget_fes: int,
    config: OcCoordinatorConfig | None = None,
    patch_config: dict | None = None,
) -> OcLoopResult:
    """Run the unified loop from a caller-supplied structure (AOB v3 path).

    The caller owns the Phase-I boundary: the structure must come from the
    Gate 42 hyperedge gate over soft-RDDSM evidence, the incumbent/error
    pair must be the strict-best archive at exactly ``phase1_fes``, and
    ``checkpoint_hash`` identifies that boundary for the state chain.
    """

    if not isinstance(structure, OverlapStructure):
        raise TypeError("structure must be OverlapStructure")
    if len(incumbent) != problem.dimension:
        raise ValueError("incumbent and problem dimensions disagree")
    return _run_oc_unified_core(
        problem,
        structure,
        checkpoint_hash=checkpoint_hash,
        total_budget_fes=total_budget_fes,
        phase1_fes=phase1_fes,
        incumbent=incumbent,
        incumbent_error=incumbent_error,
        run_seed=run_seed,
        refresh_cycles=refresh_cycles,
        sense_budget_fes=sense_budget_fes,
        config=config,
        phase1=None,
    )


def _run_oc_unified_core(
    problem: OptimizationProblem,
    structure: OverlapStructure,
    *,
    checkpoint_hash: str,
    total_budget_fes: int,
    phase1_fes: int,
    incumbent: tuple[float, ...],
    incumbent_error: float,
    run_seed: int,
    refresh_cycles: int,
    sense_budget_fes: int,
    config: OcCoordinatorConfig | None = None,
    phase1: Phase1OverlapPilotResult | None = None,
    patch_config: dict | None = None,
) -> OcLoopResult:
    if isinstance(refresh_cycles, bool) or not isinstance(refresh_cycles, int) or refresh_cycles <= 0:
        raise ValueError("refresh_cycles must be a positive integer")
    if isinstance(sense_budget_fes, bool) or not isinstance(sense_budget_fes, int) or sense_budget_fes < 8:
        raise ValueError("sense_budget_fes must be an integer of at least 8")
    config = OcCoordinatorConfig() if config is None else config
    if patch_config is not None:
        if not isinstance(patch_config, dict) or patch_config.get("mode") not in PATCH_MODES:
            raise ValueError("patch_config must be None or {'mode': one of PATCH_MODES}")
    patch_kernel = SharedPatchKernel() if patch_config is not None else None
    components = _overlap_components(structure)
    if not components:
        raise RuntimeError("Phase-I evidence contains no shared-variable component")
    overlap_groups = tuple(group for component in components for group in component)

    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=total_budget_fes,
        phase1_fes=phase1_fes,
        incumbent=incumbent,
        incumbent_error=incumbent_error,
    )
    planner = OcDispatchPlanner(
        structure, list(components), config=config, base_seed=run_seed
    )
    state = CoordinatorState(
        structure,
        list(components),
        config=config,
        checkpoint_hash=checkpoint_hash,
    )
    if patch_kernel is not None:
        patch_kernel.load(state.shared_patch)
    sense_operator = SmpSenseOperator()
    operators = {
        OC_ACTION_SMP: SmpOperator(problem),
        OC_ACTION_CTP_RESTRICTED: CtpRestrictedOperator(),
        OC_ACTION_CTP_SHARED_CORE: CtpSharedCoreOperator(),
        OC_ACTION_AOR: AorOperator(),
    }
    coordinators = {component: OverlapCoordinator(structure, ledger) for component in components}
    receipts: list[OperatorReceipt] = []
    traces: list[OcCycleTrace] = []

    for cycle_index in range(refresh_cycles):
        cycle_start = ledger.count
        error_before = float(ledger.best_error)
        probe_unavailable = False
        pool_exhausted = False
        plan: OperatorPlan | None = None
        patch_result = None
        operator_fes = 0
        smp_fes = 0
        probe_fes_used = 0
        arbitration_fes_used = 0
        error_after_arbitration = error_before
        probe_max_c = 0.0
        probe_mean_c = 0.0
        arbitration_gain = 0.0
        arbitration_value_ratio = 0.0
        operator_value_ratio = 0.0
        operator_value_gated = False
        counterfactual: CounterfactualCouplingReceipt | None = None
        counterfactual_unavailable = False

        # --- sense: the owner-local proposal lane persists through dispatch
        # stall/cooldown; those controls only gate a new operator plan. ---
        eligible_components = state.sensing_components()
        sense_groups = tuple(group for component in eligible_components for group in component)
        # Do not start a sense session that cannot be paid in full.  The
        # remaining budget is still consumed by the exact-budget terminal tail.
        if sense_groups and ledger.remaining < len(sense_groups) * sense_budget_fes:
            break
        runs = sense_operator.sense(
            structure,
            sense_groups,
            problem=problem,
            ledger=ledger,
            budget_fes_per_group=sense_budget_fes,
            seed=run_seed ^ (0x51ED * (cycle_index + 1)),
        )
        proposals = tuple(run.proposal for run in runs)
        sense_fes = ledger.count - cycle_start
        error_after_sense = float(ledger.best_error)
        contribution_by_group = {run.proposal.group: max(0.0, float(run.proposal.improvement)) for run in runs}
        smp_writeback_budget = sum(
            _smp_writeback_budget(component) for component in eligible_components
        )

        # --- priority selection over active, non-cooling components ---
        signals = [
            state.signal(
                component,
                cycle_index=cycle_index,
                proposal_contribution=max(
                    (contribution_by_group.get(group, 0.0) for group in component), default=0.0
                ),
            )
            for component in components
        ]
        ranked = planner.prioritize(signals)
        if ranked:
            component = ranked[0].component
            coordinator = coordinators[component]
            component_proposals = tuple(p for p in proposals if p.group in set(component))

            # --- scope selection under probe affordability ---
            probe_budget = int(ledger.remaining * config.probe_budget_share)
            # select_scope returns EMA-priority order; every downstream
            # consumer (probe, plan, writeback, counterfactual receipt) is
            # order-insensitive and the frozen-private counterfactual
            # validates sortedness, so canonicalize once here.
            scope = tuple(sorted(planner.select_scope(component, state.ema_c, probe_budget_fes=probe_budget)))
            if not scope:
                probe_unavailable = True
            else:
                probe_results = counted_probe(structure, ledger, scope, proposals=component_proposals)
                probe_fes_used = 2 * len(scope)
                probe_max_c = max(result.conflict_score for result in probe_results)
                probe_mean_c = sum(result.conflict_score for result in probe_results) / len(probe_results)
                state.observe_probes(
                    component, scope, {r.variable: r.conflict_score for r in probe_results}
                )

                # --- arbitration: archive incumbent + up to three new candidates ---
                arbitration_incumbent = coordinator.ledger.best_x
                arbitration = coordinator.coordinate(
                    component,
                    component_proposals,
                    ctp_budget_fes=0,
                    ctp_strategy="random",
                    reuse_incumbent=True,
                )
                if ledger.remaining > 0:
                    candidate_name, candidate, candidate_error = _counterfactual_candidate(arbitration)
                    counterfactual = evaluate_frozen_private_counterfactual(
                        ledger,
                        component=component,
                        scope=scope,
                        incumbent=arbitration_incumbent,
                        best_error_before=arbitration.best_error_before,
                        candidate_name=candidate_name,
                        candidate=candidate.vector,
                        full_candidate_error=candidate_error,
                    )
                else:
                    counterfactual_unavailable = True
                # SMP writeback is its own budget lane; do not charge those
                # evaluations to arbitration a second time.
                arbitration_fes_used = (
                    ledger.count - cycle_start - sense_fes - smp_fes - probe_fes_used
                )
                error_after_arbitration = float(ledger.best_error)
                arbitration_gain = max(0.0, error_after_sense - error_after_arbitration)
                arbitration_value_ratio = arbitration_gain / max(abs(error_after_sense), 1.0)
                state.observe_proposal_conflict(
                    component,
                    high_conflict=arbitration.conflict_level.value == "high",
                )

                # --- plan with the post-probe signal ---
                signal = state.signal(
                    component,
                    cycle_index=cycle_index,
                    proposal_contribution=max(
                        (contribution_by_group.get(group, 0.0) for group in component), default=0.0
                    ),
                )
                cycles_left = refresh_cycles - cycle_index - 1
                max_future_probe_fes = max(
                    (
                        OC_PROBE_FES_PER_VARIABLE
                        * len(planner.shared_scope_variables(candidate))
                        for candidate in components
                    ),
                    default=0,
                )
                continuation = cycles_left * (
                    len(overlap_groups) * sense_budget_fes
                    + _ARBITRATION_WINDOW_FES
                    + max_future_probe_fes
                    + sum(
                        _smp_writeback_budget(candidate)
                        for candidate in eligible_components
                    )
                )
                operator_pool = max(
                    0,
                    ledger.remaining - smp_writeback_budget - continuation,
                )
                # AOR/MMES needs a two-member population.  With one FE left,
                # leave the operator pool empty and let the exact-budget tail
                # consume it rather than emitting an executable-invalid plan.
                if operator_pool < 2 and signal.level != "low":
                    pool_exhausted = True
                else:
                    plan = planner.make_plan(
                        signal,
                        cycle_index=cycle_index,
                        scope=scope,
                        probe_widths={r.variable: r.width for r in probe_results},
                        available_fes=operator_pool,
                        arbitration_gain=arbitration_gain,
                        arbitration_reference_error=error_after_sense,
                    )
                    if plan.action == OC_ACTION_ARBITRATION:
                        execution = OperatorExecution(
                            actual_fes=0,
                            best_error_before=arbitration.best_error_before,
                            best_error_after=arbitration.best_error_after,
                        )
                    else:
                        operator = operators[plan.action]
                        exec_start = ledger.count
                        operator_error_before = float(ledger.best_error)
                        operator_archive = ledger.archive_snapshot()
                        exec_plan = plan
                        try:
                            if (
                                patch_kernel is not None
                                and plan.action
                                in (OC_ACTION_CTP_RESTRICTED, OC_ACTION_CTP_SHARED_CORE)
                                and plan.scope
                                and plan.reserved_fes >= K_PATCH_FES
                            ):
                                context_hash = patch_stable_hash(
                                    checkpoint_hash,
                                    state.snapshot().state_hash,
                                    plan.plan_hash,
                                )
                                patch_result = patch_kernel.apply(
                                    plan.component,
                                    component_proposals,
                                    plan.scope,
                                    context_hash,
                                    structure=coordinator.structure,
                                    ledger=ledger,
                                    budget_fes=K_PATCH_FES,
                                    seed=plan.seed,
                                    mode=patch_config["mode"],
                                )
                                state.shared_patch = patch_kernel.payload()
                                if patch_result.budget_status == "executed":
                                    exec_plan = dataclasses.replace(
                                        plan, reserved_fes=plan.reserved_fes - K_PATCH_FES
                                    )
                            if plan.action == OC_ACTION_SMP:
                                execution = operator.execute_plan(exec_plan, coordinator=coordinator)
                            else:
                                execution = operator.execute_plan(
                                    exec_plan, coordinator=coordinator, proposals=component_proposals
                                )
                        except Exception as exc:  # fail closed (design section 2.2)
                            failed = receipt_from_plan(
                                plan,
                                actual_fes=ledger.count - exec_start,
                                best_error_before=operator_error_before,
                                best_error_after=float(ledger.best_error),
                                state_hash=state.snapshot().state_hash,
                                remaining_fes=ledger.remaining,
                                exception_name=type(exc).__name__,
                                patch_result=patch_result,
                            )
                            raise OperatorFailure(failed) from exc
                        operator_value_ratio = max(
                            0.0,
                            execution.best_error_before - execution.best_error_after,
                        ) / max(abs(execution.best_error_before), 1.0)
                        if operator_value_ratio < config.operator_value_ratio:
                            ledger.restore_archive(operator_archive)
                            execution = OperatorExecution(
                                actual_fes=execution.actual_fes,
                                best_error_before=operator_error_before,
                                best_error_after=operator_error_before,
                                candidates=execution.candidates,
                            )
                            operator_value_gated = True
                    operator_fes = execution.actual_fes
                    if plan.action != OC_ACTION_ARBITRATION:
                        state.update_dispatch(
                            component,
                            cycle_index=cycle_index,
                            action=plan.action,
                            gained=execution.best_error_after < execution.best_error_before,
                            scope=scope,
                            realized_gain=max(
                                0.0, execution.best_error_before - execution.best_error_after
                            ),
                            predicted_gain=plan.predicted_gain,
                        )
                    elif plan.reason == "stall_guard_arbitration":
                        state.record_stall_guard(component, cycle_index=cycle_index)
                    snapshot = state.snapshot()
                    receipts.append(
                        receipt_from_plan(
                            plan,
                            actual_fes=(
                                ledger.count - exec_start
                                if patch_result is not None
                                else execution.actual_fes
                            ),
                            best_error_before=(
                                operator_error_before
                                if patch_result is not None
                                else execution.best_error_before
                            ),
                            best_error_after=execution.best_error_after,
                            candidates=execution.candidates,
                            state_hash=snapshot.state_hash,
                            remaining_fes=ledger.remaining,
                            patch_result=patch_result,
                        )
                    )
        # Preserve the proposal arm's evolving context after arbitration and
        # any explicitly reserved operator window.  This ordering makes the
        # next SMP sense start from the same post-writeback incumbent while
        # keeping the 32-FE lane independent from GCB dispatch.
        if smp_writeback_budget and ledger.remaining >= smp_writeback_budget:
            for component_index, component in enumerate(eligible_components):
                base_coordinator = coordinators[component]
                component_proposals = tuple(p for p in proposals if p.group in set(component))
                context = base_coordinator.full_context_writeback(
                    component,
                    component_proposals,
                    rounds=len(component),
                )
                smp_fes += context.consumed_fes
                writeback = base_coordinator.proposal_neighborhood_writeback(
                    component,
                    component_proposals,
                    budget_fes=_SMP_WRITEBACK_FES,
                    seed=run_seed
                    ^ (0xA0B0 * (cycle_index + 1))
                    ^ (0xC7A5 * (component_index + 1)),
                )
                smp_fes += writeback.consumed_fes
        trace_action = plan.action if plan else ("probe_budget_unavailable" if probe_unavailable else "none")
        trace_reason = (
            plan.reason
            if plan
            else (
                "probe_budget_unavailable"
                if probe_unavailable
                else "operator_pool_exhausted"
                if pool_exhausted
                else "no_active_component"
            )
        )
        traces.append(
            OcCycleTrace(
                cycle_index=cycle_index,
                component=ranked[0].component if ranked else (),
                sense_fes=sense_fes,
                smp_fes=smp_fes,
                probe_fes=probe_fes_used,
                arbitration_fes=arbitration_fes_used,
                operator_fes=operator_fes,
                tail_fes=0,
                action=trace_action,
                reason=trace_reason,
                reserved_fes=plan.reserved_fes if plan else 0,
                probe_budget_unavailable=probe_unavailable,
                operator_pool_exhausted=pool_exhausted,
                best_error_before=error_before,
                best_error_after_sense=error_after_sense,
                best_error_after_arbitration=error_after_arbitration,
                best_error_after=float(ledger.best_error),
                state_hash=state.snapshot().state_hash,
                probe_max_c=probe_max_c,
                probe_mean_c=probe_mean_c,
                arbitration_gain=arbitration_gain,
                arbitration_value_ratio=arbitration_value_ratio,
                operator_value_ratio=operator_value_ratio,
                operator_value_gated=operator_value_gated,
                counterfactual=counterfactual,
                counterfactual_unavailable=counterfactual_unavailable,
            )
        )
        if probe_unavailable:
            break

    tail_fes = ledger.remaining
    if tail_fes:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=run_seed ^ 0xE71D_3A26,
            budget_fes=tail_fes,
            population_size=24,
            restart=False,
        )
    if ledger.count != total_budget_fes:
        raise RuntimeError("unified ARAC-OC loop did not stop at the declared terminal FE")
    if any(trace.best_error_after > trace.best_error_before for trace in traces):
        raise RuntimeError("unified ARAC-OC strict-best trace regressed")
    return OcLoopResult(
        schema_version=OC_LOOP_SCHEMA,
        coordination_mode=OC_UNIFIED_MODE,
        phase1=phase1,
        cycles=tuple(traces),
        receipts=tuple(receipts),
        phase2_consumed_fes=ledger.count - phase1_fes,
        tail_fes=tail_fes,
        final_error=float(ledger.best_error),
        terminal_fes=ledger.count,
        final_state_hash=state.snapshot().state_hash,
    )


__all__ = ["OC_UNIFIED_MODE", "OcCycleTrace", "OcLoopResult", "OperatorFailure",
    "run_oc_unified", "run_oc_unified_from_structure",]
