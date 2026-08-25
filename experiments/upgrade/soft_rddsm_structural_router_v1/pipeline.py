"""Compose soft-RDDSM evidence with a candidate four-action dispatcher.

The dispatcher is deliberately confined to the upgrade lane.  It does not
replace the production selector or action registry.  Because the frozen
Phase-II actions consume only ``PhaseCheckpoint.blocks`` and ``relations``,
the dispatcher creates a deterministic *action view* of the Phase-I
checkpoint: the incumbent, FE boundary and numerical features are preserved,
while relation edges are rebuilt from the confirmed overlap sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

from arac.actions.registry import ActionRegistry
from arac.analysis.structural_router import (
    ACTION_AOR,
    ACTION_CTP,
    ACTION_GCB,
    ACTION_SMP,
    StructuralRouteDecision,
    route_from_overlap_evidence,
)
from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.overlap_adapter import Phase1OverlapAdaptation, Phase1OverlapAdapter
from arac.evidence.overlap_adapter import Phase1OverlapEvidence
from arac.runtime.contracts import ActionExecutionRegistry, ActionResult, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.upgrade.soft_rddsm_mainline_v10.phase1_v10 import Phase1V10Result, run_phase1_v10


@dataclass(frozen=True)
class SoftRddsmStructuralRun:
    """One candidate Phase-I checkpoint and its non-learning route."""

    phase1: Phase1V10Result
    decision: StructuralRouteDecision
    adaptation: Phase1OverlapAdaptation

    def __post_init__(self) -> None:
        if self.decision.action_name not in {"aor", "ctp", "smp", "gcb"}:
            raise ValueError("structural route contains an unsupported action")
        if self.adaptation.checkpoint_hash != self.phase1.checkpoint.checkpoint_hash:
            raise ValueError("overlap adaptation is not bound to the Phase-I checkpoint")


@dataclass(frozen=True)
class SoftRddsmStructuralExecution:
    """Auditable result of one candidate structural route plus one action."""

    structural_run: SoftRddsmStructuralRun
    action_checkpoint: PhaseCheckpoint
    action_result: ActionResult

    def __post_init__(self) -> None:
        source = self.structural_run.phase1.checkpoint
        if self.action_result.action_name != self.structural_run.decision.action_name:
            raise ValueError("dispatched action disagrees with structural route")
        if self.action_result.checkpoint_hash != self.action_checkpoint.checkpoint_hash:
            raise ValueError("action result is not bound to the action-view checkpoint")
        for field in ("total_budget_fes", "phase1_fes", "incumbent", "incumbent_error"):
            if getattr(self.action_checkpoint, field) != getattr(source, field):
                raise ValueError(f"action view changed Phase-I boundary field: {field}")
        if self.action_result.terminal_fes != source.total_budget_fes:
            raise ValueError("dispatched action did not reach the Phase-I run terminal budget")

    @property
    def source_checkpoint_hash(self) -> str:
        """Hash of the original Phase-I checkpoint before action projection."""

        return self.structural_run.phase1.checkpoint.checkpoint_hash

    @property
    def action_checkpoint_hash(self) -> str:
        """Hash consumed by the Phase-II action contract."""

        return self.action_checkpoint.checkpoint_hash


def _sidecar_relations(evidence: Phase1OverlapEvidence) -> tuple[RelationEvidence, ...]:
    """Convert confirmed owner memberships into deterministic relation edges."""

    confidence = {
        (variable, group): float(value)
        for variable, group, value in evidence.membership_confidences
    }
    pair_strength: dict[tuple[int, int], float] = {}
    for variable, owners in enumerate(evidence.memberships):
        for left, right in combinations(sorted(owners), 2):
            keys = ((variable, left), (variable, right))
            if any(key not in confidence for key in keys):
                raise ValueError("complete overlap evidence lacks relation confidence")
            strength = min(confidence[key] for key in keys)
            pair_strength[(left, right)] = max(pair_strength.get((left, right), 0.0), strength)
    return tuple(
        RelationEvidence(
            left_block=left,
            right_block=right,
            strength=float(strength),
            disagreement=float(1.0 - strength),
        )
        for (left, right), strength in sorted(pair_strength.items())
    )


def action_view_checkpoint(run: SoftRddsmStructuralRun) -> PhaseCheckpoint:
    """Build the checkpoint view consumed by the candidate Phase-II action.

    The frozen checkpoint contract requires disjoint blocks, so memberships
    stay in the sidecar and only their induced owner graph is projected into
    ``relations``.  This preserves the action contract while ensuring the
    CTP/GCB topology is derived from confirmed overlap evidence rather than
    incidental Phase-I screening edges.
    """

    if not isinstance(run, SoftRddsmStructuralRun):
        raise TypeError("action view requires a SoftRddsmStructuralRun")
    source = run.phase1.checkpoint
    evidence = run.phase1.overlap_evidence
    if not isinstance(evidence, Phase1OverlapEvidence):
        raise TypeError("structural run lacks a Phase1OverlapEvidence sidecar")
    if run.adaptation.checkpoint_hash != source.checkpoint_hash:
        raise ValueError("structural adaptation is bound to a different checkpoint")

    action = run.decision.action_name
    if action == ACTION_AOR:
        if evidence.complete:
            raise ValueError("AOR route requires incomplete overlap evidence")
        return source
    if not evidence.complete or not run.adaptation.ready or run.adaptation.structure is None:
        raise ValueError("non-AOR route requires complete actionable overlap evidence")
    if len(source.blocks) != len(evidence.groups):
        raise ValueError("sidecar group count does not match the Phase-I block partition")
    if action == ACTION_SMP:
        if any(len(owners) > 1 for owners in evidence.memberships):
            raise ValueError("SMP route cannot use a shared-variable sidecar")
        relations: tuple[RelationEvidence, ...] = ()
    elif action in {ACTION_CTP, ACTION_GCB}:
        relations = _sidecar_relations(evidence)
        if not relations:
            raise ValueError("overlap route requires at least one confirmed owner relation")
    else:
        raise ValueError(f"unsupported structural route: {action}")
    return replace(source, relations=relations)


def execute_soft_rddsm_structural_route(
    run: SoftRddsmStructuralRun,
    problem: OptimizationProblem,
    *,
    action_seed: int,
    registry: ActionExecutionRegistry | None = None,
) -> SoftRddsmStructuralExecution:
    """Execute exactly the structurally selected candidate action."""

    if not isinstance(run, SoftRddsmStructuralRun):
        raise TypeError("structural execution requires a SoftRddsmStructuralRun")
    if not isinstance(problem, OptimizationProblem):
        raise TypeError("structural execution requires an OptimizationProblem")
    source = run.phase1.checkpoint
    if problem.dimension != len(source.incumbent):
        raise ValueError("problem and Phase-I checkpoint dimensions disagree")
    action_checkpoint = action_view_checkpoint(run)
    active_registry = ActionRegistry() if registry is None else registry
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=action_checkpoint.total_budget_fes,
        phase1_fes=action_checkpoint.phase1_fes,
        incumbent=action_checkpoint.incumbent,
        incumbent_error=action_checkpoint.incumbent_error,
        allow_out_of_bounds=active_registry.allow_out_of_bounds,
    )
    result = execute_phase2_action(
        run.decision.action_name,
        action_checkpoint,
        problem,
        ledger,
        action_seed=action_seed,
        registry=active_registry,
    )
    return SoftRddsmStructuralExecution(
        structural_run=run,
        action_checkpoint=action_checkpoint,
        action_result=result,
    )


def run_soft_rddsm_structural_router(
    problem: OptimizationProblem,
    *,
    run_seed: int,
) -> SoftRddsmStructuralRun:
    """Run soft-RDDSM discovery and route from its structural certificate."""

    phase1 = run_phase1_v10(problem, run_seed=run_seed)
    decision = route_from_overlap_evidence(phase1.overlap_evidence)
    adaptation = Phase1OverlapAdapter().adapt(
        phase1.checkpoint,
        phase1.overlap_evidence,
    )
    return SoftRddsmStructuralRun(
        phase1=phase1,
        decision=decision,
        adaptation=adaptation,
    )


def run_and_execute_soft_rddsm_structural_route(
    problem: OptimizationProblem,
    *,
    run_seed: int,
    action_seed: int,
    registry: ActionExecutionRegistry | None = None,
) -> SoftRddsmStructuralExecution:
    """Run candidate Phase-I discovery and execute its one selected action."""

    structural_run = run_soft_rddsm_structural_router(problem, run_seed=run_seed)
    return execute_soft_rddsm_structural_route(
        structural_run,
        problem,
        action_seed=action_seed,
        registry=registry,
    )


__all__ = [
    "SoftRddsmStructuralExecution",
    "SoftRddsmStructuralRun",
    "action_view_checkpoint",
    "execute_soft_rddsm_structural_route",
    "run_and_execute_soft_rddsm_structural_route",
    "run_soft_rddsm_structural_router",
]
