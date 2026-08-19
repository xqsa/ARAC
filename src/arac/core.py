"""Active ARAC-Core policy: Phase-I evidence selects one terminal action."""

from __future__ import annotations

from dataclasses import dataclass

from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.mechanism_features import summarize_relation_topology
from arac.evidence.phase1 import Phase1Probe, run_phase1
from arac.runtime.contracts import (
    ACTION_NAMES,
    ActionExecutionRegistry,
    ActionResult,
    PhaseCheckpoint,
)
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action


@dataclass(frozen=True)
class AracCoreDecision:
    """One identity-blind action decision and its structural reason."""

    action_name: str
    reason: str
    scores: tuple[tuple[str, float], ...]
    structural_inference_complete: bool
    relation_count: int
    largest_component_fraction: float

    def __post_init__(self) -> None:
        if self.action_name not in ACTION_NAMES:
            raise ValueError("ARAC-Core selected an unsupported action")
        if tuple(action for action, _ in self.scores) != ACTION_NAMES:
            raise ValueError("ARAC-Core scores do not cover the frozen action set")
        if any(score not in {0.0, 1.0} for _, score in self.scores):
            raise ValueError("ARAC-Core scores must be one-hot")
        if sum(score == 1.0 for _, score in self.scores) != 1:
            raise ValueError("ARAC-Core scores must contain one active action")


@dataclass(frozen=True)
class AracCoreResult:
    """Auditable result of one evidence decision followed by one action."""

    decision: AracCoreDecision
    action_result: ActionResult

    def __post_init__(self) -> None:
        if self.decision.action_name != self.action_result.action_name:
            raise ValueError("ARAC-Core decision and action result drifted")


@dataclass(frozen=True)
class AracRunResult:
    """Complete Phase-I evidence and Phase-II single-action result."""

    phase1: Phase1Probe
    core: AracCoreResult

    def __post_init__(self) -> None:
        if self.phase1.checkpoint.checkpoint_hash != self.core.action_result.checkpoint_hash:
            raise ValueError("ARAC run is not bound to one Phase-I checkpoint")


def select_core_action(checkpoint: PhaseCheckpoint) -> AracCoreDecision:
    """Select from structural completeness and relation connectivity only."""

    if not isinstance(checkpoint, PhaseCheckpoint):
        raise TypeError("ARAC-Core selection requires PhaseCheckpoint")
    features = dict(zip(checkpoint.feature_names, checkpoint.feature_values, strict=True))
    if "structural_inference_complete" not in features:
        raise ValueError("checkpoint lacks structural inference completeness")
    completeness = float(features["structural_inference_complete"])
    if completeness not in {0.0, 1.0}:
        raise ValueError("structural inference completeness must be binary")

    if completeness == 0.0:
        action = "aor"
        reason = "incomplete_structure"
        largest_component = 0.0
    elif checkpoint.overlap_relation_count == 0:
        action = "smp"
        reason = "complete_zero_relation_blocks"
        largest_component = 0.0
    else:
        _, _, largest_component = summarize_relation_topology(
            checkpoint.blocks,
            checkpoint.relations,
        )
        if largest_component < 1.0:
            action = "ctp"
            reason = "complete_disconnected_relation_cover"
        else:
            action = "gcb"
            reason = "complete_connected_relation_graph"

    return AracCoreDecision(
        action_name=action,
        reason=reason,
        scores=tuple((candidate, float(candidate == action)) for candidate in ACTION_NAMES),
        structural_inference_complete=bool(completeness),
        relation_count=checkpoint.overlap_relation_count,
        largest_component_fraction=float(largest_component),
    )


def run_arac_core(
    checkpoint: PhaseCheckpoint,
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    action_seed: int,
    registry: ActionExecutionRegistry | None = None,
) -> AracCoreResult:
    """Select once at the Phase-I boundary and execute only that action."""

    active_registry = ActionRegistry() if registry is None else registry
    decision = select_core_action(checkpoint)
    action_result = execute_phase2_action(
        decision.action_name,
        checkpoint,
        problem,
        ledger,
        action_seed=action_seed,
        registry=active_registry,
    )
    return AracCoreResult(decision=decision, action_result=action_result)


def run_arac(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    action_seed: int,
    registry: ActionExecutionRegistry | None = None,
) -> AracRunResult:
    """Run the complete evidence-to-one-action ARAC-Core method."""

    active_registry = ActionRegistry() if registry is None else registry
    ledger = EvaluationLedger(
        problem,
        total_budget_fes,
        allow_out_of_bounds=active_registry.allow_out_of_bounds,
    )
    phase1 = run_phase1(problem, ledger, run_seed=run_seed)
    core = run_arac_core(
        phase1.checkpoint,
        problem,
        ledger,
        action_seed=action_seed,
        registry=active_registry,
    )
    return AracRunResult(phase1=phase1, core=core)


__all__ = [
    "AracCoreDecision",
    "AracCoreResult",
    "AracRunResult",
    "execute_phase2_action",
    "run_arac",
    "run_arac_core",
    "select_core_action",
]
