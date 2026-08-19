"""Isolated Phase-I pilot that carries explicit variable-overlap evidence."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.overlap_adapter import (
    Phase1OverlapAdaptation,
    Phase1OverlapAdapter,
    Phase1OverlapEvidence,
)
from arac.evidence.phase1 import phase1_budget
from arac.evidence.sparse_overlap_discovery import (
    DEFAULT_BUCKET_SIZE,
    DEFAULT_MAX_CANDIDATE_PAIRS,
    DEFAULT_SPARSE_MIN_SUPPORT,
    DEFAULT_SPARSE_ROUNDS,
    SparseOverlapDiscoveryResult,
    discover_overlap_sparse,
)
from arac.runtime.contracts import PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort


PHASE1_OVERLAP_PILOT_PROTOCOL = "arac-phase1-overlap-pilot-v1"


@dataclass(frozen=True)
class Phase1OverlapPilotResult:
    """Complete receipt for the isolated Phase-I overlap handoff pilot."""

    checkpoint: PhaseCheckpoint
    evidence: Phase1OverlapEvidence
    discovery: SparseOverlapDiscoveryResult
    adaptation: Phase1OverlapAdaptation
    consumed_fes: int
    target_phase1_fes: int


def _validate_anchors(
    problem: OptimizationProblem,
    anchors: Iterable[Sequence[float]] | None,
    *,
    run_seed: int,
    anchor_count: int,
) -> tuple[tuple[float, ...], ...]:
    if anchors is None:
        center = (problem.lower_array + problem.upper_array) / 2.0
        span = problem.upper_array - problem.lower_array
        rng = np.random.default_rng(int(run_seed) ^ 0x5A17_2026)
        generated = center + rng.uniform(-0.2, 0.2, size=(anchor_count, problem.dimension)) * span
        anchors = tuple(tuple(float(value) for value in row) for row in generated)
    result = tuple(tuple(float(value) for value in anchor) for anchor in anchors)
    if not result:
        raise ValueError("at least one overlap pilot anchor is required")
    for anchor in result:
        point = np.asarray(anchor, dtype=float)
        if point.shape != (problem.dimension,) or not np.all(np.isfinite(point)):
            raise ValueError("overlap pilot anchors must match the problem dimension")
        if np.any(point < problem.lower_array) or np.any(point > problem.upper_array):
            raise ValueError("overlap pilot anchor escaped the problem bounds")
    return result


def _checkpoint(
    problem: OptimizationProblem,
    *,
    run_seed: int,
    total_budget_fes: int,
    phase1_fes: int,
    evidence: Phase1OverlapEvidence,
    discovery: SparseOverlapDiscoveryResult,
    ledger: EvaluationLedger,
) -> PhaseCheckpoint:
    feature_names = (
        "overlap_discovery_complete",
        "overlap_separated_pair_fraction",
        "overlap_candidate_pair_count",
        "overlap_discovery_fes",
        "overlap_remaining_phase1_fes",
    )
    feature_values = (
        float(evidence.complete),
        float(discovery.separated_pair_fraction),
        float(discovery.candidate_pair_count),
        float(discovery.consumed_fes),
        float(ledger.remaining),
    )
    return PhaseCheckpoint(
        protocol=PHASE1_OVERLAP_PILOT_PROTOCOL,
        run_seed=int(run_seed),
        total_budget_fes=int(total_budget_fes),
        phase1_fes=int(phase1_fes),
        incumbent=tuple(float(value) for value in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=feature_names,
        feature_values=feature_values,
        # Checkpoint blocks remain a legal partition.  Variable memberships are
        # carried by the explicit evidence object and never encoded as blocks.
        blocks=tuple((variable,) for variable in range(problem.dimension)),
    )


def run_phase1_overlap_pilot(
    problem: OptimizationProblem,
    *,
    total_budget_fes: int,
    run_seed: int,
    anchors: Iterable[Sequence[float]] | None = None,
    anchor_count: int = 5,
    step: float | Sequence[float] = 0.25,
    rounds: int = DEFAULT_SPARSE_ROUNDS,
    bucket_size: int = DEFAULT_BUCKET_SIZE,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
    min_support: float = DEFAULT_SPARSE_MIN_SUPPORT,
) -> Phase1OverlapPilotResult:
    """Run sparse discovery and incumbent completion inside one Phase-I budget.

    This pilot deliberately does not replace :func:`run_phase1`.  It validates
    the future overlap-aware checkpoint boundary while preserving the current
    production Phase-I implementation as an untouched control.
    """

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if isinstance(total_budget_fes, bool) or not isinstance(total_budget_fes, int) or total_budget_fes <= 0:
        raise ValueError("total_budget_fes must be a positive integer")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError("run_seed must be a non-negative integer")
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or anchor_count <= 0:
        raise ValueError("anchor_count must be a positive integer")
    target_phase1_fes = phase1_budget(total_budget_fes)
    phase_ledger = EvaluationLedger(problem, target_phase1_fes)
    anchor_points = _validate_anchors(
        problem,
        anchors,
        run_seed=run_seed,
        anchor_count=anchor_count,
    )
    discovery = discover_overlap_sparse(
        problem,
        phase_ledger,
        anchors=anchor_points,
        step=step,
        run_seed=run_seed,
        rounds=rounds,
        bucket_size=bucket_size,
        max_candidate_pairs=max_candidate_pairs,
        min_support=min_support,
    )
    if phase_ledger.count > target_phase1_fes:
        raise RuntimeError("overlap discovery exceeded the frozen Phase-I budget")
    if phase_ledger.count == 0:
        raise RuntimeError("overlap discovery did not populate the Phase-I archive")

    if phase_ledger.remaining:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=phase_ledger,
            initial_mean=tuple(float(value) for value in phase_ledger.best_x),
            sigma=0.5,
            seed=int(run_seed) ^ 0xE71D_3A26,
            budget_fes=phase_ledger.remaining,
            population_size=24,
            restart=False,
        )
    if phase_ledger.count != target_phase1_fes:
        raise RuntimeError("overlap Phase-I pilot did not stop at its frozen FE boundary")

    checkpoint = _checkpoint(
        problem,
        run_seed=run_seed,
        total_budget_fes=total_budget_fes,
        phase1_fes=target_phase1_fes,
        evidence=discovery.evidence,
        discovery=discovery,
        ledger=phase_ledger,
    )
    adaptation = Phase1OverlapAdapter().adapt(checkpoint, discovery.evidence)
    return Phase1OverlapPilotResult(
        checkpoint=checkpoint,
        evidence=discovery.evidence,
        discovery=discovery,
        adaptation=adaptation,
        consumed_fes=phase_ledger.count,
        target_phase1_fes=target_phase1_fes,
    )


__all__ = [
    "PHASE1_OVERLAP_PILOT_PROTOCOL",
    "Phase1OverlapPilotResult",
    "run_phase1_overlap_pilot",
]
