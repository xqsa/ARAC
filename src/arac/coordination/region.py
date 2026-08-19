"""Region-level coordination interfaces (Phase-I v10.2, Gate 44).

``RegionProposal``, ``region_conflict_probe`` and a minimal
``RegionCoordinator`` operate on ``RegionStructure`` evidence only.  They
never construct an ``OverlapStructure``: the only legal bridge between
region-level and variable-level structures stays gated by resolved overlap
hyperedges (``arac.evidence.hierarchical.to_overlap_structure``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.proposals import _MirroredLedger
from arac.evidence.hierarchical import (
    Phase1Evidence,
    RegionStructure,
    VariableRegionInteraction,
)
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import ResumableOptimizerSession


@dataclass(frozen=True)
class RegionProposal:
    """One region-local optimizer proposal over the region's coordinates."""

    leaf_id: int
    values: tuple[tuple[int, float], ...]
    uncertainty: tuple[tuple[int, float], ...]
    improvement: float
    consumed_fes: int


@dataclass(frozen=True)
class RegionProbeResult:
    """Counted two-sided conflict evidence for one candidate variable."""

    variable: int
    source_region: int
    target_region: int
    step: float
    f_plus: float
    f_minus: float
    bias: float
    width: float
    conflict_score: float
    consumed_fes: int


@dataclass(frozen=True)
class RegionCycleReceipt:
    """Auditable FE and acceptance record of one region coordination cycle."""

    component: tuple[int, ...]
    proposal_leaf: int
    proposal_fes: int
    probe_fes: int
    patch_fes: int
    patched_candidates: tuple[int, ...]
    accepted: bool
    best_error_before: float
    best_error_after: float


def produce_region_proposal(
    structure: RegionStructure,
    leaf_id: int,
    *,
    problem: OptimizationProblem,
    global_ledger: EvaluationLedger,
    anchor: np.ndarray,
    anchor_error: float,
    budget_fes: int,
    seed: int,
    algorithm: str = "sepcmaes",
    population_size: int = 8,
    sigma: float = 0.5,
) -> RegionProposal:
    """Optimize one region's coordinates from a shared anchor, billed exactly."""

    if not isinstance(structure, RegionStructure):
        raise TypeError("structure must be RegionStructure")
    if problem is not global_ledger.problem or structure.evidence.dimension != problem.dimension:
        raise ValueError("region structure, problem and ledger dimensions must agree")
    if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes < 8:
        raise ValueError("budget_fes must be at least eight")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    anchor_vector = np.asarray(anchor, dtype=float)
    if anchor_vector.shape != (problem.dimension,) or not np.all(np.isfinite(anchor_vector)):
        raise ValueError("anchor must match the problem dimension and be finite")
    if budget_fes > global_ledger.remaining:
        raise ValueError("region proposal budget exceeds the global ledger remainder")
    dimensions = structure.region_variables(leaf_id)
    local_ledger = _MirroredLedger(
        problem,
        budget_fes,
        dimensions=dimensions,
        initial_incumbent=tuple(float(value) for value in anchor_vector),
        initial_error=float(anchor_error),
        global_ledger=global_ledger,
    )
    session = ResumableOptimizerSession(
        algorithm,
        problem=problem,
        ledger=local_ledger,
        initial_mean=tuple(float(anchor_vector[index]) for index in dimensions),
        sigma=sigma,
        seed=seed,
        budget_fes=budget_fes,
        population_size=population_size,
        dimensions=dimensions,
        anchor=anchor_vector,
    )
    start = global_ledger.count
    session.step(budget_fes)
    consumed = global_ledger.count - start
    if consumed != budget_fes:
        raise RuntimeError("region proposal FE accounting drifted")
    best = local_ledger.best_x
    values = tuple(
        (int(variable), float(best[variable])) for variable in dimensions
    )
    sigma_by_variable = {
        int(variable): float(value) for variable, value in zip(
            dimensions,
            np.asarray(session.sigma_vector) if hasattr(session, "sigma_vector") else np.full(len(dimensions), sigma),
            strict=True,
        )
    }
    return RegionProposal(
        leaf_id=int(leaf_id),
        values=values,
        uncertainty=tuple(sorted(sigma_by_variable.items())),
        improvement=float(anchor_error - local_ledger.best_error),
        consumed_fes=consumed,
    )


def region_conflict_probe(
    candidates: tuple[VariableRegionInteraction, ...],
    *,
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    step: float = 0.25,
) -> tuple[RegionProbeResult, ...]:
    """Two-sided counted probe for each candidate variable from the incumbent.

    Consumes exactly 2 FE per candidate; f(x0) reuses the incumbent's recorded
    error.  Formulas mirror ``arac.coordination.counted_probe``.
    """

    if problem is not ledger.problem:
        raise ValueError("probe requires the ledger for the same problem")
    if not math.isfinite(float(step)) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    seen: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, VariableRegionInteraction):
            raise TypeError("candidates must be VariableRegionInteraction instances")
        if candidate.variable in seen:
            raise ValueError("candidate variables must be unique")
        seen.add(candidate.variable)
    if 2 * len(candidates) > ledger.remaining:
        raise ValueError("region conflict probe exceeds the remaining FE budget")

    incumbent = ledger.best_x
    f0 = float(ledger.best_error)
    lower = problem.lower_array
    upper = problem.upper_array
    start = ledger.count
    batch = np.repeat(incumbent[np.newaxis, :], 2 * len(candidates), axis=0)
    for index, candidate in enumerate(candidates):
        batch[2 * index, candidate.variable] += step
        batch[2 * index + 1, candidate.variable] -= step
    np.clip(batch, lower, upper, out=batch)
    errors = np.asarray(ledger.evaluate(batch), dtype=float)
    if ledger.count - start != 2 * len(candidates):
        raise RuntimeError("region conflict probe FE accounting drifted")

    results = []
    for index, candidate in enumerate(candidates):
        f_plus = float(errors[2 * index])
        f_minus = float(errors[2 * index + 1])
        plus_delta = f_plus - f0
        minus_delta = f_minus - f0
        denominator = abs(plus_delta) + abs(minus_delta)
        bias = (minus_delta - plus_delta) / (denominator + 1e-12) if denominator > 0 else 0.0
        bias = float(np.clip(bias, -1.0, 1.0))
        width = max(abs(plus_delta), abs(minus_delta))
        scale = abs(f0) + width + 1e-12
        conflict = abs(bias) * min(1.0, width / scale)
        results.append(
            RegionProbeResult(
                variable=candidate.variable,
                source_region=candidate.source_region,
                target_region=candidate.target_region,
                step=float(step),
                f_plus=f_plus,
                f_minus=f_minus,
                bias=bias,
                width=float(width),
                conflict_score=float(conflict),
                consumed_fes=2,
            )
        )
    return tuple(results)


def region_patch(
    probes: tuple[RegionProbeResult, ...],
    proposal: RegionProposal,
    *,
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    budget_fes: int,
) -> tuple[int, tuple[int, ...]]:
    """Greedy strict-best patch of the highest-conflict candidates.

    Each candidate move evaluates the proposal value and its reflection
    (2 FE), letting the ledger's strict-best archive decide acceptance.
    """

    if problem is not ledger.problem:
        raise ValueError("patch requires the ledger for the same problem")
    if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
        raise ValueError("budget_fes must be a positive integer")
    if 2 > ledger.remaining:
        raise ValueError("region patch exceeds the remaining FE budget")
    proposal_values = {variable: value for variable, value in proposal.values}
    order = sorted(probes, key=lambda item: (-item.conflict_score, item.variable))
    lower = problem.lower_array
    upper = problem.upper_array
    consumed = 0
    patched: list[int] = []
    for probe in order:
        if consumed + 2 > budget_fes or 2 > ledger.remaining:
            break
        if probe.variable not in proposal_values:
            continue
        target = proposal_values[probe.variable]
        base = ledger.best_x
        plus = base.copy()
        minus = base.copy()
        plus[probe.variable] = target
        minus[probe.variable] = base[probe.variable] - (target - base[probe.variable])
        batch = np.clip(np.asarray((plus, minus), dtype=float), lower, upper)
        before = float(ledger.best_error)
        ledger.evaluate(batch)
        consumed += 2
        if float(ledger.best_error) < before:
            patched.append(probe.variable)
    return consumed, tuple(patched)


class RegionCoordinator:
    """Minimal HIERARCHICAL-mode coordinator over region-level evidence."""

    def __init__(self, structure: RegionStructure, ledger: EvaluationLedger) -> None:
        if not isinstance(structure, RegionStructure):
            raise TypeError("structure must be RegionStructure")
        if not isinstance(ledger, EvaluationLedger):
            raise TypeError("ledger must be EvaluationLedger")
        if structure.evidence.dimension != ledger.problem.dimension:
            raise ValueError("region structure and ledger dimensions disagree")
        self.structure = structure
        self.ledger = ledger

    def run_cycle(
        self,
        *,
        proposal_budget_fes: int,
        patch_budget_fes: int,
        probe_step: float = 0.25,
        seed: int = 0,
    ) -> RegionCycleReceipt:
        evidence: Phase1Evidence = self.structure.evidence
        relations = evidence.region_relations
        if not relations:
            raise ValueError("region coordination requires at least one relation")
        component = self.structure.components()
        target = next(
            item for item in component if any(
                relation.left in item and relation.right in item for relation in relations
            )
        )
        members = tuple(
            relation for relation in relations
            if relation.left in target and relation.right in target
        )
        degree: dict[int, int] = {leaf: 0 for leaf in target}
        for relation in members:
            degree[relation.left] += 1
            degree[relation.right] += 1
        proposal_leaf = max(target, key=lambda leaf: (degree[leaf], -leaf))
        candidates = tuple(
            interaction for interaction in evidence.variable_region_interactions
            if interaction.source_region in target and interaction.target_region in target
        )
        best_before = float(self.ledger.best_error)
        start = self.ledger.count
        proposal = produce_region_proposal(
            self.structure,
            proposal_leaf,
            problem=self.ledger.problem,
            global_ledger=self.ledger,
            anchor=self.ledger.best_x,
            anchor_error=best_before,
            budget_fes=proposal_budget_fes,
            seed=seed,
        )
        probes = region_conflict_probe(
            candidates,
            problem=self.ledger.problem,
            ledger=self.ledger,
            step=probe_step,
        )
        patch_fes, patched = region_patch(
            probes,
            proposal,
            problem=self.ledger.problem,
            ledger=self.ledger,
            budget_fes=patch_budget_fes,
        )
        best_after = float(self.ledger.best_error)
        receipt = RegionCycleReceipt(
            component=target,
            proposal_leaf=proposal_leaf,
            proposal_fes=proposal.consumed_fes,
            probe_fes=sum(probe.consumed_fes for probe in probes),
            patch_fes=patch_fes,
            patched_candidates=patched,
            accepted=best_after < best_before,
            best_error_before=best_before,
            best_error_after=best_after,
        )
        if self.ledger.count - start != receipt.proposal_fes + receipt.probe_fes + receipt.patch_fes:
            raise RuntimeError("region cycle FE accounting drifted")
        return receipt


__all__ = [
    "RegionCoordinator",
    "RegionCycleReceipt",
    "RegionProbeResult",
    "RegionProposal",
    "produce_region_proposal",
    "region_conflict_probe",
    "region_patch",
]
