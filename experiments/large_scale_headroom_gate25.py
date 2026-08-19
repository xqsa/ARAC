"""Gate 25: large-scale post-Phase-I headroom for evidence-clique repair."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination import OverlapCoordinator, OverlapStructure, produce_local_proposal
from arac.evidence import Phase1OverlapAdapter, run_phase1_overlap_pilot
from arac.runtime.ledger import EvaluationLedger


DIMENSION = 1000
ACTIVE_GROUPS = ((0, 1, 2, 3, 4, 5), (5, 6, 7, 8, 9, 10), (10, 11, 12, 13, 14, 15), (15, 16, 17, 18, 19, 20))
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
SEED = 20260825
BOUNDS = 5.0
INTERACTION_STRENGTH = 0.25
PROPOSAL_BUDGET_FES = 64
PROPOSAL_POPULATION_SIZE = 8
WRITEBACK_ROUNDS = 16
CONTINUATION_FES = 32


def _problem() -> tuple[OptimizationProblem, OverlapStructure]:
    structure = OverlapStructure(DIMENSION, ACTIVE_GROUPS + tuple((index,) for index in range(21, DIMENSION)))
    rng = np.random.default_rng(SEED ^ 0xA0B0)
    optima = rng.uniform(-3.5, 3.5, size=(len(ACTIVE_GROUPS), max(map(len, ACTIVE_GROUPS))))
    weights = 1.0 + 2.0 * rng.random(len(ACTIVE_GROUPS))

    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        batch = converted[np.newaxis, :] if converted.ndim == 1 else converted
        result = np.zeros(len(batch), dtype=float)
        result += np.sum((batch[:, 21:] / 5.0) ** 2, axis=1)
        for group_index, group in enumerate(ACTIVE_GROUPS):
            local = batch[:, np.asarray(group, dtype=int)] - optima[group_index, : len(group)]
            result += weights[group_index] * (
                10.0 * len(group) + np.sum(local**2 - 10.0 * np.cos(2.0 * np.pi * local), axis=1)
            )
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    result += INTERACTION_STRENGTH * local[:, left] ** 2 * local[:, right] ** 2
        return float(result[0]) if converted.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-BOUNDS,) * DIMENSION,
        upper_bounds=(BOUNDS,) * DIMENSION,
        optimum=0.0,
    )
    return problem, structure


@dataclass(frozen=True)
class ArmResult:
    arm: str
    selected_component: tuple[int, ...]
    final_error: float
    checkpoint_error: float
    proposal_fes: int
    arbitration_fes: int
    continuation_fes: int
    consumed_fes: int
    strict_best: bool
    trace_rounds: int
    accepted_rounds: tuple[int, ...] = ()


def _pilot(problem):
    return run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        anchor_count=5,
        step=0.25,
        rounds=12,
        bucket_size=16,
        max_candidate_pairs=128,
    )


def _proposals(problem, structure, pilot):
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        phase1_fes=pilot.checkpoint.phase1_fes,
        incumbent=pilot.checkpoint.incumbent,
        incumbent_error=pilot.checkpoint.incumbent_error,
    )
    anchor = tuple(float(value) for value in pilot.checkpoint.incumbent)
    anchor_error = float(pilot.checkpoint.incumbent_error)
    components = tuple(component for component in structure.connected_components() if len(component) > 1)
    groups = tuple(group for component in components for group in component)
    runs = []
    for group in groups:
        runs.append(
            produce_local_proposal(
                structure,
                group,
                problem=problem,
                global_ledger=ledger,
                anchor=anchor,
                anchor_error=anchor_error,
                budget_fes=PROPOSAL_BUDGET_FES,
                seed=SEED ^ (0x9E37 * (group + 1)),
                algorithm="sepcmaes",
                population_size=PROPOSAL_POPULATION_SIZE,
                sigma=0.5,
            )
        )
    return ledger, tuple(run.proposal for run in runs), components


def _arm(problem, structure, pilot, proposal_ledger, proposals, component, *, arm: str) -> ArmResult:
    ledger = EvaluationLedger(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        initial_count=proposal_ledger.count,
        initial_incumbent=tuple(float(value) for value in proposal_ledger.best_x),
        initial_error=float(proposal_ledger.best_error),
    )
    coordinator = OverlapCoordinator(structure, ledger, medium_threshold=0.0, high_threshold=0.0)
    selected = tuple(proposal for proposal in proposals if proposal.group in component)
    arbitration = coordinator.coordinate(component, selected, ctp_budget_fes=0)
    arbitration_fes = len(arbitration.candidates)
    if arm == "full_context":
        result = coordinator.full_context_writeback(component, selected, rounds=WRITEBACK_ROUNDS)
        continuation = result.consumed_fes
        accepted = tuple(item.round_index for item in result.rounds if item.accepted)
        trace_rounds = len(result.rounds)
    elif arm == "current_ctp":
        continuation = coordinator._repair_shared_core(component, selected, budget_fes=CONTINUATION_FES, seed=SEED ^ 0x51ED, base=ledger.best_x)
        accepted = ()
        trace_rounds = 0
    elif arm == "owner_full":
        rng = np.random.default_rng(SEED ^ 0xA0B0)
        candidates = np.repeat(ledger.best_x[np.newaxis, :], CONTINUATION_FES, axis=0)
        for index in range(CONTINUATION_FES):
            proposal = selected[index % len(selected)]
            for variable, value in proposal.values:
                candidates[index, variable] = value + float(rng.normal(0.0, max(np.finfo(float).eps, proposal.sigma(variable))))
        np.clip(candidates, problem.lower_array, problem.upper_array, out=candidates)
        ledger.evaluate(candidates)
        continuation = CONTINUATION_FES
        accepted = ()
        trace_rounds = 0
    else:
        raise ValueError(f"unknown arm {arm}")
    return ArmResult(
        arm=arm,
        selected_component=tuple(component),
        final_error=float(ledger.best_error),
        checkpoint_error=float(proposal_ledger.best_error),
        proposal_fes=proposal_ledger.count - pilot.checkpoint.phase1_fes,
        arbitration_fes=arbitration_fes,
        continuation_fes=continuation,
        consumed_fes=(proposal_ledger.count - pilot.checkpoint.phase1_fes) + arbitration_fes + continuation,
        strict_best=ledger.best_error <= proposal_ledger.best_error,
        trace_rounds=trace_rounds,
        accepted_rounds=accepted,
    )


def run_gate() -> dict[str, object]:
    problem, truth_structure = _problem()
    pilot = _pilot(problem)
    adaptation = Phase1OverlapAdapter().adapt(pilot.checkpoint, pilot.evidence)
    if not adaptation.ready or adaptation.structure is None:
        raise RuntimeError("Gate 25 adapter is not ready")
    structure = adaptation.structure
    proposal_ledger, proposals, components = _proposals(problem, structure, pilot)
    if not components:
        raise RuntimeError("Gate 25 discovered no overlap component")
    selected = max(components, key=lambda component: sum(len(structure.groups[group]) for group in component))
    arms = tuple(_arm(problem, structure, pilot, proposal_ledger, proposals, selected, arm=arm) for arm in ("current_ctp", "full_context", "owner_full"))
    current, full, owner = arms
    truth_shared = set(truth_structure.shared_variables)
    inferred_shared = {variable for variable, owners in enumerate(pilot.evidence.memberships) if len(owners) > 1}
    checks = {
        "phase1_boundary": pilot.consumed_fes == PHASE1_FES,
        "discovery_complete": pilot.discovery.complete,
        "adapter_ready": adaptation.ready,
        "shared_recall": truth_shared.issubset(inferred_shared),
        "proposal_coverage": all(set(variable for variable, _ in proposal.values) == set(structure.groups[proposal.group]) for proposal in proposals),
        "proposal_fes_parity": current.proposal_fes == full.proposal_fes == owner.proposal_fes,
        "arbitration_parity": current.arbitration_fes == full.arbitration_fes == owner.arbitration_fes,
        "continuation_parity": current.continuation_fes == full.continuation_fes == owner.continuation_fes == CONTINUATION_FES,
        "full_trace": full.trace_rounds == WRITEBACK_ROUNDS,
        "strict_best": all(arm.strict_best for arm in arms),
        "full_no_worse_current": full.final_error <= current.final_error,
        "full_no_worse_owner": full.final_error <= owner.final_error,
    }
    return {
        "schema_version": "arac-large-scale-headroom-gate25-v1",
        "protocol": {"dimension": DIMENSION, "phase1_fes": PHASE1_FES, "total_budget_fes": TOTAL_BUDGET_FES, "proposal_budget_fes": PROPOSAL_BUDGET_FES, "continuation_fes": CONTINUATION_FES},
        "truth_shared_variables_audit_only": tuple(sorted(truth_shared)),
        "inferred_shared_variables": tuple(sorted(inferred_shared)),
        "discovered_components": components,
        "selected_component": selected,
        "arms": [asdict(arm) for arm in arms],
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/large_scale_headroom_gate25/confirmation_fresh.json"))
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "arms": payload["arms"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
