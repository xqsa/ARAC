"""Gate 26: isolate proposal neighborhoods from online context feedback."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from arac.coordination import LocalProposal, OverlapCoordinator, OverlapStructure
from arac.evidence import Phase1OverlapAdapter
from arac.runtime.ledger import EvaluationLedger
from experiments.large_scale_headroom_gate25 import (
    CONTINUATION_FES,
    PHASE1_FES,
    SEED,
    TOTAL_BUDGET_FES,
    WRITEBACK_ROUNDS,
    _pilot,
    _problem,
    _proposals,
)


STREAM_SEED = SEED ^ 0xA0B0


@dataclass(frozen=True)
class LocalCandidate:
    sample_index: int
    group: int
    values: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ArmResult:
    arm: str
    start_error: float
    final_error: float
    arbitration_fes: int
    continuation_fes: int
    accepted_samples: tuple[int, ...]
    strict_best: bool


def build_candidate_stream(
    structure: OverlapStructure,
    component: tuple[int, ...],
    proposals: tuple[LocalProposal, ...],
    *,
    count: int = CONTINUATION_FES,
    seed: int = STREAM_SEED,
) -> tuple[LocalCandidate, ...]:
    """Freeze proposal-conditioned local values independently of global context."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    selected = {proposal.group: proposal for proposal in proposals if proposal.group in component}
    if set(selected) != set(component):
        raise ValueError("proposals must cover exactly the selected component")
    owner_order = tuple(sorted(component, key=lambda group: (-selected[group].improvement, group)))
    rng = np.random.default_rng(seed)
    lower = np.asarray(structure.dimension * (-5.0,), dtype=float)
    upper = np.asarray(structure.dimension * (5.0,), dtype=float)
    stream = []
    for sample_index in range(count):
        group = owner_order[sample_index % len(owner_order)]
        proposal = selected[group]
        values = []
        for variable in structure.groups[group]:
            sampled = proposal.value(variable) + float(rng.normal(0.0, proposal.sigma(variable)))
            values.append((variable, float(np.clip(sampled, lower[variable], upper[variable]))))
        stream.append(LocalCandidate(sample_index, group, tuple(values)))
    return tuple(stream)


def _new_arm_ledger(problem, proposal_ledger: EvaluationLedger) -> EvaluationLedger:
    return EvaluationLedger(
        problem,
        total_budget=TOTAL_BUDGET_FES,
        initial_count=proposal_ledger.count,
        initial_incumbent=tuple(float(value) for value in proposal_ledger.best_x),
        initial_error=float(proposal_ledger.best_error),
    )


def _arbitrate(
    problem,
    structure: OverlapStructure,
    proposal_ledger: EvaluationLedger,
    component: tuple[int, ...],
    proposals: tuple[LocalProposal, ...],
) -> tuple[EvaluationLedger, OverlapCoordinator, int]:
    ledger = _new_arm_ledger(problem, proposal_ledger)
    coordinator = OverlapCoordinator(structure, ledger, medium_threshold=0.0, high_threshold=0.0)
    before = ledger.count
    result = coordinator.coordinate(component, proposals, ctp_budget_fes=0)
    consumed = ledger.count - before
    if consumed != len(result.candidates):
        raise RuntimeError("arbitration FE accounting drifted")
    return ledger, coordinator, consumed


def _apply_local(base: np.ndarray, candidate: LocalCandidate) -> np.ndarray:
    vector = np.asarray(base, dtype=float).copy()
    for variable, value in candidate.values:
        vector[variable] = value
    return vector


def _run_neighborhood_arm(
    problem,
    structure: OverlapStructure,
    proposal_ledger: EvaluationLedger,
    component: tuple[int, ...],
    proposals: tuple[LocalProposal, ...],
    stream: tuple[LocalCandidate, ...],
    *,
    sequential: bool,
) -> ArmResult:
    ledger, _, arbitration_fes = _arbitrate(
        problem, structure, proposal_ledger, component, proposals
    )
    start_error = float(ledger.best_error)
    start_fes = ledger.count
    accepted = []
    if sequential:
        for candidate in stream:
            before = float(ledger.best_error)
            ledger.evaluate(_apply_local(ledger.best_x, candidate))
            if ledger.best_error < before:
                accepted.append(candidate.sample_index)
    else:
        base = ledger.best_x
        batch = np.asarray([_apply_local(base, candidate) for candidate in stream], dtype=float)
        errors = np.asarray(ledger.evaluate(batch), dtype=float)
        accepted = [int(np.argmin(errors))] if float(np.min(errors)) < start_error else []
    consumed = ledger.count - start_fes
    if consumed != len(stream):
        raise RuntimeError("neighborhood continuation FE accounting drifted")
    return ArmResult(
        arm="sequential_neighborhood" if sequential else "static_neighborhood",
        start_error=start_error,
        final_error=float(ledger.best_error),
        arbitration_fes=arbitration_fes,
        continuation_fes=consumed,
        accepted_samples=tuple(accepted),
        strict_best=ledger.best_error <= start_error,
    )


def _run_deterministic_arm(
    problem,
    structure: OverlapStructure,
    proposal_ledger: EvaluationLedger,
    component: tuple[int, ...],
    proposals: tuple[LocalProposal, ...],
) -> ArmResult:
    ledger, coordinator, arbitration_fes = _arbitrate(
        problem, structure, proposal_ledger, component, proposals
    )
    start_error = float(ledger.best_error)
    result = coordinator.full_context_writeback(component, proposals, rounds=WRITEBACK_ROUNDS)
    return ArmResult(
        arm="deterministic_full_context",
        start_error=start_error,
        final_error=float(ledger.best_error),
        arbitration_fes=arbitration_fes,
        continuation_fes=result.consumed_fes,
        accepted_samples=tuple(item.round_index for item in result.rounds if item.accepted),
        strict_best=ledger.best_error <= start_error,
    )


def run_gate() -> dict[str, object]:
    problem, _ = _problem()
    pilot = _pilot(problem)
    adaptation = Phase1OverlapAdapter().adapt(pilot.checkpoint, pilot.evidence)
    if not adaptation.ready or adaptation.structure is None:
        raise RuntimeError("Gate 26 adapter is not ready")
    structure = adaptation.structure
    proposal_ledger, all_proposals, components = _proposals(problem, structure, pilot)
    if not components:
        raise RuntimeError("Gate 26 discovered no overlap component")
    component = max(
        components,
        key=lambda item: sum(len(structure.groups[group]) for group in item),
    )
    proposals = tuple(proposal for proposal in all_proposals if proposal.group in component)
    stream = build_candidate_stream(structure, component, proposals)
    deterministic = _run_deterministic_arm(
        problem, structure, proposal_ledger, component, proposals
    )
    static = _run_neighborhood_arm(
        problem, structure, proposal_ledger, component, proposals, stream, sequential=False
    )
    sequential = _run_neighborhood_arm(
        problem, structure, proposal_ledger, component, proposals, stream, sequential=True
    )
    arms = (deterministic, static, sequential)
    checks = {
        "phase1_boundary": pilot.consumed_fes == PHASE1_FES,
        "adapter_ready": adaptation.ready,
        "candidate_count": len(stream) == CONTINUATION_FES,
        "candidate_owner_balance": max(
            sum(candidate.group == group for candidate in stream) for group in component
        ) - min(sum(candidate.group == group for candidate in stream) for group in component) <= 1,
        "arbitration_parity": len({arm.arbitration_fes for arm in arms}) == 1,
        "continuation_parity": all(arm.continuation_fes == CONTINUATION_FES for arm in arms),
        "strict_best": all(arm.strict_best for arm in arms),
        "sequential_no_worse_static": sequential.final_error <= static.final_error,
        "sequential_no_worse_deterministic": sequential.final_error <= deterministic.final_error,
    }
    return {
        "schema_version": "arac-proposal-conditioned-context-gate26-v1",
        "protocol": {
            "phase1_fes": PHASE1_FES,
            "continuation_fes": CONTINUATION_FES,
            "stream_seed": STREAM_SEED,
            "same_local_samples_static_sequential": True,
        },
        "selected_component": component,
        "stream": [asdict(candidate) for candidate in stream],
        "arms": [asdict(arm) for arm in arms],
        "gains": {
            "sequential_vs_static": static.final_error - sequential.final_error,
            "sequential_vs_deterministic": deterministic.final_error - sequential.final_error,
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/proposal_conditioned_context_gate26/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_passed": payload["gate_passed"],
                "gate_checks": payload["gate_checks"],
                "arms": payload["arms"],
                "gains": payload["gains"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
