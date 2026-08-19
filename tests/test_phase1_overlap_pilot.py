from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence import (
    INFERENCE_INCOMPLETE,
    PHASE1_OVERLAP_PILOT_PROTOCOL,
    run_phase1_overlap_pilot,
)


GROUPS = ((0, 1, 2), (2, 3, 4))


def _problem() -> OptimizationProblem:
    def objective(values: np.ndarray) -> float | np.ndarray:
        converted = np.asarray(values, dtype=float)
        batch = converted[np.newaxis, :] if converted.ndim == 1 else converted
        result = np.sum(batch**2, axis=1)
        for group in GROUPS:
            local = batch[:, group]
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    result += local[:, left] * local[:, right]
        return float(result[0]) if converted.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )


def test_overlap_pilot_carries_complete_evidence_at_phase_boundary() -> None:
    result = run_phase1_overlap_pilot(
        _problem(),
        total_budget_fes=1_000,
        run_seed=17,
        anchors=tuple(tuple(row) for row in np.zeros((2, 40))),
        rounds=4,
        bucket_size=8,
    )

    assert result.checkpoint.protocol == PHASE1_OVERLAP_PILOT_PROTOCOL
    assert result.consumed_fes == result.target_phase1_fes == 240
    assert result.checkpoint.phase1_fes == 240
    assert result.evidence.groups == GROUPS + tuple((variable,) for variable in range(5, 40))
    assert result.adaptation.ready
    assert result.adaptation.structure is not None


def test_overlap_pilot_is_deterministic() -> None:
    kwargs = dict(
        total_budget_fes=1_000,
        run_seed=17,
        anchors=tuple(tuple(row) for row in np.zeros((2, 40))),
        rounds=4,
        bucket_size=8,
    )
    first = run_phase1_overlap_pilot(_problem(), **kwargs)
    replay = run_phase1_overlap_pilot(_problem(), **kwargs)

    assert first.checkpoint.checkpoint_hash == replay.checkpoint.checkpoint_hash
    assert first.evidence == replay.evidence
    assert first.discovery == replay.discovery


def test_overlap_pilot_fail_closed_when_candidate_cap_is_too_small() -> None:
    result = run_phase1_overlap_pilot(
        _problem(),
        total_budget_fes=1_000,
        run_seed=17,
        anchors=tuple(tuple(row) for row in np.zeros((2, 40))),
        rounds=4,
        bucket_size=8,
        max_candidate_pairs=1,
    )

    assert not result.discovery.complete
    assert result.adaptation.status == INFERENCE_INCOMPLETE
    assert result.adaptation.structure is None
