from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from arac.policy.trajectory_guard import (
    make_recovery_checkpoint,
    preempt_recovery_checkpoint,
    resolve_recovery_checkpoint,
)


def test_strict_downstream_improvement_commits() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0, 2.0]), 10.0)

    resolved = resolve_recovery_checkpoint(
        checkpoint,
        downstream_candidate=np.array([3.0, 4.0]),
        downstream_fitness=8.0,
    )

    assert resolved.status == "committed"
    assert resolved.restored is False
    assert resolved.fitness == 8.0
    assert resolved.effective_delta == pytest.approx(2.0)
    assert resolved.recovery_credit == pytest.approx(0.2)
    np.testing.assert_allclose(resolved.candidate, np.array([3.0, 4.0]))


@pytest.mark.parametrize("downstream_fitness", [10.0, 12.0])
def test_non_improving_downstream_candidate_restores(
    downstream_fitness: float,
) -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0, 2.0]), 10.0)

    resolved = resolve_recovery_checkpoint(
        checkpoint,
        downstream_candidate=np.array([3.0, 4.0]),
        downstream_fitness=downstream_fitness,
    )

    assert resolved.status == "restored"
    assert resolved.restored is True
    assert resolved.fitness == 10.0
    assert resolved.effective_delta == 0.0
    np.testing.assert_allclose(resolved.candidate, np.array([1.0, 2.0]))


def test_preemption_restores_without_claiming_recovery() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0]), 10.0)

    resolved = preempt_recovery_checkpoint(checkpoint)

    assert resolved.status == "preempted_restored"
    assert resolved.restored is True
    assert resolved.recovery_credit is None
    np.testing.assert_allclose(resolved.candidate, np.array([1.0]))


def test_checkpoint_isolated_from_source_mutation() -> None:
    source = np.array([1.0, 2.0])
    checkpoint = make_recovery_checkpoint(source, 10.0)

    source[:] = -1.0

    np.testing.assert_allclose(checkpoint.candidate, np.array([1.0, 2.0]))
    assert checkpoint.candidate.flags.writeable is False


def test_resolution_isolated_from_downstream_candidate_mutation() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0, 2.0]), 10.0)
    downstream = np.array([3.0, 4.0])

    resolved = resolve_recovery_checkpoint(
        checkpoint,
        downstream_candidate=downstream,
        downstream_fitness=8.0,
    )
    downstream[:] = -1.0

    np.testing.assert_allclose(resolved.candidate, np.array([3.0, 4.0]))
    assert resolved.candidate.flags.writeable is False


def test_recovery_policy_rejects_shape_mismatch() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0]), 10.0)

    with pytest.raises(ValueError, match="shape"):
        resolve_recovery_checkpoint(
            checkpoint,
            downstream_candidate=np.array([1.0, 2.0]),
            downstream_fitness=9.0,
        )


@pytest.mark.parametrize(
    ("candidate", "fitness"),
    [
        (np.array([np.nan]), 10.0),
        (np.array([1.0]), np.inf),
    ],
)
def test_checkpoint_rejects_non_finite_values(
    candidate: np.ndarray,
    fitness: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_recovery_checkpoint(candidate, fitness)


def test_resolution_rejects_non_finite_values() -> None:
    checkpoint = make_recovery_checkpoint(np.array([1.0]), 10.0)

    with pytest.raises(ValueError, match="finite"):
        resolve_recovery_checkpoint(
            checkpoint,
            downstream_candidate=np.array([1.0]),
            downstream_fitness=np.nan,
        )


def test_recovery_dataclasses_exclude_forbidden_runtime_fields() -> None:
    from arac.policy.trajectory_guard import RecoveryCheckpoint, RecoveryResolution

    forbidden = {
        "case_id",
        "problem_id",
        "function_family",
        "paper_best",
        "historical_outcome",
        "final_error",
        "relative_gain",
    }
    for kind in (RecoveryCheckpoint, RecoveryResolution):
        assert not forbidden.intersection(field.name for field in fields(kind))
