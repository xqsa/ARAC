from __future__ import annotations

import numpy as np
import pytest

from arac.analysis.mechanism_policy import (
    run_mechanism_baseline,
    select_mechanism_action,
)
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ACTION_NAMES, ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger


def _checkpoint(
    complete: float,
    relations: tuple[RelationEvidence, ...] = (),
) -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="mechanism-policy-test-v1",
        run_seed=43,
        total_budget_fes=34,
        phase1_fes=4,
        incumbent=(1.0,) * 8,
        incumbent_error=8.0,
        feature_names=(
            "line_high_frequency_fraction_median",
            "structural_inference_complete",
        ),
        feature_values=(0.4, complete),
        blocks=((0, 1), (2, 3), (4, 5), (6, 7)),
        relations=relations,
    )


@pytest.mark.parametrize(
    ("checkpoint", "expected_action", "expected_reason"),
    (
        (
            _checkpoint(0.0, (RelationEvidence(0, 1, 0.4, 0.2),)),
            "aor",
            "incomplete_structure",
        ),
        (_checkpoint(1.0), "smp", "complete_zero_relation_blocks"),
        (
            _checkpoint(1.0, (RelationEvidence(0, 1, 0.4, 0.2),)),
            "ctp",
            "complete_disconnected_relation_cover",
        ),
        (
            _checkpoint(
                1.0,
                (
                    RelationEvidence(0, 1, 0.4, 0.2),
                    RelationEvidence(1, 2, 0.4, 0.2),
                    RelationEvidence(2, 3, 0.4, 0.2),
                ),
            ),
            "gcb",
            "complete_connected_relation_graph",
        ),
    ),
)
def test_mechanism_policy_covers_each_action_without_fitted_thresholds(
    checkpoint: PhaseCheckpoint,
    expected_action: str,
    expected_reason: str,
) -> None:
    decision = select_mechanism_action(checkpoint)

    assert decision.action_name == expected_action
    assert decision.reason == expected_reason
    assert tuple(action for action, _ in decision.scores) == ACTION_NAMES
    assert sum(score for _, score in decision.scores) == 1.0


def test_mechanism_policy_rejects_missing_or_nonbinary_completeness() -> None:
    missing = PhaseCheckpoint(
        **{
            **_checkpoint(1.0).__dict__,
            "feature_names": ("line_high_frequency_fraction_median",),
            "feature_values": (0.4,),
        }
    )
    with pytest.raises(ValueError, match="lacks"):
        select_mechanism_action(missing)
    with pytest.raises(ValueError, match="binary"):
        select_mechanism_action(_checkpoint(0.5))


def test_mechanism_baseline_selects_once_and_consumes_exact_terminal_budget() -> None:
    problem = OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=8,
        lower_bounds=(-5.0,) * 8,
        upper_bounds=(5.0,) * 8,
    )
    checkpoint = _checkpoint(0.0)
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )

    result = run_mechanism_baseline(
        ActionContext("aor", checkpoint, problem, ledger, action_seed=47)
    )

    assert result.decision.action_name == result.action_result.action_name == "aor"
    assert result.action_result.terminal_fes == 34
    assert result.numerical_repair_count >= 0
    assert ledger.count == 34
