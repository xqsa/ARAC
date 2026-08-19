from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ACTION_NAMES, ActionContext, PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=8,
        lower_bounds=(-5.0,) * 8,
        upper_bounds=(5.0,) * 8,
    )


def _checkpoint(action: str) -> PhaseCheckpoint:
    complete = float(action != "aor")
    return PhaseCheckpoint(
        protocol="recovered-registry-test-v1",
        run_seed=117,
        total_budget_fes=100,
        phase1_fes=4,
        incumbent=(1.0,) * 8,
        incumbent_error=8.0,
        feature_names=(
            "log10_center_error",
            "line_high_frequency_fraction_median",
            "structural_inference_complete",
        ),
        feature_values=(1.0, 0.4, complete),
        blocks=((0, 1), (2, 3), (4, 5), (6, 7)),
    )


def _context(action: str, *, allow_out_of_bounds: bool) -> ActionContext:
    problem = _problem()
    checkpoint = _checkpoint(action)
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=allow_out_of_bounds,
    )
    return ActionContext(action, checkpoint, problem, ledger, action_seed=117)


def test_default_and_recovered_registries_keep_distinct_profiles() -> None:
    assert ActionRegistry().allow_out_of_bounds is False
    assert RecoveredActionRegistry().allow_out_of_bounds is True
    assert RecoveredActionRegistry().action_names == ACTION_NAMES


@pytest.mark.parametrize("action", ACTION_NAMES)
def test_recovered_registry_executes_each_action_to_the_exact_terminal_fe(action: str) -> None:
    context = _context(action, allow_out_of_bounds=True)

    result = execute_phase2_action(
        action,
        context.checkpoint,
        context.problem,
        context.ledger,
        action_seed=context.action_seed,
        registry=RecoveredActionRegistry(),
    )

    assert result.action_name == action
    assert result.checkpoint_hash == context.checkpoint.checkpoint_hash
    assert result.action_seed == context.action_seed
    assert result.consumed_fes == context.checkpoint.remaining_fes
    assert result.terminal_fes == context.checkpoint.total_budget_fes
    assert result.final_error <= context.checkpoint.incumbent_error
    assert context.ledger.count == context.checkpoint.total_budget_fes


def test_recovered_registry_rejects_a_bounded_ledger() -> None:
    context = _context("smp", allow_out_of_bounds=False)

    with pytest.raises(ValueError, match="execution profile"):
        execute_phase2_action(
            "smp",
            context.checkpoint,
            context.problem,
            context.ledger,
            action_seed=context.action_seed,
            registry=RecoveredActionRegistry(),
        )


def test_phase2_handoff_rejects_a_different_problem_instance() -> None:
    context = _context("aor", allow_out_of_bounds=True)

    with pytest.raises(ValueError, match="problem and ledger problem"):
        execute_phase2_action(
            "aor",
            context.checkpoint,
            _problem(),
            context.ledger,
            action_seed=context.action_seed,
            registry=RecoveredActionRegistry(),
        )


def test_recovered_production_sources_are_identity_blind() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "src" / "arac" / "actions" / "recovered.py",
        root / "src" / "arac" / "actions" / "recovered_registry.py",
    )
    forbidden = (
        "case_id",
        "family_id",
        "elliptic",
        "AOB_DATA_ROOT",
        "vendor.hcc",
        "experiments.",
        "artifacts/",
        "results/",
    )

    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden)
