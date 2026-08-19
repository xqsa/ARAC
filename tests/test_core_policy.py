from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.core import run_arac, run_arac_core, select_core_action
from arac.runtime.contracts import PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


def _problem(dimension: int = 8) -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda values: np.sum(np.asarray(values, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )


def _checkpoint() -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol="arac-core-test-v1",
        run_seed=43,
        total_budget_fes=100,
        phase1_fes=4,
        incumbent=(1.0,) * 8,
        incumbent_error=8.0,
        feature_names=(
            "log10_center_error",
            "line_high_frequency_fraction_median",
            "structural_inference_complete",
        ),
        feature_values=(1.0, 0.4, 0.0),
        blocks=((0, 1), (2, 3), (4, 5), (6, 7)),
    )


def _ledger(problem: OptimizationProblem, checkpoint: PhaseCheckpoint) -> EvaluationLedger:
    return EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )


def test_arac_core_selects_once_and_executes_one_terminal_action(monkeypatch) -> None:
    problem = _problem()
    checkpoint = _checkpoint()
    ledger = _ledger(problem, checkpoint)
    executed_actions: list[str] = []
    original_execute = ActionRegistry.execute

    def record_execute(self, context):
        executed_actions.append(context.action_name)
        return original_execute(self, context)

    monkeypatch.setattr(ActionRegistry, "execute", record_execute)

    result = run_arac_core(
        checkpoint,
        problem,
        ledger,
        action_seed=47,
    )

    assert executed_actions == ["aor"]
    assert result.decision.action_name == result.action_result.action_name == "aor"
    assert result.decision.reason == "incomplete_structure"
    assert result.action_result.checkpoint_hash == checkpoint.checkpoint_hash
    assert result.action_result.consumed_fes == checkpoint.remaining_fes
    assert result.action_result.terminal_fes == checkpoint.total_budget_fes
    assert ledger.count == checkpoint.total_budget_fes


def test_core_selection_ignores_nonstructural_checkpoint_metadata() -> None:
    checkpoint = replace(_checkpoint(), feature_values=(1.0, 0.4, 1.0))
    changed = replace(
        checkpoint,
        protocol="different-evidence-protocol-name",
        run_seed=999,
        incumbent=(2.0,) * 8,
        incumbent_error=32.0,
        feature_values=(100.0, 0.9, 1.0),
    )

    first = select_core_action(checkpoint)
    second = select_core_action(changed)

    assert checkpoint.checkpoint_hash != changed.checkpoint_hash
    assert first == second
    assert first.action_name == "smp"


def test_arac_core_rejects_a_ledger_past_phase1_before_action_execution(monkeypatch) -> None:
    problem = _problem()
    checkpoint = _checkpoint()
    ledger = _ledger(problem, checkpoint)
    ledger.evaluate(np.zeros(problem.dimension))
    executed_actions: list[str] = []

    def record_execute(self, context):
        executed_actions.append(context.action_name)
        raise AssertionError("action execution must not start")

    monkeypatch.setattr(ActionRegistry, "execute", record_execute)

    with pytest.raises(ValueError, match="Phase-I boundary"):
        run_arac_core(checkpoint, problem, ledger, action_seed=47)

    assert executed_actions == []


def test_arac_core_exposes_action_failure_without_trying_another_action(monkeypatch) -> None:
    problem = _problem()
    checkpoint = _checkpoint()
    ledger = _ledger(problem, checkpoint)
    executed_actions: list[str] = []

    def fail_execute(self, context):
        executed_actions.append(context.action_name)
        raise RuntimeError("visible action failure")

    monkeypatch.setattr(ActionRegistry, "execute", fail_execute)

    with pytest.raises(RuntimeError, match="visible action failure"):
        run_arac_core(checkpoint, problem, ledger, action_seed=47)

    assert executed_actions == ["aor"]
    assert ledger.count == checkpoint.phase1_fes


def test_run_arac_composes_phase1_with_one_selected_phase2_action(monkeypatch) -> None:
    executed_actions: list[str] = []
    original_execute = ActionRegistry.execute

    def record_execute(self, context):
        executed_actions.append(context.action_name)
        return original_execute(self, context)

    monkeypatch.setattr(ActionRegistry, "execute", record_execute)

    result = run_arac(
        _problem(40),
        total_budget_fes=500,
        run_seed=53,
        action_seed=59,
    )

    assert len(executed_actions) == 1
    assert executed_actions == [result.core.decision.action_name]
    assert result.phase1.checkpoint.checkpoint_hash == result.core.action_result.checkpoint_hash
    assert result.core.action_result.terminal_fes == 500


def test_arac_core_executes_an_injected_registry_once() -> None:
    problem = _problem()
    checkpoint = _checkpoint()
    ledger = _ledger(problem, checkpoint)

    class SpyRegistry:
        action_names = ActionRegistry().action_names
        allow_out_of_bounds = False

        def __init__(self) -> None:
            self.contexts = []

        def execute(self, context):
            self.contexts.append(context)
            return ActionRegistry().execute(context)

    registry = SpyRegistry()
    result = run_arac_core(
        checkpoint,
        problem,
        ledger,
        action_seed=47,
        registry=registry,
    )

    assert len(registry.contexts) == 1
    context = registry.contexts[0]
    assert context.action_name == result.decision.action_name
    assert context.checkpoint is checkpoint
    assert context.problem is problem
    assert context.ledger is ledger
    assert context.action_seed == 47


def test_arac_core_rejects_a_ledger_with_a_different_checkpoint_incumbent() -> None:
    problem = _problem()
    checkpoint = _checkpoint()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=(2.0,) * problem.dimension,
        incumbent_error=32.0,
    )

    with pytest.raises(ValueError, match="incumbent"):
        run_arac_core(checkpoint, problem, ledger, action_seed=47)
