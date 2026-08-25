from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arac.actions.recovered import RecoveredHistoricalSmpExecutor, RecoveredSmpExecutor
from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.actions.smp import SmpExecutor
from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ACTION_NAMES, ActionContext, PhaseCheckpoint, RelationEvidence
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


def test_recovered_registry_uses_historical_smp_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context("smp", allow_out_of_bounds=True)
    context = ActionContext(
        "smp",
        PhaseCheckpoint(
            protocol=context.checkpoint.protocol,
            run_seed=context.checkpoint.run_seed,
            total_budget_fes=context.checkpoint.total_budget_fes,
            phase1_fes=context.checkpoint.phase1_fes,
            incumbent=context.checkpoint.incumbent,
            incumbent_error=context.checkpoint.incumbent_error,
            feature_names=context.checkpoint.feature_names,
            feature_values=context.checkpoint.feature_values,
            blocks=context.checkpoint.blocks,
            relations=(RelationEvidence(0, 1, strength=1.0, disagreement=0.8),),
        ),
        context.problem,
        EvaluationLedger.from_checkpoint(
            context.problem,
            total_budget=context.checkpoint.total_budget_fes,
            phase1_fes=context.checkpoint.phase1_fes,
            incumbent=context.checkpoint.incumbent,
            incumbent_error=context.checkpoint.incumbent_error,
            allow_out_of_bounds=True,
        ),
        action_seed=context.action_seed,
    )

    def fake_historical_execute(self: SmpExecutor, active: ActionContext):
        del self
        from arac.actions._execution import terminal_result

        while active.ledger.remaining:
            active.ledger.evaluate(active.ledger.best_x)
        return terminal_result(
            active,
            route="stateful_block_visits_1_rescue_2_global_polish_3",
        )

    monkeypatch.setattr(SmpExecutor, "execute", fake_historical_execute)
    registry = RecoveredActionRegistry()

    assert isinstance(registry._executors["smp"], RecoveredHistoricalSmpExecutor)
    result = registry.execute(context)

    assert result.route.startswith("recovered_historical_compatible_smp_v1_clip_offspring_true_")
    assert "rescue_" in result.route
    assert "global_polish_" in result.route
    assert "noop_" not in result.route


def test_recovered_registry_preserves_zero_relation_smp_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context("smp", allow_out_of_bounds=True)

    def fake_zero_relation_execute(self: RecoveredSmpExecutor, active: ActionContext):
        del self
        from arac.actions._execution import terminal_result

        while active.ledger.remaining:
            active.ledger.evaluate(active.ledger.best_x)
        return terminal_result(active, route="stateful_visits_1_zero_relation_hybrid_rescue_noop_2")

    monkeypatch.setattr(RecoveredSmpExecutor, "execute", fake_zero_relation_execute)
    result = RecoveredActionRegistry().execute(context)

    assert result.route.startswith("recovered_zero_relation_recovered_smp_v1_clip_offspring_false_")
    assert "zero_relation_hybrid_rescue" in result.route


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
