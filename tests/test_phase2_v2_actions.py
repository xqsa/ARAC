from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import ActionContext, PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger


def _make_context(
    action_name: str,
    events: list[tuple[float, ...]],
    *,
    relations=(),
    retain_trajectory: bool = True,
) -> ActionContext:
    dimension = 6

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        events.extend(tuple(float(value) for value in row) for row in batch)
        return np.sum(batch**2, axis=1) if rows.ndim == 2 else float(np.sum(batch[0] ** 2))

    problem = OptimizationProblem(
        objective=objective,
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )
    checkpoint = PhaseCheckpoint(
        protocol="phase2-v2-test-v1",
        run_seed=4,
        total_budget_fes=34,
        phase1_fes=4,
        incumbent=(1.0,) * dimension,
        incumbent_error=float(dimension),
        feature_names=("line_high_frequency_fraction_median",),
        feature_values=(0.4,),
        blocks=((0, 1), (2, 3), (4, 5)),
        relations=relations,
    )
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    return ActionContext(
        action_name,
        checkpoint,
        problem,
        ledger,
        action_seed=19,
        retain_trajectory=retain_trajectory,
    )


@pytest.mark.parametrize("action_name", ("ctp", "smp", "gcb", "aor"))
def test_v2_split_and_uninterrupted_execution_have_identical_prefix(
    action_name: str,
) -> None:
    uninterrupted_events: list[tuple[float, ...]] = []
    uninterrupted = ActionRegistry().initialize(_make_context(action_name, uninterrupted_events))
    uninterrupted.step(30)

    split_events: list[tuple[float, ...]] = []
    split_context = _make_context(action_name, split_events)
    split = ActionRegistry().initialize(split_context)
    split.step(11)
    snapshot = split.snapshot()

    resumed_context = ActionContext(
        action_name,
        split_context.checkpoint,
        split_context.problem,
        EvaluationLedger.from_phase2_snapshot(split_context.problem, snapshot),
        action_seed=19,
    )
    resumed = ActionRegistry().resume(resumed_context, snapshot)
    resumed.step(19)

    assert split_events == uninterrupted_events
    assert resumed.context.ledger.count == uninterrupted.context.ledger.count == 34
    assert resumed.context.ledger.best_error == uninterrupted.context.ledger.best_error
    assert resumed.context.ledger.best_x.tolist() == uninterrupted.context.ledger.best_x.tolist()
    assert resumed.snapshot().snapshot_hash == uninterrupted.snapshot().snapshot_hash


@pytest.mark.parametrize(
    ("action_name", "segment_boundary"),
    (("ctp", 6), ("smp", 10), ("gcb", 6)),
)
def test_block_actions_restore_at_segment_boundary_without_replaying(
    action_name: str,
    segment_boundary: int,
) -> None:
    uninterrupted_events: list[tuple[float, ...]] = []
    uninterrupted = ActionRegistry().initialize(
        _make_context(action_name, uninterrupted_events)
    )
    uninterrupted.step(30)

    split_events: list[tuple[float, ...]] = []
    split_context = _make_context(action_name, split_events)
    split = ActionRegistry().initialize(split_context)
    split.step(segment_boundary)
    boundary_snapshot = split.snapshot()
    split_prefix = list(split_events)

    resumed_context = ActionContext(
        action_name,
        split_context.checkpoint,
        split_context.problem,
        EvaluationLedger.from_phase2_snapshot(split_context.problem, boundary_snapshot),
        action_seed=19,
    )
    resumed = ActionRegistry().resume(resumed_context, boundary_snapshot)
    resumed.step(30 - segment_boundary)

    assert split_prefix == uninterrupted_events[:segment_boundary]
    assert split_events == uninterrupted_events
    assert resumed_context.ledger.count == uninterrupted.context.ledger.count == 34
    assert split_context.ledger.count == 4 + segment_boundary
    assert resumed.context.ledger.best_x.tolist() == uninterrupted.context.ledger.best_x.tolist()
    assert resumed.snapshot().snapshot_hash == uninterrupted.snapshot().snapshot_hash


def test_v2_execute_consumes_exact_terminal_budget_in_fixed_steps() -> None:
    events: list[tuple[float, ...]] = []
    context = _make_context("aor", events)

    result = ActionRegistry().execute_v2(context, step_fes=5)

    assert result.terminal_fes == 34
    assert result.consumed_fes == 30
    assert result.optimizer_package == "pypop7"
    assert len(events) == 30


def test_v2_compact_snapshot_preserves_execution_without_full_trajectory() -> None:
    full_events: list[tuple[float, ...]] = []
    full = ActionRegistry().initialize(_make_context("aor", full_events))
    full.step(11)
    full_snapshot = full.snapshot()

    compact_events: list[tuple[float, ...]] = []
    compact_context = _make_context(
        "aor", compact_events, retain_trajectory=False
    )
    compact = ActionRegistry().initialize(compact_context)
    compact.step(11)
    compact_snapshot = compact.snapshot()

    assert compact.best_trace == []
    assert compact_events == full_events
    assert len(compact_snapshot.state_payload) < len(full_snapshot.state_payload)
    full.step(19)

    resumed_context = ActionContext(
        "aor",
        compact_context.checkpoint,
        compact_context.problem,
        EvaluationLedger.from_phase2_snapshot(
            compact_context.problem, compact_snapshot
        ),
        action_seed=19,
        retain_trajectory=False,
    )
    resumed = ActionRegistry().resume(resumed_context, compact_snapshot)
    resumed.step(19)

    assert resumed.best_trace == []
    assert resumed.context.ledger.best_error == full.context.ledger.best_error
    assert resumed.context.ledger.best_x.tolist() == full.context.ledger.best_x.tolist()


def test_v2_gcb_route_is_based_only_on_relation_count() -> None:
    zero_events: list[tuple[float, ...]] = []
    positive_events: list[tuple[float, ...]] = []
    relation = (RelationEvidence(0, 1, strength=0.4, disagreement=0.2),)

    zero = ActionRegistry().initialize(_make_context("gcb", zero_events))
    positive = ActionRegistry().initialize(
        _make_context("gcb", positive_events, relations=relation)
    )

    assert zero.route == "phase2_v2_zero_relation"
    assert positive.route == "phase2_v2_positive_relation_graph"


def test_v2_resume_rejects_snapshot_from_a_different_action() -> None:
    events: list[tuple[float, ...]] = []
    source = ActionRegistry().initialize(_make_context("aor", events))
    source.step(3)
    snapshot = source.snapshot()

    wrong_context = _make_context("ctp", [])
    wrong_context = replace(wrong_context, ledger=EvaluationLedger.from_phase2_snapshot(wrong_context.problem, snapshot))
    with pytest.raises(RuntimeError, match="action"):
        ActionRegistry().resume(wrong_context, snapshot)
