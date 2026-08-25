"""Matched-host tests for the candidate soft-RDDSM four-action dispatcher."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arac.analysis.structural_router import route_from_overlap_evidence
from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.overlap_adapter import Phase1OverlapAdapter, Phase1OverlapEvidence
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from experiments.upgrade.soft_rddsm_structural_router_v1.pipeline import (
    SoftRddsmStructuralRun,
    action_view_checkpoint,
    execute_soft_rddsm_structural_route,
)


DIMENSION = 40
PHASE1_FES = 240
TOTAL_FES = 1_000


def _problem() -> OptimizationProblem:
    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        result = np.sum(batch**2, axis=1)
        return float(result[0]) if rows.ndim == 1 else result

    return OptimizationProblem(
        objective=objective,
        dimension=DIMENSION,
        lower_bounds=(-5.0,) * DIMENSION,
        upper_bounds=(5.0,) * DIMENSION,
    )


def _checkpoint() -> PhaseCheckpoint:
    blocks = tuple(
        tuple(range(start, start + 10))
        for start in range(0, DIMENSION, 10)
    )
    return PhaseCheckpoint(
        protocol="soft-rddsm-dispatch-test-v1",
        run_seed=7,
        total_budget_fes=TOTAL_FES,
        phase1_fes=PHASE1_FES,
        incumbent=(0.5,) * DIMENSION,
        incumbent_error=10.0,
        feature_names=(
            "log10_center_error",
            "line_high_frequency_fraction_median",
        ),
        feature_values=(1.0, 0.4),
        blocks=blocks,
        # Deliberately include an incidental edge: SMP must clear it when the
        # sidecar proves that no variable is shared.
        relations=(RelationEvidence(0, 1, strength=0.2, disagreement=0.1),),
    )


def _sidecar(kind: str) -> Phase1OverlapEvidence:
    if kind == "aor":
        groups = tuple(
            tuple(range(start, start + 10))
            for start in range(0, DIMENSION, 10)
        )
        complete = False
    elif kind == "smp":
        groups = tuple(
            tuple(range(start, start + 10))
            for start in range(0, DIMENSION, 10)
        )
        complete = True
    elif kind == "ctp":
        groups = (
            tuple(range(0, 11)),
            tuple(range(10, 20)),
            tuple(range(20, 31)),
            tuple(range(30, 40)),
        )
        complete = True
    elif kind == "gcb":
        groups = (
            tuple(range(0, 11)),
            tuple(range(10, 21)),
            tuple(range(20, 31)),
            tuple(range(30, 40)),
        )
        complete = True
    else:
        raise ValueError(kind)
    memberships = tuple(
        tuple(group for group, variables in enumerate(groups) if variable in variables)
        for variable in range(DIMENSION)
    )
    confidences = tuple(
        (variable, group, 1.0)
        for variable, owners in enumerate(memberships)
        for group in owners
    )
    return Phase1OverlapEvidence(
        dimension=DIMENSION,
        groups=groups,
        memberships=memberships,
        membership_confidences=confidences,
        complete=complete,
    )


def _structural_run(kind: str) -> SoftRddsmStructuralRun:
    checkpoint = _checkpoint()
    sidecar = _sidecar(kind)
    adaptation = Phase1OverlapAdapter().adapt(checkpoint, sidecar)
    decision = route_from_overlap_evidence(sidecar)
    phase1 = SimpleNamespace(
        checkpoint=checkpoint,
        overlap_evidence=sidecar,
    )
    return SoftRddsmStructuralRun(
        phase1=phase1,
        decision=decision,
        adaptation=adaptation,
    )


@pytest.mark.parametrize(
    ("kind", "action", "relation_count"),
    (
        ("aor", "aor", 1),
        ("smp", "smp", 0),
        ("ctp", "ctp", 2),
        ("gcb", "gcb", 3),
    ),
)
def test_action_view_projects_each_structural_route(
    kind: str,
    action: str,
    relation_count: int,
) -> None:
    run = _structural_run(kind)
    view = action_view_checkpoint(run)

    assert run.decision.action_name == action
    assert view.total_budget_fes == run.phase1.checkpoint.total_budget_fes
    assert view.phase1_fes == run.phase1.checkpoint.phase1_fes
    assert view.incumbent == run.phase1.checkpoint.incumbent
    assert view.incumbent_error == run.phase1.checkpoint.incumbent_error
    assert view.overlap_relation_count == relation_count


@pytest.mark.parametrize("kind", ("aor", "smp", "ctp", "gcb"))
def test_dispatch_executes_exactly_one_selected_action_to_terminal_budget(kind: str) -> None:
    problem = _problem()
    run = _structural_run(kind)
    result = execute_soft_rddsm_structural_route(run, problem, action_seed=91)

    assert result.action_result.action_name == run.decision.action_name
    assert result.action_result.terminal_fes == TOTAL_FES
    assert result.action_result.consumed_fes == TOTAL_FES - PHASE1_FES
    assert result.action_checkpoint_hash == result.action_result.checkpoint_hash
    assert result.source_checkpoint_hash == run.phase1.checkpoint.checkpoint_hash
    if kind == "aor":
        assert result.action_result.route.startswith("evidence_routed_")
    elif kind == "ctp":
        assert "relation_cover_polish" in result.action_result.route
    elif kind == "gcb":
        assert result.action_result.route.startswith("positive_relation_graph_")
    else:
        assert result.action_result.route.startswith("stateful_block_visits_")


def test_non_aor_route_rejects_incomplete_sidecar() -> None:
    run = _structural_run("aor")
    # The route is intentionally changed after evidence construction.  The
    # action-view guard must refuse an incomplete sidecar for non-AOR use.
    invalid = SoftRddsmStructuralRun(
        phase1=run.phase1,
        decision=route_from_overlap_evidence(_sidecar("smp")),
        adaptation=run.adaptation,
    )
    with pytest.raises(ValueError, match="requires complete"):
        action_view_checkpoint(invalid)
