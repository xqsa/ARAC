from __future__ import annotations

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
import arac.evidence.structural as structural
from arac.evidence.structural import infer_structure
from arac.runtime.ledger import EvaluationLedger


def _coupled_groups(groups: tuple[tuple[int, ...], ...], dimension: int):
    def objective(candidate):
        rows = np.asarray(candidate, dtype=float)
        values = np.zeros(rows.shape[:-1], dtype=float)
        for group in groups:
            selected = rows[..., np.asarray(group)]
            values += np.sum(selected**2, axis=-1) + np.sum(selected, axis=-1) ** 2
        return values

    return OptimizationProblem(
        objective=objective,
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )


def test_black_box_structure_recovers_disjoint_groups_without_identity() -> None:
    expected = tuple(tuple(range(start, start + 10)) for start in range(0, 40, 10))
    problem = _coupled_groups(expected, 40)
    ledger = EvaluationLedger(problem, 10_000)
    center = np.zeros(40)
    center_value = float(ledger.evaluate(center))

    evidence = infer_structure(
        problem,
        ledger,
        base=center,
        base_value=center_value,
        run_seed=7,
        max_fes=9_000,
        fallback_blocks=(tuple(range(40)),),
        fallback_relations=(),
    )

    assert evidence.completed is True
    assert {frozenset(block) for block in evidence.blocks} == {
        frozenset(block) for block in expected
    }
    assert evidence.relations == ()
    assert 0 < evidence.consumed_fes <= 9_000


def test_clear_separable_structure_skips_confirmation_probe() -> None:
    problem = OptimizationProblem(
        objective=lambda candidate: np.sum(np.asarray(candidate, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )
    ledger = EvaluationLedger(problem, 1_000)
    center = np.zeros(40)
    center_value = float(ledger.evaluate(center))

    evidence = infer_structure(
        problem,
        ledger,
        base=center,
        base_value=center_value,
        run_seed=13,
        max_fes=900,
        fallback_blocks=(tuple(range(40)),),
        fallback_relations=(),
    )

    assert evidence.completed is True
    assert len(evidence.blocks) == 20
    assert evidence.relations == ()
    assert evidence.consumed_fes == 330


def test_oversized_refinement_cannot_drop_the_active_seed(monkeypatch) -> None:
    dimension = 200
    problem = OptimizationProblem(
        objective=lambda candidate: np.sum(np.asarray(candidate, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )
    ledger = EvaluationLedger(problem, 1_000)

    def fake_neighbors(*, seed, **kwargs):
        del kwargs
        neighbors = tuple(range(1, dimension)) if seed == 0 else ()
        return neighbors, 1, True

    monkeypatch.setattr(structural, "_discover_neighbors", fake_neighbors)
    blocks, tests, completed = structural._discover_partition(
        order=tuple(range(dimension)),
        base=np.zeros(dimension),
        base_value=0.0,
        steps=(np.ones(dimension), np.ones(dimension)),
        ledger=ledger,
        stop_count=1_000,
    )

    assert completed is True
    assert tests == 21
    assert len(blocks) == 21
    assert any(0 in block for block in blocks)
    assert sorted(index for block in blocks for index in block) == list(range(dimension))


def test_oversized_refinement_uses_only_unassigned_anchors(monkeypatch) -> None:
    dimension = 200
    problem = OptimizationProblem(
        objective=lambda candidate: np.sum(np.asarray(candidate, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-5.0,) * dimension,
        upper_bounds=(5.0,) * dimension,
    )
    calls = []

    def fake_neighbors(*, seed, **kwargs):
        del kwargs
        calls.append(seed)
        if seed == 0:
            neighbors = tuple(range(1, 50))
        elif seed == 50:
            neighbors = tuple(index for index in range(dimension) if index != seed)
        else:
            neighbors = ()
        return neighbors, 1, True

    monkeypatch.setattr(structural, "_discover_neighbors", fake_neighbors)
    blocks, _, completed = structural._discover_partition(
        order=tuple(range(dimension)),
        base=np.zeros(dimension),
        base_value=0.0,
        steps=(np.ones(dimension), np.ones(dimension)),
        ledger=EvaluationLedger(problem, 1_000),
        stop_count=1_000,
    )

    assert completed is True
    assert calls[:4] == [0, 50, 51, 52]
    assert sorted(index for block in blocks for index in block) == list(range(dimension))


def test_black_box_structure_exposes_cross_block_overlap_relations() -> None:
    groups = (tuple(range(0, 100)), tuple(range(80, 180)), tuple(range(160, 260)))
    problem = _coupled_groups(groups, 260)
    ledger = EvaluationLedger(problem, 50_000)
    center = np.zeros(260)
    center_value = float(ledger.evaluate(center))

    evidence = infer_structure(
        problem,
        ledger,
        base=center,
        base_value=center_value,
        run_seed=9,
        max_fes=45_000,
        fallback_blocks=(tuple(range(260)),),
        fallback_relations=(),
    )

    assert evidence.completed is True
    assert len(evidence.blocks) == 3
    assert sorted(index for block in evidence.blocks for index in block) == list(range(260))
    assert evidence.relations


def test_black_box_structure_keeps_off_center_probes_inside_bounds() -> None:
    expected = tuple(tuple(range(start, start + 10)) for start in range(0, 40, 10))
    problem = _coupled_groups(expected, 40)
    ledger = EvaluationLedger(problem, 10_000)
    anchor = np.full(40, 5.0)
    anchor_value = float(ledger.evaluate(anchor))

    evidence = infer_structure(
        problem,
        ledger,
        base=anchor,
        base_value=anchor_value,
        run_seed=11,
        max_fes=9_000,
        fallback_blocks=(tuple(range(40)),),
        fallback_relations=(),
    )

    assert evidence.completed is True
    assert {frozenset(block) for block in evidence.blocks} == {
        frozenset(block) for block in expected
    }
