from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.phase1 import (
    LANDSCAPE_PROBE_FES,
    PHASE1_FEATURE_NAMES,
    PROGRESS_FEATURE_NAMES,
    StructuralEvidence,
    phase1_budget,
    run_phase1,
)
from arac.runtime.contracts import PhaseCheckpoint
from arac.runtime.ledger import EvaluationLedger


def _problem() -> OptimizationProblem:
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=40,
        lower_bounds=(-5.0,) * 40,
        upper_bounds=(5.0,) * 40,
    )


def test_phase1_consumes_exact_budget_and_freezes_identity_blind_evidence() -> None:
    problem = _problem()
    ledger = EvaluationLedger(problem, 1000)

    probe = run_phase1(problem, ledger, run_seed=19)

    expected_fes = phase1_budget(1000)
    assert ledger.count == expected_fes
    assert probe.checkpoint.phase1_fes == expected_fes
    assert probe.checkpoint.remaining_fes == 1000 - expected_fes
    assert probe.checkpoint.feature_names == PHASE1_FEATURE_NAMES
    assert len(probe.checkpoint.feature_values) == len(PHASE1_FEATURE_NAMES)
    features = dict(
        zip(
            probe.checkpoint.feature_names,
            probe.checkpoint.feature_values,
            strict=True,
        )
    )
    assert all(name in features for name in PROGRESS_FEATURE_NAMES)
    progress = tuple(features[name] for name in PROGRESS_FEATURE_NAMES)
    assert all(value >= 0.0 for value in progress)
    assert sum(progress[:3]) == pytest.approx(features["phase1_log10_improvement"])
    assert 0.0 <= progress[3] <= 1.0
    assert len(probe.checkpoint.blocks) <= 20
    assert sorted(index for block in probe.checkpoint.blocks for index in block) == list(range(40))
    forbidden = {"case_id", "family", "function_id", "permutation", "design_matrix"}
    assert not forbidden & {field.name for field in fields(PhaseCheckpoint)}


def test_phase1_is_deterministic_for_one_seed_and_changes_with_seed() -> None:
    problem = _problem()

    first = run_phase1(problem, EvaluationLedger(problem, 1000), run_seed=7)
    replay = run_phase1(problem, EvaluationLedger(problem, 1000), run_seed=7)
    other = run_phase1(problem, EvaluationLedger(problem, 1000), run_seed=8)

    assert first.checkpoint.checkpoint_hash == replay.checkpoint.checkpoint_hash
    assert first.checkpoint.blocks == replay.checkpoint.blocks
    assert first.checkpoint.checkpoint_hash != other.checkpoint.checkpoint_hash


def test_phase1_improves_anchor_before_structural_inference(monkeypatch) -> None:
    problem = _problem()
    observed: dict[str, object] = {}

    def capture_structure(problem, ledger, *, base, base_value, **kwargs):
        observed["count"] = ledger.count
        observed["base"] = np.asarray(base, dtype=float)
        observed["best_x"] = ledger.best_x
        observed["base_value"] = float(base_value)
        observed["best_value"] = ledger.best_error + problem.optimum
        return StructuralEvidence(
            blocks=kwargs["fallback_blocks"],
            relations=kwargs["fallback_relations"],
            consumed_fes=0,
            interaction_tests=0,
            completed=True,
        )

    monkeypatch.setattr("arac.evidence.phase1.infer_structure", capture_structure)
    run_phase1(problem, EvaluationLedger(problem, 80_000), run_seed=23)

    assert observed["count"] > LANDSCAPE_PROBE_FES
    np.testing.assert_allclose(observed["base"], observed["best_x"])
    assert observed["base_value"] == observed["best_value"]
