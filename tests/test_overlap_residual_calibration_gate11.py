from __future__ import annotations

import numpy as np

from experiments.overlap_residual_calibration_gate11 import (
    _calibrated_residuals,
    _context,
    _sphere_equivalence,
    _build,
    _run_replicates,
    run_gate,
)


def test_gate11_context_uses_four_independent_replicates_and_exact_budget() -> None:
    context = _context("conforming", "chain", 6, 31001)

    assert context["replicate_count_per_group"] == (4, 4, 4, 4, 4, 4)
    assert context["consumed_fes"] == context["expected_fes"] == 1153
    assert all(np.isfinite(item["heldout_standardized_score"]) for item in context["residuals"])


def test_gate11_calibration_within_variance_excludes_owner_between_variance() -> None:
    structure = __import__("arac.coordination", fromlist=["OverlapStructure"]).OverlapStructure(
        3, ((0, 1), (1, 2))
    )

    class _Proposal:
        def __init__(self, group: int, value: float) -> None:
            self.proposal = type(
                "Proposal",
                (),
                {
                    "values": ((0, 0.0), (1, value)) if group == 0 else ((1, value), (2, 0.0)),
                    "uncertainty": ((0, 0.1), (1, 0.1)) if group == 0 else ((1, 0.1), (2, 0.1)),
                },
            )()

    runs = [
        [_Proposal(0, value) for value in (0.0, 0.1, -0.1, 10.0)],
        [_Proposal(1, value) for value in (5.0, 5.1, 4.9, -10.0)],
    ]
    record = _calibrated_residuals(structure, runs)[0]

    assert np.isclose(record["within_variance"], np.var(np.asarray((0.0, 0.1, -0.1)), ddof=1))
    assert record["between_variance"] > 0.0


def test_gate11_conflicting_sphere_is_effective_quadratic_plus_constant() -> None:
    _problem, objective, _structure = _build("conflicting", "star", 12, 31002)
    equivalence = _sphere_equivalence(objective, 31002)

    assert equivalence["max_absolute_error"] <= 1.0e-8
    assert equivalence["mean_absolute_error"] <= 1.0e-8
    assert equivalence["ranking_identical"] is True
    assert equivalence["irreducible_constant"] > 0.0


def test_gate11_replicate_seed_changes_local_proposal() -> None:
    problem, _objective, structure = _build("conflicting", "random", 6, 31003)
    first_runs, _first_fes, _ = _run_replicates(problem, structure, 31003)
    second_runs, _second_fes, _ = _run_replicates(problem, structure, 31004)

    first = first_runs[0][0].proposal.values
    second = second_runs[0][0].proposal.values
    assert first != second


def test_gate11_keeps_protocol_pass_separate_from_scientific_support(monkeypatch) -> None:
    context = _context("conflicting", "random", 6, 31001)

    monkeypatch.setattr(
        "experiments.overlap_residual_calibration_gate11.TOPOLOGIES", ("random",)
    )
    monkeypatch.setattr(
        "experiments.overlap_residual_calibration_gate11.OVERLAP_BUDGETS", (6,)
    )
    monkeypatch.setattr(
        "experiments.overlap_residual_calibration_gate11.FRESH_SEEDS", (31001,)
    )
    monkeypatch.setattr(
        "experiments.overlap_residual_calibration_gate11._context",
        lambda mode, topology, overlap_budget, seed: {
            **context,
            "mode": mode,
            "equivalence": {**context["equivalence"], "ranking_identical": True},
        },
    )
    payload = run_gate()

    assert payload["gate_passed"] is False
    assert payload["scientific_findings"]["protocol_integrity_passed"] is payload["gate_passed"]
    assert payload["scientific_findings"]["residual_separation_supported"] is False
    assert payload["scientific_findings"]["hidden_conflict_identifiable_from_sphere_rankings"] is False
