from __future__ import annotations

from experiments.overlap_arac_gate30_multiseed import practical_outcome


def test_gate30_practical_outcome_ignores_numerical_noise() -> None:
    assert practical_outcome(100.0, 100.0 + 5.0e-8)[0] == "tie"
    assert practical_outcome(100.0, 101.0)[0] == "win"
    assert practical_outcome(101.0, 100.0)[0] == "loss"

