from __future__ import annotations

from experiments.historical_recovery.smp_checkpoint_handoff_ablation import (
    HISTORICAL_P90,
    VARIANTS,
    _load_checkpoint,
)


def test_handoff_ablation_is_bound_to_real_phase1_checkpoint() -> None:
    checkpoint = _load_checkpoint(phase2_fes=120_000)

    assert checkpoint.phase1_fes == 180_000
    assert checkpoint.total_budget_fes == 300_000
    assert checkpoint.remaining_fes == 120_000
    assert checkpoint.incumbent_error == 2692663045.520366
    assert len(checkpoint.blocks) == 20
    assert checkpoint.overlap_relation_count == 0


def test_recovered_variant_freezes_the_restored_smp_contract() -> None:
    assert VARIANTS["recovered"] == {
        "historical_seed": True,
        "clip_offspring": False,
        "precheck_incumbent": True,
        "strict_material_gain": True,
    }
    assert VARIANTS["identity_blind_bounded"] == {
        "historical_seed": False,
        "clip_offspring": True,
        "precheck_incumbent": True,
        "strict_material_gain": True,
    }
    assert HISTORICAL_P90 == 1.8255606813339802
