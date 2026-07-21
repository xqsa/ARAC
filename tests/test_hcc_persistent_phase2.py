from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from arac.backends.hcc_persistent_phase2 import (
    PERSISTENT_SEP_CMA_SEED_NAMESPACE,
    compile_persistent_full_space_sep_cma_action,
    execute_persistent_full_space_sep_cma_action,
    persistent_phase2_checkpoint_hash,
    persistent_relation_hash,
)


VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from HCC.OPT.CMAES.sepcmaes import SEPCMAES  # noqa: E402


def _hash(character: str) -> str:
    return character * 64


def _checkpoint(mean: np.ndarray, prefix: tuple[float, ...]) -> str:
    return persistent_phase2_checkpoint_hash(
        problem_id="R4",
        run_seed=117,
        checkpoint_fe=len(prefix),
        fitness_prefix=prefix,
        incumbent=mean,
        topology_hash=_hash("a"),
        order_hash=_hash("b"),
        action_set_hash=_hash("c"),
        start_sweep=4,
    )


def test_checkpoint_binds_prefix_incumbent_and_action_set() -> None:
    mean = np.ones(1000)
    prefix = (1001.0, 1000.0)
    checkpoint = _checkpoint(mean, prefix)

    assert len(checkpoint) == 64
    assert checkpoint == _checkpoint(mean.copy(), prefix)
    assert checkpoint != persistent_phase2_checkpoint_hash(
        problem_id="R4",
        run_seed=117,
        checkpoint_fe=2,
        fitness_prefix=(1001.0, 999.0),
        incumbent=mean,
        topology_hash=_hash("a"),
        order_hash=_hash("b"),
        action_set_hash=_hash("c"),
        start_sweep=4,
    )


def test_persistent_sep_cma_consumes_exact_remaining_budget() -> None:
    mean = np.ones(1000)
    prefix = (1001.0, 1000.0)
    checkpoint = _checkpoint(mean, prefix)
    calls = 0

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal calls
        array = np.asarray(batch, dtype=float)
        calls += len(array)
        return np.sum(np.square(array), axis=1)

    action = compile_persistent_full_space_sep_cma_action(
        problem_id="R4",
        run_seed=117,
        checkpoint_fe=2,
        checkpoint_hash=checkpoint,
        owner_group_indices=(0, 1),
        shared_variable_indices=(10,),
        incumbent=mean,
        acceptance_fitness=1000.0,
        sigma=0.5,
        lower=-5.0,
        upper=5.0,
        budget_fes=25,
        issued_sweep=3,
        start_sweep=4,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
    )

    assert calls == 0
    assert action.seed_namespace == PERSISTENT_SEP_CMA_SEED_NAMESPACE
    result = execute_persistent_full_space_sep_cma_action(
        action,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
        current_fe=2,
        current_sweep=4,
        checkpoint_hash=checkpoint,
        incumbent=mean,
    )

    assert calls == 25
    assert result.consumed_fes == 25
    assert result.lifecycle.status == "completed"
    assert result.lifecycle.completed_fe == 27
    assert len(result.incumbent) == 1000
    assert len(result.final_state_hash) == 64


def test_persistent_sep_cma_fails_closed_on_anchor_or_checkpoint_change() -> None:
    mean = np.ones(1000)
    prefix = (1001.0, 1000.0)
    checkpoint = _checkpoint(mean, prefix)

    def objective(batch: np.ndarray) -> np.ndarray:
        return np.sum(np.square(np.asarray(batch, dtype=float)), axis=1)

    action = compile_persistent_full_space_sep_cma_action(
        problem_id="R4",
        run_seed=117,
        checkpoint_fe=2,
        checkpoint_hash=checkpoint,
        owner_group_indices=(0, 1),
        shared_variable_indices=(10,),
        incumbent=mean,
        acceptance_fitness=1000.0,
        sigma=0.5,
        lower=-5.0,
        upper=5.0,
        budget_fes=24,
        issued_sweep=3,
        start_sweep=4,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
    )

    changed = mean.copy()
    changed[0] = 2.0
    with pytest.raises(ValueError, match="anchor changed"):
        execute_persistent_full_space_sep_cma_action(
            action,
            objective=objective,
            sepcmaes_factory=SEPCMAES,
            current_fe=2,
            current_sweep=4,
            checkpoint_hash=checkpoint,
            incumbent=changed,
        )
    with pytest.raises(ValueError, match="dispatch_checkpoint_hash mismatch"):
        execute_persistent_full_space_sep_cma_action(
            action,
            objective=objective,
            sepcmaes_factory=SEPCMAES,
            current_fe=2,
            current_sweep=4,
            checkpoint_hash=_hash("d"),
            incumbent=mean,
        )


def test_relation_hash_is_structural_and_ordered() -> None:
    assert persistent_relation_hash((0, 1), (3, 4)) == persistent_relation_hash(
        (0, 1),
        (3, 4),
    )
    assert persistent_relation_hash((1, 0), (3, 4)) != persistent_relation_hash(
        (0, 1),
        (3, 4),
    )
