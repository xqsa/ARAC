from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from arac.backends.hcc_persistent_phase2 import (
    FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE,
    PERSISTENT_SEP_CMA_SEED_NAMESPACE,
    compile_full_space_sep_cma_burst_action,
    compile_full_space_sep_cma_phase_boundary_action,
    compile_persistent_full_space_sep_cma_action,
    execute_full_space_sep_cma_burst_action,
    execute_persistent_full_space_sep_cma_action,
    full_space_sep_cma_burst_optimizer_seed,
    full_space_sep_cma_dispatch_checkpoint_hash,
    full_space_sep_cma_phase_boundary_action_source_hash,
    full_space_sep_cma_phase_boundary_checkpoint_hash,
    persistent_phase2_checkpoint_hash,
    persistent_relation_hash,
)
from arac.actions.full_space_sep_cma import TRIGGER_SCOPE_PHASE_BOUNDARY


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


def test_burst_dispatch_checkpoint_binds_every_runtime_input() -> None:
    incumbent = tuple(float(value) for value in np.ones(1000))
    base = {
        "problem_id": "R4",
        "run_seed": 117,
        "dispatch_fe": 2,
        "outer_iter": 4,
        "group_index": 1,
        "owner_group_indices": (0, 1),
        "shared_variable_indices": (10,),
        "incumbent": incumbent,
        "fitness_prefix": (1001.0, 1000.0),
        "topology_hash": _hash("a"),
        "order_hash": _hash("b"),
        "action_set_hash": _hash("c"),
        "previous_shared_values": (1.0,),
        "current_shared_values": (2.0,),
        "previous_delta": 3.0,
        "current_delta": 4.0,
        "completed_group_deltas": (3.0, 4.0),
        "completed_group_actual_fes": (120, 144),
        "frozen_burst_budget_fes": 240,
        "budget_source_sweep": 3,
    }
    expected = full_space_sep_cma_dispatch_checkpoint_hash(**base)
    changed_incumbent = list(incumbent)
    changed_incumbent[0] = 2.0
    mutations = {
        "problem_id": {"problem_id": "R5"},
        "run_seed": {"run_seed": 118},
        "dispatch_fe_and_prefix": {
            "dispatch_fe": 3,
            "fitness_prefix": (1001.0, 1000.0, 999.0),
        },
        "outer_iter": {"outer_iter": 5},
        "group_index": {"group_index": 2},
        "owners": {"owner_group_indices": (1, 0)},
        "shared": {"shared_variable_indices": (11,)},
        "incumbent": {"incumbent": tuple(changed_incumbent)},
        "fitness_prefix": {"fitness_prefix": (1001.0, 999.0)},
        "topology": {"topology_hash": _hash("d")},
        "order": {"order_hash": _hash("e")},
        "action_set": {"action_set_hash": _hash("f")},
        "previous_values": {"previous_shared_values": (1.5,)},
        "current_values": {"current_shared_values": (2.5,)},
        "previous_delta": {"previous_delta": 3.5},
        "current_delta": {"current_delta": 4.5},
        "completed_deltas": {"completed_group_deltas": (3.0, 5.0)},
        "completed_fes": {"completed_group_actual_fes": (120, 168)},
        "burst_budget": {"frozen_burst_budget_fes": 264},
        "budget_source_sweep": {"budget_source_sweep": 2},
    }

    assert len(expected) == 64
    assert expected == full_space_sep_cma_dispatch_checkpoint_hash(**base)
    for name, mutation in mutations.items():
        observed = full_space_sep_cma_dispatch_checkpoint_hash(
            **(base | mutation)
        )
        assert observed != expected, name


def test_burst_dispatch_checkpoint_validates_shape_prefix_and_hashes() -> None:
    base = {
        "problem_id": "R4",
        "run_seed": 117,
        "dispatch_fe": 2,
        "outer_iter": 4,
        "group_index": 1,
        "owner_group_indices": (0, 1),
        "shared_variable_indices": (10,),
        "incumbent": tuple(float(value) for value in np.ones(1000)),
        "fitness_prefix": (1001.0, 1000.0),
        "topology_hash": _hash("a"),
        "order_hash": _hash("b"),
        "action_set_hash": _hash("c"),
        "previous_shared_values": (1.0,),
        "current_shared_values": (2.0,),
        "previous_delta": 3.0,
        "current_delta": 4.0,
        "completed_group_deltas": (3.0,),
        "completed_group_actual_fes": (120,),
        "frozen_burst_budget_fes": 240,
        "budget_source_sweep": 3,
    }

    with pytest.raises(ValueError, match="1000-dimensional"):
        full_space_sep_cma_dispatch_checkpoint_hash(
            **(base | {"incumbent": tuple(np.ones(999))})
        )
    with pytest.raises(ValueError, match="length must equal dispatch_fe"):
        full_space_sep_cma_dispatch_checkpoint_hash(
            **(base | {"fitness_prefix": (1001.0,)})
        )
    with pytest.raises(ValueError, match="topology_hash"):
        full_space_sep_cma_dispatch_checkpoint_hash(
            **(base | {"topology_hash": "not-a-hash"})
        )


def test_phase_boundary_source_hash_is_stable_and_binds_action_identity() -> None:
    base = {
        "problem_id": "R1",
        "run_seed": 117,
        "issued_sweep": 3,
        "target_sweep": 4,
        "frozen_burst_budget_fes": 24,
        "topology_hash": _hash("a"),
        "order_hash": _hash("b"),
    }
    expected = full_space_sep_cma_phase_boundary_action_source_hash(**base)
    mutations = {
        "problem": {"problem_id": "R2"},
        "seed": {"run_seed": 118},
        "sweeps": {"issued_sweep": 4, "target_sweep": 5},
        "budget": {"frozen_burst_budget_fes": 48},
        "topology": {"topology_hash": _hash("c")},
        "order": {"order_hash": _hash("d")},
    }

    assert len(expected) == 64
    assert expected == full_space_sep_cma_phase_boundary_action_source_hash(**base)
    for name, mutation in mutations.items():
        observed = full_space_sep_cma_phase_boundary_action_source_hash(
            **(base | mutation)
        )
        assert observed != expected, name


def test_phase_boundary_checkpoint_hash_is_stable_and_binds_frozen_state() -> None:
    mean = tuple(float(value) for value in np.ones(1000))
    source_hash = full_space_sep_cma_phase_boundary_action_source_hash(
        problem_id="R1",
        run_seed=117,
        issued_sweep=3,
        target_sweep=4,
        frozen_burst_budget_fes=24,
        topology_hash=_hash("a"),
        order_hash=_hash("b"),
    )
    base = {
        "problem_id": "R1",
        "run_seed": 117,
        "checkpoint_fe": 2,
        "issued_sweep": 3,
        "target_sweep": 4,
        "incumbent": mean,
        "fitness_prefix": (1001.0, 1000.0),
        "topology_hash": _hash("a"),
        "order_hash": _hash("b"),
        "action_source_hash": source_hash,
        "completed_group_deltas": (3.0, 4.0),
        "completed_group_actual_fes": (12, 12),
        "frozen_burst_budget_fes": 24,
    }
    expected = full_space_sep_cma_phase_boundary_checkpoint_hash(**base)
    changed_mean = list(mean)
    changed_mean[0] = 2.0
    mutations = {
        "problem": {"problem_id": "R2"},
        "seed": {"run_seed": 118},
        "checkpoint": {
            "checkpoint_fe": 3,
            "fitness_prefix": (1001.0, 1000.0, 999.0),
        },
        "sweeps": {"issued_sweep": 4, "target_sweep": 5},
        "incumbent": {"incumbent": tuple(changed_mean)},
        "prefix": {"fitness_prefix": (1001.0, 999.0)},
        "topology": {"topology_hash": _hash("c")},
        "order": {"order_hash": _hash("d")},
        "source": {"action_source_hash": _hash("e")},
        "deltas": {"completed_group_deltas": (3.0, 5.0)},
        "group_fes": {"completed_group_actual_fes": (10, 14)},
        "budget": {
            "completed_group_actual_fes": (24, 24),
            "frozen_burst_budget_fes": 48,
        },
    }

    assert len(expected) == 64
    assert expected == full_space_sep_cma_phase_boundary_checkpoint_hash(**base)
    for name, mutation in mutations.items():
        observed = full_space_sep_cma_phase_boundary_checkpoint_hash(
            **(base | mutation)
        )
        assert observed != expected, name


def test_phase_boundary_action_compiles_without_relation_and_executes_exactly() -> None:
    mean = np.ones(1000)
    checkpoint = _hash("d")
    calls = 0

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal calls
        array = np.asarray(batch, dtype=float)
        calls += len(array)
        return np.sum(np.square(array), axis=1)

    action = compile_full_space_sep_cma_phase_boundary_action(
        problem_id="R1",
        run_seed=117,
        checkpoint_fe=2,
        checkpoint_hash=checkpoint,
        incumbent=mean,
        acceptance_fitness=1.0e9,
        sigma=0.5,
        lower=-5.12,
        upper=5.12,
        budget_fes=24,
        issued_sweep=3,
        target_sweep=4,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
    )

    assert calls == 0
    assert action.trigger_scope == TRIGGER_SCOPE_PHASE_BOUNDARY
    assert action.trigger_relation_hash == checkpoint
    assert "trigger_relation_hash" not in action.audit_payload()

    result = execute_full_space_sep_cma_burst_action(
        action,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
        current_fe=2,
        current_sweep=4,
        dispatch_checkpoint_hash=checkpoint,
        incumbent=mean,
    )

    assert calls == result.consumed_fes == action.budget_fes == 24
    assert result.lifecycle.status == "completed"
    assert result.lifecycle.started_fe == 2
    assert result.lifecycle.completed_fe == 26
    assert result.action_hash == action.action_hash
    assert result.lifecycle_hash == result.lifecycle.state_hash(action)
    assert result.accepted is True
    assert result.incumbent == result.candidate


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


def test_sep_cma_burst_uses_exp019_seed_and_compiles_without_objective_fe() -> None:
    mean = np.ones(1000)
    checkpoint = _checkpoint(mean, (1001.0, 1000.0))
    calls = 0

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal calls
        array = np.asarray(batch, dtype=float)
        calls += len(array)
        return np.sum(np.square(array), axis=1)

    expected_seed = int(
        hashlib.sha256(
            json.dumps(
                {
                    "namespace": "full_space_sep_cma",
                    "dispatch_checkpoint_hash": checkpoint,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:16],
        16,
    ) % (2**32)

    action = compile_full_space_sep_cma_burst_action(
        problem_id="R4",
        run_seed=117,
        dispatch_fe=2,
        dispatch_checkpoint_hash=checkpoint,
        owner_group_indices=(0, 1),
        shared_variable_indices=(10,),
        incumbent=mean,
        acceptance_fitness=1000.0,
        sigma=0.5,
        lower=-5.0,
        upper=5.0,
        budget_fes=25,
        issued_sweep=3,
        target_sweep=4,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
    )

    assert calls == 0
    assert action.seed_namespace == FULL_SPACE_SEP_CMA_BURST_SEED_NAMESPACE
    assert action.optimizer_seed == expected_seed
    assert full_space_sep_cma_burst_optimizer_seed(checkpoint) == expected_seed


@pytest.mark.parametrize(
    ("acceptance_fitness", "expected_accepted"),
    ((1.0e9, True), (0.0, False)),
)
def test_sep_cma_burst_consumes_exact_budget_then_resumes_native(
    acceptance_fitness: float,
    expected_accepted: bool,
) -> None:
    mean = np.ones(1000)
    checkpoint = _checkpoint(mean, (1001.0, 1000.0))
    calls = 0

    def objective(batch: np.ndarray) -> np.ndarray:
        nonlocal calls
        array = np.asarray(batch, dtype=float)
        calls += len(array)
        return np.sum(np.square(array), axis=1)

    action = compile_full_space_sep_cma_burst_action(
        problem_id="R4",
        run_seed=117,
        dispatch_fe=2,
        dispatch_checkpoint_hash=checkpoint,
        owner_group_indices=(0, 1),
        shared_variable_indices=(10,),
        incumbent=mean,
        acceptance_fitness=acceptance_fitness,
        sigma=0.5,
        lower=-5.0,
        upper=5.0,
        budget_fes=25,
        issued_sweep=3,
        target_sweep=4,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
    )
    result = execute_full_space_sep_cma_burst_action(
        action,
        objective=objective,
        sepcmaes_factory=SEPCMAES,
        current_fe=2,
        current_sweep=4,
        dispatch_checkpoint_hash=checkpoint,
        incumbent=mean,
    )

    assert calls == action.budget_fes == 25
    assert result.consumed_fes == action.budget_fes
    assert result.accepted is expected_accepted
    assert result.resume_native is True
    assert result.lifecycle.status == "completed"
    assert result.lifecycle.started_fe == 2
    assert result.lifecycle.completed_fe == 27
    assert result.lifecycle.consumed_fes == 25
    assert result.action_hash == action.action_hash
    assert result.lifecycle_hash == result.lifecycle.state_hash(action)
    assert len(result.candidate_hash) == 64
    assert len(result.post_incumbent_hash) == 64
    assert result.post_incumbent_hash == (
        result.candidate_hash if expected_accepted else action.initial_mean_hash
    )
    assert result.incumbent_fitness == (
        result.candidate_fitness if expected_accepted else acceptance_fitness
    )
    assert result.incumbent == (
        tuple(result.candidate)
        if expected_accepted
        else tuple(float(value) for value in mean)
    )


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
