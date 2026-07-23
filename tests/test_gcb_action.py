from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from arac.actions.gcb import (
    FULL_SPACE_DIMENSION,
    GCB_ACTION,
    GCB_ACTION_SPEC,
    CANONICAL_SEP_CMA_PARAMETERIZATION,
    CANONICAL_SEP_CMA_PARAMETERS_HASH,
    CANONICAL_SEP_CMA_POPULATION_SIZE,
    CANONICAL_SEP_CMA_REFERENCE_VERSION,
    TRIGGER_SCOPE_PHASE_BOUNDARY,
    TRIGGER_SCOPE_RELATION_DISPATCH,
    GcbAction,
    GcbExecutionState,
    gcb_anchor_hash,
    full_space_vector_hash,
)


def _hash(character: str) -> str:
    return character * 64


def _action(
    *,
    mean: object | None = None,
    mean_hash: str | None = None,
    anchor_hash: str | None = None,
    trigger_context_hash: str = "e" * 64,
    sigma: float = 0.5,
    lower_bound: float = -5.12,
    upper_bound: float = 5.12,
    acceptance_fitness: float = 1_000.0,
    population_size: int = 24,
    budget_fes: int = 240,
    parameterization: str = CANONICAL_SEP_CMA_PARAMETERIZATION,
    canonical_reference_version: str = CANONICAL_SEP_CMA_REFERENCE_VERSION,
    canonical_parameters_hash: str = CANONICAL_SEP_CMA_PARAMETERS_HASH,
    restart_policy: str = "none",
    issued_sweep: int = 3,
    target_sweep: int = 4,
    ttl_sweeps: int = 1,
    expires_sweep: int = 4,
    trigger_scope: str = TRIGGER_SCOPE_RELATION_DISPATCH,
) -> GcbAction:
    values = tuple(0.0 for _ in range(FULL_SPACE_DIMENSION)) if mean is None else mean
    return GcbAction(
        problem_id="R4",
        run_seed=117,
        checkpoint_fe=300_000,
        dispatch_checkpoint_hash=_hash("a"),
        trigger_context_hash=trigger_context_hash,
        anchor_hash=(
            gcb_anchor_hash("R4", values)
            if anchor_hash is None
            else anchor_hash
        ),
        initial_mean=values,  # type: ignore[arg-type]
        initial_mean_hash=(
            full_space_vector_hash(values) if mean_hash is None else mean_hash
        ),
        initial_state_hash=_hash("b"),
        initial_sigma=sigma,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        acceptance_fitness=acceptance_fitness,
        population_size=population_size,
        budget_fes=budget_fes,
        parameterization=parameterization,
        canonical_reference_version=canonical_reference_version,
        canonical_parameters_hash=canonical_parameters_hash,
        optimizer_seed=2026071901,
        seed_namespace="action-ceiling:R4:117:context-0:full-space-sep-cma",
        restart_policy=restart_policy,
        issued_sweep=issued_sweep,
        target_sweep=target_sweep,
        ttl_sweeps=ttl_sweeps,
        expires_sweep=expires_sweep,
        trigger_scope=trigger_scope,
    )


def test_canonical_parameter_identity_is_explicit_without_duplicate_formulas() -> None:
    action = _action()

    assert CANONICAL_SEP_CMA_POPULATION_SIZE == 24
    assert action.parameterization == "ros_hansen_2008_pypop7"
    assert action.canonical_reference_version.startswith("pypop7-sepcmaes@")
    assert action.canonical_parameters_hash == CANONICAL_SEP_CMA_PARAMETERS_HASH


def test_action_freezes_the_vendor_parameter_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
    monkeypatch.syspath_prepend(str(vendor_root))
    module = importlib.import_module("HCC.OPT.CMAES.sepcmaes")
    parameters = module.canonical_sep_cma_parameters(
        FULL_SPACE_DIMENSION,
        CANONICAL_SEP_CMA_POPULATION_SIZE,
    )

    action = _action(
        population_size=parameters.population_size,
        parameterization=parameters.parameterization,
        canonical_reference_version=parameters.reference_version,
        canonical_parameters_hash=parameters.parameter_hash,
    )

    assert action.canonical_parameters_hash == parameters.parameter_hash
    assert action.audit_payload()["canonical_reference_version"] == (
        parameters.reference_version
    )


def test_gcb_action_is_frozen_and_hashes_exact_initial_mean() -> None:
    action = _action(mean=[0.0] * 1000)

    assert GCB_ACTION == "gcb"
    assert GCB_ACTION_SPEC.name == GCB_ACTION
    assert isinstance(action.initial_mean, tuple)
    assert action.initial_mean_hash == full_space_vector_hash(action.initial_mean)
    assert action.anchor_hash == gcb_anchor_hash(
        action.problem_id,
        action.initial_mean,
    )
    assert len(action.action_hash) == 64
    assert action.audit_payload()["canonical_parameters_hash"] == (
        CANONICAL_SEP_CMA_PARAMETERS_HASH
    )
    assert action.audit_payload()["trigger_scope"] == (
        TRIGGER_SCOPE_RELATION_DISPATCH
    )
    assert action.audit_payload()["trigger_context_hash"] == (
        action.trigger_context_hash
    )
    with pytest.raises(FrozenInstanceError):
        action.budget_fes = 100  # type: ignore[misc]


def test_action_hash_changes_with_the_frozen_initial_mean() -> None:
    first = _action()
    changed_mean = [0.0] * 1000
    changed_mean[-1] = 1.0
    second = _action(mean=changed_mean)

    assert first.action_hash != second.action_hash
    assert first.initial_mean_hash != second.initial_mean_hash


def test_phase_boundary_action_audit_binds_context_without_a_relation() -> None:
    action = _action(trigger_scope=TRIGGER_SCOPE_PHASE_BOUNDARY)

    payload = action.audit_payload()

    assert action.trigger_scope == TRIGGER_SCOPE_PHASE_BOUNDARY
    assert payload["trigger_scope"] == TRIGGER_SCOPE_PHASE_BOUNDARY
    assert payload["trigger_context_hash"] == action.trigger_context_hash


@pytest.mark.parametrize(
    ("mean", "message"),
    [
        ([0.0] * 999, "exactly 1000"),
        ([0.0] * 999 + [float("nan")], "must be finite"),
        ([0.0] * 999 + [float("inf")], "must be finite"),
    ],
)
def test_action_rejects_invalid_initial_mean(mean: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _action(mean=mean)


def test_action_rejects_mean_and_anchor_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="initial_mean_hash does not match"):
        _action(mean_hash=_hash("c"))
    with pytest.raises(ValueError, match="anchor_hash does not match"):
        _action(anchor_hash=_hash("c"))


def test_action_rejects_invalid_context_hash_bounds_and_acceptance() -> None:
    with pytest.raises(ValueError, match="trigger_context_hash"):
        _action(trigger_context_hash="not-a-hash")

    with pytest.raises(ValueError, match="smaller than"):
        _action(lower_bound=1.0, upper_bound=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        _action(acceptance_fitness=-1.0)
    with pytest.raises(ValueError, match="must be finite"):
        _action(acceptance_fitness=float("nan"))


def test_action_preserves_native_mean_outside_nominal_bounds() -> None:
    mean = [6.0] + [0.0] * 999

    action = _action(mean=mean)

    assert action.initial_mean == tuple(mean)
    assert action.initial_mean_hash == full_space_vector_hash(mean)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_action_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="initial_sigma"):
        _action(sigma=sigma)


def test_action_rejects_noncanonical_population_and_short_budget() -> None:
    with pytest.raises(ValueError, match="canonical 1000D population"):
        _action(population_size=25)
    with pytest.raises(ValueError, match="at least one population"):
        _action(budget_fes=23)
    with pytest.raises(ValueError, match="parameterization"):
        _action(parameterization="local_formula_v1")
    with pytest.raises(ValueError, match="reference version"):
        _action(canonical_reference_version="pypop7-main")
    with pytest.raises(ValueError, match="canonical_parameters_hash"):
        _action(canonical_parameters_hash="not-a-hash")
    with pytest.raises(ValueError, match="pinned 1000D snapshot"):
        _action(canonical_parameters_hash=_hash("f"))


def test_action_rejects_restart_and_lifecycle_drift() -> None:
    with pytest.raises(ValueError, match="restart_policy='none'"):
        _action(restart_policy="ipop")
    with pytest.raises(ValueError, match="next sweep"):
        _action(target_sweep=5)
    with pytest.raises(ValueError, match="ttl_sweeps=1"):
        _action(ttl_sweeps=2, expires_sweep=5)
    with pytest.raises(ValueError, match="expires_sweep"):
        _action(expires_sweep=5)


def test_execution_state_tracks_hashes_without_duplicating_optimizer_arrays() -> None:
    action = _action(budget_fes=96)
    execution = GcbExecutionState.for_action(action)
    issued_hash = execution.state_hash(action)

    assert execution.status == "issued"
    assert not hasattr(execution, "mean")
    execution.start(
        action,
        current_fe=action.checkpoint_fe,
        current_sweep=action.target_sweep,
        dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
        trigger_context_hash=action.trigger_context_hash,
        anchor_hash=action.anchor_hash,
    )
    running_hash = execution.state_hash(action)
    execution.complete(
        action,
        consumed_fes=96,
        completed_fe=300_096,
        final_state_hash=_hash("d"),
    )

    assert execution.status == "completed"
    assert execution.consumed_fes == action.budget_fes
    assert execution.final_state_hash == _hash("d")
    assert execution.audit_payload(action)["initial_state_hash"] == (
        action.initial_state_hash
    )
    assert len(execution.state_hash(action)) == 64
    assert len({issued_hash, running_hash, execution.state_hash(action)}) == 3


def test_execution_state_can_abstain_before_start_only() -> None:
    action = _action()
    execution = GcbExecutionState.for_action(action)

    execution.abstain(action, reason="anchor_mismatch")

    assert execution.status == "abstained"
    assert execution.invalidation_reason == "anchor_mismatch"
    with pytest.raises(ValueError, match="only an issued action can start"):
        execution.start(
            action,
            current_fe=action.checkpoint_fe,
            current_sweep=action.target_sweep,
            dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
            trigger_context_hash=action.trigger_context_hash,
            anchor_hash=action.anchor_hash,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("current_fe", 300_001, "checkpoint_fe"),
        ("current_sweep", 3, "target_sweep"),
        ("current_sweep", 5, "TTL expired"),
        ("dispatch_checkpoint_hash", "f" * 64, "dispatch_checkpoint_hash mismatch"),
        ("trigger_context_hash", "f" * 64, "trigger_context_hash mismatch"),
        ("anchor_hash", "f" * 64, "anchor_hash mismatch"),
    ],
)
def test_execution_state_start_fails_closed_on_context_mismatch(
    field: str,
    value: object,
    message: str,
) -> None:
    action = _action()
    execution = GcbExecutionState.for_action(action)
    issued_hash = execution.state_hash(action)
    context: dict[str, object] = {
        "current_fe": action.checkpoint_fe,
        "current_sweep": action.target_sweep,
        "dispatch_checkpoint_hash": action.dispatch_checkpoint_hash,
        "trigger_context_hash": action.trigger_context_hash,
        "anchor_hash": action.anchor_hash,
    }
    context[field] = value

    with pytest.raises(ValueError, match=message):
        execution.start(action, **context)  # type: ignore[arg-type]

    assert execution.status == "issued"
    assert execution.started_fe is None
    assert execution.state_hash(action) == issued_hash


def test_execution_state_rejects_action_mismatch_and_fe_drift() -> None:
    action = _action(budget_fes=96)
    other = _action(budget_fes=120)
    execution = GcbExecutionState.for_action(action)

    with pytest.raises(ValueError, match="action_hash"):
        execution.validate_for(other)

    execution.start(
        action,
        current_fe=action.checkpoint_fe,
        current_sweep=action.target_sweep,
        dispatch_checkpoint_hash=action.dispatch_checkpoint_hash,
        trigger_context_hash=action.trigger_context_hash,
        anchor_hash=action.anchor_hash,
    )
    with pytest.raises(ValueError, match="frozen FE budget"):
        execution.complete(
            action,
            consumed_fes=95,
            completed_fe=300_095,
            final_state_hash=_hash("d"),
        )
    with pytest.raises(ValueError, match="completed_fe"):
        execution.complete(
            action,
            consumed_fes=96,
            completed_fe=300_095,
            final_state_hash=_hash("d"),
        )


def test_phase_boundary_lifecycle_validates_scope_and_consumes_once() -> None:
    action = _action(
        budget_fes=24,
        trigger_scope=TRIGGER_SCOPE_PHASE_BOUNDARY,
    )
    execution = GcbExecutionState.for_action(action)
    start_context = {
        "current_fe": action.checkpoint_fe,
        "current_sweep": action.target_sweep,
        "dispatch_checkpoint_hash": action.dispatch_checkpoint_hash,
        "trigger_context_hash": action.trigger_context_hash,
        "anchor_hash": action.anchor_hash,
    }

    with pytest.raises(ValueError, match="trigger_scope mismatch"):
        execution.start(
            action,
            **start_context,
            trigger_scope=TRIGGER_SCOPE_RELATION_DISPATCH,
        )

    assert execution.status == "issued"
    execution.start(
        action,
        **start_context,
        trigger_scope=TRIGGER_SCOPE_PHASE_BOUNDARY,
    )
    execution.complete(
        action,
        consumed_fes=action.budget_fes,
        completed_fe=action.checkpoint_fe + action.budget_fes,
        final_state_hash=_hash("d"),
    )

    assert execution.status == "completed"
    assert execution.consumed_fes == 24
    with pytest.raises(ValueError, match="only an issued action can start"):
        execution.start(
            action,
            **start_context,
            trigger_scope=TRIGGER_SCOPE_PHASE_BOUNDARY,
        )
