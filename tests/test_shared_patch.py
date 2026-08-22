"""Gate P0 contract tests for the Stateful Shared-Patch Kernel (12 checks)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.overlap import LocalProposal, OverlapStructure
from arac.coordination.shared_patch import K_PATCH_FES, SharedPatchKernel
from arac.coordination.state import CoordinatorState, OC_STATE_SCHEMA
from arac.runtime.ledger import EvaluationLedger


def _toy(shared_gap: float = 2.0):
    """Sphere with one shared variable (2) owned by groups 0 and 1."""

    structure = OverlapStructure(5, ((0, 1, 2), (2, 3, 4)))

    def objective(values):
        converted = np.asarray(values, dtype=float)
        shared = converted[..., 2]
        rest = converted - np.array([0.0, 0.0, 0.0, 1.0, -1.0])
        return np.sum(rest**2, axis=-1) + shared**2 + (shared - shared_gap) ** 2 * 0.25

    problem = OptimizationProblem(
        objective=objective,
        dimension=5,
        lower_bounds=(-10.0,) * 5,
        upper_bounds=(10.0,) * 5,
    )
    ledger = EvaluationLedger(
        problem, 500, initial_incumbent=(3.0, 2.0, 1.0, -2.0, 2.0), initial_error=100.0
    )
    proposals = (
        LocalProposal(
            group=0,
            values=((0, 0.5), (1, 0.2), (2, 1.5)),
            improvement=5.0,
            uncertainty=((0, 0.1), (1, 0.1), (2, 0.4)),
        ),
        LocalProposal(
            group=1,
            values=((2, -0.5), (3, 1.0), (4, -1.0)),
            improvement=3.0,
            uncertainty=((2, 0.6), (3, 0.1), (4, 0.1)),
        ),
    )
    return structure, ledger, proposals


def test_exact_fe_and_strict_best_monotone() -> None:
    structure, ledger, proposals = _toy()
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "ctx-a", structure=structure, ledger=ledger, seed=11, mode="full"
    )
    assert result.consumed_fes == K_PATCH_FES == 8
    assert ledger.count == 8
    assert result.best_error_after <= result.best_error_before + 1e-12
    assert result.budget_status == "executed"


def test_candidates_within_bounds() -> None:
    structure, ledger, proposals = _toy(shared_gap=19.0)  # huge disagreement
    kernel = SharedPatchKernel()
    before = ledger.best_x.copy()
    result = kernel.apply(
        (0, 1), proposals, (2,), "ctx", structure=structure, ledger=ledger, seed=3, mode="full"
    )
    # ledger.evaluate would raise on out-of-bounds; success implies in-bounds
    assert result.consumed_fes == 8
    assert np.all(ledger.best_x <= ledger.problem.upper_array)
    del before


def test_budget_unavailable_is_explicit() -> None:
    structure, _, proposals = _toy()
    problem = OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=5,
        lower_bounds=(-10.0,) * 5,
        upper_bounds=(10.0,) * 5,
    )
    tiny = EvaluationLedger(problem, 3, initial_incumbent=(0.0,) * 5, initial_error=1.0)
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "ctx", structure=structure, ledger=tiny, seed=1, mode="full"
    )
    assert result.budget_status == "patch_budget_unavailable"
    assert result.consumed_fes == 0
    assert tiny.count == 0


def test_u_never_enters_candidate_direction() -> None:
    structure, ledger_a, proposals = _toy()
    kernel_a = SharedPatchKernel()
    result_a = kernel_a.apply(
        (0, 1), proposals, (2,), "ctx", structure=structure, ledger=ledger_a, seed=7, mode="state"
    )
    # manually inflate u, rerun with same seed/context: candidate errors identical
    for state in kernel_a.variables.values():
        state.u = 3.9
    ledger_b = dataclasses.replace(ledger_a) if False else None
    # rebuild identical ledger deterministically by replaying same operations
    structure2, ledger2, proposals2 = _toy()
    kernel_b = SharedPatchKernel()
    result_first = kernel_b.apply(
        (0, 1), proposals2, (2,), "ctx", structure=structure2, ledger=ledger2, seed=7, mode="state"
    )
    # u is restored from kernel_a's inflated value onto b via load
    kernel_b.load(kernel_a.payload())
    structure3, ledger3, proposals3 = _toy()
    result_second = kernel_b.apply(
        (0, 1), proposals3, (2,), "ctx", structure=structure3, ledger=ledger3, seed=7, mode="state"
    )
    del result_a, result_first, ledger_b
    assert result_second.candidate_trace[0].errors == pytest.approx(
        _replay_first_errors(), abs=1e-9
    )


def _replay_first_errors() -> tuple[float, float]:
    structure, ledger, proposals = _toy()
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "ctx", structure=structure, ledger=ledger, seed=7, mode="state"
    )
    return result.candidate_trace[0].errors


def test_context_reset_reinitializes_and_decays() -> None:
    structure, ledger, proposals = _toy()
    kernel = SharedPatchKernel()
    first = kernel.apply(
        (0, 1), proposals, (2,), "ctx-1", structure=structure, ledger=ledger, seed=5, mode="full"
    )
    state_var = kernel.variables[2]
    u_before = state_var.u
    assert first.context_reset is False
    second = kernel.apply(
        (0, 1), proposals, (2,), "ctx-2", structure=structure, ledger=ledger, seed=5, mode="full"
    )
    assert second.context_reset is True
    assert kernel.reset_count == 1
    assert kernel.variables[2].u <= u_before * 0.25 + 1.0 + 1e-9
    # re-anchored first patch still runs (not disabled)
    assert second.consumed_fes == 8


def test_radius_expands_on_success_and_shrinks_on_failure() -> None:
    structure, ledger, proposals = _toy(shared_gap=0.0)  # easy: consensus at 0
    kernel = SharedPatchKernel()
    base = None
    # force acceptance by placing incumbent far from optimum on the shared var
    ledger = EvaluationLedger(
        ledger.problem, 500, initial_incumbent=(0.0, 0.0, 8.0, 0.0, 0.0), initial_error=float(
            ledger.problem.objective(np.asarray([0.0, 0.0, 8.0, 0.0, 0.0]))
        )
    )
    proposals = (
        LocalProposal(
            group=0,
            values=((0, 0.0), (1, 0.0), (2, 0.1)),
            improvement=5.0,
            uncertainty=((0, 0.05), (1, 0.05), (2, 0.05)),
        ),
        LocalProposal(
            group=1,
            values=((2, -0.1), (3, 0.0), (4, 0.0)),
            improvement=3.0,
            uncertainty=((2, 0.05), (3, 0.05), (4, 0.05)),
        ),
    )
    result = kernel.apply(
        (0, 1), proposals, (2,), "ctx", structure=structure, ledger=ledger, seed=2, mode="full"
    )
    base = kernel.variables[2].base_radius
    if any(trace.accepted for trace in result.candidate_trace):
        assert max(result.radius_trace) > kernel.variables[2].base_radius * 1.01
    else:
        assert max(result.radius_trace) < base
    # deterministic failure shrink: a flat objective never accepts
    flat_problem = OptimizationProblem(
        objective=lambda x: np.ones(np.asarray(x).shape[0] if np.asarray(x).ndim > 1 else 1),
        dimension=5,
        lower_bounds=(-10.0,) * 5,
        upper_bounds=(10.0,) * 5,
    )
    flat_ledger = EvaluationLedger(
        flat_problem, 500, initial_incumbent=(0.0, 0.0, 0.0, 0.0, 0.0), initial_error=1.0
    )
    kernel2 = SharedPatchKernel()
    kernel2.apply(
        (0, 1), proposals, (2,), "ctx", structure=structure, ledger=flat_ledger, seed=2, mode="full"
    )
    assert kernel2.variables[2].radius < kernel2.variables[2].base_radius


def test_state_hash_reproducible() -> None:
    hashes = []
    for _ in range(2):
        structure, ledger, proposals = _toy()
        kernel = SharedPatchKernel()
        kernel.apply(
            (0, 1), proposals, (2,), "ctx", structure=structure, ledger=ledger, seed=9, mode="full"
        )
        hashes.append(kernel.state_hash())
    assert hashes[0] == hashes[1]


def test_invalid_scope_fails_closed() -> None:
    structure, ledger, proposals = _toy()
    kernel = SharedPatchKernel()
    with pytest.raises(ValueError):
        kernel.apply(
            (0, 1), proposals, (0,), "ctx", structure=structure, ledger=ledger, seed=1, mode="full"
        )
    with pytest.raises(ValueError):
        kernel.apply(
            (0, 1), proposals, (2,), "ctx", structure=structure, ledger=ledger, seed=1, mode="nope"
        )


def test_state_schema_v2_and_v1_restore() -> None:
    structure, ledger, _ = _toy()
    from arac.coordination.contract import OcCoordinatorConfig

    state = CoordinatorState(
        structure, [(0, 1, 2), (2, 3, 4)], config=OcCoordinatorConfig(), checkpoint_hash=""
    )
    state.shared_patch = {"context_hash": "abc", "reset_count": 2, "variables": {}}
    snapshot = state.snapshot()
    assert b"shared_patch" in snapshot.payload
    state.restore(snapshot)
    assert state.shared_patch["reset_count"] == 2
    # v1 snapshot (no shared_patch) restores into v2 with empty patch state
    import json

    v1_payload = {
        key: value for key, value in json.loads(snapshot.payload.decode()).items()
        if key not in ("schema_version", "shared_patch")
    }
    v1_payload["schema_version"] = "arac-oc-coordinator-state-v1"
    import hashlib

    from arac.coordination.state import OcStateSnapshot

    v1_bytes = json.dumps(v1_payload, sort_keys=True, separators=(",", ":")).encode()
    v1_snapshot = OcStateSnapshot(
        payload=v1_bytes, state_hash=hashlib.sha256(v1_bytes).hexdigest()
    )
    state.shared_patch = {"stale": True}
    state.restore(v1_snapshot)
    assert state.shared_patch == {}
    assert OC_STATE_SCHEMA == "arac-oc-coordinator-state-v2"


def test_loop_integration_patch_on_off() -> None:
    import sys

    sys.path.insert(0, "tests")
    from test_oc_unified_loop import _PHASE1_KWARGS, _overlap_problem  # type: ignore
    problem = _overlap_problem()
    from arac.coordination.loop import run_oc_unified
    from arac.evidence import run_phase1_overlap_pilot

    pilot = run_phase1_overlap_pilot(problem, total_budget_fes=2_000, run_seed=55, **_PHASE1_KWARGS)
    base = run_oc_unified(problem, pilot, refresh_cycles=2, sense_budget_fes=16)
    patched = run_oc_unified(
        problem, pilot, refresh_cycles=2, sense_budget_fes=16, patch_config={"mode": "full"}
    )
    assert base.terminal_fes == patched.terminal_fes == 2_000
    patch_receipts = [
        receipt for receipt in patched.receipts if receipt.patch_enabled
    ]
    assert all(r.patch_lane_fes in (0, 8) for r in patch_receipts)
    assert all(r.patch_budget_status in ("executed", "patch_budget_unavailable") for r in patch_receipts)
