"""Gate P0 contract tests for the Stateful Shared-Patch Kernel (revised plan).

Covers docs/arac-oc-shared-patch-completion-plan.md section 10 (Gate P0):
bounds, exact FE, strict-best, no incumbent re-billing, u never steering
candidates, LOCAL context hash semantics (self-accept no reset / external
change reset / scope change reason), conforming silence, radius dynamics,
u update on budget-unavailable, state schema v2 with v1 restore, fail-closed
errors, and the loop-level integration.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from arac.benchmarks.aob import OptimizationProblem
from arac.coordination.overlap import LocalProposal, OverlapStructure
from arac.coordination.shared_patch import (
    EPS,
    K_PATCH_FES,
    RESET_REASONS,
    SharedPatchKernel,
    patch_stable_hash,
)
from arac.coordination.state import CoordinatorState, OC_STATE_SCHEMA
from arac.runtime.ledger import EvaluationLedger


def _problem(dimension: int = 6):
    return OptimizationProblem(
        objective=lambda x: np.sum(np.asarray(x, dtype=float) ** 2, axis=-1),
        dimension=dimension,
        lower_bounds=(-10.0,) * dimension,
        upper_bounds=(10.0,) * dimension,
    )


def _two_shared(shared_gap: float = 2.0):
    """Groups (0,1,2) and (2,3,4) share variable 2; groups (2,3,4) and (4,5) share 4."""

    structure = OverlapStructure(6, ((0, 1, 2), (2, 3, 4), (4, 5)))

    def objective(values):
        converted = np.asarray(values, dtype=float)
        shared = converted[..., 2]
        other = converted[..., 4]
        base = np.sum(converted**2, axis=-1)
        return base + shared_gap * shared**2 + other**2 * 0.5

    problem = OptimizationProblem(
        objective=objective,
        dimension=6,
        lower_bounds=(-10.0,) * 6,
        upper_bounds=(10.0,) * 6,
    )
    ledger = EvaluationLedger(
        problem,
        500,
        initial_incumbent=(3.0, 2.0, 1.0, -2.0, 2.0, 1.0),
        initial_error=100.0,
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
            values=((2, -0.5), (3, 1.0), (4, 1.2)),
            improvement=3.0,
            uncertainty=((2, 0.6), (3, 0.1), (4, 0.5)),
        ),
        LocalProposal(
            group=2,
            values=((4, 0.8), (5, -0.3)),
            improvement=1.0,
            uncertainty=((4, 0.3), (5, 0.1)),
        ),
    )
    return structure, ledger, proposals


def test_exact_fe_strict_best_no_incumbent_rebilling() -> None:
    structure, ledger, proposals = _two_shared()
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "stable-1", structure=structure, ledger=ledger,
        seed=11, mode="full",
    )
    assert result.consumed_fes == K_PATCH_FES == 8
    assert ledger.count == 8  # nothing else billed, incumbent not re-evaluated
    assert result.best_error_after <= result.best_error_before + 1e-12
    assert result.budget_status == "executed"


def test_candidates_within_bounds() -> None:
    structure, ledger, proposals = _two_shared(shared_gap=19.0)
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
        seed=3, mode="full",
    )
    assert result.consumed_fes == 8  # ledger.evaluate raises on out-of-bounds


def test_u_never_enters_candidate_direction() -> None:
    def run(inflate_u: bool):
        structure, ledger, proposals = _two_shared()
        kernel = SharedPatchKernel()
        kernel.apply(
            (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
            seed=7, mode="state",
        )
        if inflate_u:
            for state in kernel.variables.values():
                state.u = 3.9
        structure2, ledger2, proposals2 = _two_shared()
        second = kernel.apply(
            (0, 1), proposals2, (2,), "stable", structure=structure2, ledger=ledger2,
            seed=7, mode="state",
        )
        return second

    normal = run(inflate_u=False)
    inflated = run(inflate_u=True)
    assert normal.candidate_trace == inflated.candidate_trace
    assert inflated.u_trace != ()  # u is tracked, but never steers candidates


def test_local_context_self_accept_does_not_reset() -> None:
    structure, ledger, proposals = _two_shared()
    kernel = SharedPatchKernel()
    first = kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
        seed=5, mode="full",
    )
    # second visit with the SAME stable hash: even though the incumbent may
    # have moved (the scope write set is excluded from the local hash), no reset
    second = kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
        seed=6, mode="full",
    )
    assert second.context_reset is False
    assert second.context_reset_reason == "none"
    assert first.context_reset is False


def test_external_context_change_resets() -> None:
    structure, ledger, proposals = _two_shared()
    kernel = SharedPatchKernel()
    kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
        seed=5, mode="full",
    )
    # externally move a NON-write-set coordinate (variable 5 is outside the
    # component and the scope) and rebuild the ledger from that incumbent
    incumbent = np.array(ledger.best_x, dtype=float)
    incumbent[5] += 3.0
    new_error = float(ledger.problem.objective(incumbent))
    ledger2 = EvaluationLedger(
        ledger.problem, 500,
        initial_incumbent=tuple(float(v) for v in incumbent),
        initial_error=new_error,
    )
    result = kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger2,
        seed=5, mode="full",
    )
    assert result.context_reset is True
    assert result.context_reset_reason == "external_context_change"


def test_scope_change_reason() -> None:
    structure, ledger, proposals = _two_shared()
    kernel = SharedPatchKernel()
    kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
        seed=5, mode="full",
    )
    result = kernel.apply(
        (0, 1, 2), proposals, (2, 4), "stable", structure=structure, ledger=ledger,
        seed=5, mode="full",
    )
    assert result.context_reset is True
    assert result.context_reset_reason == "scope_change"


def test_conforming_silence_collapses_radius() -> None:
    structure, ledger, _ = _two_shared()
    conforming = (
        LocalProposal(
            group=0,
            values=((0, 0.1), (1, 0.1), (2, 0.7)),
            improvement=5.0,
            uncertainty=((0, 0.05), (1, 0.05), (2, 0.05)),
        ),
        LocalProposal(
            group=1,
            values=((2, 0.7), (3, 0.1), (4, 0.1)),
            improvement=3.0,
            uncertainty=((2, 0.05), (3, 0.05), (4, 0.05)),
        ),
        LocalProposal(
            group=2, values=((4, 0.1), (5, 0.1)), improvement=1.0,
            uncertainty=((4, 0.05), (5, 0.05)),
        ),
    )
    kernel = SharedPatchKernel()
    kernel.apply(
        (0, 1), conforming, (2,), "stable", structure=structure, ledger=ledger,
        seed=1, mode="full",
    )
    assert kernel.variables[2].base_radius == pytest.approx(EPS)


def test_radius_expands_on_success_and_shrinks_on_failure() -> None:
    structure = OverlapStructure(6, ((0, 1, 2), (2, 3, 4), (4, 5)))
    problem = _problem(6)
    ledger = EvaluationLedger(
        problem, 500, initial_incumbent=(0.0, 0.0, 8.0, 0.0, 0.0, 0.0),
        initial_error=float(problem.objective(np.asarray([0.0, 0.0, 8.0, 0.0, 0.0, 0.0]))),
    )
    proposals = (
        LocalProposal(
            group=0, values=((0, 0.0), (1, 0.0), (2, 0.1)), improvement=5.0,
            uncertainty=((0, 0.05), (1, 0.05), (2, 0.05)),
        ),
        LocalProposal(
            group=1, values=((2, -0.1), (3, 0.0), (4, 0.0)), improvement=3.0,
            uncertainty=((2, 0.05), (3, 0.05), (4, 0.05)),
        ),
    )
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
        seed=2, mode="full",
    )
    base = kernel.variables[2].base_radius
    if any(trace.accepted for trace in result.candidate_trace):
        assert max(result.radius_trace) > base * 1.01
    else:
        assert max(result.radius_trace) < base
    flat = OptimizationProblem(
        objective=lambda x: np.ones(1 if np.asarray(x).ndim == 1 else np.asarray(x).shape[0]),
        dimension=6, lower_bounds=(-10.0,) * 6, upper_bounds=(10.0,) * 6,
    )
    flat_ledger = EvaluationLedger(
        flat, 500, initial_incumbent=(0.0,) * 6, initial_error=1.0
    )
    kernel2 = SharedPatchKernel()
    kernel2.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=flat_ledger,
        seed=2, mode="full",
    )
    assert kernel2.variables[2].radius < kernel2.variables[2].base_radius


def test_budget_unavailable_still_updates_u_not_zr() -> None:
    structure, _, proposals = _two_shared()
    tiny = EvaluationLedger(
        _problem(6), 3, initial_incumbent=(0.0,) * 6, initial_error=1.0
    )
    kernel = SharedPatchKernel()
    result = kernel.apply(
        (0, 1), proposals, (2,), "stable", structure=structure, ledger=tiny,
        seed=1, mode="full",
    )
    assert result.budget_status == "patch_budget_unavailable"
    assert result.consumed_fes == 0
    assert tiny.count == 0
    assert kernel.variables[2].u > 0.0  # u still updated
    assert kernel.variables[2].radius == kernel.variables[2].base_radius  # z/r untouched


def test_state_hash_reproducible_and_modes_validated() -> None:
    hashes = []
    for _ in range(2):
        structure, ledger, proposals = _two_shared()
        kernel = SharedPatchKernel()
        kernel.apply(
            (0, 1), proposals, (2,), "stable", structure=structure, ledger=ledger,
            seed=9, mode="full",
        )
        hashes.append(kernel.state_hash())
    assert hashes[0] == hashes[1]
    kernel = SharedPatchKernel()
    with pytest.raises(ValueError):
        kernel.apply(
            (0, 1), proposals, (0,), "s", structure=structure, ledger=None,  # type: ignore[arg-type]
            seed=1, mode="full",
        )


def test_state_schema_v2_and_v1_restore() -> None:
    from arac.coordination.contract import OcCoordinatorConfig
    from arac.coordination.state import OcStateSnapshot

    structure, _, _ = _two_shared()
    state = CoordinatorState(
        structure, [(0, 1, 2), (2, 3, 4), (4, 5)],
        config=OcCoordinatorConfig(), checkpoint_hash="",
    )
    state.shared_patch = {"stable_hash": "abc", "reset_count": 2, "variables": {}}
    snapshot = state.snapshot()
    assert b"shared_patch" in snapshot.payload
    state.restore(snapshot)
    assert state.shared_patch["reset_count"] == 2

    v1_payload = {
        key: value
        for key, value in json.loads(snapshot.payload.decode()).items()
        if key not in ("schema_version", "shared_patch")
    }
    v1_payload["schema_version"] = "arac-oc-coordinator-state-v1"
    v1_bytes = json.dumps(v1_payload, sort_keys=True, separators=(",", ":")).encode()
    v1_snapshot = OcStateSnapshot(
        payload=v1_bytes, state_hash=hashlib.sha256(v1_bytes).hexdigest()
    )
    state.shared_patch = {"stale": True}
    state.restore(v1_snapshot)
    assert state.shared_patch == {}
    assert OC_STATE_SCHEMA == "arac-oc-coordinator-state-v2"
    assert "external_context_change" in RESET_REASONS
    assert len(patch_stable_hash("a", "b", "c")) == 64


def test_loop_integration_patch_on_off() -> None:
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
    patch_receipts = [r for r in patched.receipts if r.patch_enabled]
    assert all(r.patch_lane_fes in (0, 8) for r in patch_receipts)
    assert all(
        r.patch_budget_status in ("executed", "patch_budget_unavailable")
        for r in patch_receipts
    )
    assert all(r.context_reset_reason in RESET_REASONS + ("",) for r in patch_receipts)
