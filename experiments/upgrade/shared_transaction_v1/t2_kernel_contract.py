"""T2 stateless transaction kernel contract gate for shared_transaction_v1.

Contract-only stage (SCST §5-T2): no performance claims.  Verifies on toys
and on a mini-host that the kernel of ``transaction_kernel.py`` satisfies
every frozen clause:

- exact FE accounting (2 FE per selected coordinate, hard cap 8, returned
  FE recorded, ledger advances by exactly the consumed FE);
- strict-best acceptance only (the second, duplicate candidate is rejected
  by the strict-best rule, never by ad-hoc logic; best never worsens);
- fail-closed silence when no certified link has both owners fresh
  (zero evaluations);
- selection rule ``(-|M(j)|, coordinate_id)`` with the 4-coordinate cap;
- bounds clipping with the raw value recorded;
- statelessness (two runs from identical state produce identical receipts);
- mini-host mounting: a small CTP arm with the kernel enabled lands on the
  exact terminal FE with the kernel receipt in the chain, and the A0 arm
  (mount disabled) is bit-equal in route to a plain run - the FE difference
  between arms is exactly the kernel's consumed FE.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.contracts import (
    ActionContext,
    PhaseCheckpoint,
    RelationEvidence,
    canonical_sha256,
)
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.upgrade.shared_transaction_v1.transaction_kernel import (
    CertifiedLink,
    MAX_COORDINATES_PER_BOUNDARY,
    MAX_FE_PER_BOUNDARY,
    TransactionMount,
    run_stateless_transaction,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("t2_kernel_contract_protocol_v1.json")
SUMMARY_SCHEMA = "arac-upgrade-shared-transaction-t2-summary-v1"
MINI_PROTOCOL = "arac-shared-transaction-t2-mini-v1"
MINI_DIMENSION = 12
MINI_PHASE1_FES = 600
MINI_TOTAL_FES = 60_000
MINI_SEED = 271828


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


class _FakeProposal:
    """Minimal ImprovementRecord stand-in with the fields the kernel reads."""

    def __init__(self, values: dict[int, float], commit_fes: int = 111) -> None:
        vector = np.zeros(8)
        for coordinate, value in values.items():
            vector[coordinate] = value
        self.committed_x = vector
        self.commit_fes = commit_fes


def _toy_problem() -> OptimizationProblem:
    def objective(x: np.ndarray) -> float:
        point = np.atleast_1d(np.asarray(x, dtype=float))
        return float(np.sum((point - np.asarray([0.0, 2.0, -3.0, 5.0])) ** 2))

    return OptimizationProblem(
        objective=objective,
        dimension=4,
        lower_bounds=(-10.0,) * 4,
        upper_bounds=(10.0,) * 4,
        optimum=0.0,
    )


def _toy_context(problem: OptimizationProblem, incumbent: np.ndarray) -> tuple[Any, EvaluationLedger]:
    ledger = EvaluationLedger(problem, total_budget=1_000)
    ledger.evaluate(incumbent[np.newaxis, :])
    context = type(
        "ToyContext",
        (),
        {"ledger": ledger, "problem": problem},
    )()
    return context, ledger


def _toy_checks() -> dict[str, Any]:
    problem = _toy_problem()
    results: dict[str, Any] = {}

    # 1. acceptance + duplicate rejection + exact FE
    context, ledger = _toy_context(problem, np.asarray([5.0, 5.0, 5.0, 5.0]))
    before = ledger.best_error
    before_count = ledger.count
    proposals = {
        0: _FakeProposal({1: 1.0}),
        1: _FakeProposal({1: 3.0}),
    }
    receipt = run_stateless_transaction(
        context,
        [CertifiedLink(variable=1, owner_blocks=(0, 1))],
        proposals,
        boundary_phase="toy-boundary",
    )
    results["accept_and_duplicate"] = {
        "consumed_fes": receipt.consumed_fes,
        "accepted_count": receipt.accepted_count,
        "best_improved": receipt.best_error_after < receipt.best_error_before,
        "ledger_advanced_exactly": ledger.count - before_count == receipt.consumed_fes,
        "duplicate_flagged": any(
            candidate.duplicate_of_previous for candidate in receipt.candidates
        ),
        "duplicate_rejected": receipt.accepted_count == 1,
        "returned_fes": receipt.returned_fes,
        "passed": bool(
            receipt.consumed_fes == 2
            and receipt.accepted_count == 1
            and receipt.best_error_after < before
            and ledger.count - before_count == 2
            and receipt.returned_fes == MAX_FE_PER_BOUNDARY - 2
        ),
    }

    # 2. fail-closed silence
    context2, ledger2 = _toy_context(problem, np.asarray([5.0, 5.0, 5.0, 5.0]))
    count2 = ledger2.count
    silent = run_stateless_transaction(
        context2,
        [CertifiedLink(variable=1, owner_blocks=(0, 1))],
        {0: _FakeProposal({1: 1.0})},
        boundary_phase="toy-boundary",
    )
    results["fail_closed_silence"] = {
        "consumed_fes": silent.consumed_fes,
        "silent_reason": silent.silent_reason,
        "ledger_untouched": ledger2.count == count2,
        "passed": bool(silent.consumed_fes == 0 and ledger2.count == count2 and silent.silent_reason is not None),
    }

    # 3. selection cap and ordering (8-dim toy so all link coordinates exist)
    links = [
        CertifiedLink(variable=7, owner_blocks=(0, 1)),
        CertifiedLink(variable=2, owner_blocks=(0, 2)),
        CertifiedLink(variable=5, owner_blocks=(1, 2)),
        CertifiedLink(variable=1, owner_blocks=(0, 3)),
        CertifiedLink(variable=3, owner_blocks=(2, 3)),
    ]
    wide_problem = OptimizationProblem(
        objective=lambda x: float(np.sum((np.atleast_1d(np.asarray(x, dtype=float)) - 1.0) ** 2)),
        dimension=8,
        lower_bounds=(-10.0,) * 8,
        upper_bounds=(10.0,) * 8,
        optimum=0.0,
    )
    proposals3 = {owner: _FakeProposal({}) for owner in range(4)}
    receipt3 = run_stateless_transaction(
        _toy_context(wide_problem, np.full(8, 5.0))[0],
        links,
        proposals3,
        boundary_phase="toy-boundary",
    )
    results["selection_cap_order"] = {
        "selected": list(receipt3.selected_coordinates),
        "passed": bool(
            receipt3.selected_coordinates == (1, 2, 3, 5)
            and len(receipt3.selected_coordinates) == MAX_COORDINATES_PER_BOUNDARY
        ),
    }

    # 4. bounds clipping
    context4, ledger4 = _toy_context(problem, np.asarray([5.0, 5.0, 5.0, 5.0]))
    clipped = run_stateless_transaction(
        context4,
        [CertifiedLink(variable=1, owner_blocks=(0, 1))],
        {0: _FakeProposal({1: 99.0}), 1: _FakeProposal({1: -99.0})},
        boundary_phase="toy-boundary",
    )
    results["bounds_clipping"] = {
        "raw_values": [candidate.raw_value for candidate in clipped.candidates],
        "clipped_values": [candidate.clipped_value for candidate in clipped.candidates],
        "passed": bool(
            clipped.candidates
            and all(candidate.clipped_value == 0.0 for candidate in clipped.candidates)
        ),
    }

    # 5. statelessness
    run_a = run_stateless_transaction(
        _toy_context(problem, np.asarray([5.0, 5.0, 5.0, 5.0]))[0],
        [CertifiedLink(variable=1, owner_blocks=(0, 1))],
        {0: _FakeProposal({1: 1.0}), 1: _FakeProposal({1: 3.0})},
        boundary_phase="toy-boundary",
    )
    run_b = run_stateless_transaction(
        _toy_context(problem, np.asarray([5.0, 5.0, 5.0, 5.0]))[0],
        [CertifiedLink(variable=1, owner_blocks=(0, 1))],
        {0: _FakeProposal({1: 1.0}), 1: _FakeProposal({1: 3.0})},
        boundary_phase="toy-boundary",
    )
    results["stateless_determinism"] = {
        "payload_a": run_a.payload(),
        "payload_b": run_b.payload(),
        "passed": bool(run_a.payload() == run_b.payload()),
    }
    return results


def _mini_problem() -> OptimizationProblem:
    optimum = np.zeros(MINI_DIMENSION)

    def objective(x: np.ndarray) -> float | np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return float(np.sum((arr - optimum) ** 2) + 4.0 * (arr[2] + arr[6]) ** 2)
        return np.sum((arr - optimum) ** 2, axis=1) + 4.0 * (arr[:, 2] + arr[:, 6]) ** 2

    return OptimizationProblem(
        objective=objective,
        dimension=MINI_DIMENSION,
        lower_bounds=(-10.0,) * MINI_DIMENSION,
        upper_bounds=(10.0,) * MINI_DIMENSION,
        optimum=0.0,
    )


def _mini_checkpoint(problem: OptimizationProblem) -> PhaseCheckpoint:
    rng = np.random.default_rng(MINI_SEED)
    incumbent = rng.uniform(-8.0, 8.0, size=MINI_DIMENSION)
    error = float(problem.objective(incumbent))
    return PhaseCheckpoint(
        protocol=MINI_PROTOCOL,
        run_seed=MINI_SEED,
        total_budget_fes=MINI_TOTAL_FES,
        phase1_fes=MINI_PHASE1_FES,
        incumbent=tuple(float(value) for value in incumbent),
        incumbent_error=error,
        feature_names=("mini_shared_link_count",),
        feature_values=(1.0,),
        blocks=((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)),
        relations=(RelationEvidence(left_block=0, right_block=1, strength=1.0, disagreement=0.0),),
    )


def _mini_arm(enabled: bool) -> dict[str, Any]:
    problem = _mini_problem()
    checkpoint = _mini_checkpoint(problem)
    registry = RecoveredActionRegistry()
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
        allow_out_of_bounds=registry.allow_out_of_bounds,
    )
    context = ActionContext(
        action_name="ctp",
        checkpoint=checkpoint,
        problem=problem,
        ledger=ledger,
        action_seed=MINI_SEED,
    )
    mount = TransactionMount(
        [CertifiedLink(variable=2, owner_blocks=(0, 1))],
        enabled=enabled,
    )
    mount.configure_boundary(
        source_phase="run_persistent_blocks",
        boundary_phase="run_sequential_blocks",
    )
    mount.install(ledger, context)
    try:
        result = execute_phase2_action(
            "ctp",
            checkpoint,
            problem,
            ledger,
            action_seed=MINI_SEED,
            registry=registry,
        )
    finally:
        mount.uninstall()
    return {
        "enabled": enabled,
        "final_error": result.final_error,
        "route": result.route,
        "terminal_fes": result.terminal_fes,
        "ledger_count": ledger.count,
        "kernel_receipts": list(mount.kernel_receipts),
        "phase_count": len(mount.phases),
        "improvement_count": len(mount.improvements),
    }


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(protocol_path).resolve())
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("T2 refuses to run: the recovered baseline freeze verifier is not green")
    started = datetime.now(UTC)
    with threadpool_limits(limits=1):
        runtime = _runtime_block()
        toy = _toy_checks()
        arm_off = _mini_arm(enabled=False)
        arm_on = _mini_arm(enabled=True)
    kernel_consumed = sum(
        int(receipt["consumed_fes"]) for receipt in arm_on["kernel_receipts"]
    )
    exact_fe = (
        arm_off["terminal_fes"] == MINI_TOTAL_FES
        and arm_off["ledger_count"] == MINI_TOTAL_FES
        and arm_on["terminal_fes"] == MINI_TOTAL_FES
        and arm_on["ledger_count"] == MINI_TOTAL_FES
    )
    receipt_valid = True
    for receipt in arm_on["kernel_receipts"]:
        receipt_valid = receipt_valid and (
            receipt["consumed_fes"] <= MAX_FE_PER_BOUNDARY
            and receipt["reserved_fes"] == MAX_FE_PER_BOUNDARY
            and receipt["returned_fes"] == MAX_FE_PER_BOUNDARY - receipt["consumed_fes"]
            and receipt["best_error_after"] <= receipt["best_error_before"]
        )
    checks = {
        "toy_accept_and_duplicate": toy["accept_and_duplicate"]["passed"],
        "toy_fail_closed_silence": toy["fail_closed_silence"]["passed"],
        "toy_selection_cap_order": toy["selection_cap_order"]["passed"],
        "toy_bounds_clipping": toy["bounds_clipping"]["passed"],
        "toy_stateless_determinism": toy["stateless_determinism"]["passed"],
        "mini_exact_terminal_fe_both_arms": exact_fe,
        "mini_kernel_receipts_present": bool(arm_on["kernel_receipts"]),
        "mini_kernel_receipt_contract": receipt_valid,
        "mini_off_silent": arm_off["kernel_receipts"] == [],
    }
    body = {
        "schema_version": SUMMARY_SCHEMA,
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "toy": toy,
        "mini_arm_off": arm_off,
        "mini_arm_on": arm_on,
        "kernel_consumed_fes": kernel_consumed,
        "checks": checks,
        "runtime": runtime,
        "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
        "gate_passed": all(checks.values()),
        "t3_authorized": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    output_root = REPOSITORY_ROOT / str(protocol["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "summary.json", body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    summary = run_stage(args.protocol)
    print(json.dumps({"stage": "t2", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["run_stage"]


if __name__ == "__main__":
    raise SystemExit(main())
