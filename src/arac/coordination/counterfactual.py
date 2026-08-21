"""One-FE frozen counterfactual receipts for overlap coordination.

The counterfactual is deliberately a diagnostic boundary.  A complete
candidate is already evaluated by the normal arbitration path; this module
evaluates only the same candidate with the selected shared scope frozen at
the pre-arbitration incumbent.  The temporary evaluation can never replace
the current archive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger

COUNTERFACTUAL_SCHEMA = "arac-oc-coupling-counterfactual-v1"
TWO_BASELINE_COUNTERFACTUAL_SCHEMA = "arac-oc-coupling-counterfactual-v2"


def _validate_tuple(values: tuple[int, ...], name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integer indices")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted")


@dataclass(frozen=True)
class CounterfactualCouplingReceipt:
    """Auditable local interaction estimate for one complete candidate."""

    component: tuple[int, ...]
    scope: tuple[int, ...]
    candidate_name: str
    best_error_before: float
    full_candidate_error: float
    frozen_candidate_error: float
    full_gain: float
    frozen_gain: float
    coupled_gain: float
    consumed_fes: int
    archive_preserved: bool
    schema_version: str = COUNTERFACTUAL_SCHEMA

    def __post_init__(self) -> None:
        _validate_tuple(self.component, "component")
        _validate_tuple(self.scope, "scope")
        if not isinstance(self.candidate_name, str) or not self.candidate_name:
            raise ValueError("candidate_name must be a non-empty string")
        for name in (
            "best_error_before",
            "full_candidate_error",
            "frozen_candidate_error",
            "full_gain",
            "frozen_gain",
            "coupled_gain",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.consumed_fes not in (0, 1):
            raise ValueError("counterfactual receipt must consume zero or one FE")
        if not isinstance(self.archive_preserved, bool):
            raise TypeError("archive_preserved must be a bool")
        if self.consumed_fes == 1 and not self.archive_preserved:
            raise ValueError("a counted counterfactual must preserve the archive")
        if self.schema_version != COUNTERFACTUAL_SCHEMA:
            raise ValueError("unsupported counterfactual schema")

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class TwoBaselineCouplingReceipt:
    """Auditable pure interaction estimate using private/shared baselines."""

    component: tuple[int, ...]
    scope: tuple[int, ...]
    candidate_name: str
    best_error_before: float
    full_candidate_error: float
    private_candidate_error: float
    shared_candidate_error: float
    full_gain: float
    private_gain: float
    shared_gain: float
    interaction_gain: float
    consumed_fes: int
    archive_preserved: bool
    schema_version: str = TWO_BASELINE_COUNTERFACTUAL_SCHEMA

    def __post_init__(self) -> None:
        _validate_tuple(self.component, "component")
        _validate_tuple(self.scope, "scope")
        if not isinstance(self.candidate_name, str) or not self.candidate_name:
            raise ValueError("candidate_name must be a non-empty string")
        for name in (
            "best_error_before",
            "full_candidate_error",
            "private_candidate_error",
            "shared_candidate_error",
            "full_gain",
            "private_gain",
            "shared_gain",
            "interaction_gain",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.consumed_fes != 2:
            raise ValueError("two-baseline counterfactual must consume exactly two FE")
        if not isinstance(self.archive_preserved, bool) or not self.archive_preserved:
            raise ValueError("a counted two-baseline counterfactual must preserve the archive")
        if self.schema_version != TWO_BASELINE_COUNTERFACTUAL_SCHEMA:
            raise ValueError("unsupported two-baseline counterfactual schema")

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256(self.payload())


def evaluate_frozen_private_counterfactual(
    ledger: EvaluationLedger,
    *,
    component: tuple[int, ...],
    scope: tuple[int, ...],
    incumbent: np.ndarray | tuple[float, ...],
    best_error_before: float,
    candidate_name: str,
    candidate: np.ndarray | tuple[float, ...],
    full_candidate_error: float,
) -> CounterfactualCouplingReceipt:
    """Evaluate a private-only counterfactual with one counted FE.

    ``candidate`` is the already evaluated complete proposal.  Only the
    coordinates in ``scope`` are reset to ``incumbent``.  The ledger archive
    is restored in a ``finally`` block, so this diagnostic cannot become a
    hidden acceptance path.
    """

    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be an EvaluationLedger")
    _validate_tuple(component, "component")
    _validate_tuple(scope, "scope")
    before = float(best_error_before)
    full_error = float(full_candidate_error)
    if not math.isfinite(before) or not math.isfinite(full_error):
        raise ValueError("counterfactual errors must be finite")
    if not isinstance(candidate_name, str) or not candidate_name:
        raise ValueError("candidate_name must be a non-empty string")

    base = np.asarray(incumbent, dtype=float)
    full = np.asarray(candidate, dtype=float)
    expected_shape = (ledger.problem.dimension,)
    if base.shape != expected_shape or full.shape != expected_shape:
        raise ValueError("incumbent and candidate must match the problem dimension")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(full)):
        raise ValueError("incumbent and candidate must be finite")
    lower = ledger.problem.lower_array
    upper = ledger.problem.upper_array
    if np.any(base < lower) or np.any(base > upper):
        raise ValueError("incumbent escaped the problem bounds")
    if np.any(full < lower) or np.any(full > upper):
        raise ValueError("candidate escaped the problem bounds")
    if any(variable >= ledger.problem.dimension for variable in scope):
        raise ValueError("scope contains a variable outside the problem dimension")

    frozen = full.copy()
    frozen[np.asarray(scope, dtype=int)] = base[np.asarray(scope, dtype=int)]
    archive = ledger.archive_snapshot()
    try:
        frozen_error = float(ledger.evaluate(frozen))
    finally:
        ledger.restore_archive(archive)

    full_gain = before - full_error
    frozen_gain = before - frozen_error
    return CounterfactualCouplingReceipt(
        component=component,
        scope=scope,
        candidate_name=candidate_name,
        best_error_before=before,
        full_candidate_error=full_error,
        frozen_candidate_error=frozen_error,
        full_gain=full_gain,
        frozen_gain=frozen_gain,
        coupled_gain=full_gain - frozen_gain,
        consumed_fes=1,
        archive_preserved=ledger.archive_snapshot() == archive,
    )


def evaluate_two_baseline_counterfactual(
    ledger: EvaluationLedger,
    *,
    component: tuple[int, ...],
    scope: tuple[int, ...],
    incumbent: np.ndarray | tuple[float, ...],
    best_error_before: float,
    candidate_name: str,
    candidate: np.ndarray | tuple[float, ...],
    full_candidate_error: float,
) -> TwoBaselineCouplingReceipt:
    """Evaluate private-only and shared-only baselines with two counted FE.

    The complete candidate was evaluated by normal arbitration.  The two
    additional rows isolate the private and shared marginal gains; their
    inclusion-exclusion residual is the pure second-order interaction gain.
    Both temporary evaluations are charged and the archive is restored.
    """

    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be an EvaluationLedger")
    _validate_tuple(component, "component")
    _validate_tuple(scope, "scope")
    before = float(best_error_before)
    full_error = float(full_candidate_error)
    if not math.isfinite(before) or not math.isfinite(full_error):
        raise ValueError("two-baseline errors must be finite")
    if not isinstance(candidate_name, str) or not candidate_name:
        raise ValueError("candidate_name must be a non-empty string")

    base = np.asarray(incumbent, dtype=float)
    full = np.asarray(candidate, dtype=float)
    expected_shape = (ledger.problem.dimension,)
    if base.shape != expected_shape or full.shape != expected_shape:
        raise ValueError("incumbent and candidate must match the problem dimension")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(full)):
        raise ValueError("incumbent and candidate must be finite")
    lower = ledger.problem.lower_array
    upper = ledger.problem.upper_array
    if np.any(base < lower) or np.any(base > upper):
        raise ValueError("incumbent escaped the problem bounds")
    if np.any(full < lower) or np.any(full > upper):
        raise ValueError("candidate escaped the problem bounds")
    if any(variable >= ledger.problem.dimension for variable in scope):
        raise ValueError("scope contains a variable outside the problem dimension")

    indices = np.asarray(scope, dtype=int)
    private_only = full.copy()
    private_only[indices] = base[indices]
    shared_only = base.copy()
    shared_only[indices] = full[indices]
    archive = ledger.archive_snapshot()
    start = ledger.count
    try:
        errors = np.asarray(
            ledger.evaluate(np.asarray((private_only, shared_only), dtype=float)),
            dtype=float,
        )
    finally:
        ledger.restore_archive(archive)
    consumed = ledger.count - start
    if consumed != 2:
        raise RuntimeError("two-baseline counterfactual FE accounting drifted")
    private_error, shared_error = (float(value) for value in errors)
    full_gain = before - full_error
    private_gain = before - private_error
    shared_gain = before - shared_error
    interaction_gain = full_gain - private_gain - shared_gain
    return TwoBaselineCouplingReceipt(
        component=component,
        scope=scope,
        candidate_name=candidate_name,
        best_error_before=before,
        full_candidate_error=full_error,
        private_candidate_error=private_error,
        shared_candidate_error=shared_error,
        full_gain=full_gain,
        private_gain=private_gain,
        shared_gain=shared_gain,
        interaction_gain=interaction_gain,
        consumed_fes=consumed,
        archive_preserved=ledger.archive_snapshot() == archive,
    )


__all__ = [
    "COUNTERFACTUAL_SCHEMA",
    "TWO_BASELINE_COUNTERFACTUAL_SCHEMA",
    "CounterfactualCouplingReceipt",
    "TwoBaselineCouplingReceipt",
    "evaluate_frozen_private_counterfactual",
    "evaluate_two_baseline_counterfactual",
]
