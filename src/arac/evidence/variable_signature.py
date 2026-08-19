"""Billable variable interaction signatures (Phase-I v10.2, Gate 43).

Every variable is measured against one SHARED basis of ``P`` fixed random
probe batches, so signatures are mutually comparable and every component maps
to a ledger-billed evaluation:

```text
FE = 1 + d + P + d*P        (base + variable singles + probe singles + joints)
s_j = [ I(j, B_p) / scale ]_p    (signed normalized mixed differences)
```

Same-group variables fire on the probes that contain their shared partners,
so their signature vectors correlate; disjoint groups fire on disjoint probes.
The previous idea of hash-assigned per-variable batches was dropped because
vectors measured against different bases are not comparable (v10.2 design
note).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.runtime.ledger import EvaluationLedger


@dataclass(frozen=True)
class VariableSignatureResult:
    """Per-variable interaction signatures against one shared probe basis."""

    signatures: np.ndarray  # shape (dimension, probe_count)
    probe_batches: tuple[tuple[int, ...], ...]
    anchor: tuple[float, ...]
    step: float
    consumed_fes: int
    expected_fes: int

    def signature(self, variable: int) -> np.ndarray:
        return self.signatures[variable]

    def cosine_similarity(self, left: int, right: int) -> float:
        a = self.signatures[left]
        b = self.signatures[right]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 0.0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    def top_neighbors(self, variable: int, k: int) -> tuple[int, ...]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        similarities = np.asarray(
            [
                (self.cosine_similarity(variable, other) if other != variable else -np.inf)
                for other in range(self.signatures.shape[0])
            ]
        )
        order = np.argsort(-similarities, kind="stable")
        return tuple(int(index) for index in order[:k])


def _probe_batches(
    dimension: int,
    *,
    probe_count: int,
    probe_size: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(probe_count):
        batches.append(tuple(sorted(int(v) for v in rng.choice(dimension, size=probe_size, replace=False))))
    return tuple(batches)


def compute_variable_signatures(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    anchor: np.ndarray,
    step: float = 0.25,
    probe_count: int = 12,
    probe_size: int = 16,
    seed: int = 0,
) -> VariableSignatureResult:
    """Measure every variable against the shared probe basis, billed exactly.

    Consumes exactly ``1 + d + P + d*P`` FE on the provided ledger; no other
    evaluations are performed.
    """

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be EvaluationLedger")
    if ledger.problem is not problem:
        raise ValueError("signature measurement requires the ledger for the same problem")
    if not math.isfinite(float(step)) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    if isinstance(probe_count, bool) or not isinstance(probe_count, int) or probe_count <= 0:
        raise ValueError("probe_count must be a positive integer")
    if isinstance(probe_size, bool) or not isinstance(probe_size, int) or not 1 <= probe_size < problem.dimension:
        raise ValueError("probe_size must be an integer in [1, dimension)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (problem.dimension,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor must be a finite vector matching the dimension")
    dimension = problem.dimension
    corrected_count = probe_count * probe_size
    expected = 1 + dimension + probe_count + dimension * probe_count + corrected_count
    if ledger.remaining < expected:
        raise ValueError("variable signature measurement exceeds the remaining FE budget")

    batches = _probe_batches(dimension, probe_count=probe_count, probe_size=probe_size, seed=seed)
    lower = problem.lower_array
    upper = problem.upper_array
    start = ledger.count

    # One batched evaluation block: base, variable singles, probe singles,
    # joints, and corrected probe singles f(x + delta_{B \ {j}}) for every
    # (j, B) pair whose probe batch contains the measured variable.
    rows = expected
    candidates = np.repeat(anchor[np.newaxis, :], rows, axis=0)
    offset = 1
    for variable in range(dimension):
        candidates[offset + variable, variable] += step
    offset += dimension
    for index, batch in enumerate(batches):
        candidates[offset + index, list(batch)] += step
    offset += probe_count
    for variable in range(dimension):
        for index, batch in enumerate(batches):
            row = candidates[offset + variable * probe_count + index]
            row[list(batch)] += step
            # The measured variable may itself sit in the probe batch; perturb
            # it exactly once so the joint is x + delta_j + delta_{B \ {j}}.
            row[variable] = anchor[variable] + step
    offset += dimension * probe_count
    corrected_index: dict[tuple[int, int], int] = {}
    for index, batch in enumerate(batches):
        for position, variable in enumerate(batch):
            row = candidates[offset + index * probe_size + position]
            rest = [other for other in batch if other != variable]
            row[rest] += step
            corrected_index[(variable, index)] = offset + index * probe_size + position
    np.clip(candidates, lower, upper, out=candidates)
    values = np.asarray(ledger.evaluate(candidates), dtype=float)
    consumed = ledger.count - start
    if consumed != expected:
        raise RuntimeError("variable signature FE accounting drifted")

    base = float(values[0])
    variable_singles = values[1 : 1 + dimension]
    probe_singles = values[1 + dimension : 1 + dimension + probe_count]
    joints = values[
        1 + dimension + probe_count : 1 + dimension + probe_count + dimension * probe_count
    ].reshape(dimension, probe_count)
    # The mixed difference assumes j not in its probe batch; for j in B the
    # batch single must be replaced by the corrected f(x + delta_{B \ {j}}).
    batch_single_matrix = np.broadcast_to(
        probe_singles[np.newaxis, :], (dimension, probe_count)
    ).copy()
    for (variable, index), row_index in corrected_index.items():
        batch_single_matrix[variable, index] = values[row_index]
    mixed = joints - variable_singles[:, np.newaxis] - batch_single_matrix + base
    scale = (
        abs(base)
        + np.abs(variable_singles)[:, np.newaxis]
        + np.abs(batch_single_matrix)
        + np.abs(joints)
        + 1.0
    )
    signatures = mixed / scale
    if not np.all(np.isfinite(signatures)):
        raise RuntimeError("variable signature produced non-finite entries")
    return VariableSignatureResult(
        signatures=signatures,
        probe_batches=batches,
        anchor=tuple(float(value) for value in anchor),
        step=float(step),
        consumed_fes=consumed,
        expected_fes=expected,
    )


__all__ = ["VariableSignatureResult", "compute_variable_signatures"]
