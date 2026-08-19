"""Gate 43: 1000-D signal quality of variable signatures on the sparse grid.

For every gate-29 cell (and an index-permuted variant) this measures the
variable signature stage once (exact FE per the v10.2 contract) and reports
the co-membership hit rate of signature nearest neighbours among the 24
active variables, against the random baseline.  Permutation invariance is
criterion 3 of the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.variable_signature import compute_variable_signatures
from arac.runtime.ledger import EvaluationLedger
from experiments.overlap_arac_gate29_screening import (
    ACTIVE_DIMENSION,
    BOUNDS,
    build_overlap_problem,
)

GATE_SEED = 20260837
PROBE_COUNT = 12
PROBE_SIZE = 16
TOP_K = 4
OUTPUT_SCHEMA = "arac-overlap-signature-gate43-v1"


def _cells() -> list[dict[str, object]]:
    return [
        {"mode": mode, "topology": topology, "overlap_budget": overlap}
        for mode in ("conflicting", "conforming")
        for topology in ("chain", "star", "random")
        for overlap in (3, 6)
    ]


def _problem(cell: dict[str, object], cell_seed: int, permutation: np.ndarray | None):
    _, truth = build_overlap_problem(
        ACTIVE_DIMENSION,
        overlap_budget=int(cell["overlap_budget"]),
        min_group_size=5,
        max_group_size=7,
        num_groups=4,
        base_function="rastrigin",
        conflict_mode=str(cell["mode"]),
        bounds=BOUNDS,
        contiguous=True,
        rotation=False,
        transforms=False,
        seed=cell_seed,
        topology=str(cell["topology"]),
        interaction_strength=0.25,
    )

    def objective(values):
        rows = np.asarray(values, dtype=float)
        batch = rows[np.newaxis, :] if rows.ndim == 1 else rows
        if permutation is None:
            observed = batch
        else:
            observed = np.empty_like(batch)
            observed[:, permutation] = batch
        active = np.asarray(truth.evaluate(observed[:, :ACTIVE_DIMENSION]), dtype=float)
        tail = np.sum((observed[:, ACTIVE_DIMENSION:] / BOUNDS) ** 2, axis=1)
        result = active + tail
        return float(result[0]) if rows.ndim == 1 else result

    problem = OptimizationProblem(
        objective=objective,
        dimension=1000,
        lower_bounds=(-BOUNDS,) * 1000,
        upper_bounds=(BOUNDS,) * 1000,
    )
    groups = [set(group) for group in truth.structure.groups]
    if permutation is not None:
        groups = [{int(permutation[v]) for v in group} for group in groups]
    return problem, groups


def _hit_rate(result, groups) -> tuple[float, float]:
    variable_group = {v: i for i, g in enumerate(groups) for v in g}
    actives = sorted(variable_group)
    hits = total = 0
    for variable in actives:
        neighbors = [n for n in result.top_neighbors(variable, TOP_K + 1) if n != variable][:TOP_K]
        for neighbor in neighbors:
            total += 1
            if neighbor in variable_group and variable_group[neighbor] == variable_group[variable]:
                hits += 1
    baseline = (sum(len(g) for g in groups) - len(groups)) / max(1, len(groups)) / (999)
    return hits / max(1, total), baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overlap_signature_gate43/confirmation.json"),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(GATE_SEED)
    rows = []
    for cell in _cells():
        active_shuffle = rng.permutation(ACTIVE_DIMENSION)
        dummy_shuffle = ACTIVE_DIMENSION + rng.permutation(1000 - ACTIVE_DIMENSION)
        block_permutation = np.concatenate([active_shuffle, dummy_shuffle])
        for variant, permutation in (
            ("identity", None),
            ("permuted", block_permutation),
        ):
            problem, groups = _problem(cell, GATE_SEED, permutation)
            ledger = EvaluationLedger(problem, 200_000)
            anchor = rng.uniform(-2.0, 2.0, size=1000)
            result = compute_variable_signatures(
                problem,
                ledger,
                anchor=anchor,
                step=0.25,
                probe_count=PROBE_COUNT,
                probe_size=PROBE_SIZE,
                seed=int(rng.integers(0, 2**31)),
            )
            rate, baseline = _hit_rate(result, groups)
            expected = 1 + 1000 + PROBE_COUNT + 1000 * PROBE_COUNT + PROBE_COUNT * PROBE_SIZE
            assert result.consumed_fes == expected == ledger.count
            rows.append(
                {
                    "cell": cell,
                    "variant": variant,
                    "hit_rate": rate,
                    "baseline": baseline,
                    "lift": rate / max(baseline, 1e-12),
                    "consumed_fes": result.consumed_fes,
                }
            )
            print(
                f"{cell['mode']}/{cell['topology']}/ov={cell['overlap_budget']}/{variant}: "
                f"hit={rate:.3f} baseline={baseline:.3f} lift={rate / max(baseline, 1e-12):.1f}x",
                flush=True,
            )
    identity_lifts = [row["lift"] for row in rows if row["variant"] == "identity"]
    permuted_lifts = [row["lift"] for row in rows if row["variant"] == "permuted"]
    checks = {
        "billable_exact_fe_all": all(row["consumed_fes"] == 1 + 1000 + PROBE_COUNT + 1000 * PROBE_COUNT + PROBE_COUNT * PROBE_SIZE for row in rows),
        "identity_mean_lift_ge_3": float(np.mean(identity_lifts)) >= 3.0,
        "permuted_mean_lift_ge_3": float(np.mean(permuted_lifts)) >= 3.0,
        "permutation_drop_le_half": float(np.mean(permuted_lifts)) >= 0.5 * float(np.mean(identity_lifts)),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "gate_seed": GATE_SEED,
            "probe_count": PROBE_COUNT,
            "probe_size": PROBE_SIZE,
            "top_k": TOP_K,
            "step": 0.25,
            "fe_formula": "1 + d + P + d*P + P*probe_size",
        },
        "summary": {
            "identity_mean_lift": float(np.mean(identity_lifts)),
            "permuted_mean_lift": float(np.mean(permuted_lifts)),
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "checks": checks}, indent=1))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
