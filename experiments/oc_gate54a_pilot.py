"""Gate 54a instrument-validation pilot (declared, NOT judgment data).

Role: freeze the L1 classifier's discriminating signature and thresholds by
checking cross-context sensitivity separates generator-conforming from
generator-conflicting shared variables, BEFORE the protocol locks and the
judgment grid runs on fresh seeds.  Four small instances, truth scope,
5k-FE MMES burn-in, 6 FE/variable probes.  No production changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination.consistency import consistency_probe
from arac.coordination.overlap import OverlapStructure
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort

DIMENSION = 200
NUM_GROUPS = 10
OVERLAP_BUDGET = 6
GROUP_SIZE = (10, 30)
BOUNDS = 100.0
BURNIN_FES = 5_000
SEEDS = (7101, 7102)
MODES = ("conforming", "conflicting")
OUTPUT = Path("artifacts/oc_gate54a_pilot/confirmation.json")


def run_pilot() -> dict[str, object]:
    rows = []
    for mode in MODES:
        for seed in SEEDS:
            problem, objective = build_overlap_problem(
                DIMENSION,
                overlap_budget=OVERLAP_BUDGET,
                min_group_size=GROUP_SIZE[0],
                max_group_size=GROUP_SIZE[1],
                base_function="rastrigin",
                conflict_mode=mode,
                bounds=BOUNDS,
                contiguous=True,
                rotation=False,
                transforms=False,
                seed=seed,
                num_groups=NUM_GROUPS,
                topology="random",
            )
            ledger = EvaluationLedger(problem, 10 * BURNIN_FES)
            rng = np.random.default_rng(seed)
            center = (problem.lower_array + problem.upper_array) / 2.0
            start = center + rng.uniform(-0.1, 0.1, size=DIMENSION) * (
                problem.upper_array - problem.lower_array
            )
            PypopOptimizerPort().run(
                "mmes",
                problem=problem,
                ledger=ledger,
                initial_mean=tuple(float(v) for v in start),
                sigma=30.0,
                seed=seed,
                budget_fes=BURNIN_FES,
                population_size=24,
                restart=False,
            )
            truth_shared = sorted(
                {
                    int(variable)
                    for group in objective.structure.groups
                    for variable in group
                    if sum(variable in g for g in objective.structure.groups) > 1
                }
            )
            structure = OverlapStructure(
                problem.dimension,
                tuple(tuple(int(v) for v in group) for group in objective.structure.groups),
            )
            labels = consistency_probe(structure, ledger, truth_shared, seed=seed)
            for item in labels:
                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "variable": item.variable,
                        "disagreement": item.disagreement,
                        "owner_values": list(item.owner_values),
                        "label": item.label,
                        "truth": mode,
                        "match": item.label == mode,
                    }
                )
            print(
                f"{mode}/{seed}: shared={len(truth_shared)} "
                f"disagree=[{min(i.disagreement for i in labels):.4f},{max(i.disagreement for i in labels):.4f}]",
                flush=True,
            )
    conforming_cross = [r["disagreement"] for r in rows if r["mode"] == "conforming"]
    conflicting_cross = [r["disagreement"] for r in rows if r["mode"] == "conflicting"]
    conforming_bias = [0.0]
    conflicting_bias = [0.0]
    payload = {
        "rows": rows,
        "summary": {
            "n_conforming": len(conforming_cross),
            "n_conflicting": len(conflicting_cross),
            "instability_conforming_median": float(np.median(conforming_cross)),
            "instability_conflicting_median": float(np.median(conflicting_cross)),
            "match_rate_conforming": sum(1 for r in rows if r["mode"] == "conforming" and r["match"]) / max(1, sum(1 for r in rows if r["mode"] == "conforming")),
            "match_rate_conflicting": sum(1 for r in rows if r["mode"] == "conflicting" and r["match"]) / max(1, sum(1 for r in rows if r["mode"] == "conflicting")),
            "separation_instability": float(np.median(conflicting_cross) - np.median(conforming_cross)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=1, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
