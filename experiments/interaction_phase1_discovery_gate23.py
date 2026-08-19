"""Gate 23: Phase-I group recovery on a nonseparable overlap benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arac.benchmarks.overlap_objective import build_overlap_problem
from arac.coordination import OverlapStructure
from arac.evidence import run_phase1_overlap_pilot


DIMENSION = 24
MODE = "conflicting"
TOPOLOGY = "chain"
OVERLAP_BUDGET = 6
SEED = 31001
INTERACTION_STRENGTH = 0.25
TOTAL_BUDGET_FES = 3_000_000
PHASE1_FES = 180_000
ANCHOR_COUNT = 5
STEP = 0.25
ROUNDS = 12
BUCKET_SIZE = 4
MAX_CANDIDATE_PAIRS = 128


def _problem():
    return build_overlap_problem(
        DIMENSION,
        overlap_budget=OVERLAP_BUDGET,
        min_group_size=3,
        max_group_size=5,
        num_groups=6,
        base_function="sphere",
        conflict_mode=MODE,
        bounds=10.0,
        contiguous=True,
        rotation=False,
        transforms=False,
        interaction_strength=INTERACTION_STRENGTH,
        seed=SEED,
        topology=TOPOLOGY,
    )


def _components(groups: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return OverlapStructure(DIMENSION, groups).connected_components()


def run_gate() -> dict[str, object]:
    problem, objective = _problem()
    pilot = run_phase1_overlap_pilot(
        problem,
        total_budget_fes=TOTAL_BUDGET_FES,
        run_seed=SEED,
        anchor_count=ANCHOR_COUNT,
        step=STEP,
        rounds=ROUNDS,
        bucket_size=BUCKET_SIZE,
        max_candidate_pairs=MAX_CANDIDATE_PAIRS,
    )
    truth_groups = tuple(tuple(group) for group in objective.structure.groups)
    truth_shared = tuple(objective.structure.shared_variables)
    inferred_groups = tuple(pilot.evidence.groups)
    inferred_shared = tuple(
        variable for variable, owners in enumerate(pilot.evidence.memberships) if len(owners) > 1
    )
    truth_set = set(truth_shared)
    inferred_set = set(inferred_shared)
    precision = len(truth_set & inferred_set) / max(len(inferred_set), 1)
    recall = len(truth_set & inferred_set) / max(len(truth_set), 1)
    truth_components = _components(truth_groups)
    inferred_components = _components(inferred_groups) if pilot.adaptation.ready else ()
    checks = {
        "exact_phase1_boundary": pilot.consumed_fes == pilot.checkpoint.phase1_fes == PHASE1_FES,
        "discovery_complete": pilot.discovery.complete,
        "adapter_ready": pilot.adaptation.ready,
        "groups_exact": set(inferred_groups) == set(truth_groups),
        "shared_precision_one": precision == 1.0,
        "shared_recall_one": recall == 1.0,
        "components_exact": inferred_components == truth_components,
        "coverage_complete": pilot.discovery.separated_pair_fraction == 1.0,
        "candidate_cap_respected": pilot.discovery.candidate_pair_count <= MAX_CANDIDATE_PAIRS,
    }
    return {
        "schema_version": "arac-interaction-phase1-discovery-gate23-v1",
        "protocol": {
            "dimension": DIMENSION,
            "mode": MODE,
            "topology": TOPOLOGY,
            "overlap_budget": OVERLAP_BUDGET,
            "seed": SEED,
            "interaction_strength": INTERACTION_STRENGTH,
            "total_budget_fes": TOTAL_BUDGET_FES,
            "phase1_fes": PHASE1_FES,
            "anchor_count": ANCHOR_COUNT,
            "step": STEP,
            "rounds": ROUNDS,
            "bucket_size": BUCKET_SIZE,
            "max_candidate_pairs": MAX_CANDIDATE_PAIRS,
        },
        "truth_groups": truth_groups,
        "inferred_groups": inferred_groups,
        "truth_shared": truth_shared,
        "inferred_shared": inferred_shared,
        "shared_precision": precision,
        "shared_recall": recall,
        "truth_components": truth_components,
        "inferred_components": inferred_components,
        "discovery_complete_reason": pilot.discovery.complete_reason,
        "discovery_consumed_fes": pilot.discovery.consumed_fes,
        "discovery_expected_fes": pilot.discovery.expected_fes,
        "candidate_pair_count": pilot.discovery.candidate_pair_count,
        "separated_pair_fraction": pilot.discovery.separated_pair_fraction,
        "phase1_consumed_fes": pilot.consumed_fes,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/interaction_phase1_discovery_gate23/confirmation_fresh.json"),
    )
    args = parser.parse_args()
    payload = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": payload["gate_passed"], "gate_checks": payload["gate_checks"], "shared_precision": payload["shared_precision"], "shared_recall": payload["shared_recall"], "reason": payload["discovery_complete_reason"]}, indent=2, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
