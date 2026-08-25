"""Composed Phase-I v10 candidate: soft-RDDSM discovery in the main chain.

Replaces the v9 warmup + ``infer_structure`` stages with the C1-validated
soft-RDDSM branch (default ``SoftDsmConfig``, exactly the configuration the
frozen C1 baseline ran on AOB), keeps the v9 landscape probe block verbatim,
and tops the ledger up to exactly 180,000 FE with the v9 tail MMES recipe:

```text
240-FE v9 landscape probes (identical points/features to a v9 run of the
  same seed - same rng namespaces, same partition, same probe design)
  -> discover_hierarchical_soft (default config; C1's instrument)
  -> checkpoint blocks/relations from the discovery evidence
     (leaves + certificate-derived relations, the T0 mapping)
  -> MMES top-up to exactly 180,000 FE (v9 tail seed namespace)
  -> features = landscape (bitwise v9) + structural (discovery-fed, v9
     formulas) + progress (mapped stages; tail gain is the top-up gain)
```

The checkpoint keeps the v9 ``PHASE1_FEATURE_NAMES`` tuple so downstream
consumers (the C3 dispatch rule) read features by name unchanged.  This
module never modifies frozen sources; it imports the v9 probe helpers and
mirrors their exact rng usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.mechanism_features import (
    PhaseProgressErrors,
    summarize_phase_progress,
)
from arac.evidence.hierarchical import Phase1Evidence
from arac.evidence.overlap_adapter import Phase1OverlapEvidence
from arac.evidence.phase1 import (
    PHASE1_FEATURE_NAMES,
    PHASE1_FES,
    STRUCTURE_WARMUP_FES,
    _SEED_NAMESPACE,
    _TAIL_SEED_NAMESPACE,
    _WARMUP_SEED_NAMESPACE,
    _derive_relations,
    _feature_vector,
    _partition,
    _probe_candidates,
    _structural_features,
)
from arac.evidence.soft_rddsm import discover_hierarchical_soft
from arac.evidence.soft_rddsm_adapter import (
    overlap_evidence_hash,
    soft_evidence_to_overlap_evidence,
)
from arac.runtime.contracts import PhaseCheckpoint, RelationEvidence
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import PypopOptimizerPort


V10_PROTOCOL = "arac-soft-rddsm-mainline-v10-candidate-1"
V10_DISCOVERY_WINDOW = 174_000  # v9 STRUCTURE_MAX_FES: same formulas, same constants
V10_MIN_TAIL_FES = 2_000
TOTAL_BUDGET_FES = 3_000_000


@dataclass(frozen=True)
class Phase1V10Result:
    checkpoint: PhaseCheckpoint
    landscape_feature_parity_values: tuple[float, ...]
    discovery_fes: int
    topup_fes: int
    shared_candidates: tuple[int, ...]
    evidence_complete: bool
    evidence: Phase1Evidence
    overlap_evidence: Phase1OverlapEvidence
    overlap_evidence_hash: str
    discovery_start_fes: int
    discovery_consumed_fes: int
    discovery_end_fes: int


def run_phase1_v10(
    problem: OptimizationProblem,
    *,
    run_seed: int,
    ledger: EvaluationLedger | None = None,
) -> Phase1V10Result:
    if ledger is None:
        ledger = EvaluationLedger(problem, TOTAL_BUDGET_FES)
    if ledger.count != 0 or ledger.problem is not problem:
        raise ValueError("v10 Phase-I requires a fresh ledger bound to the problem")

    # v9 landscape probe block, reproduced verbatim (same rng namespaces and
    # consumption order as run_phase1), so landscape features are bitwise
    # identical to a v9 run of the same seed.
    rng = np.random.default_rng(int(run_seed) ^ _SEED_NAMESPACE)
    blocks = _partition(problem.dimension, rng)
    candidates, pairs, _line_axes = _probe_candidates(problem, blocks, rng)
    values = np.asarray(ledger.evaluate(candidates), dtype=float)
    errors = np.maximum(values - problem.optimum, 0.0)
    probe_best_error = float(ledger.best_error)
    fallback_relations, strengths = _derive_relations(errors, pairs)
    landscape_features = _feature_vector(
        values,
        problem.optimum,
        pairs,
        fallback_relations,
        strengths,
    )

    # v9 warmup stage, mirrored verbatim: same seed namespace, budget, and
    # starting incumbent, so this stage is bitwise identical to a v9 run of
    # the same seed (the divergence begins at the structure/discovery stage).
    warmup_fes = min(
        STRUCTURE_WARMUP_FES,
        max(0, PHASE1_FES - ledger.count - V10_MIN_TAIL_FES),
    )
    if warmup_fes:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=int(run_seed) ^ _WARMUP_SEED_NAMESPACE,
            budget_fes=warmup_fes,
            population_size=24,
            restart=False,
        )
    warmup_best_error = float(ledger.best_error)

    # soft-RDDSM discovery replaces infer_structure (C1 config); it occupies
    # the v9 "structure" stage slot.
    discovery_start_fes = ledger.count
    discovery_budget = PHASE1_FES - discovery_start_fes - V10_MIN_TAIL_FES
    if discovery_budget <= 0:
        raise RuntimeError(
            "v10 Phase-I cannot reserve the minimum discovery/tail headroom: "
            f"start={discovery_start_fes}, phase1={PHASE1_FES}, tail={V10_MIN_TAIL_FES}"
        )
    discovery = discover_hierarchical_soft(
        problem,
        ledger,
        run_seed=run_seed,
        budget_fes=discovery_budget,
    )
    discovery_end_fes = ledger.count
    discovery_consumed_fes = discovery_end_fes - discovery_start_fes
    if discovery_consumed_fes > discovery_budget:
        raise RuntimeError("soft-RDDSM exceeded its reserved discovery budget")
    if (
        discovery.discovery_start_fes != discovery_start_fes
        or discovery.discovery_consumed_fes != discovery_consumed_fes
        or discovery.discovery_end_fes != discovery_end_fes
    ):
        raise RuntimeError("soft-RDDSM discovery receipt disagrees with the Phase-I ledger")
    evidence = discovery.evidence
    overlap_evidence = soft_evidence_to_overlap_evidence(evidence)
    leaf_variables = [tuple(leaf.variables) for leaf in evidence.region_tree.leaves]
    leaf_index = {leaf.node_id: index for index, leaf in enumerate(evidence.region_tree.leaves)}
    v10_blocks = tuple(tuple(int(variable) for variable in members) for members in leaf_variables)
    v10_relations = []
    for relation in evidence.region_relations:
        left = leaf_index.get(relation.left)
        right = leaf_index.get(relation.right)
        if left is None or right is None or left == right:
            continue
        v10_relations.append(
            RelationEvidence(
                left_block=min(left, right),
                right_block=max(left, right),
                strength=float(relation.score),
                disagreement=0.0,
            )
        )
    discovery_end_best = float(ledger.best_error)

    # v9 tail recipe: MMES top-up to exactly the frozen boundary.
    if ledger.count < PHASE1_FES:
        PypopOptimizerPort().run(
            "mmes",
            problem=problem,
            ledger=ledger,
            initial_mean=tuple(float(value) for value in ledger.best_x),
            sigma=0.5,
            seed=int(run_seed) ^ _TAIL_SEED_NAMESPACE,
            budget_fes=PHASE1_FES - ledger.count,
            population_size=24,
            restart=False,
        )
    if ledger.count != PHASE1_FES:
        raise RuntimeError("v10 Phase-I ledger did not stop at the frozen FE boundary")
    tail_best_error = float(ledger.best_error)

    @dataclass(frozen=True)
    class _Structural:
        blocks: tuple[tuple[int, ...], ...]
        relations: tuple[RelationEvidence, ...]
        consumed_fes: int
        completed: bool

    structural = _Structural(
        blocks=v10_blocks,
        relations=tuple(v10_relations),
        consumed_fes=discovery_consumed_fes,
        completed=bool(discovery.level_budgets) and bool(overlap_evidence.complete),
    )
    structural_features = _structural_features(
        structural,
        dimension=problem.dimension,
        # The receipt and the normalized feature must use the same reserved
        # increment.  V10_DISCOVERY_WINDOW is the legacy nominal ceiling;
        # the actual reservation starts after probes and warmup.
        structural_budget=discovery_budget,
        probe_best_error=probe_best_error,
        phase1_best_error=tail_best_error,
    )
    progress_features = summarize_phase_progress(
        PhaseProgressErrors(
            probe=probe_best_error,
            warmup=warmup_best_error,
            structure=discovery_end_best,
            tail=tail_best_error,
        )
    )
    features = landscape_features + structural_features + progress_features
    if len(features) != len(PHASE1_FEATURE_NAMES):
        raise RuntimeError(
            f"v10 feature count drifted: {len(features)} vs {len(PHASE1_FEATURE_NAMES)}"
        )
    checkpoint = PhaseCheckpoint(
        protocol=V10_PROTOCOL,
        run_seed=int(run_seed),
        total_budget_fes=ledger.total_budget,
        phase1_fes=ledger.count,
        incumbent=tuple(float(value) for value in ledger.best_x),
        incumbent_error=float(ledger.best_error),
        feature_names=PHASE1_FEATURE_NAMES,
        feature_values=tuple(float(value) for value in features),
        blocks=v10_blocks,
        relations=tuple(v10_relations),
    )
    return Phase1V10Result(
        checkpoint=checkpoint,
        landscape_feature_parity_values=tuple(float(value) for value in landscape_features),
        discovery_fes=int(discovery_consumed_fes),
        topup_fes=int(PHASE1_FES - discovery_end_fes),
        shared_candidates=tuple(int(variable) for variable in discovery.shared_candidates),
        evidence_complete=overlap_evidence.complete,
        evidence=evidence,
        overlap_evidence=overlap_evidence,
        overlap_evidence_hash=overlap_evidence_hash(overlap_evidence),
        discovery_start_fes=int(discovery_start_fes),
        discovery_consumed_fes=int(discovery_consumed_fes),
        discovery_end_fes=int(discovery_end_fes),
    )


def landscape_feature_names() -> tuple[str, ...]:
    """The prefix of PHASE1_FEATURE_NAMES that depends only on probe points."""

    from arac.evidence.phase1 import LANDSCAPE_FEATURE_NAMES, LINE_FEATURE_NAMES

    return tuple(LANDSCAPE_FEATURE_NAMES) + tuple(LINE_FEATURE_NAMES)


__all__ = [
    "Phase1V10Result",
    "TOTAL_BUDGET_FES",
    "V10_DISCOVERY_WINDOW",
    "V10_PROTOCOL",
    "landscape_feature_names",
    "run_phase1_v10",
]
