"""Stage-1 region-merge split repair and per-link target validation for the
shared_transaction_v1 candidate.

Derived from ``arac.evidence.soft_rddsm.discover_hierarchical_soft`` (frozen
instrument, never modified).  Two behavioural changes are inserted, both
preregistered in the T0 protocol and both consuming already-registered
thresholds only:

1. **Endpoint-refinement merge repair.**  Stage 1 seeds in ascending variable
   index order; when a shared variable j is the lowest-index ungrouped
   variable of its owner union (~8%/pair/seed on generator v3), its recursive
   interactor search absorbs both planted blocks into one ~200-variable
   region (the v5.1 P0 failure signature).  For every Stage-1 block R the
   module refines R's first variable AND R's last variable against the full
   variable set (the first refinements are exactly the pipeline's own Stage
   2; the last-variable refinements are the added cost).  A block is split
   into ``core`` and ``R - core`` only when:

   - ``Gf = G_first ∩ R`` and ``Gl = G_last ∩ R`` are both strict subsets
     of R, or exactly one is strict while the other equals R;
   - ``Gf ∪ Gl == R`` (exact cover - a budget-truncated refinement cannot
     satisfy this against a complete opposite-endpoint group);
   - ``|Gf ∩ Gl| < min_residual_size`` (the overlap is the coupling shard);
   - ``|R - core| >= min_residual_size``.

   The core keeps the coupling variables, mirroring how ascending Stage 1
   assigns shared variables to their lower-index owner.  On clean covers
   both endpoint groups equal R and the repair is a no-op.

2. **Per-link target validation.**  In a chain topology the raw target set of
   a middle link's shared variable is polluted by co-member shared variables
   of neighbouring links (the v5.0 H0 three-region rejection, reproduced in
   the first smoke: 16/24 hyperedges rejected).  Before a hyperedge is
   materialized, every candidate target region t is validated with one
   moderate-range residual probe - ``_rdg_interact([j], residual(t))`` where
   the residual excludes all current shared suspects - exactly the
   chain_pair_isolation diagnostic pattern.  Targets that fail the probe are
   dropped; a variable with no surviving targets is not a shared candidate.

3. **Certificate-derived checkpoint relations.**  The soft-DSM edge pool
   contains no cross-block edges on generator v3 (p0 receipts: relation_count
   0), so the candidate checkpoint derives its block-level relations from the
   validated hyperedge graph: one RelationEvidence per certified region pair
   with strength = number of certificates on that pair.  This is the only
   channel by which certified structure enters the frozen PhaseCheckpoint
   contract, and it is recorded in the sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arac.benchmarks.aob import OptimizationProblem
from arac.evidence.hierarchical import (
    Phase1Evidence,
    RegionNode,
    RegionRelation,
    RegionTree,
    ResolvedOverlapHyperedge,
    VariableRegionInteraction,
)
from arac.evidence.soft_rddsm import (
    SoftDsmConfig,
    SoftDiscoveryResult,
    _block_separable,
    _rdg_interact,
    _rdg_refine_interactors,
    build_soft_dsm,
)
from arac.evidence.variable_signature import compute_variable_signatures
from arac.runtime.ledger import EvaluationLedger


@dataclass(frozen=True)
class SplitRepairRecord:
    """Audit trail for one coarse block's split-repair decision."""

    block_index: int
    block_size: int
    first_group_inside: tuple[int, ...]
    last_group_inside: tuple[int, ...]
    third_group_inside: tuple[int, ...]
    pieces: tuple[tuple[int, ...], ...]
    adopted: bool
    reason: str


@dataclass(frozen=True)
class SplitRepairDiscoveryResult:
    """Soft-RDDSM receipt plus the split-repair audit trail."""

    discovery: SoftDiscoveryResult
    repair_records: tuple[SplitRepairRecord, ...]
    last_endpoint_refinement_fes: int
    target_validation_fes: int
    target_validation: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]


def _refine_group(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    seed: int,
    region_size: int,
    base_point: np.ndarray,
    base_value: float,
    threshold: float,
    budget_start: int,
    budget: int,
) -> set[int]:
    candidates = [variable for variable in range(problem.dimension) if variable != seed]
    return {seed} | _rdg_refine_interactors(
        problem,
        ledger,
        seed=seed,
        candidates=candidates,
        region_size=region_size,
        base_point=base_point,
        base_value=base_value,
        threshold=threshold,
        budget_start=budget_start,
        budget=budget,
    )


def discover_hierarchical_soft_split_repair(
    problem: OptimizationProblem,
    ledger: EvaluationLedger,
    *,
    run_seed: int,
    config: SoftDsmConfig | None = None,
    signature_probe_count: int = 12,
    signature_probe_size: int = 16,
    step: float = 0.25,
) -> SplitRepairDiscoveryResult:
    """Soft-RDDSM discovery with endpoint-refinement merge repair and
    per-link target validation inserted."""

    if not isinstance(problem, OptimizationProblem):
        raise TypeError("problem must be OptimizationProblem")
    if not isinstance(ledger, EvaluationLedger):
        raise TypeError("ledger must be EvaluationLedger")
    if ledger.problem is not problem:
        raise ValueError("discovery requires the ledger for the same problem")
    config = SoftDsmConfig() if config is None else config
    dimension = problem.dimension
    center = (problem.lower_array + problem.upper_array) / 2.0
    span = problem.upper_array - problem.lower_array
    anchor_rng = np.random.default_rng(run_seed ^ 0x5A17)
    signature_anchor = center + anchor_rng.uniform(-0.2, 0.2, size=dimension) * span

    signature_result = compute_variable_signatures(
        problem,
        ledger,
        anchor=signature_anchor,
        step=step,
        probe_count=signature_probe_count,
        probe_size=signature_probe_size,
        seed=run_seed ^ 0x516,
    )
    edges, _candidates, dsm_fes = build_soft_dsm(
        problem,
        ledger,
        signature_result.signatures,
        config=config,
        step=step,
        seed=run_seed ^ 0x50F7,
    )

    centre_point = (problem.lower_array + problem.upper_array) / 2.0
    rdg_start = ledger.count
    centre_value = float(
        np.asarray(ledger.evaluate(centre_point[np.newaxis, :])).reshape(-1)[0]
    )

    def _find_interactors(seed: int, candidates: list[int], budget_left: int) -> set[int]:
        if not candidates or budget_left < 3:
            return set()
        if _rdg_interact(
            problem, ledger,
            set_a=[seed], set_b=candidates,
            base_point=centre_point, base_value=centre_value,
            threshold=config.edge_threshold,
        ):
            if len(candidates) == 1:
                return set(candidates)
            mid = len(candidates) // 2
            return (
                _find_interactors(seed, candidates[:mid], budget_left - 3)
                | _find_interactors(seed, candidates[mid:], budget_left - 3)
            )
        return set()

    # Stage 1: disjoint coarse cover (unchanged from the frozen instrument).
    rdg_budget = config.dsm_budget
    grouped: set[int] = set()
    rdg_blocks: list[tuple[int, ...]] = []
    for seed in range(dimension):
        if seed in grouped:
            continue
        remaining_budget = rdg_budget - (ledger.count - rdg_start)
        if remaining_budget < 3:
            break
        candidates = [v for v in range(dimension) if v not in grouped and v != seed]
        interactors = _find_interactors(seed, candidates, remaining_budget)
        group = tuple(sorted({seed} | interactors))
        if len(group) >= 2:
            rdg_blocks.append(group)
        grouped.update(group)

    # Stage 2 refinements: first-variable groups (the pipeline's own), plus
    # last-variable groups for the endpoint-refinement merge repair.
    def _budget_ok(extra: int) -> bool:
        return ledger.count - rdg_start + extra <= rdg_budget

    first_groups: list[set[int]] = []
    refinement_complete = True
    for block in rdg_blocks:
        if not block:
            continue
        if not _budget_ok(3):
            refinement_complete = False
            break
        group = _refine_group(
            problem, ledger,
            seed=block[0],
            region_size=config.rdg_region_size,
            base_point=centre_point,
            base_value=centre_value,
            threshold=config.edge_threshold,
            budget_start=rdg_start,
            budget=rdg_budget,
        )
        first_groups.append(group)
    if len(first_groups) != len(rdg_blocks):
        refinement_complete = False

    last_endpoint_fes_before = ledger.count
    last_groups: list[set[int]] = []
    for block in rdg_blocks:
        if not block:
            last_groups.append(set())
            continue
        if not _budget_ok(3):
            last_groups.append(set(block))
            continue
        group = _refine_group(
            problem, ledger,
            seed=block[-1],
            region_size=config.rdg_region_size,
            base_point=centre_point,
            base_value=centre_value,
            threshold=config.edge_threshold,
            budget_start=rdg_start,
            budget=rdg_budget,
        )
        last_groups.append(group)
    last_endpoint_refinement_fes = ledger.count - last_endpoint_fes_before

    # Merge repair: a block splits when its endpoint refinements expose two
    # strict interaction groups inside it.  When both endpoints see strict
    # subsets, the pieces are (Gf, R - Gf) and the consistency requirements
    # are exact cover and a sub-residual overlap.  When exactly one endpoint
    # covers R (the merged-block signature: the first variable IS the shared
    # variable that caused the merge, so its refinement returns the whole
    # merge), the strict side becomes the core and the complement is verified
    # by a third refinement seeded at the complement's first variable:
    # adopt only if core | (G3 & R) == R and |core & (G3 & R)| < min_residual.
    # On adoption the confirmation soft cover also drops the merged group in
    # favour of the two verified groups, otherwise the link's shared variable
    # would never appear in two refined groups and could not be confirmed.
    repair_records: list[SplitRepairRecord] = []
    blocks: list[tuple[int, ...]] = []
    piece_groups: list[list[set[int]]] = []
    for index, block in enumerate(rdg_blocks):
        members = set(block)
        gf = (first_groups[index] & members) if index < len(first_groups) else members
        gl = (last_groups[index] & members) if index < len(last_groups) else members
        gf_covers = gf == members
        gl_covers = gl == members
        base = SplitRepairRecord(
            block_index=index,
            block_size=len(block),
            first_group_inside=tuple(sorted(gf)),
            last_group_inside=tuple(sorted(gl)),
            third_group_inside=(),
            pieces=(block,),
            adopted=False,
            reason="no_split_signal" if (gf_covers and gl_covers) else "kept",
        )
        if gf_covers and gl_covers:
            repair_records.append(base)
            blocks.append(block)
            piece_groups.append([first_groups[index]])
            continue
        if not gf_covers and not gl_covers:
            core = gf
            core_group = first_groups[index]
            other_group = last_groups[index]
            exact_cover = (gf | gl) == members
            # Non-containment: a budget-truncated group is contained in the
            # complete opposite group, so its overlap equals the smaller
            # group; a genuine split overlaps only on the link's shared
            # variables (shared_width can exceed min_residual_size).
            small_overlap = len(gf & gl) < min(len(gf), len(gl))
        elif not gf_covers:
            core = gf
            core_group = first_groups[index]
            other_group = None
            exact_cover = False
            small_overlap = False
        else:
            core = gl
            core_group = last_groups[index]
            other_group = None
            exact_cover = False
            small_overlap = False
        other = members - core
        if (not gf_covers and not gl_covers) and other_group is not None:
            if exact_cover and small_overlap and len(other) >= config.min_residual_size:
                repair_records.append(
                    SplitRepairRecord(
                        block_index=index,
                        block_size=len(block),
                        first_group_inside=tuple(sorted(gf)),
                        last_group_inside=tuple(sorted(gl)),
                        third_group_inside=(),
                        pieces=(tuple(sorted(core)), tuple(sorted(other))),
                        adopted=True,
                        reason="both_endpoints_strict_split",
                    )
                )
                blocks.extend((tuple(sorted(core)), tuple(sorted(other))))
                piece_groups.append([core_group, other_group])
                continue
            repair_records.append(
                SplitRepairRecord(
                    block_index=index,
                    block_size=len(block),
                    first_group_inside=tuple(sorted(gf)),
                    last_group_inside=tuple(sorted(gl)),
                    third_group_inside=(),
                    pieces=(block,),
                    adopted=False,
                    reason=(
                        "consistency_failed_exact_cover" if not exact_cover
                        else "consistency_failed_overlap" if not small_overlap
                        else "fragment_too_small"
                    ),
                )
            )
            blocks.append(block)
            piece_groups.append([first_groups[index]])
            continue
        # Asymmetric case: one endpoint covers R, the other exposes a strict
        # core.  Verify the complement with a third refinement.
        complement_seed = min(other) if other else None
        if complement_seed is None or not _budget_ok(3):
            repair_records.append(
                SplitRepairRecord(
                    block_index=index,
                    block_size=len(block),
                    first_group_inside=tuple(sorted(gf)),
                    last_group_inside=tuple(sorted(gl)),
                    third_group_inside=(),
                    pieces=(block,),
                    adopted=False,
                    reason="complement_verification_unavailable",
                )
            )
            blocks.append(block)
            piece_groups.append([first_groups[index]])
            continue
        g3 = _refine_group(
            problem, ledger,
            seed=complement_seed,
            region_size=config.rdg_region_size,
            base_point=centre_point,
            base_value=centre_value,
            threshold=config.edge_threshold,
            budget_start=rdg_start,
            budget=rdg_budget,
        )
        g3_inside = g3 & members
        exact_cover = (core | g3_inside) == members
        small_overlap = len(core & g3_inside) < min(len(core), len(g3_inside))
        if exact_cover and small_overlap and len(other) >= config.min_residual_size:
            # The overlap is the link's shared variables.  Keep the ascending
            # primary-assignment convention: they stay with the lower-index
            # (complement) piece, exactly as an unmerged ascending cover would
            # assign them.
            overlap = core & g3_inside
            piece_core = core - overlap
            piece_other = other | overlap
            repair_records.append(
                SplitRepairRecord(
                    block_index=index,
                    block_size=len(block),
                    first_group_inside=tuple(sorted(gf)),
                    last_group_inside=tuple(sorted(gl)),
                    third_group_inside=tuple(sorted(g3_inside)),
                    pieces=(tuple(sorted(piece_core)), tuple(sorted(piece_other))),
                    adopted=True,
                    reason="asymmetric_endpoint_split",
                )
            )
            blocks.extend((tuple(sorted(piece_core)), tuple(sorted(piece_other))))
            piece_groups.append([core_group, g3])
            continue
        repair_records.append(
            SplitRepairRecord(
                block_index=index,
                block_size=len(block),
                first_group_inside=tuple(sorted(gf)),
                last_group_inside=tuple(sorted(gl)),
                third_group_inside=tuple(sorted(g3_inside)),
                pieces=(block,),
                adopted=False,
                reason=(
                    "consistency_failed_exact_cover" if not exact_cover
                    else "consistency_failed_overlap" if not small_overlap
                    else "fragment_too_small"
                ),
            )
        )
        blocks.append(block)
        piece_groups.append([first_groups[index]])
    blocks = tuple(blocks)

    # Shared-candidate confirmation over the repaired soft cover: merged
    # blocks contribute their two verified groups instead of the merged one.
    refined_groups: list[tuple[int, ...]] = []
    for groups in piece_groups:
        for group in groups:
            if group:
                refined_groups.append(tuple(sorted(group)))
    primary: dict[int, int] = {}
    for block_index, block in enumerate(blocks):
        for variable in block:
            primary.setdefault(variable, block_index)
    primary_members: list[list[int]] = [[] for _ in blocks]
    for variable, block_index in primary.items():
        primary_members[block_index].append(variable)
    leaf_variables: list[tuple[int, ...]] = []
    for members in primary_members:
        if members:
            leaf_variables.append(tuple(sorted(members)))
    for variable in range(dimension):
        if variable not in primary:
            leaf_variables.append((variable,))
    nodes = [RegionNode(0, None, 0, tuple(range(dimension)))]
    for index, members in enumerate(leaf_variables, start=1):
        nodes.append(RegionNode(index, 0, 1, members))
    tree = RegionTree(dimension=dimension, nodes=tuple(nodes))
    leaf_of_variable = {
        variable: leaf.node_id for leaf in tree.leaves for variable in leaf.variables
    }
    leaf_variables_by_id = {leaf.node_id: tuple(leaf.variables) for leaf in tree.leaves}

    relations: list[RegionRelation] = []
    for (left, right), edge in sorted(edges.items()):
        left_leaf = leaf_of_variable.get(left)
        right_leaf = leaf_of_variable.get(right)
        if left_leaf is None or right_leaf is None or left_leaf == right_leaf:
            continue
        relations.append(
            RegionRelation(
                left=left_leaf,
                right=right_leaf,
                score=edge.score,
                stability=edge.support / edge.probes,
                depth=1,
            )
        )

    interactions_by_key: dict[tuple[int, int, int], VariableRegionInteraction] = {}
    shared_targets: dict[int, set[int]] = {}
    separability_rng = np.random.default_rng(run_seed ^ 0x5EED)
    separability_anchor = center + separability_rng.uniform(-0.2, 0.2, size=dimension) * span
    candidate_groups = tuple(refined_groups) if refinement_complete else ()
    for left_index, left_group in enumerate(candidate_groups):
        left_set = set(left_group)
        for right_index, right_group in enumerate(
            candidate_groups[left_index + 1 :], start=left_index + 1
        ):
            right_set = set(right_group)
            common = left_set & right_set
            left_residual = sorted(left_set - common)
            right_residual = sorted(right_set - common)
            if not common or not left_residual or not right_residual:
                continue
            confirmed_common: list[int] = []
            for variable in sorted(common):
                if not _budget_ok(6):
                    break
                interacts_left = _rdg_interact(
                    problem, ledger,
                    set_a=[variable],
                    set_b=left_residual,
                    base_point=centre_point,
                    base_value=centre_value,
                    threshold=config.edge_threshold,
                )
                interacts_right = _rdg_interact(
                    problem, ledger,
                    set_a=[variable],
                    set_b=right_residual,
                    base_point=centre_point,
                    base_value=centre_value,
                    threshold=config.edge_threshold,
                )
                if interacts_left and interacts_right:
                    confirmed_common.append(variable)
            common = set(confirmed_common)
            left_residual = sorted(left_set - common)
            right_residual = sorted(right_set - common)
            if (
                not common
                or len(left_residual) < config.min_residual_size
                or len(right_residual) < config.min_residual_size
            ):
                continue
            if not _block_separable(
                problem,
                ledger,
                separability_anchor,
                left_residual,
                right_residual,
                step=step,
                threshold=config.edge_threshold,
                probes=config.block_separability_probes,
                seed=run_seed ^ (0x7EA5 * (left_index + 1) ^ right_index),
            ):
                continue
            if not _budget_ok(3):
                continue
            if _rdg_interact(
                problem, ledger,
                set_a=left_residual,
                set_b=right_residual,
                base_point=centre_point,
                base_value=centre_value,
                threshold=config.edge_threshold,
            ):
                continue

            for variable in sorted(left_set | right_set):
                if variable in left_set - common:
                    continue
                source_leaf = leaf_of_variable[variable]
                opposite_members: set[int] = set()
                if variable in common:
                    opposite_members.update((left_set | right_set) - {variable})
                elif variable in left_set:
                    opposite_members.update(right_set - common)
                elif variable in right_set:
                    opposite_members.update(left_set)
                target_leaves = {
                    leaf_of_variable[target]
                    for target in opposite_members
                    if target != variable and leaf_of_variable[target] != source_leaf
                }
                for target_leaf in sorted(target_leaves):
                    key = (variable, source_leaf, target_leaf)
                    interactions_by_key.setdefault(
                        key,
                        VariableRegionInteraction(
                            variable=variable,
                            source_region=source_leaf,
                            target_region=target_leaf,
                            q_lb=1.0,
                            support=1,
                            sign_stability=1.0,
                        ),
                    )
                if variable in common:
                    shared_targets.setdefault(variable, set()).update(target_leaves)

    # Per-link target validation: keep only targets whose shared-suspect-free
    # residual still interacts with the variable (chain_pair_isolation pattern).
    target_validation_fes = 0
    target_validation: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    shared_suspects = set(shared_targets)
    for variable in sorted(shared_targets):
        source_leaf = leaf_of_variable[variable]
        raw_targets = set(shared_targets[variable])
        raw_targets.discard(source_leaf)
        validated: set[int] = set()
        for target_leaf in sorted(raw_targets):
            residual = [
                member
                for member in leaf_variables_by_id[target_leaf]
                if member not in shared_suspects and member != variable
            ]
            if len(residual) < config.min_residual_size or not _budget_ok(3):
                continue
            target_validation_fes += 3
            if _rdg_interact(
                problem, ledger,
                set_a=[variable],
                set_b=residual,
                base_point=centre_point,
                base_value=centre_value,
                threshold=config.edge_threshold,
            ):
                validated.add(target_leaf)
        target_validation.append(
            (variable, tuple(sorted(raw_targets)), tuple(sorted(validated)))
        )
        if validated:
            shared_targets[variable] = validated
        else:
            del shared_targets[variable]
    validated_targets_union = {
        (variable, target)
        for variable, targets in shared_targets.items()
        for target in targets
    }
    interactions_by_key = {
        key: interaction
        for key, interaction in interactions_by_key.items()
        if (key[0], key[2]) in validated_targets_union
    }

    interactions = [interactions_by_key[key] for key in sorted(interactions_by_key)]
    hyperedges: list[ResolvedOverlapHyperedge] = []
    shared_candidates: list[int] = []
    for variable, targets in sorted(shared_targets.items()):
        source_leaf = leaf_of_variable[variable]
        targets.discard(source_leaf)
        if not targets:
            continue
        evidence_for_variable = tuple(
            interaction
            for interaction in interactions
            if interaction.variable == variable
            and interaction.source_region == source_leaf
            and interaction.target_region in targets
        )
        covered = {interaction.target_region for interaction in evidence_for_variable}
        if covered != targets:
            continue
        shared_candidates.append(variable)
        hyperedges.append(
            ResolvedOverlapHyperedge(
                variable=variable,
                regions=(source_leaf, *sorted(targets)),
                evidence=evidence_for_variable,
            )
        )

    # Certificate-derived relations: one relation per validated hyperedge
    # region pair, strength = certificate count on the pair, merged with any
    # DSM-derived relations (empty on generator v3).
    certificate_edges: dict[tuple[int, int], int] = {}
    for hyperedge in hyperedges:
        regions = sorted(int(region) for region in hyperedge.regions)
        for left in regions:
            for right in regions:
                if left < right:
                    certificate_edges[(left, right)] = certificate_edges.get((left, right), 0) + 1
    existing_pairs = {(min(relation.left, relation.right), max(relation.left, relation.right)) for relation in relations}
    for (left, right), count in sorted(certificate_edges.items()):
        if (left, right) in existing_pairs:
            continue
        relations.append(
            RegionRelation(
                left=left,
                right=right,
                score=float(count),
                stability=1.0,
                depth=1,
            )
        )
    relations.sort(key=lambda item: (item.left, item.right))

    member_variables = {interaction.variable for interaction in interactions}
    variable_status = tuple(
        (
            variable,
            "member_candidate" if variable in member_variables else "observed_separable",
        )
        for variable in range(dimension)
    )
    adjacency: dict[int, set[int]] = {node.node_id: set() for node in tree.leaves}
    for relation in relations:
        adjacency[relation.left].add(relation.right)
        adjacency[relation.right].add(relation.left)
    unseen = set(adjacency)
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack_c, component = [root], set()
        while stack_c:
            node = stack_c.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack_c.extend(adjacency[node] - component)
        components.append(tuple(sorted(component)))
    per_component_mode = []
    for component in components:
        if len(component) < 2:
            per_component_mode.append((component, "SPARSE"))
        elif hyperedges:
            per_component_mode.append((component, "SPARSE"))
        else:
            per_component_mode.append((component, "HIERARCHICAL"))

    evidence = Phase1Evidence(
        dimension=dimension,
        region_tree=tree,
        region_relations=tuple(relations),
        variable_region_interactions=tuple(
            sorted(interactions, key=lambda item: item.variable)
        ),
        resolved_hyperedges=tuple(hyperedges),
        variable_status=variable_status,
        per_component_mode=tuple(per_component_mode),
        level_budgets=(
            ("signature", signature_result.consumed_fes),
            ("dsm", dsm_fes),
            ("rdg", ledger.count - rdg_start),
        ),
    )
    discovery = SoftDiscoveryResult(
        evidence=evidence,
        signature_result=signature_result,
        edges=tuple(sorted(edges.values(), key=lambda edge: (edge.left, edge.right))),
        blocks=blocks,
        level_budgets=evidence.level_budgets,
        shared_candidates=tuple(shared_candidates),
    )
    return SplitRepairDiscoveryResult(
        discovery=discovery,
        repair_records=tuple(repair_records),
        last_endpoint_refinement_fes=last_endpoint_refinement_fes,
        target_validation_fes=target_validation_fes,
        target_validation=tuple(target_validation),
    )


__all__ = [
    "SplitRepairDiscoveryResult",
    "SplitRepairRecord",
    "discover_hierarchical_soft_split_repair",
]
