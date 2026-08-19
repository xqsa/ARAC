"""Overlapping variable grouping for continuous LSGO benchmarks.

A Python port of the "empty-slot filling" overlap construction by W.Y.Q.
(``generateGroups.m`` + ``repeat.m`` / ``G_network.m``), generalised so that a
caller can request an arbitrary overlap degree ``C``, subgroup size range
``[M, N]`` and random seed, and receive a list of variable-index lists in which
the same variable may legitimately appear in more than one group.

The MATLAB original produces a *binary deceptive* problem.  This module only
implements the grouping logic; the continuous per-subgroup objective (shift,
rotation, base function) lives in :mod:`arac.benchmarks.overlap_objective`.
Keeping the two concerns separate mirrors the AOB split between the benchmark
data bundle and the four ``ackley`` / ``elliptic`` / ``rastrigin`` /
``schwefel`` evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


__all__ = [
    "MAX_OCCURRENCE_CAP_RATIO",
    "OverlapGrouping",
    "OverlapStructure",
    "generate_overlap_groups",
    "membership",
    "shared_variables",
]


# Upper bound on how many groups a single shared variable may be injected into.
# In the MATLAB original ``num_delta = max(1, min(round(C*delta), round(K*delta)))``
# so a variable can be reused at most ``num_delta`` times; we expose the same
# idea as a fraction of the overlap budget ``C``.
MAX_OCCURRENCE_CAP_RATIO = 1.0


@dataclass(frozen=True)
class OverlapGrouping:
    """Parameters describing how an overlapping decomposition is built."""

    dimension: int
    overlap_budget: int
    min_group_size: int
    max_group_size: int
    contiguous: bool
    seed: int
    num_groups: int | None = None
    topology: str = "random"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.overlap_budget < 0:
            raise ValueError("overlap_budget must be non-negative")
        if self.overlap_budget >= self.dimension:
            raise ValueError("overlap_budget must be smaller than dimension")
        if self.min_group_size <= 0 or self.max_group_size < self.min_group_size:
            raise ValueError("group size range is invalid")
        if self.topology not in ("random", "chain", "star"):
            raise ValueError("topology must be 'random', 'chain' or 'star'")
        if self.num_groups is not None:
            if self.num_groups < 2:
                raise ValueError("num_groups must be at least 2")
            if self.num_groups * self.min_group_size > self.dimension:
                raise ValueError("num_groups * min_group_size exceeds dimension")
            if self.num_groups * self.max_group_size < self.dimension:
                raise ValueError("num_groups * max_group_size cannot cover dimension")
        else:
            count = math.ceil(2 * self.dimension / (self.min_group_size + self.max_group_size))
            if count * self.min_group_size > self.dimension:
                raise ValueError("min_group_size is too large for the base partition")


@dataclass(frozen=True)
class OverlapStructure:
    """A frozen overlapping decomposition plus its membership audit.

    ``groups[i]`` is the tuple of variable indices owned (possibly jointly with
    other groups) by subgroup ``i``.  The same variable index may occur in
    several tuples; those are the shared/overlapping variables.
    """

    grouping: OverlapGrouping
    groups: tuple[tuple[int, ...], ...]
    group_sizes: tuple[int, ...]
    overlap_shares: tuple[int, ...]
    membership: tuple[tuple[int, ...], ...]
    shared_variables: tuple[int, ...]
    occurrence: dict[int, int]

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("overlap structure has no groups")
        if len(self.group_sizes) != len(self.groups):
            raise ValueError("group_sizes length drifted")
        if len(self.overlap_shares) not in (0, len(self.groups)):
            raise ValueError("overlap_shares length drifted")
        if any(len(group) != len(set(group)) for group in self.groups):
            raise ValueError("each group must contain unique variables")
        if tuple(len(group) for group in self.groups) != self.group_sizes:
            raise ValueError("group_sizes disagree with group contents")
        # An empty audit tuple means the caller supplied explicit groups but
        # no per-group share decomposition (the Gate 5/6 oracle fixtures use
        # this form).  Membership below remains the authoritative overlap
        # check; enforce slot conservation only when the audit is present.
        if self.overlap_shares and sum(self.group_sizes) - self.grouping.dimension != sum(self.overlap_shares):
            raise ValueError("overlap slots do not match overlap shares")
        if len(self.membership) != self.grouping.dimension:
            raise ValueError("membership must cover every variable")
        if set(self.occurrence) != set(range(self.grouping.dimension)):
            raise ValueError("occurrence must cover every variable")
        for index, owners in enumerate(self.membership):
            if not owners:
                raise ValueError("every variable must belong to at least one group")
            for owner in owners:
                if not 0 <= owner < len(self.groups):
                    raise ValueError("membership owner is out of range")
                if index not in self.groups[owner]:
                    raise ValueError("membership disagrees with group contents")
            if self.occurrence[index] != len(owners):
                raise ValueError("occurrence disagrees with membership")
        expected_shared = tuple(
            index for index, owners in enumerate(self.membership) if len(owners) > 1
        )
        if tuple(self.shared_variables) != expected_shared:
            raise ValueError("shared_variables disagrees with membership")
        for owner, group in enumerate(self.groups):
            if any(owner not in self.membership[variable] for variable in group):
                raise ValueError("membership is missing a group owner")


def _base_group_sizes(
    dimension: int,
    min_size: int,
    max_size: int,
    rng: np.random.Generator,
    *,
    num_groups: int | None = None,
) -> np.ndarray:
    """Pick per-group sizes in ``[min_size, max_size]`` summing to ``dimension``.

    When ``num_groups`` is given (the AOB / CEC'2013 convention: fix K, then
    distribute sizes) it is used directly; otherwise the count is inferred as
    ``ceil(2n / (M+N))`` following ``rand_sum.m``.
    """

    if num_groups is not None:
        count = int(num_groups)
        if count * min_size > dimension:
            raise ValueError("num_groups * min_group_size exceeds dimension")
        if count * max_size < dimension:
            raise ValueError("num_groups * max_group_size cannot cover dimension")
    else:
        count = math.ceil(2 * dimension / (min_size + max_size))
        if count * min_size > dimension:
            raise ValueError("min_group_size is too large for the base partition")
        if count * max_size < dimension:
            count = math.ceil(dimension / max_size)
    raw = min_size + rng.integers(0, max_size - min_size + 1, size=count)
    raw = np.clip(raw, min_size, max_size).astype(int)
    while raw.sum() != dimension:
        if raw.sum() < dimension:
            candidates = np.where(raw < max_size)[0]
            choice = int(rng.choice(candidates))
            raw[choice] += 1
        else:
            candidates = np.where(raw > min_size)[0]
            choice = int(rng.choice(candidates))
            raw[choice] -= 1
    return raw


def _split_overlap_budget(
    overlap_budget: int,
    max_group_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Split ``overlap_budget`` into positive shares, one per injected group.

    Mirrors the ``overlap`` array loop in ``generateGroups.m``: each share is
    ``max(1, min(round(C*beta), max_group_size))``; the final share is trimmed
    so the total equals ``overlap_budget`` exactly.
    """

    shares: list[int] = []
    while sum(shares) != overlap_budget:
        beta = float(rng.random())
        share = max(1, min(round(overlap_budget * beta), max_group_size))
        shares.append(int(share))
        if sum(shares) > overlap_budget:
            shares[-1] = overlap_budget - sum(shares[:-1])
            if shares[-1] <= 0:
                shares = shares[:-1]
                break
    if not shares:
        raise RuntimeError("overlap budget split produced no shares")
    return shares


def _choose_special_groups(
    shares: list[int],
    num_groups: int,
    rng: np.random.Generator,
    *,
    topology: str,
) -> list[int]:
    """Pick, for each overlap share, a distinct base group to receive it.

    Shares are appended on top of the base size, so any group is eligible; we
    still process larger shares first (mirroring ``G_network.m``'s descending
    sort) purely to keep the assignment ordering stable across runs.
    """

    if topology == "star":
        return [1 + (index % max(1, num_groups - 1)) for index in range(len(shares))]
    if topology == "chain":
        return [index % num_groups for index in range(len(shares))]
    return [int(value) for value in rng.integers(0, num_groups, size=len(shares))]


def _inject_shared_variables(
    base_groups: list[list[int]],
    shares: list[int],
    special_groups: list[int],
    rng: np.random.Generator,
    *,
    topology: str,
) -> tuple[list[list[int]], list[int], dict[int, int]]:
    """Append borrowed variables to each special group to create overlap.

    Reinterprets ``repeat.m``'s empty-slot idea in the "no inert dummy"
    setting: every base member is a real variable, so instead of emptying
    slots we *append* ``share`` variables borrowed from donor groups.  The
    borrower samples without replacement from the donor pool, skips variables
    it already owns, and respects a per-variable reuse cap so the overlap
    graph stays diverse.
    """

    groups = [list(group) for group in base_groups]
    cap = max(1, int(round(len(groups) * MAX_OCCURRENCE_CAP_RATIO)))
    occurrence: dict[int, int] = {variable: 1 for group in groups for variable in group}
    occurrence = {variable: 0 for variable in occurrence}
    for group in groups:
        for variable in group:
            occurrence[variable] = occurrence.get(variable, 0) + 1
    injected: list[int] = []

    for share, special in zip(shares, special_groups, strict=True):
        own_variables = set(groups[special])

        num_segments = max(
            1,
            min(len(groups) - 1, min(share, max(1, int(round(share * rng.random()))))),
        )
        if topology in ("chain", "star"):
            num_segments = 1
        split_points = sorted(
            rng.choice(share - 1, size=num_segments - 1, replace=False)
            if share > 1
            else np.array([], dtype=int)
        )
        boundaries = list(split_points) + [share]
        segment_lengths = []
        previous = 0
        for boundary in boundaries:
            segment_lengths.append(boundary - previous)
            previous = boundary

        donors: list[int] = []
        for _segment_length in segment_lengths:
            if topology == "star":
                eligible = [0] if special != 0 else [1]
            elif topology == "chain":
                eligible = [index for index in (special - 1, special + 1) if 0 <= index < len(groups)]
            else:
                eligible = [
                    index
                    for index in range(len(groups))
                    if index != special and index not in donors
                ]
            if topology == "random":
                eligible = [index for index in eligible if index not in donors]
            if not eligible:
                raise RuntimeError("no donor group available for an overlap segment")
            donors.append(int(eligible[0] if topology in ("chain", "star") else rng.choice(eligible)))

        pool: list[int] = []
        for donor in donors:
            pool.extend(groups[donor])
        deduped_pool = list(dict.fromkeys(variable for variable in pool if variable not in own_variables))
        if len(deduped_pool) < share:
            extras = [
                variable
                for index, group in enumerate(groups)
                if index not in donors and index != special
                for variable in group
                if variable not in own_variables
            ]
            deduped_pool.extend(variable for variable in extras if variable not in deduped_pool)
        if len(deduped_pool) < share:
            raise RuntimeError("overlap pool is too small to fill the share")

        chosen = list(rng.choice(np.asarray(deduped_pool, dtype=int), size=share, replace=False))
        accepted: list[int] = []
        for value in chosen:
            value = int(value)
            attempts = 0
            while occurrence.get(value, 0) >= cap and attempts < len(deduped_pool):
                alternatives = [
                    candidate
                    for candidate in deduped_pool
                    if candidate not in accepted and occurrence.get(candidate, 0) < cap
                ]
                if not alternatives:
                    break
                value = int(rng.choice(alternatives))
                attempts += 1
            occurrence[value] = occurrence.get(value, 0) + 1
            accepted.append(value)
            if value not in injected:
                injected.append(value)
            own_variables.add(value)

        groups[special].extend(accepted)

    return groups, injected, occurrence


def _inject_topology_variables(
    base_groups: list[list[int]],
    overlap_budget: int,
    rng: np.random.Generator,
    *,
    topology: str,
) -> tuple[list[list[int]], list[int]]:
    """Inject exact overlap slots using only the requested group graph edges."""

    group_count = len(base_groups)
    if topology == "chain":
        edges = [(index, index + 1) for index in range(group_count - 1)]
    elif topology == "star":
        edges = [(0, index) for index in range(1, group_count)]
    else:
        raise ValueError("structured injection requires chain or star topology")

    groups = [list(group) for group in base_groups]
    shares = [0] * group_count
    edge_offset = 0
    for _ in range(overlap_budget):
        candidates_by_edge: list[tuple[int, list[tuple[int, int]]]] = []
        for edge_index in range(len(edges)):
            left, right = edges[(edge_offset + edge_index) % len(edges)]
            orientations = []
            for borrower, donor in ((left, right), (right, left)):
                available = [
                    variable
                    for variable in base_groups[donor]
                    if variable not in groups[borrower]
                    and sum(variable in group for group in groups) == 1
                ]
                orientations.extend((borrower, variable) for variable in available)
            if orientations:
                candidates_by_edge.append((edge_index, orientations))
        if not candidates_by_edge:
            raise RuntimeError(
                f"{topology} topology cannot realize overlap_budget={overlap_budget} "
                "without duplicate variables"
            )

        # Prefer the next graph edge so low budgets cover the requested topology;
        # skip exhausted edges without introducing non-topological fallback edges.
        edge_index, orientations = candidates_by_edge[0]
        borrower, variable = orientations[int(rng.integers(0, len(orientations)))]
        groups[borrower].append(variable)
        shares[borrower] += 1
        edge_offset = (edge_offset + edge_index + 1) % len(edges)
    return groups, shares


def _build_membership(
    groups: list[list[int]],
    dimension: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], dict[int, int]]:
    """Derive per-variable ownership, the shared-variable list and occurrence counts."""

    membership: list[list[int]] = [[] for _ in range(dimension)]
    for group_index, group in enumerate(groups):
        for variable in group:
            if not 0 <= variable < dimension:
                raise ValueError("group references an out-of-range variable")
            if group_index not in membership[variable]:
                membership[variable].append(group_index)
    occurrence: dict[int, int] = {}
    shared: list[int] = []
    for variable in range(dimension):
        count = len(membership[variable])
        occurrence[variable] = count
        if count > 1:
            shared.append(variable)
    return (
        tuple(tuple(owners) for owners in membership),
        tuple(shared),
        occurrence,
    )


def generate_overlap_groups(
    dimension: int,
    *,
    overlap_budget: int,
    min_group_size: int,
    max_group_size: int,
    contiguous: bool = True,
    seed: int = 0,
    num_groups: int | None = None,
    topology: str = "random",
) -> OverlapStructure:
    """Generate one overlapping decomposition of ``dimension`` variables.

    Every one of the ``dimension`` variables participates in at least one
    group.  ``overlap_budget`` extra *slots* are then distributed across a few
    "special" groups and filled with variables already owned by other groups,
    so that ``sum(group_sizes) == dimension + overlap_budget`` and exactly
    ``overlap_budget`` cross-group memberships are introduced.  This keeps the
    MATLAB "empty-slot filling" idea while guaranteeing no variable is left as
    an inert dummy.

    ``num_groups`` fixes the subgroup count K explicitly (the AOB / CEC'2013
    convention, e.g. K=20); when ``None`` the count is inferred from the size
    range as in ``rand_sum.m``.
    """

    grouping = OverlapGrouping(
        dimension=dimension,
        overlap_budget=overlap_budget,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        contiguous=contiguous,
        seed=seed,
        num_groups=num_groups,
        topology=topology,
    )
    rng = np.random.default_rng(seed)

    base_sizes = _base_group_sizes(
        dimension, min_group_size, max_group_size, rng, num_groups=num_groups
    )
    order = np.arange(dimension) if contiguous else rng.permutation(dimension)
    cursor = 0
    base_groups: list[list[int]] = []
    for size in base_sizes:
        base_groups.append([int(value) for value in order[cursor : cursor + int(size)]])
        cursor += int(size)
    if cursor != dimension:
        raise RuntimeError("base partition did not exhaust every variable")

    shares: list[int] = []
    if overlap_budget == 0:
        groups = base_groups
        share_audit = [0] * len(groups)
    elif topology in ("chain", "star"):
        groups, shares = _inject_topology_variables(
            base_groups, overlap_budget, rng, topology=topology
        )
        share_audit = shares
    else:
        shares = _split_overlap_budget(overlap_budget, max_group_size, rng)
        special_groups = _choose_special_groups(
            shares, len(base_groups), rng, topology=topology
        )
        groups, _injected, _occurrence = _inject_shared_variables(
            base_groups, shares, special_groups, rng, topology=topology
        )
        share_audit = [0] * len(groups)
        for special, share in zip(special_groups, shares, strict=True):
            share_audit[special] += int(share)

    membership_tuple, shared, occurrence_full = _build_membership(groups, dimension)
    if overlap_budget > 0 and not shared:
        raise RuntimeError("construction produced no shared variables")

    return OverlapStructure(
        grouping=grouping,
        groups=tuple(tuple(group) for group in groups),
        group_sizes=tuple(len(group) for group in groups),
        overlap_shares=tuple(share_audit),
        membership=membership_tuple,
        shared_variables=shared,
        occurrence=occurrence_full,
    )


def membership(structure: OverlapStructure) -> tuple[tuple[int, ...], ...]:
    """Return, for every variable, the groups that own it (length-1 for exclusive)."""

    return structure.membership


def shared_variables(structure: OverlapStructure) -> tuple[int, ...]:
    """Return the variables that belong to more than one group."""

    return structure.shared_variables
