"""Oracle overlap coordination primitives for the first Phase-II gate.

The structures in this module are intentionally independent from the current
identity-blind Phase-I checkpoint contract.  They describe a known overlap
topology so the coordination hypothesis can be tested without conflating
structure discovery errors with shared-variable repair errors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping

import numpy as np

from arac.runtime.ledger import EvaluationLedger
from arac.runtime.optimizers import ResumableOptimizerSession


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class OverlapStructure:
    """Known variable-to-group memberships used by the oracle Phase-II test."""

    dimension: int
    groups: tuple[tuple[int, ...], ...]
    member_confidences: tuple[tuple[int, int, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not self.groups or any(not group for group in self.groups):
            raise ValueError("groups must be non-empty")
        normalized = []
        for group in self.groups:
            if any(isinstance(index, bool) or not isinstance(index, int) for index in group):
                raise ValueError("group variable indices must be integers")
            values = tuple(group)
            if len(set(values)) != len(values) or any(index < 0 or index >= self.dimension for index in values):
                raise ValueError("group variable indices must be unique and in bounds")
            normalized.append(values)
        if tuple(normalized) != self.groups:
            raise ValueError("group indices must be supplied as integer tuples")
        covered = {index for group in self.groups for index in group}
        if covered != set(range(self.dimension)):
            raise ValueError("groups must cover every variable at least once")
        for variable, group, confidence in self.member_confidences:
            if (
                isinstance(variable, bool)
                or not isinstance(variable, int)
                or isinstance(group, bool)
                or not isinstance(group, int)
                or variable not in range(self.dimension)
                or group not in range(len(self.groups))
            ):
                raise ValueError("member confidence references an unknown variable or group")
            value = _finite(confidence, "member confidence")
            if not 0.0 <= value <= 1.0:
                raise ValueError("member confidence must be in [0, 1]")
            if variable not in self.groups[group]:
                raise ValueError("member confidence must reference a group member")
        keys = [(variable, group) for variable, group, _ in self.member_confidences]
        if len(set(keys)) != len(keys):
            raise ValueError("member confidences must be unique per variable and group")

    @property
    def memberships(self) -> dict[int, tuple[int, ...]]:
        result: dict[int, list[int]] = defaultdict(list)
        for group, variables in enumerate(self.groups):
            for variable in variables:
                result[variable].append(group)
        return {variable: tuple(groups) for variable, groups in sorted(result.items())}

    @property
    def shared_variables(self) -> tuple[int, ...]:
        return tuple(variable for variable, groups in self.memberships.items() if len(groups) > 1)

    def owners(self, variable: int) -> tuple[int, ...]:
        if variable not in self.memberships:
            raise ValueError(f"variable {variable} is not covered by the overlap structure")
        return self.memberships[variable]

    def confidence(self, variable: int, group: int) -> float:
        if group not in self.owners(variable):
            raise ValueError(f"group {group} does not own variable {variable}")
        for item_variable, item_group, value in self.member_confidences:
            if (item_variable, item_group) == (variable, group):
                return float(value)
        return 1.0

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        """Return connected group components induced by shared variables."""

        adjacency = {group: set() for group in range(len(self.groups))}
        for owners in self.memberships.values():
            for left in owners:
                adjacency[left].update(right for right in owners if right != left)
        components: list[tuple[int, ...]] = []
        unseen = set(adjacency)
        while unseen:
            root = min(unseen)
            stack = [root]
            component = set()
            while stack:
                group = stack.pop()
                if group in component:
                    continue
                component.add(group)
                unseen.discard(group)
                stack.extend(adjacency[group] - component)
            components.append(tuple(sorted(component)))
        return tuple(components)


@dataclass(frozen=True)
class LocalProposal:
    """A component-local proposal for one or more variables."""

    group: int
    values: tuple[tuple[int, float], ...]
    improvement: float
    uncertainty: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if isinstance(self.group, bool) or not isinstance(self.group, int) or self.group < 0:
            raise ValueError("proposal group must be a non-negative integer")
        if not self.values:
            raise ValueError("proposal values must be non-empty")
        value_keys = [variable for variable, _ in self.values]
        uncertainty_keys = [variable for variable, _ in self.uncertainty]
        if len(set(value_keys)) != len(value_keys) or len(set(uncertainty_keys)) != len(uncertainty_keys):
            raise ValueError("proposal variables must be unique")
        for variable, value in self.values:
            if isinstance(variable, bool) or not isinstance(variable, int) or variable < 0:
                raise ValueError("proposal variable must be a non-negative integer")
            _finite(value, "proposal value")
        for variable, value in self.uncertainty:
            if (
                isinstance(variable, bool)
                or not isinstance(variable, int)
                or variable not in value_keys
                or _finite(value, "proposal uncertainty") < 0.0
            ):
                raise ValueError("proposal uncertainty must cover proposal variables and be non-negative")
        if set(value_keys) != set(uncertainty_keys):
            raise ValueError("proposal uncertainty must cover exactly the proposal variables")
        _finite(self.improvement, "proposal improvement")

    def value(self, variable: int) -> float:
        for item_variable, value in self.values:
            if item_variable == variable:
                return float(value)
        raise KeyError(variable)

    def sigma(self, variable: int) -> float:
        for item_variable, value in self.uncertainty:
            if item_variable == variable:
                return float(value)
        raise KeyError(variable)


@dataclass(frozen=True)
class ProposalResidual:
    variable: int
    weighted_mean: float
    between_variance: float
    within_variance: float
    conflict_score: float
    weights: tuple[tuple[int, float], ...]


class ConflictLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _proposal_weight(structure: OverlapStructure, proposal: LocalProposal, variable: int, epsilon: float) -> float:
    confidence = structure.confidence(variable, proposal.group)
    contribution = max(0.0, float(proposal.improvement))
    stability = 1.0 / (proposal.sigma(variable) + epsilon)
    return max(epsilon, confidence) * (1.0 + contribution) * stability


def compute_proposal_residuals(
    structure: OverlapStructure,
    proposals: Iterable[LocalProposal],
    *,
    variables: Iterable[int] | None = None,
    epsilon: float = 1e-12,
) -> dict[int, ProposalResidual]:
    """Compute weighted between/within variance for each shared variable."""

    if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    proposal_list = tuple(proposals)
    by_group = {proposal.group: proposal for proposal in proposal_list}
    if len(by_group) != len(proposal_list):
        raise ValueError("proposals must contain at most one proposal per group")
    unknown = set(by_group) - set(range(len(structure.groups)))
    if unknown:
        raise ValueError(f"proposals reference unknown groups: {sorted(unknown)}")
    if variables is None:
        selected_variables = structure.shared_variables
    else:
        selected_variables = tuple(variables)
        if len(set(selected_variables)) != len(selected_variables):
            raise ValueError("residual variables must be unique")
        if any(variable not in structure.shared_variables for variable in selected_variables):
            raise ValueError("residual variables must be shared variables in the structure")
    result: dict[int, ProposalResidual] = {}
    for variable in selected_variables:
        owners = structure.owners(variable)
        for group in owners:
            if group not in by_group:
                raise ValueError(f"proposals must cover every owner of shared variable {variable}")
            proposal_variables = {item[0] for item in by_group[group].values}
            if variable not in proposal_variables:
                raise ValueError(f"proposals must cover every owner of shared variable {variable}")
        weights = np.asarray(
            [_proposal_weight(structure, by_group[group], variable, epsilon) for group in owners],
            dtype=float,
        )
        values = np.asarray([by_group[group].value(variable) for group in owners], dtype=float)
        sigmas = np.asarray([by_group[group].sigma(variable) for group in owners], dtype=float)
        normalized = weights / float(np.sum(weights))
        mean = float(np.dot(normalized, values))
        between = float(np.dot(normalized, (values - mean) ** 2))
        within = float(np.dot(normalized, sigmas**2))
        result[variable] = ProposalResidual(
            variable=variable,
            weighted_mean=mean,
            between_variance=between,
            within_variance=within,
            conflict_score=between / (within + epsilon),
            weights=tuple((group, float(weight)) for group, weight in zip(owners, normalized, strict=True)),
        )
    return result


@dataclass(frozen=True)
class CoordinationCandidate:
    name: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class CoordinationResult:
    component: tuple[int, ...]
    conflict_level: ConflictLevel
    residuals: tuple[ProposalResidual, ...]
    candidates: tuple[CoordinationCandidate, ...]
    candidate_errors: tuple[tuple[str, float], ...]
    accepted: bool
    accepted_candidate: str | None
    best_error_before: float
    best_error_after: float
    conflict_streak: int = 0
    ctp_triggered: bool = False
    ctp_consumed_fes: int = 0
    ctp_best_error_before: float | None = None


@dataclass(frozen=True)
class FullContextWritebackRound:
    """One counted proposal/reflection decision from the shared context."""

    round_index: int
    group: int
    proposal_error: float
    reflection_error: float
    best_error_before: float
    best_error_after: float
    accepted: bool


@dataclass(frozen=True)
class FullContextWritebackResult:
    """Auditable result of sequential complete-proposal context updates."""

    component: tuple[int, ...]
    rounds: tuple[FullContextWritebackRound, ...]
    consumed_fes: int
    best_error_before: float
    best_error_after: float


@dataclass(frozen=True)
class ProposalNeighborhoodRound:
    """One proposal-conditioned local-neighborhood evaluation."""

    round_index: int
    group: int
    error: float
    best_error_before: float
    best_error_after: float
    accepted: bool


@dataclass(frozen=True)
class ProposalNeighborhoodResult:
    """Auditable result of proposal-conditioned complete-context search."""

    component: tuple[int, ...]
    rounds: tuple[ProposalNeighborhoodRound, ...]
    consumed_fes: int
    best_error_before: float
    best_error_after: float


class OverlapCoordinator:
    """Generate and strictly arbitrate complete candidates for one overlap component."""

    def __init__(
        self,
        structure: OverlapStructure,
        ledger: EvaluationLedger,
        *,
        medium_threshold: float = 1.0,
        high_threshold: float = 2.0,
        epsilon: float = 1e-12,
    ) -> None:
        if not isinstance(structure, OverlapStructure):
            raise TypeError("structure must be OverlapStructure")
        if not isinstance(ledger, EvaluationLedger):
            raise TypeError("ledger must be EvaluationLedger")
        if structure.dimension != ledger.problem.dimension:
            raise ValueError("structure and ledger dimensions disagree")
        if not 0.0 <= medium_threshold <= high_threshold:
            raise ValueError("conflict thresholds must be ordered and non-negative")
        if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
            raise ValueError("epsilon must be finite and positive")
        self.structure = structure
        self.ledger = ledger
        self.medium_threshold = float(medium_threshold)
        self.high_threshold = float(high_threshold)
        self.epsilon = float(epsilon)
        self._conflict_streaks: dict[tuple[int, ...], int] = {}

    def conflict_streak(self, component: tuple[int, ...]) -> int:
        """Return the number of consecutive high observations for a component."""

        return self._conflict_streaks.get(tuple(component), 0)

    def full_context_writeback(
        self,
        component: tuple[int, ...],
        proposals: Iterable[LocalProposal],
        *,
        rounds: int = 16,
    ) -> FullContextWritebackResult:
        """Write complete group proposals into one evolving global context.

        Each round evaluates the selected group proposal and its reflection
        around the current strict-best incumbent.  The ledger owns acceptance,
        so a rejected round cannot degrade the shared context.
        """

        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
            raise ValueError("rounds must be a positive integer")
        component = tuple(component)
        self._component_variables(component)
        proposal_list = tuple(proposals)
        by_group = {proposal.group: proposal for proposal in proposal_list}
        if len(by_group) != len(proposal_list) or set(by_group) != set(component):
            raise ValueError("proposals must cover exactly one overlap component")
        for group in component:
            proposal_variables = {variable for variable, _ in by_group[group].values}
            if proposal_variables != set(self.structure.groups[group]):
                raise ValueError(f"proposal {group} must cover exactly its group variables")
        requested_fes = 2 * rounds
        if requested_fes > self.ledger.remaining:
            raise ValueError("full-context write-back exceeds the remaining FE budget")

        owner_order = tuple(
            sorted(component, key=lambda group: (-by_group[group].improvement, group))
        )
        lower = self.ledger.problem.lower_array
        upper = self.ledger.problem.upper_array
        before_all = float(self.ledger.best_error)
        start_fes = self.ledger.count
        trace = []
        for round_index in range(rounds):
            group = owner_order[round_index % len(owner_order)]
            proposal = by_group[group]
            base = self.ledger.best_x
            variables = np.asarray(self.structure.groups[group], dtype=int)
            target = np.asarray(
                [proposal.value(int(variable)) for variable in variables],
                dtype=float,
            )
            proposal_candidate = base.copy()
            proposal_candidate[variables] = target
            reflection_candidate = base.copy()
            reflection_candidate[variables] = base[variables] - (target - base[variables])
            batch = np.asarray((proposal_candidate, reflection_candidate), dtype=float)
            np.clip(batch, lower, upper, out=batch)
            before = float(self.ledger.best_error)
            errors = np.asarray(self.ledger.evaluate(batch), dtype=float)
            after = float(self.ledger.best_error)
            trace.append(
                FullContextWritebackRound(
                    round_index=round_index,
                    group=group,
                    proposal_error=float(errors[0]),
                    reflection_error=float(errors[1]),
                    best_error_before=before,
                    best_error_after=after,
                    accepted=after < before,
                )
            )
        consumed = self.ledger.count - start_fes
        if consumed != requested_fes:
            raise RuntimeError("full-context write-back FE accounting drifted")
        return FullContextWritebackResult(
            component=component,
            rounds=tuple(trace),
            consumed_fes=consumed,
            best_error_before=before_all,
            best_error_after=float(self.ledger.best_error),
        )

    def proposal_neighborhood_writeback(
        self,
        component: tuple[int, ...],
        proposals: Iterable[LocalProposal],
        *,
        budget_fes: int,
        seed: int,
    ) -> ProposalNeighborhoodResult:
        """Search proposal-conditioned complete group neighborhoods.

        Each FE samples the selected group's full proposal coordinates around
        its proposal uncertainty, writes them into the current global context,
        and lets the strict-best ledger decide whether to retain the candidate.
        The same evolving context is therefore used by every subsequent sample.
        """

        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        component = tuple(component)
        self._component_variables(component)
        proposal_list = tuple(proposals)
        by_group = {proposal.group: proposal for proposal in proposal_list}
        if len(by_group) != len(proposal_list) or set(by_group) != set(component):
            raise ValueError("proposals must cover exactly one overlap component")
        for group in component:
            proposal_variables = {variable for variable, _ in by_group[group].values}
            if proposal_variables != set(self.structure.groups[group]):
                raise ValueError(f"proposal {group} must cover exactly its group variables")
        if budget_fes > self.ledger.remaining:
            raise ValueError("proposal neighborhood write-back exceeds the remaining FE budget")

        owner_order = tuple(
            sorted(component, key=lambda group: (-by_group[group].improvement, group))
        )
        rng = np.random.default_rng(seed)
        lower = self.ledger.problem.lower_array
        upper = self.ledger.problem.upper_array
        before_all = float(self.ledger.best_error)
        start_fes = self.ledger.count
        trace: list[ProposalNeighborhoodRound] = []
        for round_index in range(budget_fes):
            group = owner_order[round_index % len(owner_order)]
            proposal = by_group[group]
            candidate = self.ledger.best_x
            variables = np.asarray(self.structure.groups[group], dtype=int)
            target = np.asarray(
                [proposal.value(int(variable)) for variable in variables],
                dtype=float,
            )
            scales = np.asarray(
                [max(np.finfo(float).eps, proposal.sigma(int(variable))) for variable in variables],
                dtype=float,
            )
            candidate[variables] = target + rng.normal(0.0, scales)
            np.clip(candidate, lower, upper, out=candidate)
            before = float(self.ledger.best_error)
            error = float(self.ledger.evaluate(candidate))
            after = float(self.ledger.best_error)
            trace.append(
                ProposalNeighborhoodRound(
                    round_index=round_index,
                    group=group,
                    error=error,
                    best_error_before=before,
                    best_error_after=after,
                    accepted=after < before,
                )
            )
        consumed = self.ledger.count - start_fes
        if consumed != budget_fes:
            raise RuntimeError("proposal neighborhood write-back FE accounting drifted")
        return ProposalNeighborhoodResult(
            component=component,
            rounds=tuple(trace),
            consumed_fes=consumed,
            best_error_before=before_all,
            best_error_after=float(self.ledger.best_error),
        )

    def _level(self, residuals: Iterable[ProposalResidual]) -> ConflictLevel:
        maximum = max((item.conflict_score for item in residuals), default=0.0)
        if maximum >= self.high_threshold:
            return ConflictLevel.HIGH
        if maximum >= self.medium_threshold:
            return ConflictLevel.MEDIUM
        return ConflictLevel.LOW

    def _component_variables(self, component: tuple[int, ...]) -> tuple[int, ...]:
        component_set = set(component)
        if (
            not component_set
            or len(component_set) != len(component)
            or not component_set.issubset(range(len(self.structure.groups)))
        ):
            raise ValueError("component must contain unique known groups")
        variables = tuple(
            variable
            for variable in self.structure.shared_variables
            if component_set.intersection(self.structure.owners(variable))
        )
        if any(not set(self.structure.owners(variable)).issubset(component_set) for variable in variables):
            raise ValueError("component must contain every owner of its shared variables")
        return variables

    def _repair_scope(
        self,
        component: tuple[int, ...],
        scope: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """Return the explicit GCB repair scope without silent expansion."""

        available = set(self._component_variables(tuple(component)))
        if scope is None:
            return tuple(sorted(available))
        selected = tuple(scope)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("repair scope must be a non-empty tuple of unique variables")
        if selected != tuple(sorted(selected)):
            raise ValueError("repair scope must be sorted")
        if set(selected) - available:
            raise ValueError("repair scope must be shared variables of the component")
        return selected

    def candidates(
        self,
        component: tuple[int, ...],
        incumbent: np.ndarray,
        proposals: Iterable[LocalProposal],
    ) -> tuple[CoordinationCandidate, ...]:
        """Build incumbent, owner, weighted-mean and weighted-median candidates."""

        vector = np.asarray(incumbent, dtype=float)
        if vector.shape != (self.structure.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError("incumbent must be a finite vector matching the structure dimension")
        proposal_list = tuple(proposals)
        variables = self._component_variables(component)
        residuals = compute_proposal_residuals(
            self.structure,
            proposal_list,
            variables=variables,
            epsilon=self.epsilon,
        )
        by_group = {proposal.group: proposal for proposal in proposal_list}

        for group in component:
            if group not in by_group:
                raise ValueError(f"proposals must cover component group {group}")
            proposal_variables = {variable for variable, _ in by_group[group].values}
            if proposal_variables != set(self.structure.groups[group]):
                raise ValueError(f"proposal {group} must cover exactly its group variables")

        def make(name: str, values: Mapping[int, float]) -> CoordinationCandidate:
            candidate = vector.copy()
            for variable, value in values.items():
                candidate[variable] = float(value)
            # Local proposal optimizers may use an unbounded offspring model.
            # Consensus is a public objective-evaluation boundary, so close
            # every assembled candidate to the problem box before charging FE.
            np.clip(
                candidate,
                self.ledger.problem.lower_array,
                self.ledger.problem.upper_array,
                out=candidate,
            )
            return CoordinationCandidate(name, tuple(float(value) for value in candidate))

        # Keep the complete-candidate assembly explicit: local proposals first
        # update unique variables, then the shared-variable policy overwrites
        # only coordinates that need consensus.
        component_variables_seen: set[int] = set()
        for group in component:
            component_variables_seen.update(self.structure.groups[group])
        base_values: dict[int, float] = {}
        for variable in sorted(component_variables_seen):
            owners = self.structure.owners(variable)
            if len(owners) == 1:
                base_values[variable] = by_group[owners[0]].value(variable)

        candidates = [make("incumbent", {})]
        owner_values: dict[int, float] = {}
        mean_values: dict[int, float] = {}
        median_values: dict[int, float] = {}
        owner_group = max(
            component,
            key=lambda group: (by_group[group].improvement, -group),
        )
        for variable in variables:
            residual = residuals[variable]
            owners = self.structure.owners(variable)
            owner_values[variable] = (
                by_group[owner_group].value(variable)
                if owner_group in owners
                else residual.weighted_mean
            )
            mean_values[variable] = residual.weighted_mean
            values = sorted(by_group[group].value(variable) for group in owners)
            weights = sorted(
                ((by_group[group].value(variable), weight) for group, weight in residual.weights),
                key=lambda item: item[0],
            )
            midpoint = 0.5
            cumulative = 0.0
            median_values[variable] = values[-1]
            for value, weight in weights:
                cumulative += weight
                if cumulative >= midpoint:
                    median_values[variable] = value
                    break
        candidates.extend(
            [
                make("owner", {**base_values, **owner_values}),
                make("weighted_mean", {**base_values, **mean_values}),
                make("weighted_median", {**base_values, **median_values}),
            ]
        )
        return tuple(candidates)

    def coordinate(
        self,
        component: tuple[int, ...],
        proposals: Iterable[LocalProposal],
        *,
        ctp_budget_fes: int = 0,
        ctp_seed: int = 0,
        ctp_strategy: str = "random",
        search_base: np.ndarray | None = None,
    ) -> CoordinationResult:
        """Arbitrate proposals and optionally repair a persistent high-conflict core.

        CTP is intentionally opt-in through an explicit budget.  The first high
        residual establishes a conflict; only a second consecutive high residual
        for the same group component is persistent enough to trigger repair.
        """

        if isinstance(ctp_budget_fes, bool) or not isinstance(ctp_budget_fes, int) or ctp_budget_fes < 0:
            raise ValueError("ctp_budget_fes must be a non-negative integer")
        if isinstance(ctp_seed, bool) or not isinstance(ctp_seed, int) or ctp_seed < 0:
            raise ValueError("ctp_seed must be a non-negative integer")
        if ctp_strategy not in {
            "random",
            "joint_cmaes",
            "sequential_joint_patch",
            "sequential_shared_patch",
            "sequential_coordinate_patch",
        }:
            raise ValueError(f"unsupported ctp_strategy: {ctp_strategy}")

        proposal_list = tuple(proposals)
        base = self.ledger.best_x if search_base is None else np.asarray(search_base, dtype=float)
        if base.shape != (self.structure.dimension,) or not np.all(np.isfinite(base)):
            raise ValueError("search_base must be a finite vector matching the structure dimension")
        if np.any(base < self.ledger.problem.lower_array) or np.any(base > self.ledger.problem.upper_array):
            raise ValueError("search_base must stay inside the problem bounds")
        candidates = self.candidates(component, base, proposal_list)
        variables = self._component_variables(component)
        residuals = compute_proposal_residuals(
            self.structure,
            proposal_list,
            variables=variables,
            epsilon=self.epsilon,
        )
        level = self._level(residuals.values())
        if level is ConflictLevel.LOW:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.name in {"incumbent", "owner", "weighted_mean"}
            )
        component_key = tuple(component)
        if level is ConflictLevel.HIGH:
            streak = self._conflict_streaks.get(component_key, 0) + 1
        else:
            streak = 0
        self._conflict_streaks[component_key] = streak
        before = float(self.ledger.best_error)
        errors = np.asarray(self.ledger.evaluate(np.asarray([item.vector for item in candidates])), dtype=float)
        after = float(self.ledger.best_error)
        candidate_errors = tuple((candidate.name, float(error)) for candidate, error in zip(candidates, errors, strict=True))
        improving = [(name, error) for name, error in candidate_errors if error < before]
        accepted_name = min(improving, key=lambda item: item[1])[0] if improving else None
        ctp_triggered = level is ConflictLevel.HIGH and streak >= 2 and ctp_budget_fes > 0
        ctp_consumed = 0
        ctp_error: float | None = None
        ctp_before = after if ctp_triggered else None
        if ctp_triggered:
            repair_budget = min(ctp_budget_fes, self.ledger.remaining)
            if ctp_strategy == "joint_cmaes":
                ctp_consumed = self._optimize_shared_core(
                    component,
                    proposal_list,
                    budget_fes=repair_budget,
                    seed=ctp_seed,
                    base=self.ledger.best_x,
                )
            elif ctp_strategy == "sequential_joint_patch":
                ctp_consumed = self._repair_sequential_joint_patch(
                    component,
                    proposal_list,
                    budget_fes=repair_budget,
                    seed=ctp_seed,
                    base=self.ledger.best_x,
                )
            elif ctp_strategy == "sequential_shared_patch":
                ctp_consumed = self._repair_sequential_shared_patch(
                    component,
                    proposal_list,
                    budget_fes=repair_budget,
                    seed=ctp_seed,
                    base=self.ledger.best_x,
                )
            elif ctp_strategy == "sequential_coordinate_patch":
                ctp_consumed = self._repair_sequential_coordinate_patch(
                    component,
                    proposal_list,
                    budget_fes=repair_budget,
                    seed=ctp_seed,
                    base=self.ledger.best_x,
                )
            else:
                ctp_consumed = self._repair_shared_core(
                    component,
                    proposal_list,
                    budget_fes=repair_budget,
                    seed=ctp_seed,
                    base=base,
                )
            after = float(self.ledger.best_error)
            ctp_error = after
            if ctp_error < before and (
                accepted_name is None
                or ctp_error < min(error for _, error in candidate_errors)
            ):
                accepted_name = "ctp_shared_core"
                candidate_errors = (*candidate_errors, (accepted_name, ctp_error))
        return CoordinationResult(
            component=tuple(component),
            conflict_level=level,
            residuals=tuple(residuals[variable] for variable in sorted(residuals)),
            candidates=candidates,
            candidate_errors=candidate_errors,
            accepted=accepted_name is not None,
            accepted_candidate=accepted_name,
            best_error_before=before,
            best_error_after=after,
            conflict_streak=streak,
            ctp_triggered=ctp_triggered,
            ctp_consumed_fes=ctp_consumed,
            ctp_best_error_before=ctp_before,
            )

    def dispatch_repair(
        self,
        component: tuple[int, ...],
        proposals: Iterable[LocalProposal],
        *,
        budget_fes: int,
        seed: int,
        strategy: str,
        scope: tuple[int, ...] | None = None,
    ) -> int:
        """Execute one pre-planned CTP repair without re-arbitrating candidates.

        The GCB dispatch planner calls this after a separate arbitration pass.
        The budget is reserved by the plan before execution and the repair may
        not implicitly consume any other budget category.
        """

        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if strategy not in {
            "joint_cmaes",
            "sequential_joint_patch",
            "sequential_shared_patch",
            "sequential_coordinate_patch",
        }:
            raise ValueError(f"unsupported dispatch strategy: {strategy}")
        if budget_fes > self.ledger.remaining:
            raise ValueError("dispatched repair exceeds the remaining FE budget")
        proposal_list = tuple(proposals)
        if scope is not None:
            self._repair_scope(component, scope)
        if strategy == "joint_cmaes":
            return self._optimize_shared_core(
                component,
                proposal_list,
                budget_fes=budget_fes,
                seed=seed,
                base=self.ledger.best_x,
                scope=scope,
            )
        if strategy == "sequential_joint_patch":
            return self._repair_sequential_joint_patch(
                component,
                proposal_list,
                budget_fes=budget_fes,
                seed=seed,
                base=self.ledger.best_x,
                scope=scope,
            )
        if strategy == "sequential_shared_patch":
            return self._repair_sequential_shared_patch(
                component,
                proposal_list,
                budget_fes=budget_fes,
                seed=seed,
                base=self.ledger.best_x,
                scope=scope,
            )
        return self._repair_sequential_coordinate_patch(
            component,
            proposal_list,
            budget_fes=budget_fes,
            seed=seed,
            base=self.ledger.best_x,
            scope=scope,
        )

    def _repair_shared_core(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
        *,
        budget_fes: int,
        seed: int,
        base: np.ndarray | None = None,
        scope: tuple[int, ...] | None = None,
    ) -> int:
        """Run a bounded derivative-free search over only shared core variables."""

        if budget_fes <= 0:
            return 0
        variables = self._repair_scope(component, scope)
        if not variables:
            return 0
        residuals = compute_proposal_residuals(
            self.structure,
            proposals,
            variables=variables,
            epsilon=self.epsilon,
        )
        search_base = self.ledger.best_x if base is None else np.asarray(base, dtype=float)
        rng = np.random.default_rng(seed)
        centers = np.asarray([residuals[variable].weighted_mean for variable in variables], dtype=float)
        by_group = {proposal.group: proposal for proposal in proposals}
        spreads = np.asarray(
            [
                max(
                    self.epsilon,
                    max(
                        abs(
                            by_group[group].value(variable)
                            - residuals[variable].weighted_mean
                        )
                        for group in self.structure.owners(variable)
                    ),
                    max(
                        by_group[group].sigma(variable)
                        for group in self.structure.owners(variable)
                    ),
                )
                for variable in variables
            ],
            dtype=float,
        )
        batch = np.repeat(search_base[np.newaxis, :], budget_fes, axis=0)
        batch[:, variables] = centers + rng.normal(0.0, spreads, size=(budget_fes, len(variables)))
        batch[0, variables] = centers
        batch = np.clip(batch, self.ledger.problem.lower_array, self.ledger.problem.upper_array)
        self.ledger.evaluate(batch)
        return budget_fes

    def _optimize_shared_core(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
        *,
        budget_fes: int,
        seed: int,
        base: np.ndarray | None = None,
        scope: tuple[int, ...] | None = None,
    ) -> int:
        """Optimize the shared core and its one-hop boundary jointly.

        This opt-in path gives CTP a population state that can learn correlations
        between overlapping variables while allowing the component's unique
        coordinates to co-adapt.  The legacy random repair remains available for
        historical gates and direct diagnostics.
        """

        if budget_fes <= 0:
            return 0
        if scope is None:
            variables = tuple(
                sorted(
                    {
                        variable
                        for group in component
                        for variable in self.structure.groups[group]
                    }
                )
            )
        else:
            variables = self._repair_scope(component, scope)
        if not variables:
            return 0
        if budget_fes < 2:
            return self._repair_shared_core(
                component,
                proposals,
                budget_fes=budget_fes,
                seed=seed,
                base=base,
                scope=scope,
            )
        shared_variables = self._component_variables(component) if scope is None else variables
        residuals = compute_proposal_residuals(
            self.structure,
            proposals,
            variables=shared_variables,
            epsilon=self.epsilon,
        )
        by_group = {proposal.group: proposal for proposal in proposals}
        centers: list[float] = []
        spreads: list[float] = []
        for variable in variables:
            owners = self.structure.owners(variable)
            if variable in residuals:
                center = residuals[variable].weighted_mean
                spread = max(
                    max(
                        abs(by_group[group].value(variable) - center)
                        for group in owners
                    ),
                    max(by_group[group].sigma(variable) for group in owners),
                )
            else:
                owner = owners[0]
                center = by_group[owner].value(variable)
                spread = by_group[owner].sigma(variable)
            centers.append(float(center))
            spreads.append(max(self.epsilon, float(spread)))
        center_array = np.asarray(centers, dtype=float)
        spread_array = np.asarray(spreads, dtype=float)
        search_base = self.ledger.best_x if base is None else np.asarray(base, dtype=float)
        if search_base.shape != (self.structure.dimension,):
            raise ValueError("base must match the structure dimension")
        sigma = max(self.epsilon, float(np.median(spread_array)))
        dimensions = tuple(variables)
        algorithm = "cmaes" if len(dimensions) <= 256 else "sepcmaes"
        session = ResumableOptimizerSession(
            algorithm,
            problem=self.ledger.problem,
            ledger=self.ledger,
            initial_mean=tuple(float(value) for value in center_array),
            sigma=sigma,
            seed=seed,
            budget_fes=budget_fes,
            population_size=max(2, min(8, budget_fes)),
            dimensions=dimensions,
            anchor=search_base,
        )
        start = self.ledger.count
        session.step(budget_fes)
        if self.ledger.count - start != budget_fes:
            raise RuntimeError("joint shared-core CTP drifted from its exact FE request")
        return budget_fes

    def _repair_sequential_joint_patch(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
        *,
        budget_fes: int,
        seed: int,
        base: np.ndarray | None = None,
        include_boundary: bool = True,
        scope: tuple[int, ...] | None = None,
    ) -> int:
        """Apply a deterministic, feedback-driven joint patch to one component.

        The patch treats the component's shared variables, and optionally its
        unique boundary variables, as one coupled block. Each owner supplies a
        directional proposal; the weighted consensus direction is used for
        coordinates the current owner does not cover. Positive and reflected
        candidates are evaluated from the current strict-best context. An
        accepted candidate becomes the next center, so later rounds receive
        actual objective feedback instead of repeatedly sampling around a stale
        anchor.

        ``seed`` is accepted for a common CTP interface but intentionally unused:
        this mechanism is deterministic and therefore reproducible across runs.
        """

        del seed
        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be a positive integer")
        component = tuple(component)
        if scope is not None:
            variables = self._repair_scope(component, scope)
        elif include_boundary:
            variables = tuple(
                sorted(
                    {
                        variable
                        for group in component
                        for variable in self.structure.groups[group]
                    }
                )
            )
        else:
            variables = self._component_variables(component)
        if not variables:
            return 0
        selected = tuple(proposals)
        by_group = {proposal.group: proposal for proposal in selected}
        if len(by_group) != len(selected) or set(by_group) != set(component):
            raise ValueError("proposals must cover exactly one overlap component")
        for group in component:
            proposal_variables = {variable for variable, _ in by_group[group].values}
            if proposal_variables != set(self.structure.groups[group]):
                raise ValueError(f"proposal {group} must cover exactly its group variables")

        shared_variables = tuple(
            variable for variable in variables if len(self.structure.owners(variable)) > 1
        )
        residuals = compute_proposal_residuals(
            self.structure,
            selected,
            variables=shared_variables,
            epsilon=self.epsilon,
        )
        owner_order = tuple(
            sorted(component, key=lambda group: (-by_group[group].improvement, group))
        )
        search_base = self.ledger.best_x if base is None else np.asarray(base, dtype=float)
        if search_base.shape != (self.structure.dimension,) or not np.all(np.isfinite(search_base)):
            raise ValueError("base must be a finite vector matching the structure dimension")
        if np.any(search_base < self.ledger.problem.lower_array) or np.any(
            search_base > self.ledger.problem.upper_array
        ):
            raise ValueError("base must stay inside the problem bounds")

        # The initial radius reflects observed disagreement/noise.  For unique
        # boundary coordinates, the proposal displacement is also actionable.
        centers = search_base[np.asarray(variables, dtype=int)].copy()
        initial_radii: list[float] = []
        for variable in variables:
            owners = self.structure.owners(variable)
            if variable in residuals:
                target = residuals[variable].weighted_mean
                disagreement = max(
                    abs(by_group[group].value(variable) - target) for group in owners
                )
                uncertainty = max(by_group[group].sigma(variable) for group in owners)
                radius = max(self.epsilon, disagreement, uncertainty)
            else:
                owner = owners[0]
                proposal = by_group[owner]
                radius = max(
                    self.epsilon,
                    abs(proposal.value(variable) - search_base[variable]),
                    proposal.sigma(variable),
                )
            initial_radii.append(float(radius))
        radii = np.asarray(initial_radii, dtype=float)
        # These factors only govern local trust-region adaptation; they are
        # independent of the benchmark and do not alter the FE allocation.
        growth = 1.25
        shrink = 0.5
        cap = 4.0
        indices = np.asarray(variables, dtype=int)
        lower = self.ledger.problem.lower_array
        upper = self.ledger.problem.upper_array
        start = self.ledger.count
        rounds = budget_fes // 2

        for round_index in range(rounds):
            group = owner_order[round_index % len(owner_order)]
            proposal = by_group[group]
            fallback = np.asarray(
                [
                    (
                        residuals[variable].weighted_mean - centers[index]
                        if variable in residuals
                        else by_group[self.structure.owners(variable)[0]].value(variable)
                        - centers[index]
                    )
                    for index, variable in enumerate(variables)
                ],
                dtype=float,
            )
            direction = np.asarray(
                [
                    proposal.value(variable) - centers[index]
                    if group in self.structure.owners(variable)
                    else fallback[index]
                    for index, variable in enumerate(variables)
                ],
                dtype=float,
            )
            direction = np.where(np.abs(direction) > self.epsilon, direction, fallback)
            direction = np.where(np.abs(direction) > self.epsilon, direction, 1.0)
            step = radii * np.sign(direction)
            current = self.ledger.best_x
            plus = current.copy()
            minus = current.copy()
            plus[indices] = centers + step
            minus[indices] = centers - step
            batch = np.asarray((plus, minus), dtype=float)
            np.clip(batch, lower, upper, out=batch)
            before = float(self.ledger.best_error)
            self.ledger.evaluate(batch)
            if self.ledger.best_error < before:
                centers = self.ledger.best_x[indices].copy()
                radii = np.minimum(radii * growth, np.asarray(initial_radii) * cap)
            else:
                radii *= shrink

        if budget_fes % 2:
            group = owner_order[rounds % len(owner_order)]
            proposal = by_group[group]
            fallback = np.asarray(
                [
                    (
                        residuals[variable].weighted_mean - centers[index]
                        if variable in residuals
                        else by_group[self.structure.owners(variable)[0]].value(variable)
                        - centers[index]
                    )
                    for index, variable in enumerate(variables)
                ],
                dtype=float,
            )
            direction = np.asarray(
                [
                    proposal.value(variable) - centers[index]
                    if group in self.structure.owners(variable)
                    else fallback[index]
                    for index, variable in enumerate(variables)
                ],
                dtype=float,
            )
            direction = np.where(np.abs(direction) > self.epsilon, direction, fallback)
            direction = np.where(np.abs(direction) > self.epsilon, direction, 1.0)
            candidate = self.ledger.best_x
            candidate[indices] = centers + radii * np.sign(direction)
            np.clip(candidate, lower, upper, out=candidate)
            self.ledger.evaluate(candidate)

        consumed = self.ledger.count - start
        if consumed != budget_fes:
            raise RuntimeError("sequential joint-patch CTP drifted from its exact FE request")
        return consumed

    def _repair_sequential_shared_patch(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
        *,
        budget_fes: int,
        seed: int,
        base: np.ndarray | None = None,
        scope: tuple[int, ...] | None = None,
    ) -> int:
        """Repair only shared coordinates while preserving proposal boundaries."""

        return self._repair_sequential_joint_patch(
            component,
            proposals,
            budget_fes=budget_fes,
            seed=seed,
            base=base,
            include_boundary=False,
            scope=scope,
        )

    def _repair_sequential_coordinate_patch(
        self,
        component: tuple[int, ...],
        proposals: tuple[LocalProposal, ...],
        *,
        budget_fes: int,
        seed: int,
        base: np.ndarray | None = None,
        scope: tuple[int, ...] | None = None,
    ) -> int:
        """Patch one conflict-ranked shared variable per feedback step."""

        del seed
        if isinstance(budget_fes, bool) or not isinstance(budget_fes, int) or budget_fes <= 0:
            raise ValueError("budget_fes must be a positive integer")
        component = tuple(component)
        variables = self._repair_scope(component, scope)
        if not variables:
            return 0
        selected = tuple(proposals)
        by_group = {proposal.group: proposal for proposal in selected}
        if len(by_group) != len(selected) or set(by_group) != set(component):
            raise ValueError("proposals must cover exactly one overlap component")
        for group in component:
            proposal_variables = {variable for variable, _ in by_group[group].values}
            if proposal_variables != set(self.structure.groups[group]):
                raise ValueError(f"proposal {group} must cover exactly its group variables")
        residuals = compute_proposal_residuals(
            self.structure,
            selected,
            variables=variables,
            epsilon=self.epsilon,
        )
        search_base = self.ledger.best_x if base is None else np.asarray(base, dtype=float)
        if search_base.shape != (self.structure.dimension,) or not np.all(np.isfinite(search_base)):
            raise ValueError("base must be a finite vector matching the structure dimension")
        if np.any(search_base < self.ledger.problem.lower_array) or np.any(
            search_base > self.ledger.problem.upper_array
        ):
            raise ValueError("base must stay inside the problem bounds")

        # Spend the fixed CTP budget on the most conflicting coordinates first.
        variable_order = tuple(
            sorted(variables, key=lambda variable: (-residuals[variable].conflict_score, variable))
        )
        owner_order = tuple(
            sorted(component, key=lambda group: (-by_group[group].improvement, group))
        )
        centers = search_base[np.asarray(variables, dtype=int)].copy()
        initial_radii = np.asarray(
            [
                max(
                    self.epsilon,
                    max(
                        abs(
                            by_group[group].value(variable)
                            - residuals[variable].weighted_mean
                        )
                        for group in self.structure.owners(variable)
                    ),
                    max(
                        by_group[group].sigma(variable)
                        for group in self.structure.owners(variable)
                    ),
                )
                for variable in variables
            ],
            dtype=float,
        )
        radii = initial_radii.copy()
        indices = np.asarray(variables, dtype=int)
        lower = self.ledger.problem.lower_array
        upper = self.ledger.problem.upper_array
        start = self.ledger.count
        rounds = budget_fes // 2

        def evaluate_step(round_index: int, *, reflected: bool = False) -> None:
            variable = variable_order[round_index % len(variable_order)]
            variable_index = variables.index(variable)
            owners = self.structure.owners(variable)
            owner_candidates = tuple(group for group in owner_order if group in owners)
            group = owner_candidates[round_index % len(owner_candidates)]
            proposal = by_group[group]
            fallback = residuals[variable].weighted_mean - centers[variable_index]
            direction = proposal.value(variable) - centers[variable_index]
            if abs(direction) <= self.epsilon:
                direction = fallback
            if abs(direction) <= self.epsilon:
                direction = 1.0
            sign = -1.0 if reflected else 1.0
            candidate = self.ledger.best_x
            candidate[variable] = centers[variable_index] + sign * radii[variable_index] * np.sign(direction)
            np.clip(candidate, lower, upper, out=candidate)
            before = float(self.ledger.best_error)
            self.ledger.evaluate(candidate)
            if self.ledger.best_error < before:
                centers[:] = self.ledger.best_x[indices]
                radii[variable_index] = min(radii[variable_index] * 1.25, initial_radii[variable_index] * 4.0)
            elif not reflected:
                radii[variable_index] *= 0.5

        for round_index in range(rounds):
            variable = variable_order[round_index % len(variable_order)]
            variable_index = variables.index(variable)
            owners = self.structure.owners(variable)
            owner_candidates = tuple(group for group in owner_order if group in owners)
            group = owner_candidates[round_index % len(owner_candidates)]
            proposal = by_group[group]
            fallback = residuals[variable].weighted_mean - centers[variable_index]
            direction = proposal.value(variable) - centers[variable_index]
            if abs(direction) <= self.epsilon:
                direction = fallback
            if abs(direction) <= self.epsilon:
                direction = 1.0
            current = self.ledger.best_x
            plus = current.copy()
            minus = current.copy()
            plus[variable] = centers[variable_index] + radii[variable_index] * np.sign(direction)
            minus[variable] = centers[variable_index] - radii[variable_index] * np.sign(direction)
            batch = np.asarray((plus, minus), dtype=float)
            np.clip(batch, lower, upper, out=batch)
            before = float(self.ledger.best_error)
            self.ledger.evaluate(batch)
            if self.ledger.best_error < before:
                centers[:] = self.ledger.best_x[indices]
                radii[variable_index] = min(radii[variable_index] * 1.25, initial_radii[variable_index] * 4.0)
            else:
                radii[variable_index] *= 0.5

        if budget_fes % 2:
            evaluate_step(rounds)
        consumed = self.ledger.count - start
        if consumed != budget_fes:
            raise RuntimeError("coordinate-wise shared CTP drifted from its exact FE request")
        return consumed


__all__ = [
    "ConflictLevel",
    "CoordinationCandidate",
    "CoordinationResult",
    "FullContextWritebackResult",
    "FullContextWritebackRound",
    "ProposalNeighborhoodResult",
    "ProposalNeighborhoodRound",
    "LocalProposal",
    "OverlapCoordinator",
    "OverlapStructure",
    "ProposalResidual",
    "compute_proposal_residuals",
]
