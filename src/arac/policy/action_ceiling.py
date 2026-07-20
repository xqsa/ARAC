"""Frozen offline contracts and statistics for the G1 action-ceiling audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from random import Random
from typing import Mapping, Sequence

from arac.actions.full_space_sep_cma import FULL_SPACE_SEP_CMA_ACTION
from arac.policy.evidence_overlay import (
    BridgeWeights,
    ProbeUtilities,
    RelationKey,
    UTILITY_EPSILON,
    runtime_probe_anchor_hash,
    runtime_probe_shared_values_hash,
)


ACTION_CEILING_PROTOCOL_VERSION = "exp019-action-ceiling-v5"
ACTION_CEILING_ARMS = (
    "native_eq8",
    "true_no_writeback",
    "exact_left",
    "exact_right",
    "exact_bridge",
    "efficiency_budget_reallocation",
    "delta_priority_scan",
    "stagnation_cross_group_warm_start",
    FULL_SPACE_SEP_CMA_ACTION,
)
ACTION_CEILING_HORIZONS = ("immediate", "sweep_1", "sweep_3")
ACTION_CEILING_CONTEXT_FIELDS = (
    "protocol_version",
    "cohort",
    "problem_id",
    "seed",
    "context_id",
    "relation_id",
    "action_set_hash",
    "checkpoint_hash",
    "dispatch_checkpoint_hash",
    "phase_boundary_fe",
    "dispatch_fe",
    "issued_sweep",
    "target_sweep",
    "group_index",
    "efficiency_ewma",
    "completed_efficiency_sweeps",
    "stagnation_streaks",
    "population_sizes",
    "uniform_group_budgets",
    "horizon_fe",
    "full_space_action_hash",
    "full_space_action_payload",
    "full_space_initial_mean_hash",
    "full_space_parameter_hash",
    "full_space_optimizer_seed",
    "full_space_population_size",
    "full_space_budget_fes",
    "full_space_acceptance_fitness",
    "selector_arm",
    "selector_reason",
    "anchor_values",
    "left_values",
    "right_values",
    "bridge_values",
    "bridge_weights",
    "native_parity",
    "runtime_authorized",
    "status",
    "invalidation_reason",
)
ACTION_CEILING_ARM_RESULT_FIELDS = (
    "protocol_version",
    "cohort",
    "problem_id",
    "seed",
    "context_id",
    "arm",
    "horizon",
    "target_fe",
    "natural_endpoint_fe",
    "native_error",
    "arm_error",
    "delta",
    "action_budget_fes",
    "action_actual_fes",
    "action_instance_hash",
    "action_lifecycle_payload",
    "action_lifecycle_hash",
    "action_accepted",
    "action_candidate_hash",
    "action_candidate_fitness",
    "action_post_incumbent_hash",
    "optimizer_scope",
    "optimizer_parameter_hash",
    "optimizer_initial_state_hash",
    "optimizer_final_state_hash",
    "optimizer_population_size",
    "optimizer_generation_count",
    "counterfactual_applied",
    "mutation_norm",
    "optimizer_mean_mutation_norm",
    "continuation_policy_applied",
    "execution_sweep_trace",
    "execution_order_trace",
    "group_budget_trace",
    "execution_start_fe_trace",
    "warm_start_trigger_count",
    "warm_start_mean_shift_norm",
    "selected_candidate",
    "runtime_authorized",
    "status",
    "invalidation_reason",
)
PRIMARY_HORIZON = "sweep_1"
MATERIAL_POSITIVE_DELTA = math.log(1.01)
CATASTROPHIC_DELTA = -math.log(1.20)
SPARSE_POSITIVE_THRESHOLD = 0.20
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2026071901
EFFICIENCY_EWMA_ALPHA = 0.3
BUDGET_MAX_UNIFORM_MULTIPLIER = 3
STAGNATION_EPSILON = 1e-6
STAGNATION_TRIGGER_STREAK = 3
WARM_START_COOLDOWN_SWEEPS = 3
_HASH_LENGTH = 64
_TIE_TOLERANCE = 1e-15


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


@dataclass(frozen=True)
class FrozenProbeCandidate:
    """One exact Phase1 shared block and its full-candidate provenance."""

    name: str
    shared_values: tuple[float, ...]
    shared_values_hash: str
    candidate_hash: str
    fitness: float
    utility: float

    def __post_init__(self) -> None:
        if self.name not in {"anchor", "left_owner", "right_owner", "bridge"}:
            raise ValueError("unsupported frozen probe candidate")
        values = tuple(_finite(value, "shared value") for value in self.shared_values)
        object.__setattr__(self, "shared_values", values)
        _validate_hash(self.shared_values_hash, "shared_values_hash")
        _validate_hash(self.candidate_hash, "candidate_hash")
        fitness = _finite(self.fitness, "candidate fitness")
        if fitness < 0.0:
            raise ValueError("candidate fitness must be non-negative")
        _finite(self.utility, "candidate utility")


@dataclass(frozen=True)
class RelationActionSet:
    """Offline-only Phase1 action set for one selected relation."""

    relation: RelationKey
    anchor: FrozenProbeCandidate
    left_owner: FrozenProbeCandidate
    right_owner: FrozenProbeCandidate
    bridge: FrozenProbeCandidate
    bridge_weights: BridgeWeights
    probe_utilities: ProbeUtilities
    selector_winner: str
    selector_utility: float
    selector_reason: str
    checkpoint_fe: int
    checkpoint_hash: str
    issued_sweep: int
    target_sweep: int

    def __post_init__(self) -> None:
        expected_names = ("anchor", "left_owner", "right_owner", "bridge")
        candidates = (self.anchor, self.left_owner, self.right_owner, self.bridge)
        if tuple(candidate.name for candidate in candidates) != expected_names:
            raise ValueError("relation action candidates are not in frozen order")
        shared_count = len(self.relation.shared_variable_indices)
        if shared_count == 0:
            raise ValueError("action-ceiling relation must have shared variables")
        for candidate in candidates:
            if len(candidate.shared_values) != shared_count:
                raise ValueError("candidate shared values do not match relation")
            expected_hash = (
                runtime_probe_anchor_hash(self.relation, candidate.shared_values)
                if candidate.name == "anchor"
                else runtime_probe_shared_values_hash(
                    self.relation,
                    candidate.shared_values,
                )
            )
            if candidate.shared_values_hash != expected_hash:
                label = "anchor hash" if candidate.name == "anchor" else "candidate hash"
                raise ValueError(f"{label} does not match shared values")
        if self.anchor.shared_values_hash != runtime_probe_anchor_hash(
            self.relation,
            self.anchor.shared_values,
        ):
            raise ValueError("anchor hash does not match relation-local anchor")
        if self.selector_winner not in {
            "left_owner",
            "right_owner",
            "bridge",
            "none",
        }:
            raise ValueError("unsupported selector winner")
        _finite(self.selector_utility, "selector utility")
        if not self.selector_reason:
            raise ValueError("selector reason is required")
        if isinstance(self.checkpoint_fe, bool) or int(self.checkpoint_fe) < 0:
            raise ValueError("checkpoint_fe must be a non-negative integer")
        _validate_hash(self.checkpoint_hash, "checkpoint_hash")
        if isinstance(self.issued_sweep, bool) or int(self.issued_sweep) < 0:
            raise ValueError("issued_sweep must be a non-negative integer")
        if self.target_sweep != self.issued_sweep + 1:
            raise ValueError("target_sweep must be the next sweep")

    @property
    def action_set_hash(self) -> str:
        return _sha256(
            {
                "relation": {
                    "owners": self.relation.owner_group_indices,
                    "shared": self.relation.shared_variable_indices,
                },
                "candidates": [
                    {
                        "name": candidate.name,
                        "shared_values_hash": candidate.shared_values_hash,
                        "candidate_hash": candidate.candidate_hash,
                        "fitness": candidate.fitness,
                        "utility": candidate.utility,
                    }
                    for candidate in (
                        self.anchor,
                        self.left_owner,
                        self.right_owner,
                        self.bridge,
                    )
                ],
                "bridge_weights": {
                    "left": self.bridge_weights.left_owner,
                    "right": self.bridge_weights.right_owner,
                },
                "selector_winner": self.selector_winner,
                "selector_utility": self.selector_utility,
                "checkpoint_fe": self.checkpoint_fe,
                "checkpoint_hash": self.checkpoint_hash,
                "issued_sweep": self.issued_sweep,
                "target_sweep": self.target_sweep,
            }
        )

    def candidate_for_arm(self, arm: str) -> FrozenProbeCandidate:
        mapping = {
            "exact_left": self.left_owner,
            "exact_right": self.right_owner,
            "exact_bridge": self.bridge,
        }
        try:
            return mapping[arm]
        except KeyError as error:
            raise ValueError(f"arm has no frozen exact candidate: {arm}") from error


def actionability_delta(
    native_error: float,
    arm_error: float,
    *,
    epsilon: float = UTILITY_EPSILON,
) -> float:
    native = _finite(native_error, "native error")
    arm = _finite(arm_error, "arm error")
    eps = _finite(epsilon, "epsilon")
    if native < 0.0 or arm < 0.0 or eps <= 0.0:
        raise ValueError("actionability errors must be non-negative and epsilon positive")
    return math.log((native + eps) / (arm + eps))


@dataclass(frozen=True)
class ActionCeilingObservation:
    context_id: str
    cohort: str
    problem_id: str
    seed: int
    arm: str
    horizon: str
    delta: float
    selector_arm: str

    def __post_init__(self) -> None:
        if not self.context_id or not self.cohort or not self.problem_id:
            raise ValueError("observation identity fields are required")
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.arm not in ACTION_CEILING_ARMS:
            raise ValueError("unsupported action-ceiling arm")
        if self.horizon not in ACTION_CEILING_HORIZONS:
            raise ValueError("unsupported action-ceiling horizon")
        _finite(self.delta, "actionability delta")
        if self.selector_arm not in ACTION_CEILING_ARMS:
            raise ValueError("selector arm is unsupported")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(float(value) for value in values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _best_arm(deltas: Mapping[str, float]) -> tuple[str, float]:
    missing = set(ACTION_CEILING_ARMS) - set(deltas)
    if missing:
        raise ValueError(f"context is missing action-ceiling arms: {sorted(missing)}")
    best_value = max(float(deltas[arm]) for arm in ACTION_CEILING_ARMS)
    for arm in ACTION_CEILING_ARMS:
        if math.isclose(
            float(deltas[arm]),
            best_value,
            rel_tol=0.0,
            abs_tol=_TIE_TOLERANCE,
        ):
            return arm, float(deltas[arm])
    raise RuntimeError("action-ceiling tie-break failed")


def summarize_action_ceiling(
    observations: Sequence[ActionCeilingObservation],
    *,
    cohort: str,
    horizon: str = PRIMARY_HORIZON,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Compute SBS, VBS/oracle, current-selector utility, and the G1 gate."""

    selected = tuple(
        row for row in observations if row.cohort == cohort and row.horizon == horizon
    )
    if not selected:
        raise ValueError("no action-ceiling observations match the requested cohort")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    contexts: dict[str, dict[str, ActionCeilingObservation]] = {}
    for row in selected:
        arm_rows = contexts.setdefault(row.context_id, {})
        if row.arm in arm_rows:
            raise ValueError("duplicate context arm observation")
        arm_rows[row.arm] = row

    context_metrics: dict[str, dict[str, object]] = {}
    for context_id, arm_rows in contexts.items():
        deltas = {arm: row.delta for arm, row in arm_rows.items()}
        vbs_arm, vbs_delta = _best_arm(deltas)
        if not math.isclose(
            deltas["native_eq8"],
            0.0,
            rel_tol=0.0,
            abs_tol=_TIE_TOLERANCE,
        ):
            raise ValueError("native Eq.8 must be the zero-delta reference")
        selector_arms = {row.selector_arm for row in arm_rows.values()}
        if len(selector_arms) != 1:
            raise ValueError("selector arm differs within one context")
        selector_arm = next(iter(selector_arms))
        reference = next(iter(arm_rows.values()))
        context_metrics[context_id] = {
            "problem_id": reference.problem_id,
            "seed": reference.seed,
            "deltas": deltas,
            "vbs_arm": vbs_arm,
            "vbs_delta": vbs_delta,
            "selector_arm": selector_arm,
            "selector_delta": deltas[selector_arm],
        }

    arm_means = {
        arm: _mean(
            [float(metrics["deltas"][arm]) for metrics in context_metrics.values()]
        )
        for arm in ACTION_CEILING_ARMS
    }
    sbs_arm, sbs_delta = _best_arm(arm_means)
    vbs_values = [float(item["vbs_delta"]) for item in context_metrics.values()]
    selector_values = [
        float(item["selector_delta"]) for item in context_metrics.values()
    ]
    material_values = [
        float(value > MATERIAL_POSITIVE_DELTA) for value in vbs_values
    ]
    selector_material_values = [
        float(value > MATERIAL_POSITIVE_DELTA) for value in selector_values
    ]

    clusters: dict[tuple[str, int], list[str]] = {}
    for context_id, metrics in context_metrics.items():
        key = (str(metrics["problem_id"]), int(metrics["seed"]))
        clusters.setdefault(key, []).append(context_id)
    cluster_keys = tuple(sorted(clusters))
    rng = Random(int(bootstrap_seed))
    bootstrap_vbs: list[float] = []
    bootstrap_selector: list[float] = []
    bootstrap_material: list[float] = []
    for _ in range(int(bootstrap_replicates)):
        sampled_contexts: list[str] = []
        for _cluster in cluster_keys:
            sampled_key = cluster_keys[rng.randrange(len(cluster_keys))]
            sampled_contexts.extend(clusters[sampled_key])
        bootstrap_vbs.append(
            _mean([float(context_metrics[key]["vbs_delta"]) for key in sampled_contexts])
        )
        bootstrap_selector.append(
            _mean(
                [float(context_metrics[key]["selector_delta"]) for key in sampled_contexts]
            )
        )
        bootstrap_material.append(
            _mean(
                [
                    float(
                        float(context_metrics[key]["vbs_delta"])
                        > MATERIAL_POSITIVE_DELTA
                    )
                    for key in sampled_contexts
                ]
            )
        )

    vbs_mean = _mean(vbs_values)
    selector_mean = _mean(selector_values)
    vbs_lcb = _quantile(bootstrap_vbs, 0.025)
    selector_lcb = _quantile(bootstrap_selector, 0.025)
    material_rate = _mean(material_values)
    material_ucb = _quantile(bootstrap_material, 0.975)
    catastrophic_count = sum(value <= CATASTROPHIC_DELTA for value in selector_values)

    if vbs_mean <= 0.0:
        recommendation = "redesign_actions"
    elif vbs_lcb <= 0.0:
        recommendation = "collect_more_ceiling_contexts"
    elif material_ucb < SPARSE_POSITIVE_THRESHOLD:
        recommendation = "force_abstain_sparse_headroom"
    elif selector_lcb <= 0.0 or catastrophic_count > 0:
        recommendation = "upgrade_evidence"
    else:
        recommendation = "small_runtime_validation_only"

    return {
        "protocol_version": ACTION_CEILING_PROTOCOL_VERSION,
        "cohort": cohort,
        "horizon": horizon,
        "context_count": len(context_metrics),
        "cluster_count": len(clusters),
        "sbs_arm": sbs_arm,
        "sbs_mean_delta": sbs_delta,
        "vbs_mean_delta": vbs_mean,
        "vbs_lcb": vbs_lcb,
        "selector_mean_delta": selector_mean,
        "selector_lcb": selector_lcb,
        "selector_vbs_regret": _mean(
            [
                float(item["vbs_delta"]) - float(item["selector_delta"])
                for item in context_metrics.values()
            ]
        ),
        "selector_material_positive_count": int(sum(selector_material_values)),
        "selector_material_positive_rate": _mean(selector_material_values),
        "selector_catastrophic_rate": catastrophic_count / len(selector_values),
        "vbs_material_positive_rate": material_rate,
        "vbs_material_positive_ucb": material_ucb,
        "selector_catastrophic_count": catastrophic_count,
        "recommendation": recommendation,
    }
