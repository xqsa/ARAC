"""Observer-only HCC runtime for the RDDSM evidence overlay pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from arac.backends.hcc import EVIDENCE_OVERLAY_MODES
from arac.policy.evidence_overlay import (
    LOCAL_OPTIMUM_TOP_K,
    PROPOSAL_DISAGREEMENT_METRIC,
    TOP_RELATION_COUNT,
    FourPointProbe,
    RelationKey,
    RelationSelection,
    RuntimeProbeAction,
    ScoredRelation,
    ShadowDecision,
    build_four_point_probe,
    build_reference_blind_ordering,
    build_relation_candidates,
    relation_cohen_d,
    runtime_probe_anchor_hash,
    runtime_probe_shared_values_hash,
    decide_shadow_action,
    score_relations,
    select_top_relations,
    shuffle_relation_scores,
    summarize_probe_utilities,
)
from arac.policy.overlap_hypergraph import (
    FINAL_OWNER_PROPOSAL_WATERMARK,
    CompletedSweepSnapshot,
    GroupCycleObservation,
    HyperedgeScore,
    build_closed_owner_credit,
    build_delayed_hyperedge_credit,
    build_group_cycle_observation,
    build_hyperedge_cycle_states,
    build_overlap_hypergraph,
    score_hyperedge_states,
)


EVIDENCE_OVERLAY_PROTOCOL_VERSION = "rddsm-evidence-overlay-pilot-v3"
EVIDENCE_OVERLAY_SOURCE_MODE = "fresh_runtime_probe"
TERMINAL_TOLERANCE_RULE = "maximum_native_group_population"
RUNTIME_INPUT_FIELDS = (
    "rddsm_topology",
    "rddsm_order",
    "group_cycle_observations",
    "closed_owner_credit",
    "phase_boundary_anchor",
    "remaining_fe",
    "normal_sweep_fe",
    "tolerance_fe",
)
PROBE_CANDIDATES = ("x0", "left_owner", "right_owner", "bridge")

CHECKPOINT_FIELDS = (
    "problem_id",
    "seed",
    "mode",
    "checkpoint_fe",
    "fitness_prefix_hash",
    "incumbent_hash",
    "rddsm_topology_hash",
    "rddsm_order_hash",
    "phase_boundary_fe",
    "history_sweeps",
    "previous_survival_closed",
    "plan_status",
    "plan_reason",
    "runtime_authorized",
)
PLAN_FIELDS = (
    "problem_id",
    "seed",
    "mode",
    "relation_id",
    "owner_groups",
    "shared_variables",
    "selected",
    "voi",
    "native_voi",
    "proposal_disagreement",
    "cohen_d",
    "disagreement_metric",
    "left_top_k_count",
    "right_top_k_count",
    "left_distribution_centers",
    "right_distribution_centers",
    "left_distribution_standard_deviations",
    "right_distribution_standard_deviations",
    "owner_priority",
    "left_owner_reliability",
    "right_owner_reliability",
    "score_source_relation_id",
    "phase_boundary_fe",
    "runtime_authorized",
)
PROBE_EVIDENCE_FIELDS = (
    "problem_id",
    "seed",
    "mode",
    "relation_id",
    "candidate",
    "fitness",
    "utility",
    "owner_reliability",
    "candidate_hash",
    "phase_boundary_fe",
    "actual_fe",
    "runtime_authorized",
)
DELAYED_OUTCOME_FIELDS = (
    "problem_id",
    "seed",
    "mode",
    "relation_id",
    "owner",
    "action_sweep_index",
    "resolution_sweep_index",
    "survival_label",
    "overwrite_label",
    "next_sweep_log_improvement",
    "overwrite_penalized_credit",
    "label_closed",
    "label_status",
    "resolution_fe",
    "runtime_authorized",
)
SHADOW_DECISION_FIELDS = (
    "problem_id",
    "seed",
    "mode",
    "relation_id",
    "action",
    "winner",
    "utility",
    "reason",
    "runtime_authorized",
)
RUNTIME_ACTION_FIELDS = (
    "problem_id",
    "seed",
    "mode",
    "relation_id",
    "owner_groups",
    "shared_variables",
    "winner",
    "canonical_action",
    "shared_values",
    "shared_values_hash",
    "candidate_hash",
    "bridge_weight_left",
    "bridge_weight_right",
    "utility",
    "anchor_hash",
    "checkpoint_fe",
    "checkpoint_hash",
    "issued_sweep",
    "ttl_sweeps",
    "expires_sweep",
    "runtime_authorized",
    "runtime_consumed",
    "consumed_sweep",
    "consumed_fe",
    "status",
    "invalidation_reason",
)


class EvidenceOverlayRuntimeError(RuntimeError):
    """Raised when an active probe fails after its FE bundle was admitted."""


@dataclass(frozen=True)
class EvidenceOverlayArtifactPaths:
    manifest: Path
    checkpoint: Path
    plan: Path
    probe_evidence: Path
    delayed_outcomes: Path
    shadow_decisions: Path
    runtime_actions: Path

    @classmethod
    def under(cls, directory: Path | str) -> "EvidenceOverlayArtifactPaths":
        root = Path(directory)
        return cls(
            manifest=root / "evidence_overlay_manifest.json",
            checkpoint=root / "checkpoint.csv",
            plan=root / "plan.csv",
            probe_evidence=root / "probe_evidence.csv",
            delayed_outcomes=root / "delayed_outcomes.csv",
            shadow_decisions=root / "shadow_decisions.csv",
            runtime_actions=root / "evidence_overlay_runtime_actions.csv",
        )


@dataclass(frozen=True)
class EvidenceOverlayBarrierResult:
    status: str
    reason: str
    requested_fe: int
    actual_fe: int
    selected_relation_count: int
    phase_boundary_fe: int
    anchor_hash: str
    anchor_unchanged: bool
    runtime_authorized: bool = False


@dataclass(frozen=True)
class _RelationProbeResult:
    scored_relation: ScoredRelation
    probe: FourPointProbe
    fitness: tuple[float, float, float, float]
    decision: ShadowDecision


@dataclass(frozen=True)
class _PendingOwnerOutcome:
    relation: RelationKey
    action_sweep_index: int
    anchor_error: float
    anchor_shared_values: tuple[float, ...]
    left_shared_values: tuple[float, ...]
    right_shared_values: tuple[float, ...]


@dataclass
class RuntimeProbeActionRecord:
    action: RuntimeProbeAction
    status: str = "issued"
    runtime_consumed: bool = False
    consumed_sweep: int | None = None
    consumed_fe: int | None = None
    invalidation_reason: str = ""


@dataclass(frozen=True)
class RuntimeProbeConsumption:
    action: RuntimeProbeAction | None
    consumed: bool
    reason: str


class RuntimeProbeActionLedger:
    """Own one-shot runtime actions and their auditable lifecycle."""

    def __init__(self) -> None:
        self._records: dict[RelationKey, RuntimeProbeActionRecord] = {}

    def issue(self, actions: Sequence[RuntimeProbeAction]) -> None:
        records: dict[RelationKey, RuntimeProbeActionRecord] = {}
        for action in actions:
            if action.relation in records:
                raise ValueError("duplicate runtime probe relation action")
            records[action.relation] = RuntimeProbeActionRecord(action=action)
        self._records = records

    @property
    def records(self) -> tuple[RuntimeProbeActionRecord, ...]:
        return tuple(
            self._records[key]
            for key in sorted(self._records)
        )

    def action_for(self, relation: RelationKey) -> RuntimeProbeAction | None:
        record = self._records.get(relation)
        return None if record is None else record.action

    def consume(
        self,
        *,
        action: RuntimeProbeAction | None,
        relation: RelationKey,
        anchor_hash: str,
        checkpoint_hash: str,
        current_sweep: int,
        current_fe: int,
        write_shared_values: Callable[[tuple[float, ...]], None],
    ) -> RuntimeProbeConsumption:
        if action is None:
            return RuntimeProbeConsumption(None, False, "no_action_for_relation")
        record = self._records.get(action.relation)
        if record is None or record.action != action:
            return RuntimeProbeConsumption(None, False, "unknown_action")
        sweep = _integer(current_sweep, name="current_sweep")
        consumed_fe = _integer(current_fe, name="current_fe")
        if record.runtime_consumed:
            record.invalidation_reason = "already_consumed"
            return RuntimeProbeConsumption(None, False, "already_consumed")
        if record.status == "abstained":
            return RuntimeProbeConsumption(
                None,
                False,
                record.invalidation_reason,
            )
        if checkpoint_hash != action.checkpoint_hash:
            record.status = "abstained"
            record.invalidation_reason = "checkpoint_mismatch"
            return RuntimeProbeConsumption(None, False, record.invalidation_reason)
        if sweep < action.expires_sweep:
            record.invalidation_reason = "not_next_sweep"
            return RuntimeProbeConsumption(None, False, "not_next_sweep")
        if sweep > action.expires_sweep:
            record.status = "abstained"
            record.invalidation_reason = "ttl_expired"
            return RuntimeProbeConsumption(None, False, record.invalidation_reason)
        if relation != action.relation:
            record.status = "abstained"
            record.invalidation_reason = "relation_mismatch"
            return RuntimeProbeConsumption(None, False, record.invalidation_reason)
        if anchor_hash != action.anchor_hash:
            record.status = "abstained"
            record.invalidation_reason = "anchor_mismatch"
            return RuntimeProbeConsumption(None, False, record.invalidation_reason)
        try:
            write_shared_values(action.shared_values)
        except Exception:
            record.status = "abstained"
            record.invalidation_reason = "write_failed"
            raise
        record.status = "consumed"
        record.runtime_consumed = True
        record.consumed_sweep = sweep
        record.consumed_fe = consumed_fe
        record.invalidation_reason = ""
        return RuntimeProbeConsumption(action, True, "consumed")


def _finite(value: float, *, name: str, non_negative: bool = False) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if non_negative and converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _integer(value: int, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector_hash(values: Sequence[float]) -> str:
    return _canonical_hash([float(value) for value in values])


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relation_id(key: RelationKey) -> str:
    owners = "-".join(str(value) for value in key.owner_group_indices)
    shared = "-".join(str(value) for value in key.shared_variable_indices)
    return f"g{owners}:v{shared}"


def _csv_values(values: Sequence[int]) -> str:
    return ";".join(str(value) for value in values)


def _csv_float_values(values: Sequence[float]) -> str:
    return ";".join(_format_float(value) for value in values)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{float(value):.17e}"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _objective_value(raw: object) -> float:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return _finite(raw, name="objective value", non_negative=True)
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("objective must return one scalar value") from exc
    if len(values) != 1:
        raise ValueError("objective must return exactly one scalar value")
    return _finite(values[0], name="objective value", non_negative=True)


class HccEvidenceOverlayObserver:
    """Collect native evidence and expose explicit actions after the Phase1 probe."""

    def __init__(
        self,
        *,
        mode: str,
        grouping_result: Sequence[Sequence[int]],
        problem_id: str,
        seed: int,
        run_id: str,
        configured_max_fes: int,
        terminal_tolerance_fe: int,
        lower_bound: float,
        upper_bound: float,
        fresh_optimizer_execution: bool = True,
        top_relation_count: int | None = TOP_RELATION_COUNT,
    ) -> None:
        if mode not in EVIDENCE_OVERLAY_MODES:
            raise ValueError("unsupported evidence overlay mode")
        if not problem_id or not run_id:
            raise ValueError("problem_id and run_id are required")
        if not isinstance(fresh_optimizer_execution, bool):
            raise ValueError("fresh_optimizer_execution must be boolean")
        if not fresh_optimizer_execution:
            raise ValueError("evidence overlay requires a fresh optimizer execution")
        lower = _finite(lower_bound, name="lower_bound")
        upper = _finite(upper_bound, name="upper_bound")
        if upper <= lower:
            raise ValueError("upper_bound must exceed lower_bound")

        self.mode = mode
        self.problem_id = str(problem_id)
        self.seed = _integer(seed, name="seed")
        self.run_id = str(run_id)
        self.configured_max_fes = _integer(
            configured_max_fes,
            name="configured_max_fes",
            minimum=1,
        )
        self.terminal_tolerance_fe = _integer(
            terminal_tolerance_fe,
            name="terminal_tolerance_fe",
        )
        self.lower_bound = lower
        self.upper_bound = upper
        self.fresh_optimizer_execution = fresh_optimizer_execution
        self.top_relation_count: int | None = (
            int(top_relation_count) if top_relation_count is not None else None
        )
        self.ordering = build_reference_blind_ordering(grouping_result)
        self.groups = self.ordering.groups
        self.topology = build_overlap_hypergraph(self.groups)

        self._groups: dict[tuple[int, int], GroupCycleObservation] = {}
        self._snapshots: list[CompletedSweepSnapshot] = []
        self._consecutive_complete_sweep_count = 0
        self._last_complete_sweep_index: int | None = None
        self._sweep_errors: dict[int, float] = {}
        self._fitness_prefix_hashes: dict[int, str] = {}
        self._states: tuple[object, ...] = ()
        self._group_scores: tuple[HyperedgeScore, ...] = ()
        self._native_relations: tuple[ScoredRelation, ...] = ()
        self._selection_relations: tuple[ScoredRelation, ...] = ()
        self._selection = RelationSelection((), True, "history_not_complete")
        self._plan_ready = False
        self._phase_boundary_snapshot: CompletedSweepSnapshot | None = None
        self._phase_boundary_history: tuple[int, ...] = ()
        self._barrier_result: EvidenceOverlayBarrierResult | None = None
        self._relation_probe_results: list[_RelationProbeResult] = []
        self._pending_outcomes: list[_PendingOwnerOutcome] = []
        self._delayed_rows: list[dict[str, str]] = []
        self._objective_calls = 0
        self._optimizer_calls = 0
        self._rng_calls = 0
        self._runtime_fingerprint_before: str | None = None
        self._runtime_fingerprint_after: str | None = None
        self._state_fingerprints: dict[str, dict[str, str]] = {}
        self._probe_start_fe: int | None = None
        self._probe_end_fe: int | None = None
        self._failure: dict[str, str] | None = None

    @property
    def evidence_overlay_fe(self) -> int:
        return self._objective_calls

    @property
    def plan_ready(self) -> bool:
        return self._plan_ready

    @property
    def consecutive_complete_sweep_count(self) -> int:
        return self._consecutive_complete_sweep_count

    @property
    def barrier_result(self) -> EvidenceOverlayBarrierResult | None:
        return self._barrier_result

    @property
    def runtime_probe_checkpoint_hash(self) -> str | None:
        snapshot = self._phase_boundary_snapshot
        if snapshot is None:
            return None
        return _canonical_hash(
            {
                "problem_id": self.problem_id,
                "seed": self.seed,
                "checkpoint_fe": snapshot.sweep_end_fe,
                "fitness_prefix_hash": self._fitness_prefix_hashes[
                    snapshot.sweep_index
                ],
                "incumbent_hash": _vector_hash(snapshot.sweep_end_candidate),
                "rddsm_topology_hash": self.ordering.topology_sha256,
                "rddsm_order_hash": self.ordering.ordering_sha256,
            }
        )

    @property
    def runtime_probe_actions(self) -> tuple[RuntimeProbeAction, ...]:
        """Compile public runtime actions without exposing raw probe internals."""

        result = self._barrier_result
        snapshot = self._phase_boundary_snapshot
        checkpoint_hash = self.runtime_probe_checkpoint_hash
        if (
            result is None
            or result.status != "probed"
            or snapshot is None
            or checkpoint_hash is None
        ):
            return ()
        actions: list[RuntimeProbeAction] = []
        for probe_result in self._relation_probe_results:
            winner = probe_result.decision.winner
            if winner not in {"left_owner", "right_owner", "bridge"}:
                continue
            probe = probe_result.probe
            candidate = {
                "left_owner": probe.x_left,
                "right_owner": probe.x_right,
                "bridge": probe.x_bridge,
            }[winner]
            relation = probe_result.scored_relation.relation.key
            shared = relation.shared_variable_indices
            shared_values = tuple(candidate[index] for index in shared)
            anchor_shared_values = tuple(probe.x0[index] for index in shared)
            actions.append(
                RuntimeProbeAction(
                    relation=relation,
                    winner=winner,
                    shared_values=shared_values,
                    shared_values_hash=runtime_probe_shared_values_hash(
                        relation,
                        shared_values,
                    ),
                    candidate_hash=_vector_hash(candidate),
                    bridge_weights=probe.weights,
                    utility=probe_result.decision.utility,
                    anchor_hash=runtime_probe_anchor_hash(
                        relation,
                        anchor_shared_values,
                    ),
                    checkpoint_fe=snapshot.sweep_end_fe,
                    checkpoint_hash=checkpoint_hash,
                    issued_sweep=snapshot.sweep_index,
                    ttl_sweeps=1,
                    expires_sweep=snapshot.sweep_index + 1,
                )
            )
        return tuple(actions)

    def record_group(
        self,
        *,
        sweep_index: int,
        group_index: int,
        pre_error: float,
        best_error: float,
        primary_requested_fe: int,
        primary_actual_fe: int,
        full_interval_actual_fe: int,
        full_interval_start_fe: int,
        full_interval_end_fe: int,
        pre_block_candidate: Sequence[float],
        final_owner_candidate: Sequence[float],
        local_top_candidates: Sequence[Sequence[float]],
    ) -> None:
        key = (
            _integer(sweep_index, name="sweep_index"),
            _integer(group_index, name="group_index"),
        )
        if key in self._groups:
            raise ValueError("duplicate evidence-overlay group observation")
        self._groups[key] = build_group_cycle_observation(
            self.topology,
            sweep_index=key[0],
            group_index=key[1],
            pre_error=pre_error,
            best_error=best_error,
            primary_requested_fe=primary_requested_fe,
            primary_actual_fe=primary_actual_fe,
            full_interval_actual_fe=full_interval_actual_fe,
            full_interval_start_fe=full_interval_start_fe,
            full_interval_end_fe=full_interval_end_fe,
            pre_block_candidate=pre_block_candidate,
            final_owner_candidate=final_owner_candidate,
            local_top_candidates=local_top_candidates,
            capture_stage=FINAL_OWNER_PROPOSAL_WATERMARK,
            capture_fe=full_interval_end_fe,
        )

    def complete_sweep(
        self,
        *,
        sweep_index: int,
        sweep_end_fe: int,
        sweep_end_candidate: Sequence[float],
        sweep_end_error: float,
        fitness_record: Sequence[float],
        all_raw_groups_completed: bool,
        native_sweep_end_completed: bool,
    ) -> bool:
        sweep = _integer(sweep_index, name="sweep_index")
        if not isinstance(all_raw_groups_completed, bool):
            raise ValueError("all_raw_groups_completed must be boolean")
        if not isinstance(native_sweep_end_completed, bool):
            raise ValueError("native_sweep_end_completed must be boolean")
        if not all_raw_groups_completed or not native_sweep_end_completed:
            self._drop_sweep(sweep)
            self._mark_missed_delayed_closure(sweep)
            self._consecutive_complete_sweep_count = 0
            self._last_complete_sweep_index = None
            return False

        observations = tuple(
            self._groups[(sweep, group)] for group in range(len(self.groups))
        )
        endpoint = tuple(float(value) for value in sweep_end_candidate)
        snapshot = CompletedSweepSnapshot(
            topology=self.topology,
            sweep_index=sweep,
            observations=observations,
            sweep_end_candidate=endpoint,
            native_sweep_end_completed=True,
            sweep_end_fe=_integer(sweep_end_fe, name="sweep_end_fe"),
        )
        record = tuple(float(value) for value in fitness_record)
        if len(record) < snapshot.sweep_end_fe:
            raise ValueError("fitness_record does not reach the sweep-end FE")
        self._snapshots.append(snapshot)
        if self._last_complete_sweep_index == sweep - 1:
            self._consecutive_complete_sweep_count += 1
        else:
            self._consecutive_complete_sweep_count = 1
        self._last_complete_sweep_index = sweep
        self._sweep_errors[sweep] = _finite(
            sweep_end_error,
            name="sweep_end_error",
            non_negative=True,
        )
        self._fitness_prefix_hashes[sweep] = _vector_hash(
            record[: snapshot.sweep_end_fe]
        )
        self._drop_sweep(sweep)
        self._resolve_delayed_outcomes(snapshot)
        if not self._plan_ready:
            self._prepare_plan()
        return True

    def _drop_sweep(self, sweep: int) -> None:
        for key in tuple(self._groups):
            if key[0] == sweep:
                del self._groups[key]

    def _latest_consecutive_snapshots(self) -> tuple[CompletedSweepSnapshot, ...]:
        if len(self._snapshots) < 3:
            return ()
        latest = tuple(self._snapshots[-3:])
        indices = tuple(snapshot.sweep_index for snapshot in latest)
        if indices != tuple(range(indices[0], indices[0] + 3)):
            return ()
        return latest

    def _prepare_plan(self) -> None:
        snapshots = self._latest_consecutive_snapshots()
        if not snapshots:
            return
        current = snapshots[-1]
        eligible = self.topology.eligible_group_indices
        credits = tuple(
            build_closed_owner_credit(
                proposal_observation=snapshots[-2].observation_for_group(group),
                resolution_snapshot=current,
            )
            for group in eligible
        )
        states = build_hyperedge_cycle_states(
            self.topology,
            snapshots,
            closed_owner_credits=credits,
            decision_fe=current.sweep_end_fe,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        group_scores = score_hyperedge_states(states)
        priorities = [0.0] * len(self.groups)
        reliabilities = [0.0] * len(self.groups)
        for group, score in zip(eligible, group_scores, strict=True):
            priorities[group] = score.focal_priority
            reliabilities[group] = score.owner_reliability
        owner_proposals = {
            (group, variable): value
            for group in eligible
            for variable, value in current.observation_for_group(
                group
            ).shared_proposal.proposed_values
        }
        owner_population_samples = {
            (group, variable): samples
            for group in eligible
            for variable, samples in current.observation_for_group(
                group
            ).shared_top_k_population.variable_samples
        }
        candidates = build_relation_candidates(
            self.groups,
            owner_proposals,
            owner_population_samples,
            priorities,
            reliabilities,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        native = score_relations(candidates)
        native_selection = select_top_relations(native, count=self.top_relation_count)

        selected_scores = native
        selection = native_selection
        if self.mode == "shuffled_owner" and not native_selection.abstained:
            try:
                shuffled = shuffle_relation_scores(native, seed=self.seed)
            except ValueError:
                selection = RelationSelection((), True, "shuffle_derangement_unavailable")
            else:
                shuffled_selection = select_top_relations(shuffled, count=self.top_relation_count)
                native_keys = {item.relation.key for item in native_selection.selected}
                shuffled_keys = {
                    item.relation.key for item in shuffled_selection.selected
                }
                if shuffled_selection.abstained or shuffled_keys == native_keys:
                    selection = RelationSelection(
                        (),
                        True,
                        "shuffled_selection_not_deranged",
                    )
                else:
                    selected_scores = shuffled
                    selection = shuffled_selection

        self._states = states
        self._group_scores = group_scores
        self._native_relations = native
        self._selection_relations = selected_scores
        self._selection = selection
        self._phase_boundary_snapshot = current
        self._phase_boundary_history = tuple(
            snapshot.sweep_index for snapshot in snapshots
        )
        self._plan_ready = True

    def execute_barrier(
        self,
        objective: Callable[[tuple[float, ...]], object],
        anchor: Sequence[float],
        remaining_fe: int,
        normal_sweep_fe: int,
        tolerance_fe: int,
    ) -> EvidenceOverlayBarrierResult:
        if self._barrier_result is not None:
            raise ValueError("evidence overlay barrier may execute only once")
        if not callable(objective):
            raise TypeError("objective must be callable")
        remaining = _integer(remaining_fe, name="remaining_fe")
        normal_sweep = _integer(normal_sweep_fe, name="normal_sweep_fe", minimum=1)
        tolerance = _integer(tolerance_fe, name="tolerance_fe")
        if tolerance != self.terminal_tolerance_fe:
            raise ValueError(
                "tolerance_fe must match the observer's frozen terminal tolerance"
            )
        anchor_tuple = tuple(_finite(value, name="anchor value") for value in anchor)
        if not anchor_tuple:
            raise ValueError("anchor must be non-empty")
        anchor_hash = _vector_hash(anchor_tuple)
        snapshot = self._phase_boundary_snapshot
        phase_boundary_fe = 0 if snapshot is None else snapshot.sweep_end_fe

        if not self._plan_ready or snapshot is None:
            return self._abstain_barrier(
                "history_not_complete",
                requested_fe=0,
                phase_boundary_fe=phase_boundary_fe,
                anchor_hash=anchor_hash,
            )
        if anchor_tuple != snapshot.sweep_end_candidate:
            raise ValueError("barrier anchor must equal the frozen phase-boundary incumbent")
        if self.mode == "off":
            return self._abstain_barrier(
                "overlay_off",
                requested_fe=0,
                phase_boundary_fe=phase_boundary_fe,
                anchor_hash=anchor_hash,
            )
        if self._selection.abstained:
            return self._abstain_barrier(
                self._selection.reason,
                requested_fe=0,
                phase_boundary_fe=phase_boundary_fe,
                anchor_hash=anchor_hash,
            )

        selected = self._selection.selected
        requested_fe = 4 * len(selected)
        if self.mode == "native_audit":
            return self._abstain_barrier(
                "native_audit_zero_probe_fe",
                requested_fe=0,
                phase_boundary_fe=phase_boundary_fe,
                anchor_hash=anchor_hash,
            )
        if remaining < requested_fe + normal_sweep + tolerance:
            return self._abstain_barrier(
                "insufficient_budget_for_probe_and_native_sweep",
                requested_fe=requested_fe,
                phase_boundary_fe=phase_boundary_fe,
                anchor_hash=anchor_hash,
            )

        try:
            for scored in selected:
                self._execute_relation_probe(objective, anchor_tuple, scored)
            if self._objective_calls != requested_fe:
                raise EvidenceOverlayRuntimeError(
                    "probe FE does not match its reservation"
                )
            anchor_fitnesses = tuple(
                result.fitness[0] for result in self._relation_probe_results
            )
            if any(
                value != anchor_fitnesses[0]
                for value in anchor_fitnesses[1:]
            ):
                raise EvidenceOverlayRuntimeError(
                    "repeated x0 fitness values are not deterministic"
                )
            if tuple(anchor) != anchor_tuple:
                raise EvidenceOverlayRuntimeError(
                    "evidence overlay mutated the caller anchor"
                )
        except EvidenceOverlayRuntimeError as error:
            if self._failure is None:
                self._failure = {
                    "stage": "four_point_probe_integrity",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "objective_calls": str(self._objective_calls),
                }
            self._barrier_result = EvidenceOverlayBarrierResult(
                status="failed",
                reason=(
                    "four_point_probe_objective_failure"
                    if self._failure["stage"] == "four_point_probe"
                    else "four_point_probe_integrity_failure"
                ),
                requested_fe=requested_fe,
                actual_fe=self._objective_calls,
                selected_relation_count=len(selected),
                phase_boundary_fe=phase_boundary_fe,
                anchor_hash=anchor_hash,
                anchor_unchanged=tuple(anchor) == anchor_tuple,
            )
            raise
        result = EvidenceOverlayBarrierResult(
            status="probed",
            reason="four_point_probe_complete",
            requested_fe=requested_fe,
            actual_fe=self._objective_calls,
            selected_relation_count=len(selected),
            phase_boundary_fe=phase_boundary_fe,
            anchor_hash=anchor_hash,
            anchor_unchanged=True,
        )
        self._barrier_result = result
        return result

    def _abstain_barrier(
        self,
        reason: str,
        *,
        requested_fe: int,
        phase_boundary_fe: int,
        anchor_hash: str,
    ) -> EvidenceOverlayBarrierResult:
        result = EvidenceOverlayBarrierResult(
            status="abstained",
            reason=reason,
            requested_fe=requested_fe,
            actual_fe=0,
            selected_relation_count=len(self._selection.selected),
            phase_boundary_fe=phase_boundary_fe,
            anchor_hash=anchor_hash,
            anchor_unchanged=True,
        )
        self._barrier_result = result
        return result

    def record_runtime_audit(
        self,
        *,
        fingerprints_before: Mapping[str, str],
        fingerprints_after: Mapping[str, str],
        probe_start_fe: int,
        probe_end_fe: int,
    ) -> None:
        if self._runtime_fingerprint_before is not None:
            raise ValueError("runtime audit may be recorded only once")
        result = self._barrier_result
        if result is None:
            raise ValueError("runtime audit requires an executed barrier")
        before = dict(fingerprints_before)
        after = dict(fingerprints_after)
        if not before or set(before) != set(after):
            raise ValueError("runtime state fingerprint components must align")
        for component, values in (
            (component, (before[component], after[component]))
            for component in sorted(before)
        ):
            for value in values:
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                ):
                    raise ValueError(
                        f"{component} must use lowercase SHA-256 digests"
                    )
        start = _integer(probe_start_fe, name="probe_start_fe")
        end = _integer(probe_end_fe, name="probe_end_fe")
        if end < start or end - start != result.actual_fe:
            raise ValueError("probe FE slice must match the barrier's actual FE")

        self._runtime_fingerprint_before = _canonical_hash(before)
        self._runtime_fingerprint_after = _canonical_hash(after)
        self._state_fingerprints = {
            component: {
                "before": before[component],
                "after": after[component],
            }
            for component in sorted(before)
        }
        self._probe_start_fe = start
        self._probe_end_fe = end
        if before == after:
            return

        error = EvidenceOverlayRuntimeError(
            "evidence overlay probe mutated frozen runtime state"
        )
        self._failure = {
            "stage": "runtime_state_integrity",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "objective_calls": str(self._objective_calls),
        }
        self._barrier_result = replace(
            result,
            status="failed",
            reason="runtime_state_fingerprint_changed",
        )
        raise error

    def _evaluate(
        self,
        objective: Callable[[tuple[float, ...]], object],
        candidate: tuple[float, ...],
    ) -> float:
        self._objective_calls += 1
        try:
            return _objective_value(objective(candidate))
        except Exception as error:
            self._failure = {
                "stage": "four_point_probe",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "objective_calls": str(self._objective_calls),
            }
            raise EvidenceOverlayRuntimeError(
                "evidence overlay objective failed after probe admission"
            ) from error

    def _execute_relation_probe(
        self,
        objective: Callable[[tuple[float, ...]], object],
        anchor: tuple[float, ...],
        scored: ScoredRelation,
    ) -> None:
        probe = build_four_point_probe(anchor, scored.relation)
        candidates = (
            probe.x0,
            probe.x_left,
            probe.x_right,
            probe.x_bridge,
        )
        fitness = tuple(self._evaluate(objective, candidate) for candidate in candidates)
        utilities = summarize_probe_utilities(
            anchor_fitness=fitness[0],
            left_fitness=fitness[1],
            right_fitness=fitness[2],
            bridge_fitness=fitness[3],
        )
        decision = decide_shadow_action(utilities)
        self._relation_probe_results.append(
            _RelationProbeResult(scored, probe, fitness, decision)
        )
        snapshot = self._phase_boundary_snapshot
        if snapshot is None:
            raise EvidenceOverlayRuntimeError("probe executed without a phase boundary")
        shared = scored.relation.key.shared_variable_indices
        self._pending_outcomes.append(
            _PendingOwnerOutcome(
                relation=scored.relation.key,
                action_sweep_index=snapshot.sweep_index,
                anchor_error=fitness[0],
                anchor_shared_values=tuple(anchor[index] for index in shared),
                left_shared_values=tuple(probe.x_left[index] for index in shared),
                right_shared_values=tuple(probe.x_right[index] for index in shared),
            )
        )

    def _resolve_delayed_outcomes(self, snapshot: CompletedSweepSnapshot) -> None:
        pending = tuple(self._pending_outcomes)
        for item in pending:
            expected = item.action_sweep_index + 1
            if snapshot.sweep_index < expected:
                continue
            if snapshot.sweep_index > expected:
                self._append_unclosed_owner_rows(item, "missed_next_complete_sweep")
                self._pending_outcomes.remove(item)
                continue
            next_values = tuple(
                snapshot.sweep_end_candidate[index]
                for index in item.relation.shared_variable_indices
            )
            next_error = self._sweep_errors[snapshot.sweep_index]
            for owner, candidate_values in (
                ("left", item.left_shared_values),
                ("right", item.right_shared_values),
            ):
                credit = build_delayed_hyperedge_credit(
                    action_sweep_index=item.action_sweep_index,
                    resolution_sweep_index=snapshot.sweep_index,
                    all_groups_completed=True,
                    native_sweep_end_completed=True,
                    anchor_error=item.anchor_error,
                    next_sweep_error=next_error,
                    anchor_shared_values=item.anchor_shared_values,
                    candidate_shared_values=candidate_values,
                    next_sweep_shared_values=next_values,
                )
                self._delayed_rows.append(
                    {
                        "problem_id": self.problem_id,
                        "seed": str(self.seed),
                        "mode": self.mode,
                        "relation_id": _relation_id(item.relation),
                        "owner": owner,
                        "action_sweep_index": str(item.action_sweep_index),
                        "resolution_sweep_index": str(snapshot.sweep_index),
                        "survival_label": _format_float(credit.survival),
                        "overwrite_label": _format_float(credit.overwrite),
                        "next_sweep_log_improvement": _format_float(
                            credit.next_sweep_log_improvement
                        ),
                        "overwrite_penalized_credit": _format_float(
                            credit.penalized_credit
                        ),
                        "label_closed": "1",
                        "label_status": "closed_next_complete_sweep",
                        "resolution_fe": str(snapshot.sweep_end_fe),
                        "runtime_authorized": "0",
                    }
                )
            self._pending_outcomes.remove(item)

    def _mark_missed_delayed_closure(self, sweep_index: int) -> None:
        missed = False
        for item in tuple(self._pending_outcomes):
            if sweep_index == item.action_sweep_index + 1:
                self._append_unclosed_owner_rows(item, "incomplete_next_native_sweep")
                self._pending_outcomes.remove(item)
                missed = True
        if missed and self._barrier_result is not None:
            error = EvidenceOverlayRuntimeError(
                "next native sweep was incomplete after the probe barrier"
            )
            self._failure = {
                "stage": "delayed_sweep_closure",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "objective_calls": str(self._objective_calls),
            }
            self._barrier_result = replace(
                self._barrier_result,
                status="failed",
                reason="incomplete_next_native_sweep",
            )

    def _append_unclosed_owner_rows(
        self,
        item: _PendingOwnerOutcome,
        status: str,
    ) -> None:
        for owner in ("left", "right"):
            self._delayed_rows.append(
                {
                    "problem_id": self.problem_id,
                    "seed": str(self.seed),
                    "mode": self.mode,
                    "relation_id": _relation_id(item.relation),
                    "owner": owner,
                    "action_sweep_index": str(item.action_sweep_index),
                    "resolution_sweep_index": "",
                    "survival_label": "",
                    "overwrite_label": "",
                    "next_sweep_log_improvement": "",
                    "overwrite_penalized_credit": "",
                    "label_closed": "0",
                    "label_status": status,
                    "resolution_fe": "",
                    "runtime_authorized": "0",
                }
            )

    @property
    def delayed_outcomes_pending(self) -> bool:
        return bool(self._pending_outcomes)

    def _checkpoint_rows(self) -> list[dict[str, str]]:
        snapshot = self._phase_boundary_snapshot
        if snapshot is None:
            return []
        return [
            {
                "problem_id": self.problem_id,
                "seed": str(self.seed),
                "mode": self.mode,
                "checkpoint_fe": str(snapshot.sweep_end_fe),
                "fitness_prefix_hash": self._fitness_prefix_hashes[snapshot.sweep_index],
                "incumbent_hash": _vector_hash(snapshot.sweep_end_candidate),
                "rddsm_topology_hash": self.ordering.topology_sha256,
                "rddsm_order_hash": self.ordering.ordering_sha256,
                "phase_boundary_fe": str(snapshot.sweep_end_fe),
                "history_sweeps": ";".join(
                    str(sweep_index)
                    for sweep_index in self._phase_boundary_history
                ),
                "previous_survival_closed": "1",
                "plan_status": "abstained" if self._selection.abstained else "selected",
                "plan_reason": self._selection.reason,
                "runtime_authorized": "0",
            }
        ]

    def _plan_rows(self) -> list[dict[str, str]]:
        native_by_key = {
            item.relation.key: item for item in self._native_relations
        }
        selected = {item.relation.key for item in self._selection.selected}
        snapshot = self._phase_boundary_snapshot
        boundary_fe = "" if snapshot is None else str(snapshot.sweep_end_fe)
        return [
            {
                "problem_id": self.problem_id,
                "seed": str(self.seed),
                "mode": self.mode,
                "relation_id": _relation_id(item.relation.key),
                "owner_groups": _csv_values(item.relation.key.owner_group_indices),
                "shared_variables": _csv_values(item.relation.key.shared_variable_indices),
                "selected": str(int(item.relation.key in selected)),
                "voi": _format_float(item.voi_score),
                "native_voi": _format_float(
                    native_by_key[item.relation.key].voi_score
                ),
                "proposal_disagreement": _format_float(
                    item.relation.proposal_disagreement
                ),
                "cohen_d": _format_float(relation_cohen_d(item.relation)),
                "disagreement_metric": PROPOSAL_DISAGREEMENT_METRIC,
                "left_top_k_count": str(item.relation.owner_population_sizes[0]),
                "right_top_k_count": str(item.relation.owner_population_sizes[1]),
                "left_distribution_centers": _csv_float_values(
                    item.relation.owner_population_centers[0]
                ),
                "right_distribution_centers": _csv_float_values(
                    item.relation.owner_population_centers[1]
                ),
                "left_distribution_standard_deviations": _csv_float_values(
                    item.relation.owner_population_standard_deviations[0]
                ),
                "right_distribution_standard_deviations": _csv_float_values(
                    item.relation.owner_population_standard_deviations[1]
                ),
                "owner_priority": _format_float(item.relation.owner_priority),
                "left_owner_reliability": _format_float(
                    item.relation.owner_reliabilities[0]
                ),
                "right_owner_reliability": _format_float(
                    item.relation.owner_reliabilities[1]
                ),
                "score_source_relation_id": _relation_id(item.score_source),
                "phase_boundary_fe": boundary_fe,
                "runtime_authorized": "0",
            }
            for item in self._selection_relations
        ]

    def _probe_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if not self._relation_probe_results:
            return rows
        snapshot = self._phase_boundary_snapshot
        if snapshot is None:
            raise EvidenceOverlayRuntimeError("probe evidence has no phase boundary")
        for result in self._relation_probe_results:
            relation = result.scored_relation.relation
            utilities = summarize_probe_utilities(
                anchor_fitness=result.fitness[0],
                left_fitness=result.fitness[1],
                right_fitness=result.fitness[2],
                bridge_fitness=result.fitness[3],
            )
            candidates = (
                result.probe.x0,
                result.probe.x_left,
                result.probe.x_right,
                result.probe.x_bridge,
            )
            utility_values = (
                0.0,
                utilities.left_owner,
                utilities.right_owner,
                utilities.bridge,
            )
            reliability_values: tuple[float | None, ...] = (
                None,
                relation.owner_reliabilities[0],
                relation.owner_reliabilities[1],
                (
                    result.probe.weights.left_owner
                    * relation.owner_reliabilities[0]
                    + result.probe.weights.right_owner
                    * relation.owner_reliabilities[1]
                ),
            )
            for candidate_name, candidate, fitness, utility, reliability in zip(
                PROBE_CANDIDATES,
                candidates,
                result.fitness,
                utility_values,
                reliability_values,
                strict=True,
            ):
                rows.append(
                    {
                        "problem_id": self.problem_id,
                        "seed": str(self.seed),
                        "mode": self.mode,
                        "relation_id": _relation_id(relation.key),
                        "candidate": candidate_name,
                        "fitness": _format_float(fitness),
                        "utility": _format_float(utility),
                        "owner_reliability": _format_float(reliability),
                        "candidate_hash": _vector_hash(candidate),
                        "phase_boundary_fe": str(snapshot.sweep_end_fe),
                        "actual_fe": "1",
                        "runtime_authorized": "0",
                    }
                )
        return rows

    def _shadow_rows(self) -> list[dict[str, str]]:
        return [
            {
                "problem_id": self.problem_id,
                "seed": str(self.seed),
                "mode": self.mode,
                "relation_id": _relation_id(result.scored_relation.relation.key),
                "action": result.decision.shadow_action,
                "winner": result.decision.winner,
                "utility": _format_float(result.decision.utility),
                "reason": result.decision.reason,
                "runtime_authorized": "0",
            }
            for result in self._relation_probe_results
        ]

    def _runtime_action_rows(
        self,
        ledger: RuntimeProbeActionLedger | None,
    ) -> list[dict[str, str]]:
        if ledger is None:
            return []
        return [
            {
                "problem_id": self.problem_id,
                "seed": str(self.seed),
                "mode": self.mode,
                "relation_id": _relation_id(record.action.relation),
                "owner_groups": _csv_values(
                    record.action.relation.owner_group_indices
                ),
                "shared_variables": _csv_values(
                    record.action.relation.shared_variable_indices
                ),
                "winner": record.action.winner,
                "canonical_action": record.action.canonical_action,
                "shared_values": _csv_float_values(record.action.shared_values),
                "shared_values_hash": record.action.shared_values_hash,
                "candidate_hash": record.action.candidate_hash,
                "bridge_weight_left": _format_float(
                    record.action.bridge_weights.left_owner
                ),
                "bridge_weight_right": _format_float(
                    record.action.bridge_weights.right_owner
                ),
                "utility": _format_float(record.action.utility),
                "anchor_hash": record.action.anchor_hash,
                "checkpoint_fe": str(record.action.checkpoint_fe),
                "checkpoint_hash": record.action.checkpoint_hash,
                "issued_sweep": str(record.action.issued_sweep),
                "ttl_sweeps": str(record.action.ttl_sweeps),
                "expires_sweep": str(record.action.expires_sweep),
                "runtime_authorized": "1",
                "runtime_consumed": str(int(record.runtime_consumed)),
                "consumed_sweep": (
                    "" if record.consumed_sweep is None else str(record.consumed_sweep)
                ),
                "consumed_fe": (
                    "" if record.consumed_fe is None else str(record.consumed_fe)
                ),
                "status": (
                    record.status if record.runtime_consumed else "abstained"
                ),
                "invalidation_reason": (
                    record.invalidation_reason
                    or ("" if record.runtime_consumed else "not_dispatched")
                ),
            }
            for record in ledger.records
        ]

    def write_artifacts(
        self,
        *,
        paths: EvidenceOverlayArtifactPaths,
        native_terminal_error: float,
        all_evaluation_best_error: float,
        runtime_action_ledger: RuntimeProbeActionLedger | None = None,
    ) -> dict[str, object]:
        native_terminal = _finite(
            native_terminal_error,
            name="native_terminal_error",
            non_negative=True,
        )
        all_best = _finite(
            all_evaluation_best_error,
            name="all_evaluation_best_error",
            non_negative=True,
        )
        for item in tuple(self._pending_outcomes):
            self._append_unclosed_owner_rows(item, "terminal_censored")
            self._pending_outcomes.remove(item)

        runtime_action_rows = self._runtime_action_rows(runtime_action_ledger)
        artifact_rows = (
            (paths.checkpoint, CHECKPOINT_FIELDS, self._checkpoint_rows()),
            (paths.plan, PLAN_FIELDS, self._plan_rows()),
            (paths.probe_evidence, PROBE_EVIDENCE_FIELDS, self._probe_rows()),
            (paths.delayed_outcomes, DELAYED_OUTCOME_FIELDS, self._delayed_rows),
            (paths.shadow_decisions, SHADOW_DECISION_FIELDS, self._shadow_rows()),
            (paths.runtime_actions, RUNTIME_ACTION_FIELDS, runtime_action_rows),
        )
        for path, fields, rows in artifact_rows:
            _write_csv(path, fields, rows)
        artifact_paths = {
            "checkpoint": paths.checkpoint.name,
            "plan": paths.plan.name,
            "probe_evidence": paths.probe_evidence.name,
            "delayed_outcomes": paths.delayed_outcomes.name,
            "shadow_decisions": paths.shadow_decisions.name,
            "runtime_actions": paths.runtime_actions.name,
        }
        artifact_hashes = {
            path.name: _file_hash(path) for path, _, _ in artifact_rows
        }
        result = self._barrier_result
        delayed_expected = 2 * len(self._relation_probe_results)
        delayed_closed = sum(row["label_closed"] == "1" for row in self._delayed_rows)
        integrity = bool(
            self._failure is None
            and self._optimizer_calls == 0
            and self._rng_calls == 0
            and (result is None or result.actual_fe == self._objective_calls)
            and (
                result is None
                or (
                    self._runtime_fingerprint_before is not None
                    and self._runtime_fingerprint_before
                    == self._runtime_fingerprint_after
                )
            )
            and (
                not self._relation_probe_results
                or delayed_closed == delayed_expected
            )
        )
        applicable = bool(
            result is not None
            and result.status == "probed"
            and self._failure is None
            and integrity
        )
        abstain_reason = ""
        if not applicable:
            abstain_reason = (
                "barrier_not_executed" if result is None else result.reason
            )
        manifest: dict[str, object] = {
            "protocol_version": EVIDENCE_OVERLAY_PROTOCOL_VERSION,
            "schema_version": 2,
            "problem_id": self.problem_id,
            "seed": self.seed,
            "evidence_overlay_mode": self.mode,
            "configured_max_fes": self.configured_max_fes,
            "source_mode": EVIDENCE_OVERLAY_SOURCE_MODE,
            "terminal_tolerance_rule": TERMINAL_TOLERANCE_RULE,
            "terminal_tolerance_fe": self.terminal_tolerance_fe,
            "run_id": self.run_id,
            "fresh_optimizer_execution": int(self.fresh_optimizer_execution),
            "runtime_authorized": int(bool(runtime_action_rows)),
            "runtime_consumed": int(
                any(row["runtime_consumed"] == "1" for row in runtime_action_rows)
            ),
            "runtime_actions_issued": len(runtime_action_rows),
            "runtime_actions_authorized": len(runtime_action_rows),
            "runtime_actions_consumed": sum(
                row["runtime_consumed"] == "1" for row in runtime_action_rows
            ),
            "runtime_actions_abstained": sum(
                row["runtime_consumed"] != "1" for row in runtime_action_rows
            ),
            "aob_truth_runtime_used": 0,
            "runtime_input_fields": list(RUNTIME_INPUT_FIELDS),
            "proposal_disagreement_metric": PROPOSAL_DISAGREEMENT_METRIC,
            "local_optimum_top_k": LOCAL_OPTIMUM_TOP_K,
            "runtime_fingerprint_before": self._runtime_fingerprint_before,
            "runtime_fingerprint_after": self._runtime_fingerprint_after,
            "state_fingerprints": self._state_fingerprints,
            "native_state_unchanged": int(
                self._runtime_fingerprint_before is not None
                and self._runtime_fingerprint_before
                == self._runtime_fingerprint_after
            ),
            "probe_start_fe": self._probe_start_fe,
            "probe_end_fe": self._probe_end_fe,
            "applicable": int(applicable),
            "abstain_reason": abstain_reason,
            "rddsm_topology_hash": self.ordering.topology_sha256,
            "rddsm_order_hash": self.ordering.ordering_sha256,
            "phase_boundary_fe": (
                None
                if self._phase_boundary_snapshot is None
                else self._phase_boundary_snapshot.sweep_end_fe
            ),
            "barrier_status": "not_executed" if result is None else result.status,
            "barrier_reason": "" if result is None else result.reason,
            "selected_relation_count": len(self._selection.selected),
            "objective_calls": self._objective_calls,
            "optimizer_calls": self._optimizer_calls,
            "rng_calls": self._rng_calls,
            "evidence_overlay_fe": self._objective_calls,
            "native_terminal_error": native_terminal,
            "all_evaluation_best_error": all_best,
            "complete_sweep_count": len(self._snapshots),
            "delayed_label_expected": delayed_expected,
            "delayed_label_closed": delayed_closed,
            "artifacts": artifact_paths,
            "artifact_sha256": artifact_hashes,
            "failure": self._failure,
            "observer_integrity": int(integrity),
        }
        paths.manifest.parent.mkdir(parents=True, exist_ok=True)
        paths.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest


__all__ = [
    "CHECKPOINT_FIELDS",
    "DELAYED_OUTCOME_FIELDS",
    "EVIDENCE_OVERLAY_PROTOCOL_VERSION",
    "EVIDENCE_OVERLAY_SOURCE_MODE",
    "EvidenceOverlayArtifactPaths",
    "EvidenceOverlayBarrierResult",
    "EvidenceOverlayRuntimeError",
    "HccEvidenceOverlayObserver",
    "PLAN_FIELDS",
    "PROBE_EVIDENCE_FIELDS",
    "RUNTIME_ACTION_FIELDS",
    "RUNTIME_INPUT_FIELDS",
    "RuntimeProbeActionLedger",
    "RuntimeProbeActionRecord",
    "RuntimeProbeConsumption",
    "SHADOW_DECISION_FIELDS",
    "TERMINAL_TOLERANCE_RULE",
]
