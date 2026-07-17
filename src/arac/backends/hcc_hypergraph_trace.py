"""Trace-only v37 overlap-hypergraph observer.

The observer receives already-computed optimizer state from the ARAC-owned HCC
runner.  It has no objective, RNG, or optimizer handle, so enabling it cannot
consume FE or perturb the v37 execution path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from arac.policy.overlap_hypergraph import (
    FINAL_OWNER_PROPOSAL_WATERMARK,
    HYPERGRAPH_SCHEMA_VERSION,
    ClosedOwnerCredit,
    CompletedSweepSnapshot,
    GroupCycleObservation,
    HyperedgeCycleState,
    HyperedgeScore,
    OverlapHypergraphTopology,
    build_group_cycle_observation,
    build_hyperedge_cycle_states,
    build_closed_owner_credit,
    score_hyperedge_states,
)


HYPERGRAPH_TRACE_MODES = frozenset({"off", "observer"})
HYPERGRAPH_NATIVE_SWEEP_END_STAGE = (
    "after_search_state_and_component_credit_before_outer_iter_increment"
)

HYPERGRAPH_FEATURE_FIELDS = (
    "decision_id",
    "current_unit_fe_contribution",
    "ewma_unit_fe_contribution_3",
    "zero_gain_difficulty",
    "stagnation_ratio_3",
    "direct_owner_proposal_disagreement",
    "prior_next_sweep_overwrite",
    "contribution_score",
    "need_score",
    "focal_priority",
    "owner_reliability",
)
HYPERGRAPH_AUDIT_FIELDS = (
    "protocol_version",
    "decision_id",
    "problem_id",
    "seed",
    "sweep_index",
    "group_index",
    "cohort_locked",
    "state_complete",
    "unique_focal",
    "applicable",
    "not_applicable_reason",
    "source_end_fe",
    "decision_fe",
    "full_interval_start_fe",
    "full_interval_end_fe",
    "primary_requested_fe",
    "primary_actual_fe",
    "full_interval_actual_fe",
    "pre_error",
    "best_error",
    "successful",
    "unit_fe_contribution",
    "feature_sha256",
    "topology_sha256",
    "rng_descriptor_sha256",
    "fitness_record_sha256",
    "proposal_capture_watermark",
    "all_raw_groups_completed",
    "native_sweep_end_completed",
    "native_sweep_end_stage",
    "watermark_valid",
    "observer_integrity",
)
HYPERGRAPH_PROPOSAL_FIELDS = (
    "protocol_version",
    "decision_id",
    "problem_id",
    "seed",
    "sweep_index",
    "group_index",
    "variable_index",
    "capture_watermark",
    "proposal_source_end_fe",
    "anchor_value",
    "proposed_value",
    "sweep_end_value",
    "next_sweep_value",
    "sweep_end_fe",
    "next_sweep_end_fe",
    "topology_sha256",
    "observer_integrity",
)
HYPERGRAPH_OUTCOME_FIELDS = (
    "decision_id",
    "problem_id",
    "seed",
    "sweep_index",
    "resolution_sweep_index",
    "next_sweep_unit_fe_contribution",
    "next_sweep_survival",
    "next_sweep_overwrite",
    "resolution_end_fe",
    "all_groups_completed",
    "native_sweep_end_completed",
    "outcome_complete",
    "terminal_censored",
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_float(value: float) -> str:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("trace values must be finite")
    return f"{converted:.17e}"


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_hypergraph_initialization_failure_manifest(
    *,
    path: Path,
    problem_id: str,
    seed: int | None,
    run_id: str,
    fresh_optimizer_execution: bool,
    terminal_target_fe: int,
    terminal_completion_tolerance_fe: int,
    error: BaseException,
    source_fe: int | None,
) -> dict[str, object]:
    """Best-effort explicit evidence failure when observer construction fails."""

    if not isinstance(fresh_optimizer_execution, bool):
        raise TypeError("fresh_optimizer_execution must be boolean")
    manifest: dict[str, object] = {
        "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
        "schema_version": 1,
        "hypergraph_trace_mode": "observer",
        "fresh_optimizer_execution": int(fresh_optimizer_execution),
        "problem_id": str(problem_id),
        "seed": seed,
        "run_id": str(run_id),
        "terminal_target_fe": int(terminal_target_fe),
        "terminal_observed_fe": "" if source_fe is None else int(source_fe),
        "terminal_completion_tolerance_fe": int(
            terminal_completion_tolerance_fe
        ),
        "artifact_sha256": {},
        "observer_status": "failed",
        "observer_error_stage": "initialization",
        "observer_error_type": type(error).__name__,
        "observer_error_message": str(error)[:1000],
        "observer_error_source_fe": "" if source_fe is None else str(int(source_fe)),
        "observer_integrity": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class HypergraphTraceArtifactPaths:
    manifest: Path
    features: Path
    audit: Path
    proposals: Path
    outcomes: Path


@dataclass(frozen=True)
class _ObservedGroup:
    observation: GroupCycleObservation
    decision_id: str
    pre_error: float
    best_error: float


class HypergraphTraceObserver:
    """Collect complete-sweep evidence without owning any executable handle."""

    def __init__(
        self,
        *,
        topology: OverlapHypergraphTopology,
        problem_id: str,
        seed: int | None,
        run_id: str,
        fresh_optimizer_execution: bool,
        lower_bound: float,
        upper_bound: float,
        rng_descriptor_sha256: str,
        protocol_config_path: Path,
        protocol_spec_path: Path,
        runner_source_path: Path,
        terminal_target_fe: int,
        terminal_completion_tolerance_fe: int,
    ) -> None:
        if not isinstance(topology, OverlapHypergraphTopology):
            raise TypeError("topology must be OverlapHypergraphTopology")
        lower = float(lower_bound)
        upper = float(upper_bound)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError("observer bounds must be finite and ordered")
        if len(rng_descriptor_sha256) != 64:
            raise ValueError("rng_descriptor_sha256 must be a full SHA-256")
        if not isinstance(fresh_optimizer_execution, bool):
            raise TypeError("fresh_optimizer_execution must be boolean")
        if isinstance(terminal_target_fe, bool) or not isinstance(
            terminal_target_fe, int
        ):
            raise TypeError("terminal_target_fe must be an integer")
        if isinstance(terminal_completion_tolerance_fe, bool) or not isinstance(
            terminal_completion_tolerance_fe, int
        ):
            raise TypeError("terminal_completion_tolerance_fe must be an integer")
        if terminal_target_fe <= 0:
            raise ValueError("terminal_target_fe must be positive")
        if not 1 <= terminal_completion_tolerance_fe <= terminal_target_fe:
            raise ValueError(
                "terminal_completion_tolerance_fe must be between 1 and "
                "terminal_target_fe"
            )
        source_paths = (
            Path(protocol_config_path),
            Path(protocol_spec_path),
            Path(runner_source_path),
        )
        if not all(path.is_file() for path in source_paths):
            raise FileNotFoundError("observer source binding is incomplete")

        self.topology = topology
        self.problem_id = str(problem_id)
        self.seed = seed
        self.run_id = str(run_id)
        self.fresh_optimizer_execution = fresh_optimizer_execution
        self.lower_bound = lower
        self.upper_bound = upper
        self.rng_descriptor_sha256 = rng_descriptor_sha256
        self.protocol_config_path = source_paths[0]
        self.protocol_spec_path = source_paths[1]
        self.runner_source_path = source_paths[2]
        self.terminal_target_fe = terminal_target_fe
        self.terminal_completion_tolerance_fe = terminal_completion_tolerance_fe
        self.topology_sha256 = _canonical_sha256(
            {
                "hyperedges": topology.hyperedges,
                "variable_owner_groups": topology.variable_owner_groups,
            }
        )

        self._groups: dict[tuple[int, int], _ObservedGroup] = {}
        self._complete_sweeps: dict[int, CompletedSweepSnapshot] = {}
        self._incomplete_sweeps: set[int] = set()
        self._sweep_end_fe: dict[int, int] = {}
        self._audit_rows: dict[tuple[int, int], dict[str, str]] = {}
        self._feature_rows: dict[tuple[int, int], dict[str, str]] = {}
        self._proposal_rows: dict[tuple[int, int, int], dict[str, str]] = {}
        self._outcomes: dict[tuple[int, int], dict[str, str]] = {}
        self._closed_credits: dict[tuple[int, int], ClosedOwnerCredit] = {}
        self._decision_lock_consumed = False
        self._decision_snapshot_sweep: int | None = None
        self._decision_status = "pending_three_complete_sweeps"
        self._decision_reason = ""
        self._fitness_record_hasher = hashlib.sha256()
        self._fitness_record_hashed_fe = 0
        self._failure: dict[str, str] | None = None

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def record_failure(
        self,
        *,
        stage: str,
        error: BaseException,
        source_fe: int | None,
    ) -> None:
        """Seal the first observer failure without raising into v37."""

        if self._failure is not None:
            return
        self._failure = {
            "observer_error_stage": str(stage) or "unknown",
            "observer_error_type": type(error).__name__,
            "observer_error_message": str(error)[:1000],
            "observer_error_source_fe": (
                "" if source_fe is None else str(int(source_fe))
            ),
        }

    def _advance_fitness_record_hash(
        self,
        fitness_record: Sequence[float],
        end_fe: int,
    ) -> str:
        endpoint = int(end_fe)
        if endpoint < self._fitness_record_hashed_fe:
            raise ValueError("fitness-record hash watermark cannot move backwards")
        if len(fitness_record) < endpoint:
            raise ValueError("fitness record does not reach the requested watermark")
        for index in range(self._fitness_record_hashed_fe, endpoint):
            value = float(fitness_record[index])
            if not math.isfinite(value):
                raise ValueError("fitness record must contain only finite values")
            self._fitness_record_hasher.update(repr(value).encode("ascii"))
            self._fitness_record_hasher.update(b"\n")
        self._fitness_record_hashed_fe = endpoint
        return self._fitness_record_hasher.copy().hexdigest()

    def _decision_id(self, sweep_index: int, group_index: int) -> str:
        return _canonical_sha256(
            {
                "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
                "run_id": self.run_id,
                "problem_id": self.problem_id,
                "seed": self.seed,
                "sweep_index": int(sweep_index),
                "group_index": int(group_index),
                "topology_sha256": self.topology_sha256,
            }
        )

    def record_group(
        self,
        *,
        sweep_index: int,
        group_index: int,
        pre_error: float,
        best_error: float,
        primary_requested_fe: int,
        primary_actual_fe: int,
        full_interval_start_fe: int,
        full_interval_end_fe: int,
        pre_block_candidate: Sequence[float],
        final_owner_candidate: Sequence[float],
    ) -> GroupCycleObservation:
        """Seal a group interval after recovery and before relation writeback."""

        if self.failed:
            raise RuntimeError("hypergraph observer is disabled after failure")
        key = (int(sweep_index), int(group_index))
        if key in self._groups:
            raise ValueError("duplicate observer group interval")
        if key[0] in self._complete_sweeps or key[0] in self._incomplete_sweeps:
            raise ValueError("cannot append a group to a closed sweep")
        full_actual = int(full_interval_end_fe) - int(full_interval_start_fe)
        observation = build_group_cycle_observation(
            self.topology,
            sweep_index=key[0],
            group_index=key[1],
            pre_error=pre_error,
            best_error=best_error,
            primary_requested_fe=primary_requested_fe,
            primary_actual_fe=primary_actual_fe,
            full_interval_actual_fe=full_actual,
            full_interval_start_fe=full_interval_start_fe,
            full_interval_end_fe=full_interval_end_fe,
            pre_block_candidate=pre_block_candidate,
            final_owner_candidate=final_owner_candidate,
            capture_stage=FINAL_OWNER_PROPOSAL_WATERMARK,
            capture_fe=full_interval_end_fe,
        )
        observed = _ObservedGroup(
            observation=observation,
            decision_id=self._decision_id(*key),
            pre_error=float(pre_error),
            best_error=float(best_error),
        )
        self._groups[key] = observed
        return observation

    def _resolve_previous_sweep(
        self,
        resolution_snapshot: CompletedSweepSnapshot,
    ) -> None:
        proposal_sweep = resolution_snapshot.sweep_index - 1
        proposal_snapshot = self._complete_sweeps.get(proposal_sweep)
        if proposal_snapshot is None:
            return
        for group_index in self.topology.eligible_group_indices:
            proposal_observation = proposal_snapshot.observation_for_group(group_index)
            current_observation = resolution_snapshot.observation_for_group(group_index)
            credit = build_closed_owner_credit(
                proposal_observation=proposal_observation,
                resolution_snapshot=resolution_snapshot,
            )
            key = (proposal_sweep, group_index)
            self._closed_credits[key] = credit
            proposal = proposal_observation.shared_proposal
            for variable in proposal.variables:
                row = self._proposal_rows[(proposal_sweep, group_index, variable)]
                row["next_sweep_value"] = _format_float(
                    resolution_snapshot.sweep_end_candidate[variable]
                )
                row["next_sweep_end_fe"] = str(resolution_snapshot.sweep_end_fe)
            if key not in self._feature_rows:
                continue
            prior = self._groups[key]
            self._outcomes[key] = {
                "decision_id": prior.decision_id,
                "problem_id": self.problem_id,
                "seed": "" if self.seed is None else str(self.seed),
                "sweep_index": str(proposal_sweep),
                "resolution_sweep_index": str(resolution_snapshot.sweep_index),
                "next_sweep_unit_fe_contribution": _format_float(
                    current_observation.unit_fe_contribution
                ),
                "next_sweep_survival": _format_float(credit.survival),
                "next_sweep_overwrite": _format_float(credit.overwrite),
                "resolution_end_fe": str(resolution_snapshot.sweep_end_fe),
                "all_groups_completed": "1",
                "native_sweep_end_completed": "1",
                "outcome_complete": "1",
                "terminal_censored": "0",
            }

    def _consume_decision_opportunity(self, sweep_index: int) -> None:
        if self._decision_lock_consumed:
            return
        history_sweeps = (sweep_index - 2, sweep_index - 1, sweep_index)
        if any(value not in self._complete_sweeps for value in history_sweeps):
            return
        self._decision_lock_consumed = True
        self._decision_snapshot_sweep = sweep_index
        eligible = self.topology.eligible_group_indices
        if len(eligible) < 2:
            self._decision_status = "inapplicable"
            self._decision_reason = (
                "no_shared_hyperedge"
                if not eligible
                else "insufficient_comparison_hyperedges"
            )
            for group in range(len(self.topology.hyperedges)):
                audit = self._audit_rows[(sweep_index, group)]
                audit["cohort_locked"] = "1"
                audit["state_complete"] = "1"
                audit["unique_focal"] = "0"
                audit["applicable"] = "0"
                audit["not_applicable_reason"] = self._decision_reason
            return

        snapshots = tuple(self._complete_sweeps[value] for value in history_sweeps)
        credits = tuple(
            self._closed_credits[(sweep_index - 1, group)] for group in eligible
        )
        states = build_hyperedge_cycle_states(
            self.topology,
            snapshots,
            closed_owner_credits=credits,
            decision_fe=snapshots[-1].sweep_end_fe,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        scores = score_hyperedge_states(states)
        highest = max(score.focal_priority for score in scores)
        unique_focal = sum(score.focal_priority == highest for score in scores) == 1
        self._decision_status = "applicable" if unique_focal else "inapplicable"
        self._decision_reason = "" if unique_focal else "focal_priority_tie"
        eligible_set = set(eligible)
        for group in range(len(self.topology.hyperedges)):
            audit = self._audit_rows[(sweep_index, group)]
            audit["cohort_locked"] = "1"
            audit["state_complete"] = "1"
            audit["unique_focal"] = str(int(unique_focal))
            if group not in eligible_set:
                audit["applicable"] = "0"
                audit["not_applicable_reason"] = "no_shared_variables"
        for group, state, score in zip(eligible, states, scores, strict=True):
            observed = self._groups[(sweep_index, group)]
            row = self._feature_row(observed.decision_id, state, score)
            feature_payload = {
                field: row[field] for field in HYPERGRAPH_FEATURE_FIELDS[1:]
            }
            self._feature_rows[(sweep_index, group)] = row
            audit = self._audit_rows[(sweep_index, group)]
            audit["applicable"] = str(int(unique_focal))
            audit["not_applicable_reason"] = self._decision_reason
            audit["feature_sha256"] = _canonical_sha256(feature_payload)

    @staticmethod
    def _feature_row(
        decision_id: str,
        state: HyperedgeCycleState,
        score: HyperedgeScore,
    ) -> dict[str, str]:
        return {
            "decision_id": decision_id,
            "current_unit_fe_contribution": _format_float(
                state.current_unit_fe_contribution
            ),
            "ewma_unit_fe_contribution_3": _format_float(
                state.ewma_unit_fe_contribution_3
            ),
            "zero_gain_difficulty": _format_float(state.zero_gain_difficulty),
            "stagnation_ratio_3": _format_float(state.stagnation_ratio_3),
            "direct_owner_proposal_disagreement": _format_float(
                state.direct_owner_proposal_disagreement
            ),
            "prior_next_sweep_overwrite": _format_float(
                state.prior_next_sweep_overwrite
            ),
            "contribution_score": _format_float(score.contribution_score),
            "need_score": _format_float(score.need_score),
            "focal_priority": _format_float(score.focal_priority),
            "owner_reliability": _format_float(score.owner_reliability),
        }

    def complete_sweep(
        self,
        *,
        sweep_index: int,
        optimized_group_count: int,
        all_raw_groups_completed: bool,
        native_sweep_end_completed: bool,
        native_sweep_end_stage: str,
        sweep_end_fe: int,
        sweep_end_candidate: Sequence[float],
        fitness_record: Sequence[float],
    ) -> bool:
        """Close evidence only after every native sweep-end handler has run."""

        sweep = int(sweep_index)
        if self.failed:
            raise RuntimeError("hypergraph observer is disabled after failure")
        if not isinstance(all_raw_groups_completed, bool):
            raise TypeError("all_raw_groups_completed must be boolean")
        if not isinstance(native_sweep_end_completed, bool):
            raise TypeError("native_sweep_end_completed must be boolean")
        if native_sweep_end_stage != HYPERGRAPH_NATIVE_SWEEP_END_STAGE:
            raise ValueError("invalid native sweep-end completion stage")
        if sweep in self._complete_sweeps or sweep in self._incomplete_sweeps:
            raise ValueError("observer sweep was already closed")
        group_count = len(self.topology.hyperedges)
        observed_groups = {
            group for observed_sweep, group in self._groups if observed_sweep == sweep
        }
        observed_all_groups = bool(
            int(optimized_group_count) == group_count
            and observed_groups == set(range(group_count))
        )
        if all_raw_groups_completed is not observed_all_groups:
            raise ValueError("all-raw-groups closure disagrees with observed groups")
        if native_sweep_end_completed is True and not all_raw_groups_completed:
            raise ValueError("native sweep-end closure requires every raw group")
        if not all_raw_groups_completed or not native_sweep_end_completed:
            self._incomplete_sweeps.add(sweep)
            self._materialize_incomplete_audits(
                sweep,
                all_raw_groups_completed=all_raw_groups_completed,
                native_sweep_end_completed=native_sweep_end_completed,
                native_sweep_end_stage=native_sweep_end_stage,
            )
            return False

        endpoint = tuple(float(value) for value in sweep_end_candidate)
        if len(endpoint) <= max(star.variable_index for star in self.topology.stars):
            raise ValueError("sweep endpoint does not cover hypergraph variables")
        if not all(math.isfinite(value) for value in endpoint):
            raise ValueError("sweep endpoint must be finite")
        decision_fe = int(sweep_end_fe)
        prefix_sha256 = self._advance_fitness_record_hash(
            fitness_record,
            decision_fe,
        )
        observations = tuple(
            self._groups[(sweep, group)].observation
            for group in range(group_count)
        )
        snapshot = CompletedSweepSnapshot(
            topology=self.topology,
            sweep_index=sweep,
            observations=observations,
            sweep_end_candidate=endpoint,
            native_sweep_end_completed=True,
            sweep_end_fe=decision_fe,
        )
        self._complete_sweeps[sweep] = snapshot
        self._sweep_end_fe[sweep] = decision_fe

        for group in range(group_count):
            observed = self._groups[(sweep, group)]
            observation = observed.observation
            shared = observation.shared_proposal.variables
            reason = (
                "no_shared_variables"
                if not shared
                else (
                    "decision_lock_already_consumed"
                    if self._decision_lock_consumed
                    else "insufficient_three_sweep_history"
                )
            )
            watermark_valid = bool(
                observation.full_interval_end_fe <= decision_fe
                and observation.shared_proposal.capture_stage
                == FINAL_OWNER_PROPOSAL_WATERMARK
                and observation.shared_proposal.capture_fe
                == observation.full_interval_end_fe
            )
            self._audit_rows[(sweep, group)] = {
                "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
                "decision_id": observed.decision_id,
                "problem_id": self.problem_id,
                "seed": "" if self.seed is None else str(self.seed),
                "sweep_index": str(sweep),
                "group_index": str(group),
                "cohort_locked": "0",
                "state_complete": "0",
                "unique_focal": "0",
                "applicable": "0",
                "not_applicable_reason": reason,
                "source_end_fe": str(decision_fe),
                "decision_fe": str(decision_fe),
                "full_interval_start_fe": str(observation.full_interval_start_fe),
                "full_interval_end_fe": str(observation.full_interval_end_fe),
                "primary_requested_fe": str(observation.primary_requested_fe),
                "primary_actual_fe": str(observation.primary_actual_fe),
                "full_interval_actual_fe": str(observation.full_interval_actual_fe),
                "pre_error": _format_float(observed.pre_error),
                "best_error": _format_float(observed.best_error),
                "successful": str(int(observation.successful)),
                "unit_fe_contribution": _format_float(
                    observation.unit_fe_contribution
                ),
                "feature_sha256": "",
                "topology_sha256": self.topology_sha256,
                "rng_descriptor_sha256": self.rng_descriptor_sha256,
                "fitness_record_sha256": prefix_sha256,
                "proposal_capture_watermark": (
                    observation.shared_proposal.capture_stage
                ),
                "all_raw_groups_completed": "1",
                "native_sweep_end_completed": "1",
                "native_sweep_end_stage": native_sweep_end_stage,
                "watermark_valid": str(int(watermark_valid)),
                "observer_integrity": str(int(watermark_valid)),
            }
            for variable, anchor_value in observation.shared_proposal.anchor_values:
                proposed_value = observation.shared_proposal.proposed_value(variable)
                self._proposal_rows[(sweep, group, variable)] = {
                    "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
                    "decision_id": observed.decision_id,
                    "problem_id": self.problem_id,
                    "seed": "" if self.seed is None else str(self.seed),
                    "sweep_index": str(sweep),
                    "group_index": str(group),
                    "variable_index": str(variable),
                    "capture_watermark": FINAL_OWNER_PROPOSAL_WATERMARK,
                    "proposal_source_end_fe": str(
                        observation.full_interval_end_fe
                    ),
                    "anchor_value": _format_float(anchor_value),
                    "proposed_value": _format_float(proposed_value),
                    "sweep_end_value": _format_float(
                        snapshot.sweep_end_candidate[variable]
                    ),
                    "next_sweep_value": "",
                    "sweep_end_fe": str(decision_fe),
                    "next_sweep_end_fe": "",
                    "topology_sha256": self.topology_sha256,
                    "observer_integrity": str(int(watermark_valid)),
                }

        self._resolve_previous_sweep(snapshot)
        self._consume_decision_opportunity(sweep)
        return True

    def _materialize_incomplete_audits(
        self,
        sweep_index: int,
        *,
        all_raw_groups_completed: bool,
        native_sweep_end_completed: bool,
        native_sweep_end_stage: str,
    ) -> None:
        for (sweep, group), observed in sorted(self._groups.items()):
            if sweep != sweep_index:
                continue
            observation = observed.observation
            watermark_valid = bool(
                observation.shared_proposal.capture_stage
                == FINAL_OWNER_PROPOSAL_WATERMARK
                and observation.shared_proposal.capture_fe
                == observation.full_interval_end_fe
                and observation.full_interval_end_fe
                - observation.full_interval_start_fe
                == observation.full_interval_actual_fe
            )
            self._audit_rows[(sweep, group)] = {
                "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
                "decision_id": observed.decision_id,
                "problem_id": self.problem_id,
                "seed": "" if self.seed is None else str(self.seed),
                "sweep_index": str(sweep),
                "group_index": str(group),
                "cohort_locked": "0",
                "state_complete": "0",
                "unique_focal": "0",
                "applicable": "0",
                "not_applicable_reason": "incomplete_native_sweep",
                "source_end_fe": str(observation.full_interval_end_fe),
                "decision_fe": "",
                "full_interval_start_fe": str(observation.full_interval_start_fe),
                "full_interval_end_fe": str(observation.full_interval_end_fe),
                "primary_requested_fe": str(observation.primary_requested_fe),
                "primary_actual_fe": str(observation.primary_actual_fe),
                "full_interval_actual_fe": str(observation.full_interval_actual_fe),
                "pre_error": _format_float(observed.pre_error),
                "best_error": _format_float(observed.best_error),
                "successful": str(int(observation.successful)),
                "unit_fe_contribution": _format_float(
                    observation.unit_fe_contribution
                ),
                "feature_sha256": "",
                "topology_sha256": self.topology_sha256,
                "rng_descriptor_sha256": self.rng_descriptor_sha256,
                "fitness_record_sha256": "",
                "proposal_capture_watermark": (
                    observation.shared_proposal.capture_stage
                ),
                "all_raw_groups_completed": str(int(all_raw_groups_completed)),
                "native_sweep_end_completed": str(
                    int(native_sweep_end_completed)
                ),
                "native_sweep_end_stage": native_sweep_end_stage,
                "watermark_valid": str(int(watermark_valid)),
                "observer_integrity": str(int(watermark_valid)),
            }
        observed_groups = {
            group
            for observed_sweep, group in self._groups
            if observed_sweep == sweep_index
        }
        for group in range(len(self.topology.hyperedges)):
            if group in observed_groups:
                continue
            # A naturally censored group has no group-level FE watermark. Recording
            # that absence explicitly is valid observer evidence, not a trace failure.
            self._audit_rows[(sweep_index, group)] = {
                "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
                "decision_id": self._decision_id(sweep_index, group),
                "problem_id": self.problem_id,
                "seed": "" if self.seed is None else str(self.seed),
                "sweep_index": str(sweep_index),
                "group_index": str(group),
                "cohort_locked": "0",
                "state_complete": "0",
                "unique_focal": "0",
                "applicable": "0",
                "not_applicable_reason": "incomplete_native_sweep",
                "source_end_fe": "",
                "decision_fe": "",
                "full_interval_start_fe": "",
                "full_interval_end_fe": "",
                "primary_requested_fe": "",
                "primary_actual_fe": "",
                "full_interval_actual_fe": "",
                "pre_error": "",
                "best_error": "",
                "successful": "",
                "unit_fe_contribution": "",
                "feature_sha256": "",
                "topology_sha256": self.topology_sha256,
                "rng_descriptor_sha256": self.rng_descriptor_sha256,
                "fitness_record_sha256": "",
                "proposal_capture_watermark": "",
                "all_raw_groups_completed": str(int(all_raw_groups_completed)),
                "native_sweep_end_completed": str(
                    int(native_sweep_end_completed)
                ),
                "native_sweep_end_stage": native_sweep_end_stage,
                "watermark_valid": "0",
                "observer_integrity": "1",
            }
        if (
            not self._decision_lock_consumed
            and sweep_index - 2 in self._complete_sweeps
            and sweep_index - 1 in self._complete_sweeps
        ):
            self._decision_lock_consumed = True
            self._decision_snapshot_sweep = sweep_index
            self._decision_status = "inapplicable"
            self._decision_reason = "incomplete_native_sweep"
            for (row_sweep, _), audit in self._audit_rows.items():
                if row_sweep == sweep_index:
                    audit["cohort_locked"] = "1"

    def _outcome_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        pending = tuple(
            key for key in self._feature_rows if key not in self._outcomes
        )
        if pending:
            terminal_sweep = max(self._complete_sweeps, default=-1)
            if any(sweep != terminal_sweep for sweep, _ in pending):
                raise ValueError(
                    "only the final complete state may be terminal-censored"
                )
        for key, feature in sorted(self._feature_rows.items()):
            if key in self._outcomes:
                rows.append(self._outcomes[key])
                continue
            sweep, _ = key
            rows.append(
                {
                    "decision_id": feature["decision_id"],
                    "problem_id": self.problem_id,
                    "seed": "" if self.seed is None else str(self.seed),
                    "sweep_index": str(sweep),
                    "resolution_sweep_index": "",
                    "next_sweep_unit_fe_contribution": "",
                    "next_sweep_survival": "",
                    "next_sweep_overwrite": "",
                    "resolution_end_fe": "",
                    "all_groups_completed": "0",
                    "native_sweep_end_completed": "0",
                    "outcome_complete": "0",
                    "terminal_censored": "1",
                }
            )
        return rows

    def write_artifacts(
        self,
        *,
        paths: HypergraphTraceArtifactPaths,
        final_fitness_record: Sequence[float],
    ) -> dict[str, object]:
        """Write independent observer artifacts; never touch action_trace.csv."""

        feature_rows = [row for _, row in sorted(self._feature_rows.items())]
        audit_rows = [row for _, row in sorted(self._audit_rows.items())]
        proposal_rows = [row for _, row in sorted(self._proposal_rows.items())]
        outcome_rows = self._outcome_rows()
        _write_csv(paths.features, HYPERGRAPH_FEATURE_FIELDS, feature_rows)
        _write_csv(paths.audit, HYPERGRAPH_AUDIT_FIELDS, audit_rows)
        _write_csv(paths.proposals, HYPERGRAPH_PROPOSAL_FIELDS, proposal_rows)
        _write_csv(paths.outcomes, HYPERGRAPH_OUTCOME_FIELDS, outcome_rows)

        artifact_hashes = {
            paths.features.name: _file_sha256(paths.features),
            paths.audit.name: _file_sha256(paths.audit),
            paths.proposals.name: _file_sha256(paths.proposals),
            paths.outcomes.name: _file_sha256(paths.outcomes),
        }
        terminal_observed_fe = len(final_fitness_record)
        terminal_completion_valid = (
            self.terminal_target_fe - self.terminal_completion_tolerance_fe
            <= terminal_observed_fe
            <= self.terminal_target_fe
        )
        integrity = bool(
            all(row["observer_integrity"] == "1" for row in audit_rows)
            and all(row["observer_integrity"] == "1" for row in proposal_rows)
            and terminal_completion_valid
        )
        if outcome_rows and all(
            row["outcome_complete"] == "1" for row in outcome_rows
        ):
            label_closure = "closed"
        elif outcome_rows and all(
            row["terminal_censored"] == "1" for row in outcome_rows
        ):
            label_closure = "terminal_censored"
        elif self._decision_lock_consumed and not feature_rows:
            label_closure = "not_applicable"
        else:
            label_closure = "not_reached"
        manifest: dict[str, object] = {
            "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
            "schema_version": 1,
            "hypergraph_trace_mode": "observer",
            "fresh_optimizer_execution": int(self.fresh_optimizer_execution),
            "problem_id": self.problem_id,
            "seed": self.seed,
            "run_id": self.run_id,
            "terminal_target_fe": self.terminal_target_fe,
            "terminal_observed_fe": terminal_observed_fe,
            "terminal_completion_tolerance_fe": (
                self.terminal_completion_tolerance_fe
            ),
            "topology_source": "raw_grouping_result_direct_no_transitive_closure",
            "transitive_closure_used": 0,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "raw_hyperedges": [list(group) for group in self.topology.hyperedges],
            "variable_owner_groups": [
                [variable, list(owners)]
                for variable, owners in self.topology.variable_owner_groups
            ],
            "raw_group_count": len(self.topology.hyperedges),
            "eligible_hyperedge_count": len(self.topology.eligible_group_indices),
            "shared_variable_count": len(self.topology.overlap_variables),
            "complete_sweep_count": len(self._complete_sweeps),
            "incomplete_sweep_count": len(self._incomplete_sweeps),
            "group_observation_count": len(self._groups),
            "feature_row_count": len(feature_rows),
            "audit_row_count": len(audit_rows),
            "shared_proposal_row_count": len(proposal_rows),
            "outcome_row_count": len(outcome_rows),
            "complete_outcome_count": sum(
                row["outcome_complete"] == "1" for row in outcome_rows
            ),
            "terminal_censored_outcome_count": sum(
                row["terminal_censored"] == "1" for row in outcome_rows
            ),
            "decision_lock_consumed": int(self._decision_lock_consumed),
            "decision_snapshot_sweep": self._decision_snapshot_sweep,
            "decision_status": self._decision_status,
            "decision_reason": self._decision_reason,
            "decision_feature_row_count": len(feature_rows),
            "cohort_locked_sweep": self._decision_snapshot_sweep,
            "label_closure": label_closure,
            "observer_objective_calls": 0,
            "observer_rng_calls": 0,
            "observer_optimizer_calls": 0,
            "observer_fe": 0,
            "topology_sha256": self.topology_sha256,
            "rng_descriptor_sha256": self.rng_descriptor_sha256,
            "protocol_config_sha256": _file_sha256(self.protocol_config_path),
            "protocol_spec_sha256": _file_sha256(self.protocol_spec_path),
            "runner_source_sha256": _file_sha256(self.runner_source_path),
            "fitness_record_sha256": self._advance_fitness_record_hash(
                final_fitness_record,
                len(final_fitness_record),
            ),
            "artifact_sha256": artifact_hashes,
            "observer_status": "failed" if self.failed else "complete",
            "observer_error_stage": (
                "" if self._failure is None else self._failure["observer_error_stage"]
            ),
            "observer_error_type": (
                "" if self._failure is None else self._failure["observer_error_type"]
            ),
            "observer_error_message": (
                "" if self._failure is None else self._failure["observer_error_message"]
            ),
            "observer_error_source_fe": (
                "" if self._failure is None else self._failure["observer_error_source_fe"]
            ),
            "observer_integrity": int(integrity and not self.failed),
        }
        paths.manifest.parent.mkdir(parents=True, exist_ok=True)
        paths.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest

    def write_failure_manifest(
        self,
        *,
        path: Path,
        final_fitness_record: Sequence[float],
    ) -> dict[str, object]:
        """Best-effort minimal manifest used when normal artifact writing fails."""

        if not self.failed:
            raise ValueError("failure manifest requires a recorded observer failure")
        try:
            fitness_sha256 = self._advance_fitness_record_hash(
                final_fitness_record,
                len(final_fitness_record),
            )
        except Exception:
            fitness_sha256 = ""
        manifest: dict[str, object] = {
            "protocol_version": HYPERGRAPH_SCHEMA_VERSION,
            "schema_version": 1,
            "hypergraph_trace_mode": "observer",
            "fresh_optimizer_execution": int(self.fresh_optimizer_execution),
            "problem_id": self.problem_id,
            "seed": self.seed,
            "run_id": self.run_id,
            "terminal_target_fe": self.terminal_target_fe,
            "terminal_observed_fe": len(final_fitness_record),
            "terminal_completion_tolerance_fe": (
                self.terminal_completion_tolerance_fe
            ),
            "topology_source": "raw_grouping_result_direct_no_transitive_closure",
            "transitive_closure_used": 0,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "raw_hyperedges": [list(group) for group in self.topology.hyperedges],
            "variable_owner_groups": [
                [variable, list(owners)]
                for variable, owners in self.topology.variable_owner_groups
            ],
            "topology_sha256": self.topology_sha256,
            "rng_descriptor_sha256": self.rng_descriptor_sha256,
            "fitness_record_sha256": fitness_sha256,
            "artifact_sha256": {},
            "observer_status": "failed",
            **self._failure,
            "observer_integrity": 0,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest


__all__ = [
    "HYPERGRAPH_AUDIT_FIELDS",
    "HYPERGRAPH_FEATURE_FIELDS",
    "HYPERGRAPH_NATIVE_SWEEP_END_STAGE",
    "HYPERGRAPH_OUTCOME_FIELDS",
    "HYPERGRAPH_PROPOSAL_FIELDS",
    "HYPERGRAPH_TRACE_MODES",
    "HypergraphTraceArtifactPaths",
    "HypergraphTraceObserver",
    "write_hypergraph_initialization_failure_manifest",
]
