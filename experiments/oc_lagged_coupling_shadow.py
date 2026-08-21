"""Replay action-conditioned coupling as a strictly lagged component EMA.

The replay is deliberately offline.  It consumes the v3 semantic-gate artifact
and never evaluates a candidate or calls the production selector.  A coupling
value from one context can only affect the next context in the same structural
scope stream.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
OUTPUT_SCHEMA = "arac-oc-lagged-coupling-shadow-v1"
ARMS = (
    "owner_control",
    "shared_core",
    "expanded_shared_private",
    "duplicated_shared_competition",
    "duplicated_shared_local_competition",
)
MODE_ORDER = {"conforming": 0, "conflicting": 1}
EMA_ALPHA = 0.5
AUTHORITY_THRESHOLD = 0.30
MIN_ELIGIBLE_ROWS = 10


@dataclass(frozen=True)
class ReplayRow:
    context_index: int
    topology: str
    overlap_budget: int
    seed: int
    mode: str
    scope: tuple[int, ...]
    scope_stream: str
    source_order: int
    predicted_arm: str | None
    prediction_source: str
    oracle_action_arm: str
    oracle_end_to_end_arm: str
    action_hit: bool | None
    end_to_end_hit: bool | None
    predicted_action_gain: float | None
    oracle_action_gain: float
    action_regret: float | None
    predicted_end_to_end_gain: float | None
    oracle_end_to_end_gain: float
    end_to_end_regret: float | None
    prior_ema: tuple[tuple[str, float | None], ...]
    observed_coupled_gain: tuple[tuple[str, float], ...]
    updated_ema: tuple[tuple[str, float], ...]
    ticket_fes: int


def _scope_key(context: dict[str, Any]) -> tuple[str, int, tuple[int, ...]]:
    topology = context.get("topology")
    budget = context.get("overlap_budget")
    scope = context.get("selected_component")
    if not isinstance(topology, str) or not topology:
        raise ValueError("context topology is invalid")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("context overlap_budget is invalid")
    if not isinstance(scope, list) or not scope:
        raise ValueError("context selected_component is invalid")
    component = tuple(scope)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in component):
        raise ValueError("context selected_component contains an invalid variable")
    if component != tuple(sorted(set(component))):
        raise ValueError("context selected_component must be sorted and unique")
    return topology, budget, component


def _scope_stream(key: tuple[str, int, tuple[int, ...]]) -> str:
    topology, budget, component = key
    return f"{topology}|budget={budget}|component={','.join(str(v) for v in component)}"


def _finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_input(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("input artifact schema drifted")
    contexts = payload.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 60:
        raise ValueError("input artifact must contain exactly 60 contexts")
    if payload.get("context_count") != len(contexts):
        raise ValueError("input context_count drifted")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("production_selector_modified") is not False:
        raise ValueError("production selector modification flag is not false")
    for context in contexts:
        if not isinstance(context, dict):
            raise ValueError("context row must be an object")
        _scope_key(context)
        if context.get("strict_best") is not True:
            raise ValueError("context strict-best contract failed")
        if context.get("promotion_applied") is not False:
            raise ValueError("context promotion flag is not false")
        if context.get("coupling_receipt_parity") is not True:
            raise ValueError("context coupling receipt parity failed")
        for arm in ARMS:
            record = context.get(arm)
            if not isinstance(record, dict):
                raise ValueError(f"missing arm record: {arm}")
            if record.get("coupling_fes") != 1 or record.get("coupling_archive_preserved") is not True:
                raise ValueError(f"{arm} coupling receipt contract failed")
            for field in ("coupled_gain", "action_gain", "end_to_end_gain"):
                _finite_number(record.get(field), f"{arm}.{field}")


def _oracle_arm(context: dict[str, Any], field: str) -> str:
    return max(
        ARMS,
        key=lambda arm: (_finite_number(context[arm][field], f"{arm}.{field}"), -ARMS.index(arm)),
    )


def _ordered_contexts(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(payload["contexts"]))
    indexed.sort(
        key=lambda item: (
            *_scope_key(item[1]),
            int(item[1]["seed"]),
            MODE_ORDER.get(item[1]["mode"], 99),
            str(item[1]["mode"]),
            item[0],
        )
    )
    return indexed


def _replay(payload: dict[str, Any]) -> tuple[ReplayRow, ...]:
    states: dict[tuple[str, int, tuple[int, ...]], dict[str, float | None]] = {}
    rows: list[ReplayRow] = []
    for source_order, (context_index, context) in enumerate(_ordered_contexts(payload)):
        key = _scope_key(context)
        state = states.setdefault(key, {arm: None for arm in ARMS})
        prior = tuple((arm, state[arm]) for arm in ARMS)
        available = tuple(arm for arm in ARMS if state[arm] is not None)
        predicted = (
            max(available, key=lambda arm: (float(state[arm]), -ARMS.index(arm)))
            if available
            else None
        )
        action_oracle = _oracle_arm(context, "action_gain")
        end_oracle = _oracle_arm(context, "end_to_end_gain")
        action_gain = {
            arm: _finite_number(context[arm]["action_gain"], f"{arm}.action_gain")
            for arm in ARMS
        }
        end_gain = {
            arm: _finite_number(context[arm]["end_to_end_gain"], f"{arm}.end_to_end_gain")
            for arm in ARMS
        }
        if predicted is None:
            prediction_source = "cold_start"
            action_hit = None
            end_hit = None
            predicted_action_gain = None
            predicted_end_gain = None
            action_regret = None
            end_regret = None
        else:
            prediction_source = "prior_ema"
            action_hit = predicted == action_oracle
            end_hit = predicted == end_oracle
            predicted_action_gain = action_gain[predicted]
            predicted_end_gain = end_gain[predicted]
            action_regret = action_gain[action_oracle] - predicted_action_gain
            end_regret = end_gain[end_oracle] - predicted_end_gain
        observed_values: list[tuple[str, float]] = []
        updated: list[tuple[str, float]] = []
        for arm in ARMS:
            observed = _finite_number(context[arm]["coupled_gain"], f"{arm}.coupled_gain")
            observed_values.append((arm, observed))
            previous = state[arm]
            next_value = observed if previous is None else EMA_ALPHA * observed + (1.0 - EMA_ALPHA) * previous
            state[arm] = float(next_value)
            updated.append((arm, float(next_value)))
        rows.append(
            ReplayRow(
                context_index=context_index,
                topology=str(context["topology"]),
                overlap_budget=int(context["overlap_budget"]),
                seed=int(context["seed"]),
                mode=str(context["mode"]),
                scope=key[2],
                scope_stream=_scope_stream(key),
                source_order=source_order,
                predicted_arm=predicted,
                prediction_source=prediction_source,
                oracle_action_arm=action_oracle,
                oracle_end_to_end_arm=end_oracle,
                action_hit=action_hit,
                end_to_end_hit=end_hit,
                predicted_action_gain=predicted_action_gain,
                oracle_action_gain=action_gain[action_oracle],
                action_regret=action_regret,
                predicted_end_to_end_gain=predicted_end_gain,
                oracle_end_to_end_gain=end_gain[end_oracle],
                end_to_end_regret=end_regret,
                prior_ema=prior,
                observed_coupled_gain=tuple(observed_values),
                updated_ema=tuple(updated),
                ticket_fes=int(payload["protocol"]["action_fes"]),
            )
        )
    return tuple(rows)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _metrics(rows: tuple[ReplayRow, ...]) -> dict[str, Any]:
    eligible = tuple(row for row in rows if row.predicted_arm is not None)
    action_hits = [bool(row.action_hit) for row in eligible]
    end_hits = [bool(row.end_to_end_hit) for row in eligible]
    action_regret = [float(row.action_regret) for row in eligible if row.action_regret is not None]
    end_regret = [float(row.end_to_end_regret) for row in eligible if row.end_to_end_regret is not None]
    stream_counts: dict[str, int] = {}
    for row in rows:
        stream_counts[row.scope_stream] = stream_counts.get(row.scope_stream, 0) + 1
    return {
        "context_rows": len(rows),
        "eligible_rows": len(eligible),
        "cold_start_rows": sum(row.predicted_arm is None for row in rows),
        "scope_stream_count": len(stream_counts),
        "scope_stream_counts": dict(sorted(stream_counts.items())),
        "action_ticket_hit_rate": _mean([float(value) for value in action_hits]),
        "end_to_end_ticket_hit_rate": _mean([float(value) for value in end_hits]),
        "action_mean_regret": _mean(action_regret),
        "end_to_end_mean_regret": _mean(end_regret),
        "chance_hit_rate": 1.0 / len(ARMS),
    }


def _replay_digest(rows: tuple[ReplayRow, ...]) -> str:
    encoded = json.dumps([asdict(row) for row in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _replay_contract_holds(rows: tuple[ReplayRow, ...]) -> bool:
    previous_by_stream: dict[str, dict[str, float | None]] = {}
    for row in rows:
        previous = previous_by_stream.setdefault(row.scope_stream, {arm: None for arm in ARMS})
        available = tuple(arm for arm in ARMS if previous[arm] is not None)
        expected_prediction = (
            max(available, key=lambda arm: (float(previous[arm]), -ARMS.index(arm)))
            if available
            else None
        )
        if row.predicted_arm != expected_prediction:
            return False
        prior = dict(row.prior_ema)
        observed = dict(row.observed_coupled_gain)
        updated = dict(row.updated_ema)
        if any(prior[arm] != previous[arm] for arm in ARMS):
            return False
        for arm in ARMS:
            expected = observed[arm] if previous[arm] is None else EMA_ALPHA * observed[arm] + (1.0 - EMA_ALPHA) * previous[arm]
            if not np.isclose(updated[arm], expected, rtol=0.0, atol=1e-12):
                return False
            previous[arm] = updated[arm]
    return True


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    rows = _replay(payload)
    second_rows = _replay(payload)
    metrics = _metrics(rows)
    replay_equal = rows == second_rows
    eligible = int(metrics["eligible_rows"])
    action_hit = float(metrics["action_ticket_hit_rate"])
    end_hit = float(metrics["end_to_end_ticket_hit_rate"])
    promotion = bool(
        eligible >= MIN_ELIGIBLE_ROWS
        and action_hit >= AUTHORITY_THRESHOLD
        and end_hit >= AUTHORITY_THRESHOLD
        and float(metrics["action_mean_regret"]) <= 0.0
        and float(metrics["end_to_end_mean_regret"]) <= 0.0
    )
    checks = {
        "input_contracts_valid": True,
        "context_count_60": len(rows) == 60,
        "replay_rows_complete": len(rows) == len(payload["contexts"]),
        "replay_deterministic": replay_equal,
        "prior_only_prediction_and_update": _replay_contract_holds(rows),
        "ema_values_finite": all(
            np.isfinite(value)
            for row in rows
            for _, value in row.updated_ema
        ),
        "one_fe_receipts_preserved": all(
            context["coupling_receipt_parity"] is True for context in payload["contexts"]
        ),
        "production_selector_unchanged": payload["protocol"]["production_selector_modified"] is False,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "contexts": len(rows),
            "arms": ARMS,
            "scope_key": "topology|overlap_budget|selected_component",
            "ordering": "scope, seed, conforming-before-conflicting, mode, source index",
            "ema_alpha": EMA_ALPHA,
            "ticket_fes": int(payload["protocol"]["action_fes"]),
            "coupling_fes_already_charged_per_arm": int(payload["protocol"]["coupling_fes"]),
            "min_eligible_rows": MIN_ELIGIBLE_ROWS,
            "authority_threshold": AUTHORITY_THRESHOLD,
            "production_selector_modified": False,
        },
        "replay_digest": _replay_digest(rows),
        "rows": [asdict(row) for row in rows],
        "summary": {
            **metrics,
            "promotion_recommended": promotion,
            "interpretation": (
                "lagged shadow only; prior EMA selects an offline one-ticket arm, "
                "then current G_coupled updates state; production selector unchanged"
            ),
        },
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oc_lagged_coupling_shadow/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": result["gate_passed"], "summary": result["summary"], "gate_checks": result["gate_checks"]}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
