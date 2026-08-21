"""Scale-normalized replay of the lagged coupling EMA shadow (G1).

Pre-registration (frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md
before this gate ran):

- three EMA input variants share the identical replay ordering, scope
  streams, alpha=0.5, and prior-only update contract of
  ``oc_lagged_coupling_shadow``:
    raw              coupled_gain unchanged (reproduces the v1 shadow)
    error_normalized coupled_gain / checkpoint_error
    rank             within-context average rank of coupled_gain in [0, 1]
- all variants are judged on the *raw* action/end-to-end gains; only the
  EMA input is normalized;
- PRIMARY variant = error_normalized, PRIMARY endpoint = end-to-end
  ticket hit rate.  The original negative conclusion is revoked only if
  the primary endpoint's bootstrap 95% CI lower bound exceeds 0.20
  (5-arm chance).  rank and raw are secondary/reference and cannot
  trigger revocation alone (multiplicity guard);
- ``raw`` must reproduce the v1 shadow hit rates exactly, otherwise the
  gate fails.

The replay is offline: no candidate evaluation, no production selector
access, no new search FE.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.oc_lagged_coupling_shadow import (
    ARMS,
    EMA_ALPHA,
    MODE_ORDER,
    _finite_number,
    _oracle_arm,
    _ordered_contexts,
    _scope_key,
    _scope_stream,
    validate_input,
)

INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
OUTPUT_SCHEMA = "arac-oc-lagged-coupling-normalized-v1"
V1_REFERENCE_ARTIFACT = Path("artifacts/oc_lagged_coupling_shadow/confirmation.json")
VARIANTS = ("raw", "error_normalized", "rank")
PRIMARY_VARIANT = "error_normalized"
PRIMARY_ENDPOINT = "end_to_end_ticket_hit_rate"
CHANCE_HIT_RATE = 1.0 / len(ARMS)
MIN_ELIGIBLE_ROWS = 10
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260820
REVOKE_RULE = "primary variant end-to-end hit-rate bootstrap 95% CI lower bound > 0.20"


@dataclass(frozen=True)
class VariantBlock:
    predicted_arm: str | None
    prediction_source: str
    action_hit: bool | None
    end_to_end_hit: bool | None
    action_regret: float | None
    end_to_end_regret: float | None
    prior_ema: tuple[tuple[str, float | None], ...]
    observed_input: tuple[tuple[str, float], ...]
    updated_ema: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class NormalizedReplayRow:
    context_index: int
    topology: str
    overlap_budget: int
    seed: int
    mode: str
    scope: tuple[int, ...]
    scope_stream: str
    source_order: int
    checkpoint_error: float
    oracle_action_arm: str
    oracle_end_to_end_arm: str
    oracle_action_gain: float
    oracle_end_to_end_gain: float
    variants: dict[str, VariantBlock]
    ticket_fes: int


def _error_scale(context: dict[str, Any]) -> float:
    error = _finite_number(context.get("checkpoint_error"), "checkpoint_error")
    if error <= 0.0:
        raise ValueError("checkpoint_error must be positive for error normalization")
    return float(error)


def _within_context_rank(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(ARMS, key=lambda arm: (values[arm], -ARMS.index(arm)))
    total = len(ARMS)
    ranks: dict[str, float] = {}
    index = 0
    while index < total:
        end = index
        while end + 1 < total and values[ordered[end + 1]] == values[ordered[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[ordered[position]] = (average_rank - 1.0) / (total - 1.0)
        index = end + 1
    return ranks


def _variant_inputs(context: dict[str, Any]) -> dict[str, dict[str, float]]:
    coupled = {
        arm: _finite_number(context[arm]["coupled_gain"], f"{arm}.coupled_gain")
        for arm in ARMS
    }
    scale = _error_scale(context)
    ranks = _within_context_rank(coupled)
    return {
        "raw": dict(coupled),
        "error_normalized": {arm: value / scale for arm, value in coupled.items()},
        "rank": ranks,
    }


def _replay(payload: dict[str, Any]) -> tuple[NormalizedReplayRow, ...]:
    states: dict[str, dict[tuple[str, int, tuple[int, ...]], dict[str, float | None]]] = {
        variant: {} for variant in VARIANTS
    }
    rows: list[NormalizedReplayRow] = []
    for source_order, (context_index, context) in enumerate(_ordered_contexts(payload)):
        key = _scope_key(context)
        action_gain = {
            arm: _finite_number(context[arm]["action_gain"], f"{arm}.action_gain")
            for arm in ARMS
        }
        end_gain = {
            arm: _finite_number(context[arm]["end_to_end_gain"], f"{arm}.end_to_end_gain")
            for arm in ARMS
        }
        action_oracle = _oracle_arm(context, "action_gain")
        end_oracle = _oracle_arm(context, "end_to_end_gain")
        inputs = _variant_inputs(context)
        blocks: dict[str, VariantBlock] = {}
        for variant in VARIANTS:
            state = states[variant].setdefault(key, {arm: None for arm in ARMS})
            prior = tuple((arm, state[arm]) for arm in ARMS)
            available = tuple(arm for arm in ARMS if state[arm] is not None)
            predicted = (
                max(available, key=lambda arm: (float(state[arm]), -ARMS.index(arm)))
                if available
                else None
            )
            observed = tuple((arm, inputs[variant][arm]) for arm in ARMS)
            updated: list[tuple[str, float]] = []
            for arm in ARMS:
                observed_value = inputs[variant][arm]
                previous = state[arm]
                next_value = (
                    observed_value
                    if previous is None
                    else EMA_ALPHA * observed_value + (1.0 - EMA_ALPHA) * previous
                )
                state[arm] = float(next_value)
                updated.append((arm, float(next_value)))
            if predicted is None:
                block = VariantBlock(
                    predicted_arm=None,
                    prediction_source="cold_start",
                    action_hit=None,
                    end_to_end_hit=None,
                    action_regret=None,
                    end_to_end_regret=None,
                    prior_ema=prior,
                    observed_input=observed,
                    updated_ema=tuple(updated),
                )
            else:
                block = VariantBlock(
                    predicted_arm=predicted,
                    prediction_source="prior_ema",
                    action_hit=predicted == action_oracle,
                    end_to_end_hit=predicted == end_oracle,
                    action_regret=action_gain[action_oracle] - action_gain[predicted],
                    end_to_end_regret=end_gain[end_oracle] - end_gain[predicted],
                    prior_ema=prior,
                    observed_input=observed,
                    updated_ema=tuple(updated),
                )
            blocks[variant] = block
        rows.append(
            NormalizedReplayRow(
                context_index=context_index,
                topology=str(context["topology"]),
                overlap_budget=int(context["overlap_budget"]),
                seed=int(context["seed"]),
                mode=str(context["mode"]),
                scope=key[2],
                scope_stream=_scope_stream(key),
                source_order=source_order,
                checkpoint_error=_error_scale(context),
                oracle_action_arm=action_oracle,
                oracle_end_to_end_arm=end_oracle,
                oracle_action_gain=action_gain[action_oracle],
                oracle_end_to_end_gain=end_gain[end_oracle],
                variants=blocks,
                ticket_fes=int(payload["protocol"]["action_fes"]),
            )
        )
    return tuple(rows)


def _hits(rows: tuple[NormalizedReplayRow, ...], variant: str, endpoint: str) -> list[bool]:
    return [
        bool(getattr(row.variants[variant], endpoint))
        for row in rows
        if row.variants[variant].predicted_arm is not None
    ]


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _bootstrap_ci(hit_values: list[bool]) -> tuple[float, float]:
    if not hit_values:
        return (0.0, 0.0)
    sample = np.asarray([float(value) for value in hit_values], dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, sample.size, size=(BOOTSTRAP_DRAWS, sample.size))
    means = sample[indices].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _kappa(rows: tuple[NormalizedReplayRow, ...], variant: str, endpoint: str) -> float:
    predicted: list[str] = []
    oracle: list[str] = []
    oracle_field = (
        "oracle_action_arm" if endpoint == "action_hit" else "oracle_end_to_end_arm"
    )
    for row in rows:
        block = row.variants[variant]
        if block.predicted_arm is None:
            continue
        predicted.append(block.predicted_arm)
        oracle.append(str(getattr(row, oracle_field)))
    if not predicted:
        return 0.0
    total = len(predicted)
    observed = sum(1 for a, b in zip(predicted, oracle) if a == b) / total
    if observed >= 1.0:
        return 1.0
    marginal_pred = {arm: predicted.count(arm) / total for arm in ARMS}
    marginal_oracle = {arm: oracle.count(arm) / total for arm in ARMS}
    expected = sum(marginal_pred[arm] * marginal_oracle[arm] for arm in ARMS)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def _variant_metrics(rows: tuple[NormalizedReplayRow, ...], variant: str) -> dict[str, Any]:
    action_hits = _hits(rows, variant, "action_hit")
    end_hits = _hits(rows, variant, "end_to_end_hit")
    action_regret = [
        float(row.variants[variant].action_regret)
        for row in rows
        if row.variants[variant].action_regret is not None
    ]
    end_regret = [
        float(row.variants[variant].end_to_end_regret)
        for row in rows
        if row.variants[variant].end_to_end_regret is not None
    ]
    action_ci = _bootstrap_ci(action_hits)
    end_ci = _bootstrap_ci(end_hits)
    return {
        "eligible_rows": len(action_hits),
        "cold_start_rows": sum(
            row.variants[variant].predicted_arm is None for row in rows
        ),
        "action_ticket_hit_rate": _mean([float(v) for v in action_hits]),
        "action_hit_rate_ci95": [action_ci[0], action_ci[1]],
        "end_to_end_ticket_hit_rate": _mean([float(v) for v in end_hits]),
        "end_to_end_hit_rate_ci95": [end_ci[0], end_ci[1]],
        "action_mean_regret": _mean(action_regret),
        "end_to_end_mean_regret": _mean(end_regret),
        "action_cohens_kappa": _kappa(rows, variant, "action_hit"),
        "end_to_end_cohens_kappa": _kappa(rows, variant, "end_to_end_hit"),
    }


def _raw_reproduces_v1(rows: tuple[NormalizedReplayRow, ...]) -> bool:
    if not V1_REFERENCE_ARTIFACT.exists():
        return False
    reference = json.loads(V1_REFERENCE_ARTIFACT.read_text(encoding="utf-8"))
    metrics = _variant_metrics(rows, "raw")
    return (
        abs(metrics["action_ticket_hit_rate"] - float(reference["summary"]["action_ticket_hit_rate"])) < 1e-12
        and abs(
            metrics["end_to_end_ticket_hit_rate"]
            - float(reference["summary"]["end_to_end_ticket_hit_rate"])
        )
        < 1e-12
        and metrics["eligible_rows"] == int(reference["summary"]["eligible_rows"])
    )


def _prior_only_contract(rows: tuple[NormalizedReplayRow, ...]) -> bool:
    for variant in VARIANTS:
        previous: dict[str, dict[str, float | None]] = {}
        for row in rows:
            block = row.variants[variant]
            state = previous.setdefault(row.scope_stream, {arm: None for arm in ARMS})
            available = tuple(arm for arm in ARMS if state[arm] is not None)
            expected = (
                max(available, key=lambda arm: (float(state[arm]), -ARMS.index(arm)))
                if available
                else None
            )
            if block.predicted_arm != expected:
                return False
            prior = dict(block.prior_ema)
            observed = dict(block.observed_input)
            updated = dict(block.updated_ema)
            if any(prior[arm] != state[arm] for arm in ARMS):
                return False
            for arm in ARMS:
                expected_value = (
                    observed[arm]
                    if state[arm] is None
                    else EMA_ALPHA * observed[arm] + (1.0 - EMA_ALPHA) * state[arm]
                )
                if not np.isclose(updated[arm], expected_value, rtol=0.0, atol=1e-12):
                    return False
                state[arm] = updated[arm]
    return True


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    rows = _replay(payload)
    deterministic = rows == _replay(payload)
    metrics = {variant: _variant_metrics(rows, variant) for variant in VARIANTS}
    primary = metrics[PRIMARY_VARIANT]
    eligible_enough = primary["eligible_rows"] >= MIN_ELIGIBLE_ROWS
    primary_ci_low = float(primary["end_to_end_hit_rate_ci95"][0])
    scale_confound_revoke = bool(eligible_enough and primary_ci_low > CHANCE_HIT_RATE)
    checks = {
        "input_contracts_valid": True,
        "context_count_60": len(rows) == 60,
        "replay_deterministic": deterministic,
        "prior_only_prediction_and_update": _prior_only_contract(rows),
        "ema_values_finite": all(
            np.isfinite(value)
            for row in rows
            for variant in VARIANTS
            for _, value in row.variants[variant].updated_ema
        ),
        "one_fe_receipts_preserved": all(
            context["coupling_receipt_parity"] is True for context in payload["contexts"]
        ),
        "production_selector_unchanged": payload["protocol"]["production_selector_modified"] is False,
        "raw_reproduces_v1_shadow": _raw_reproduces_v1(rows),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "variants": VARIANTS,
            "primary_variant": PRIMARY_VARIANT,
            "primary_endpoint": PRIMARY_ENDPOINT,
            "chance_hit_rate": CHANCE_HIT_RATE,
            "ema_alpha": EMA_ALPHA,
            "scope_key": "topology|overlap_budget|selected_component",
            "ordering": "scope, seed, conforming-before-conflicting, mode, source index",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "min_eligible_rows": MIN_ELIGIBLE_ROWS,
            "revoke_rule": REVOKE_RULE,
            "v1_reference_artifact": str(V1_REFERENCE_ARTIFACT),
            "production_selector_modified": False,
        },
        "variant_metrics": metrics,
        "summary": {
            "primary_endpoint_value": float(primary[PRIMARY_ENDPOINT]),
            "primary_endpoint_ci95": primary["end_to_end_hit_rate_ci95"],
            "eligible_enough": eligible_enough,
            "scale_confound_revoke": scale_confound_revoke,
            "lagged_negative_confirmed": not scale_confound_revoke,
            "interpretation": (
                "primary variant (error-normalized) end-to-end CI lower bound "
                f"{primary_ci_low:.4f} vs chance {CHANCE_HIT_RATE:.2f}; "
                + (
                    "original negative conclusion REVOKED (scale confound confirmed)"
                    if scale_confound_revoke
                    else "original negative conclusion CONFIRMED after scale normalization"
                )
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
        default=Path("artifacts/oc_lagged_coupling_normalized_gate/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_passed": result["gate_passed"], "summary": result["summary"], "variant_metrics": result["variant_metrics"]}, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
