"""Action-level oracle-gap quantification (G3a).

Pre-registration (frozen in .codex-tasks/arac-oc-evidence-closure/EPIC.md):

- input is the frozen five-arm matched artifact
  ``artifacts/oc_action_semantic_gate_v3/confirmation_fresh.json``;
- oracle arm per context = argmax raw gain (action and end-to-end reported
  separately);
- best fixed arm = argmax over arms of the mean raw gain across all 60
  contexts (tie broken by arm order);
- per-context best-fixed regret = oracle gain - best-fixed-arm gain,
  normalized by that context's checkpoint_error;
- classification (end-to-end is primary): with r = mean normalized regret
  and CI its percentile bootstrap 95% interval,
    MATERIAL          CI lower bound > 0 and r >= 0.05
    PRESENT_BUT_SMALL CI lower bound > 0 and r < 0.05
    NOT_MATERIAL      CI lower bound <= 0
- oracle-arm identity flip rate is measured between consecutive contexts
  of the same scope stream under the lagged-shadow ordering.

Offline replay only; no candidate evaluation; production selector untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.oc_lagged_coupling_shadow import (
    ARMS,
    _finite_number,
    _ordered_contexts,
    _scope_key,
    _scope_stream,
    validate_input,
)

INPUT_SCHEMA = "arac-oc-action-semantic-gate-v3"
OUTPUT_SCHEMA = "arac-oc-oracle-gap-action-v1"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260821
MATERIAL_MAGNITUDE = 0.05


def _arm_gains(context: dict[str, Any], field: str) -> dict[str, float]:
    return {
        arm: _finite_number(context[arm][field], f"{arm}.{field}") for arm in ARMS
    }


def _oracle(gains: dict[str, float]) -> tuple[str, float]:
    arm = max(ARMS, key=lambda a: (gains[a], -ARMS.index(a)))
    return arm, gains[arm]


def _best_fixed(contexts: list[dict[str, Any]], field: str) -> tuple[str, float]:
    means = {
        arm: float(np.mean([_arm_gains(c, field)[arm] for c in contexts]))
        for arm in ARMS
    }
    arm = max(ARMS, key=lambda a: (means[a], -ARMS.index(a)))
    return arm, means[arm]


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _classify(ci_low: float, point: float) -> str:
    if ci_low <= 0.0:
        return "NOT_MATERIAL"
    return "MATERIAL" if point >= MATERIAL_MAGNITUDE else "PRESENT_BUT_SMALL"


def _flip_rate(payload: dict[str, Any], field: str) -> dict[str, Any]:
    previous_stream: str | None = None
    previous_arm: str | None = None
    flips = 0
    pairs = 0
    distribution = {arm: 0 for arm in ARMS}
    for _, context in _ordered_contexts(payload):
        arm, _ = _oracle(_arm_gains(context, field))
        distribution[arm] += 1
        stream = _scope_stream(_scope_key(context))
        if stream == previous_stream and previous_arm is not None:
            pairs += 1
            flips += int(arm != previous_arm)
        previous_stream, previous_arm = stream, arm
    return {
        "adjacent_pairs": pairs,
        "identity_flips": flips,
        "flip_rate": (flips / pairs) if pairs else 0.0,
        "oracle_arm_distribution": distribution,
    }


def _endpoint_report(payload: dict[str, Any], field: str) -> dict[str, Any]:
    contexts = payload["contexts"]
    per_context = []
    for context in contexts:
        gains = _arm_gains(context, field)
        oracle_arm, oracle_gain = _oracle(gains)
        error = float(context["checkpoint_error"])
        if error <= 0.0:
            raise ValueError("checkpoint_error must be positive")
        per_context.append(
            {
                "topology": str(context["topology"]),
                "overlap_budget": int(context["overlap_budget"]),
                "seed": int(context["seed"]),
                "mode": str(context["mode"]),
                "oracle_arm": oracle_arm,
                "oracle_gain": oracle_gain,
                "worst_arm_gain": min(gains.values()),
                "oracle_range": oracle_gain - min(gains.values()),
                "mean_arm_gain": float(np.mean(list(gains.values()))),
                "checkpoint_error": error,
            }
        )
    best_fixed_arm, best_fixed_mean = _best_fixed(contexts, field)
    regrets = []
    normalized = []
    for context, record in zip(contexts, per_context):
        gains = _arm_gains(context, field)
        regret = record["oracle_gain"] - gains[best_fixed_arm]
        regrets.append(regret)
        normalized.append(regret / record["checkpoint_error"])
    regrets_arr = np.asarray(regrets, dtype=float)
    normalized_arr = np.asarray(normalized, dtype=float)
    ci = _bootstrap_ci(regrets_arr)
    normalized_ci = _bootstrap_ci(normalized_arr)
    oracle_mean = float(np.mean([r["oracle_gain"] for r in per_context]))
    return {
        "best_fixed_arm": best_fixed_arm,
        "best_fixed_mean_gain": best_fixed_mean,
        "oracle_mean_gain": oracle_mean,
        "mean_best_fixed_regret": float(regrets_arr.mean()),
        "mean_best_fixed_regret_ci95": [ci[0], ci[1]],
        "mean_normalized_regret": float(normalized_arr.mean()),
        "mean_normalized_regret_ci95": [normalized_ci[0], normalized_ci[1]],
        "random_arm_mean_gain": float(
            np.mean([r["mean_arm_gain"] for r in per_context])
        ),
        "opportunity_captured_by_best_fixed": (
            best_fixed_mean / oracle_mean if oracle_mean else 0.0
        ),
        "opportunity_captured_by_random_arm": (
            float(np.mean([r["mean_arm_gain"] for r in per_context])) / oracle_mean
            if oracle_mean
            else 0.0
        ),
        "mean_normalized_oracle_range": float(
            np.mean([r["oracle_range"] / r["checkpoint_error"] for r in per_context])
        ),
        "classification": _classify(normalized_ci[0], float(normalized_arr.mean())),
        "oracle_identity": _flip_rate(payload, field),
        "per_context": per_context,
    }


def run_gate(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    validate_input(payload)
    end_report = _endpoint_report(payload, "end_to_end_gain")
    action_report = _endpoint_report(payload, "action_gain")
    action_report.pop("per_context")
    checks = {
        "input_contracts_valid": True,
        "context_count_60": len(payload["contexts"]) == 60,
        "production_selector_unchanged": payload["protocol"]["production_selector_modified"]
        is False,
        "classification_rule_unchanged": MATERIAL_MAGNITUDE == 0.05,
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": {
            "input_schema": INPUT_SCHEMA,
            "input_artifact": str(input_path),
            "primary_endpoint": "end_to_end_gain",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "material_magnitude_threshold": MATERIAL_MAGNITUDE,
            "classification_rule": (
                "MATERIAL if normalized-regret CI low > 0 and mean >= 0.05; "
                "PRESENT_BUT_SMALL if CI low > 0 and mean < 0.05; else NOT_MATERIAL"
            ),
            "production_selector_modified": False,
        },
        "end_to_end": end_report,
        "action": action_report,
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
        default=Path("artifacts/oc_oracle_gap_action_gate/confirmation.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "gate_passed": result["gate_passed"],
        "end_to_end": {
            k: v
            for k, v in result["end_to_end"].items()
            if k not in ("per_context", "oracle_identity")
        },
        "end_to_end_oracle_identity": result["end_to_end"]["oracle_identity"],
        "action": result["action"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
