"""Read-only mechanism diagnosis for the failed Gate 10 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


DEFAULT_INPUT = Path("artifacts/overlap_coordination_effectiveness_gate10/confirmation_fresh.json")
DEFAULT_OUTPUT = Path("artifacts/overlap_coordination_mechanism_diagnosis_v1/diagnostic.json")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": float(median(values)) if values else float("nan"),
        "p10": _percentile(values, 10.0),
        "p90": _percentile(values, 90.0),
        "mean": float(np.mean(values)) if values else float("nan"),
    }


def _proposal_metrics(context: dict[str, object]) -> dict[str, list[float]]:
    proposals = context["coordination"]["proposal_runs"]
    residuals = [
        item
        for result in context["coordination"]["prime_results"]
        for item in result["residuals"]
    ]
    proposal_by_group = {item["group"]: item for item in proposals}
    raw_disagreement: list[float] = []
    uncertainty: list[float] = []
    improvement: list[float] = []
    proposal_weight: list[float] = []
    normalized_weight_max: list[float] = []
    for residual in residuals:
        variable = int(residual["variable"])
        values = []
        sigmas = []
        weights = []
        for proposal in proposal_by_group.values():
            values_by_variable = dict(proposal["values"])
            sigma_by_variable = dict(proposal["uncertainty"])
            if variable in values_by_variable:
                values.append(float(values_by_variable[variable]))
                sigma = float(sigma_by_variable[variable])
                contribution = float(proposal["improvement"])
                sigmas.append(sigma)
                weights.append((1.0 + max(0.0, contribution)) / (sigma + 1.0e-12))
        if len(values) >= 2:
            raw_disagreement.append(float(max(values) - min(values)))
            uncertainty.append(float(np.mean(sigmas)))
            proposal_weight.extend(weights)
            normalized_weight_max.append(float(max(weights) / sum(weights)))
        for proposal in proposal_by_group.values():
            if variable in dict(proposal["values"]):
                improvement.append(float(proposal["improvement"]))
    return {
        "raw_disagreement": raw_disagreement,
        "uncertainty": uncertainty,
        "improvement": improvement,
        "proposal_weight": proposal_weight,
        "normalized_weight_max": normalized_weight_max,
        "between_variance": [float(item["between_variance"]) for item in residuals],
        "within_variance": [float(item["within_variance"]) for item in residuals],
        "conflict_score": [float(item["conflict_score"]) for item in residuals],
    }


def _mode_summary(contexts: list[dict[str, object]], mode: str) -> dict[str, object]:
    rows = [context for context in contexts if context["mode"] == mode]
    keys = (
        "raw_disagreement",
        "uncertainty",
        "improvement",
        "proposal_weight",
        "normalized_weight_max",
        "between_variance",
        "within_variance",
        "conflict_score",
    )
    metrics = {key: [] for key in keys}
    for context in rows:
        current = _proposal_metrics(context)
        for key in keys:
            metrics[key].extend(current[key])
    return {key: _summary(values) for key, values in metrics.items()}


def _per_context(contexts: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for context in contexts:
        metrics = _proposal_metrics(context)
        rows.append(
            {
                "mode": context["mode"],
                "topology": context["topology"],
                "overlap_budget": context["overlap_budget"],
                "seed": context["seed"],
                "max_raw_disagreement": max(metrics["raw_disagreement"], default=0.0),
                "max_between_variance": max(metrics["between_variance"], default=0.0),
                "median_within_variance": float(median(metrics["within_variance"])) if metrics["within_variance"] else 0.0,
                "max_conflict_score": max(metrics["conflict_score"], default=0.0),
                "ctp_triggered": bool(context["ctp_triggered"]),
                "gain": float(context["gain"]),
            }
        )
    return rows


def run_diagnosis(input_path: Path = DEFAULT_INPUT) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("gate_passed") is not False:
        raise ValueError("diagnosis requires a Gate 10 artifact with gate_passed=false")
    contexts = list(payload["contexts"])
    per_context = _per_context(contexts)
    conforming = _mode_summary(contexts, "conforming")
    conflicting = _mode_summary(contexts, "conflicting")
    conforming_trigger = [row for row in per_context if row["mode"] == "conforming" and row["ctp_triggered"]]
    conflicting_trigger = [row for row in per_context if row["mode"] == "conflicting" and row["ctp_triggered"]]
    gain_triggered = [row["gain"] for row in conflicting_trigger]
    return {
        "schema_version": "arac-overlap-coordination-mechanism-diagnosis-v1",
        "input": str(input_path),
        "context_count": len(contexts),
        "mode_summary": {"conforming": conforming, "conflicting": conflicting},
        "trigger_gain": {
            "conforming_trigger_count": len(conforming_trigger),
            "conflicting_trigger_count": len(conflicting_trigger),
            "conflicting_trigger_gain": _summary(gain_triggered),
        },
        "per_context": per_context,
        "interpretation": {
            "residual_failure": (
                "Conforming and conflicting raw proposal disagreement and normalized residuals overlap; "
                "the current B/(W+epsilon) score is not a reliable conflict discriminator under real proposals."
            ),
            "likely_mechanism": (
                "The local proposal producer optimizes the full objective from a common anchor, while sigma is "
                "estimated from each proposal's elite samples. A small within variance can inflate conforming "
                "scores, so disagreement is being conflated with optimizer certainty."
            ),
            "ctp_failure": (
                "CTP is selected after the misclassified high residual and competes with an owner continuation; "
                "strict-best protects the archive but does not make shared-core search value-positive."
            ),
            "next_interface_to_test": (
                "Before changing thresholds, test an independently calibrated residual using repeated local "
                "proposal replicates or held-out proposal noise, and separately audit CTP candidate value "
                "against an equal-FE shared-core random control."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run_diagnosis(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"context_count": payload["context_count"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
