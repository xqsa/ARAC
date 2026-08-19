"""Diagnose protocol and source drift between historical experts and v9 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "config.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "experiments" / "historical_recovery" / "fixed_expert_drift.json"
)
DEFAULT_REPORT = (
    REPOSITORY_ROOT / "experiments" / "historical_recovery" / "fixed_expert_drift.md"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_summary(path: Path, case_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(path.parent.glob(f"runs/{case_id}/seed_117/**/run_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"representative historical summary not found: {path}")
    summary_path = candidates[0]
    return summary_path, _load_json(summary_path)


def _historical_protocols(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    representatives: dict[str, Any] = {}
    for action, case_id in (("aor", "A1"), ("ctp", "S1"), ("gcb", "R1")):
        manifest_path = root / config["evidence"][action]["manifest"]
        summary_path, summary = _representative_summary(manifest_path, case_id)
        fields = (
            "protocol_version",
            "worker_protocol_version",
            "backend",
            "optimizer_route",
            "policy_action",
            "policy_protocol",
            "execution_source",
            "initial_mean",
            "phase1_fes",
            "action_actual_fes",
            "decision_available_fes",
            "decision_fe",
            "configured_max_fes",
            "fitness_evaluations",
            "runtime_action",
            "runtime_policy_action",
            "runtime_policy_protocol",
            "trigger_scope",
        )
        representatives[action] = {
            "manifest": str(manifest_path.relative_to(root)),
            "summary": str(summary_path.relative_to(root)),
            "fields": {name: summary[name] for name in fields if name in summary},
        }
    representatives["smp"] = {
        "status": "historical_complete_lane_absent",
        "manifest": config["evidence"]["smp"]["frozen_candidate_manifest"],
    }
    return representatives


def _source_drift(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    current = _load_json(root / config["current_fixed_expert"]["campaign_manifest"])
    frozen = _load_json(root / config["frozen_independent_matrix"]["manifest"])
    current_hashes = current["source_hashes"]
    frozen_hashes = frozen["source_hashes"]
    common = []
    for name in sorted(set(current_hashes) & set(frozen_hashes)):
        common.append(
            {
                "component": name,
                "matches": current_hashes[name] == frozen_hashes[name],
                "current_sha256": current_hashes[name],
                "frozen_matrix_sha256": frozen_hashes[name],
            }
        )
    return {
        "current_manifest": config["current_fixed_expert"]["campaign_manifest"],
        "frozen_matrix_manifest": config["frozen_independent_matrix"]["manifest"],
        "current_phase1_protocol": current["phase1_protocol"],
        "frozen_matrix_phase1_protocol": frozen["phase1_protocol"],
        "common_component_count": len(common),
        "matching_component_count": sum(row["matches"] for row in common),
        "components": common,
    }


def run_diagnosis(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_json(config_path.resolve())
    root = REPOSITORY_ROOT
    current = _load_json(root / config["current_fixed_expert"]["summary"])
    checkpoint = _load_json(
        root
        / config["current_fixed_expert"]["results_csv"]
        .replace("results.csv", "arms/A1/seed_117/aor.json")
    )
    checkpoint_receipt = _load_json(
        root
        / config["current_fixed_expert"]["results_csv"]
        .replace("results.csv", "checkpoints/A1/seed_117/checkpoint.json")
    )
    action_result = checkpoint["action_result"]
    checkpoint_payload = checkpoint_receipt["checkpoint"]
    return {
        "schema_version": "arac-historical-fixed-expert-drift-v1",
        "historical_table": config["historical_reference"]["table_csv"],
        "current_campaign": {
            "summary": config["current_fixed_expert"]["summary"],
            "campaign_manifest": config["current_fixed_expert"]["campaign_manifest"],
            "gate_passed": current["gate_passed"],
            "context_count": current["context_count"],
            "all_terminal_fes_exact": current["all_terminal_fes_exact"],
            "phase1_fes": checkpoint_payload["phase1_fes"],
            "action_consumed_fes": action_result["consumed_fes"],
            "action_optimizer_package": action_result["optimizer_package"],
            "action_optimizer_version": action_result["optimizer_version"],
            "action_route": action_result["route"],
        },
        "historical_representatives": _historical_protocols(root, config),
        "source_drift": _source_drift(root, config),
        "confirmed_findings": [
            "The historical AOR representative is a fresh full-space 3,000,000-FE run from initial_mean=0 using vendor.HCC.OPT.CMAES.sepcmaes.",
            "The current fixed-expert campaign spends 180,000 FE in an identity-blind Phase-I checkpoint and gives the mapped action only the remaining 2,820,000 FE.",
            "Historical CTP and GCB representatives use hcc-run-summary-v3 action-specific decision and routing fields that are not represented by the v9 shared checkpoint contract.",
            "The current fixed-expert manifest and frozen v5 matrix disagree on the Phase-I protocol and on most common source hashes.",
            "The completed v9 campaign therefore establishes a valid current Phase-I fixed-action result, but it is not a fair bitwise or aggregate replay oracle for the historical table.",
        ],
        "decision": "hold_selector_and_end_to_end_claims_until_historical_action_protocol_is_reconstructed_or_the_target_is_rebound_to_the_v9_checkpoint_protocol",
    }


def render_report(diagnosis: dict[str, Any]) -> str:
    current = diagnosis["current_campaign"]
    drift = diagnosis["source_drift"]
    lines = [
        "# Fixed-expert historical drift diagnosis",
        "",
        "## Current v9 campaign",
        "",
        f"- Summary: `{current['summary']}`",
        f"- Complete contexts: **{current['context_count']}**",
        f"- Exact terminal FE: **{str(current['all_terminal_fes_exact']).lower()}**",
        f"- Phase-I FE in the shared checkpoint: **{current['phase1_fes']}**",
        f"- Mapped action FE after Phase-I: **{current['action_consumed_fes']}**",
        f"- Current optimizer port: `{current['action_optimizer_package']}=={current['action_optimizer_version']}`",
        "",
        "## Historical representatives",
        "",
        "| Lane | Evidence | Key protocol fields |",
        "|---|---|---|",
    ]
    for action, row in diagnosis["historical_representatives"].items():
        if "fields" not in row:
            fields = row["status"]
        else:
            fields = "; ".join(f"{key}={value}" for key, value in row["fields"].items())
        lines.append(f"| {action.upper()} | `{row['summary'] if 'summary' in row else row['manifest']}` | {fields} |")
    lines.extend(
        [
            "",
            "## Source and protocol drift",
            "",
            f"- Current Phase-I protocol: `{drift['current_phase1_protocol']}`",
            f"- Frozen matrix Phase-I protocol: `{drift['frozen_matrix_phase1_protocol']}`",
            f"- Common source components: **{drift['matching_component_count']}/{drift['common_component_count']}** match",
            "",
            "| Component | Match |",
            "|---|---|",
        ]
    )
    for row in drift["components"]:
        lines.append(f"| {row['component']} | {row['matches']} |")
    lines.extend(
        [
            "",
            "## Confirmed findings",
            "",
        ]
    )
    lines.extend(f"- {finding}" for finding in diagnosis["confirmed_findings"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The v9 fixed-action campaign is valid evidence for the current "
            "Phase-I-plus-mapped-action protocol, but it cannot certify recovery "
            "of the historical action-specific table. Reconstruct the historical "
            "action protocol, or explicitly rebind the target table to v9, before "
            "running selector correctness or ARAC-Core end-to-end experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    diagnosis = run_diagnosis(args.config)
    args.output.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(render_report(diagnosis), encoding="utf-8")
    print(json.dumps({"decision": diagnosis["decision"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
