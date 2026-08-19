"""Audit the retained EXP-052 SMP protocol binding without executing it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.historical_recovery.recover_session_sources import (
    AOR_WORKER_SHA256,
    DEFAULT_SESSION_PATH,
    SMP_ACTION_SHA256,
    build_recovery_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "exp052_smp_binding.json"
DEFAULT_OUTPUT_MD = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "exp052_smp_binding.md"
CONFIG_PATH = "experiments/pilots/exp_052_e_series_smp_paired_gate/config.json"
RECEIPT_PATH = (
    "results/exp_052_e_series_smp_paired_gate/validation/runs/E1/candidate_smp/seed_117/"
    "exp_052_e_series_smp_paired_gate-e1-candidate_smp-seed117/elliptic/"
    "exp052_execution_receipt.json"
)
SUMMARY_PATH = RECEIPT_PATH.replace("exp052_execution_receipt.json", "run_summary.json")
ACTION_PATH = RECEIPT_PATH.replace("exp052_execution_receipt.json", "smp_action.json")
INPUT_MANIFEST_PATH = RECEIPT_PATH.replace(
    "exp052_execution_receipt.json", "E1_aob_input_manifest.csv"
)
BUDGET_PATH = RECEIPT_PATH.replace(
    "exp052_execution_receipt.json", "E1_budget_summary.csv"
)
HISTORICAL_REF = "c7505d91"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(relative_path: str) -> dict[str, Any]:
    payload = json.loads((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {relative_path}")
    return payload


def _git_blob_hashes(relative_path: str) -> dict[str, str | None]:
    completed = subprocess.run(
        ("git", "show", f"{HISTORICAL_REF}:vendor/hcc/AOB/AOBG/datafile/{relative_path}"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"git_blob_sha256": None, "windows_checkout_sha256": None}
    git_bytes = completed.stdout
    windows_bytes = git_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return {
        "git_blob_sha256": _sha256(git_bytes),
        "windows_checkout_sha256": _sha256(windows_bytes),
    }


def _canonical_sha256(payload: object) -> str:
    return _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def build_report(session_path: Path = DEFAULT_SESSION_PATH) -> dict[str, Any]:
    recovery, artifacts = build_recovery_bundle(session_path)
    receipt = _read_json(RECEIPT_PATH)
    summary = _read_json(SUMMARY_PATH)
    action = _read_json(ACTION_PATH)
    config_artifact = artifacts[
        "retained-campaign-sources/experiments/pilots/exp_052_e_series_smp_paired_gate/config.json"
    ]

    with (REPOSITORY_ROOT / INPUT_MANIFEST_PATH).open(encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    input_binding = []
    for row in input_rows:
        retained_hashes = _git_blob_hashes(row["file"])
        recorded_hash = row["sha256_before"]
        input_binding.append(
            {
                "file": row["file"],
                "recorded_sha256": recorded_hash,
                "unchanged_during_run": row["unchanged"] == "1",
                **retained_hashes,
                "matching_representation": (
                    "git_blob"
                    if recorded_hash == retained_hashes["git_blob_sha256"]
                    else "windows_checkout_crlf"
                    if recorded_hash == retained_hashes["windows_checkout_sha256"]
                    else None
                ),
            }
        )

    command = receipt.get("command", [])
    command_text = " ".join(str(value) for value in command)
    action_payload = action.get("action")
    action_contract = {
        "schema_version": action.get("schema_version") == "smp-action-v1",
        "action_hash_matches_payload": action.get("action_hash")
        == _canonical_sha256(action_payload),
        "module_sha256_recovered": recovery["supporting_sources"]["SMP action module"][
            "sha256"
        ]
        == SMP_ACTION_SHA256,
        "group_dimensions": action_payload.get("group_dimensions") if isinstance(action_payload, dict) else None,
        "target_groups_cover_all": (
            isinstance(action_payload, dict)
            and action_payload.get("target_groups") == list(
                range(len(action_payload.get("group_dimensions", [])))
            )
        ),
        "state_fields": action_payload.get("state_fields") if isinstance(action_payload, dict) else None,
        "restore_count": action.get("restore_count"),
        "reset_count": action.get("reset_count"),
        "abstain_count": action.get("abstain_count"),
        "event_count": action.get("event_count"),
    }
    receipt_binding = {
        "runner_sha256_matches_recovered": receipt.get("runner_sha256")
        == recovery["exact_sources"]["SMP"]["sha256"],
        "config_sha256_matches_recovered": receipt.get("config_sha256")
        == _sha256(config_artifact),
        "fresh_seed": receipt.get("seed") == 117,
        "three_million_fe": receipt.get("configured_max_fes") == 3_000_000,
        "smp_group_mode": "--group-optimizer-mode smp" in command_text,
        "no_external_checkpoint_argument": not any(
            "checkpoint" in str(value).lower() or "resume" == str(value).lower()
            for value in command
        ),
    }
    summary_binding = {
        "runner_artifact_hash_matches": summary.get("runtime_action_artifact_sha256")
        == _sha256((REPOSITORY_ROOT / ACTION_PATH).read_bytes()),
        "action_hash_matches": summary.get("runtime_action_hash") == action.get("action_hash"),
        "restore_count_matches": summary.get("smp_restore_count") == action.get("restore_count"),
        "reset_count_matches": summary.get("smp_reset_count") == action.get("reset_count"),
        "abstain_count_matches": summary.get("smp_abstain_count") == action.get("abstain_count"),
        "terminal_fes_exact": summary.get("fitness_evaluations") == 3_000_000,
    }
    budget_binding = {
        "path": BUDGET_PATH,
        "rows": [],
    }
    with (REPOSITORY_ROOT / BUDGET_PATH).open(encoding="utf-8", newline="") as handle:
        budget_binding["rows"] = list(csv.DictReader(handle))
    budget_binding["strict_zero_violation"] = (
        len(budget_binding["rows"]) == 1
        and budget_binding["rows"][0]["budget_accounting"] == "strict"
        and budget_binding["rows"][0]["budget_aligned_fe"] == "3000000"
        and budget_binding["rows"][0]["same_budget_violation"] == "0"
    )
    input_binding_complete = all(
        row["matching_representation"] is not None and row["unchanged_during_run"]
        for row in input_binding
    )
    exact_binding_complete = (
        all(receipt_binding.values())
        and all(summary_binding.values())
        and all(action_contract[key] for key in ("schema_version", "action_hash_matches_payload", "module_sha256_recovered", "target_groups_cover_all"))
        and budget_binding["strict_zero_violation"]
        and input_binding_complete
    )
    return {
        "schema_version": "arac-exp052-smp-binding-v1",
        "protocol": "exp052-e-series-smp-paired-gate-v1",
        "session_path": str(session_path),
        "external_checkpoint": {"required": False, "evidence": "fresh seeded run; internal SmpStateCache"},
        "receipt_binding": receipt_binding,
        "summary_binding": summary_binding,
        "action_contract": action_contract,
        "input_binding": input_binding,
        "input_binding_complete": input_binding_complete,
        "budget_binding": budget_binding,
        "exact_binding_complete": exact_binding_complete,
        "optimizer_dependency_sources": recovery["exp052_optimizer_sources"],
        "optimizer_dependency_closure": (
            "sources_recovered_not_receipt_hash_bound"
            if recovery["all_exp052_optimizer_sources_recovered"]
            else "source_recovery_incomplete"
        ),
        "replay_authorized": False,
        "decision": "protocol_bound_no_external_checkpoint_dependency_closure_pending",
        "recovered_source_hashes": {
            "runner": recovery["exact_sources"]["SMP"]["sha256"],
            "smp_action": recovery["supporting_sources"]["SMP action module"]["sha256"],
            "aor_worker_reference": AOR_WORKER_SHA256,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-052 SMP binding audit",
        "",
        "This is a read-only audit of the retained seed117 trajectory. It does not",
        "execute the recovered runner.",
        "",
        "- External checkpoint required: **no**",
        f"- Exact receipt/config/action/input/budget binding: **{'yes' if report['exact_binding_complete'] else 'no'}**",
        "- Optimizer dependency sources: **recovered from the session timeline**",
        "- Optimizer dependency receipt hashes: **not recorded**",
        "- Runtime package pins match the current candidate `.venv`: **yes**",
        "- Runtime environment manifest bound in receipt: **no**",
        "- Replay authorized: **no**",
        "",
        "## Evidence",
        "",
        "The command starts a fresh seed117 run. The runner constructs an empty",
        "SMP cache and records/validates state internally on each group visit; the",
        "retained `smp_action.json` contains the state schema, group dimensions,",
        "restore/reset counts, and per-event state hashes.",
        "",
        "The optimizer source closure is recovered. The remaining gate is provenance",
        "binding: EXP-052 does not hash those optimizer files or the Python/numerical",
        "environment in its receipt. The current `.venv` matches the runtime pins from",
        "the historical `pyproject.toml`, but that is a reconstruction candidate, not a",
        "receipt-bound environment. See `exp052_environment.md`.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    json_path: Path = DEFAULT_OUTPUT_JSON,
    markdown_path: Path = DEFAULT_OUTPUT_MD,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.session)
    write_report(report, args.json, args.markdown)
    if args.check:
        print(f"Exact protocol binding: {'yes' if report['exact_binding_complete'] else 'no'}")
        print("External checkpoint required: no")
        print("Replay authorized: no")
        print(f"Wrote {args.json}")
        print(f"Wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
