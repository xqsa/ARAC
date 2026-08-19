"""Audit historical AOB action lanes as numerical golden references.

The historical result directories are treated as read-only evidence. Deleted
source is inspected through ``git show``; no checkout or restoration is used.
The report deliberately distinguishes an auditable result from a replayable
protocol. Missing source blocks bitwise HCC replay, but it does not authorize
restoring HCC as a production dependency; ARAC must recover action semantics in
its independent runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "historical_protocol_reconstruction.json"
DEFAULT_OUTPUT_MD = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "historical_protocol_reconstruction.md"
HISTORICAL_REF = "c7505d91"
SEEDS = tuple(range(117, 142))
MAX_FES = 3_000_000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(ref: str, path: str) -> dict[str, Any]:
    completed = subprocess.run(
        ("git", "show", f"{ref}:{path}"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"ref": ref, "path": path, "present": False}
    data = completed.stdout
    return {
        "ref": ref,
        "path": path,
        "present": True,
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact(relative_path: str) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative_path
    payload = _read_json(path)
    return {
        "path": relative_path,
        "present": path.is_file(),
        "json_object": payload is not None,
        "sha256": _sha256_bytes(path.read_bytes()) if path.is_file() else None,
        "payload": payload,
    }


def _field(payload: dict[str, Any] | None, key: str) -> Any:
    return None if payload is None else payload.get(key)


def _lane(
    *,
    name: str,
    family: str,
    action: str,
    summary_path: str,
    receipt_path: str,
    action_artifact_paths: tuple[str, ...],
    source_objects: tuple[tuple[str, str], ...],
    required_fields: tuple[str, ...],
    blockers: tuple[str, ...],
    notes: tuple[str, ...],
) -> dict[str, Any]:
    summary = _artifact(summary_path)
    receipt = _artifact(receipt_path)
    summary_payload = summary["payload"]
    receipt_payload = receipt["payload"]
    artifacts = [_artifact(path) for path in action_artifact_paths]
    sources = [_git_blob(ref, path) for ref, path in source_objects]

    observed = {
        field: _field(summary_payload, field)
        for field in required_fields
    }
    missing_fields = [field for field, value in observed.items() if value is None]
    missing_sources = [
        f"{item['ref']}:{item['path']}"
        for item in sources
        if not item["present"]
    ]
    receipt_runner_sha = _field(receipt_payload, "runner_sha256") or _field(
        receipt_payload, "worker_sha256"
    )
    runner_sources = [item for item in sources if item["path"].endswith(("run.py", "_worker.py", "hcc_smoke_runner.py"))]
    runner_sha_matches = any(
        receipt_runner_sha is not None
        and item.get("sha256") == receipt_runner_sha
        for item in runner_sources
    )
    all_artifacts_present = all(item["present"] and item["json_object"] for item in artifacts)
    execution_contract_complete = (
        summary["present"]
        and receipt["present"]
        and all_artifacts_present
        and not missing_fields
        and not missing_sources
        and (receipt_runner_sha is None or runner_sha_matches)
        and not blockers
    )
    if execution_contract_complete:
        verdict = "reconstructable"
    elif summary["present"] or receipt["present"] or any(item["present"] for item in artifacts):
        verdict = "partial"
    else:
        verdict = "missing"
    return {
        "lane": name,
        "family": family,
        "action": action,
        "verdict": verdict,
        "replay_authorized": verdict == "reconstructable",
        "historical_ref": HISTORICAL_REF,
        "required_fields": list(required_fields),
        "observed_fields": observed,
        "missing_fields": missing_fields,
        "missing_sources": missing_sources,
        "blockers": list(blockers),
        "runner_sha256": receipt_runner_sha,
        "runner_sha_matches_git_source": runner_sha_matches,
        "summary": summary,
        "receipt": receipt,
        "action_artifacts": artifacts,
        "source_objects": sources,
        "notes": list(notes),
    }


def build_report() -> dict[str, Any]:
    aor = _lane(
        name="AOR",
        family="A",
        action="aor",
        summary_path="results/exp_057_a_series_aor_25seed/a1-a6-25seed-v1/runs/A1/seed_117/run_summary.json",
        receipt_path="results/exp_057_a_series_aor_25seed/a1-a6-25seed-v1/runs/A1/seed_117/execution_receipt.json",
        action_artifact_paths=(),
        source_objects=(
            (HISTORICAL_REF, "experiments/pilots/exp_057_a_series_aor_25seed/_worker.py"),
            (HISTORICAL_REF, "experiments/pilots/exp_057_a_series_aor_25seed/run.py"),
            ("b53c7121", "src/arac/actions/full_space_sep_cma.py"),
            (HISTORICAL_REF, "vendor/hcc/HCC/OPT/CMAES/sepcmaes.py"),
            (HISTORICAL_REF, "vendor/hcc/AOB/AOB.py"),
        ),
        required_fields=(
            "case", "seed", "dimension", "configured_max_fes", "fitness_evaluations",
            "initial_mean", "lower", "upper", "sigma", "population_size",
            "backend", "optimizer_route", "policy_action", "policy_protocol",
        ),
        blockers=("exact_exp057_worker_source_missing",),
        notes=(
            "The result is a fresh 3,000,000-FE full-space vendor Sep-CMA run from initial_mean=0.",
            "The exact exp057 worker source is absent from the checked historical Git ref; worker_sha256 is retained in the receipt.",
        ),
    )
    ctp = _lane(
        name="CTP",
        family="S",
        action="ctp_stable",
        summary_path="results/exp_058_ctp_stable_v2_25seed/validation/runs/S1/seed_117/exp_058_ctp_stable_v2_25seed-s1-seed117/schwefel/run_summary.json",
        receipt_path="results/exp_058_ctp_stable_v2_25seed/validation/runs/S1/seed_117/exp_058_ctp_stable_v2_25seed-s1-seed117/schwefel/exp058_execution_receipt.json",
        action_artifact_paths=(
            "results/exp_058_ctp_stable_v2_25seed/validation/runs/S1/seed_117/exp_058_ctp_stable_v2_25seed-s1-seed117/schwefel/ctp_stable_action.json",
        ),
        source_objects=(
            (HISTORICAL_REF, "scripts/hcc_smoke_runner.py"),
            (HISTORICAL_REF, "src/arac/actions/mmes_resume.py"),
            (HISTORICAL_REF, "src/arac/backends/hcc_budget.py"),
            (HISTORICAL_REF, "vendor/hcc/HCC/NDAs/MMES/mmes.py"),
            (HISTORICAL_REF, "vendor/hcc/AOB/AOB.py"),
        ),
        required_fields=(
            "problem_id", "seed", "configured_max_fes", "fitness_evaluations",
            "decision_fe", "decision_available_fes", "coverage_sweeps",
            "runtime_action", "protocol_version", "group_optimizer_mode",
        ),
        blockers=("exact_exp058_runner_sha_unavailable",),
        notes=(
            "The action artifact binds one decision at FE 1,389,598 after four coverage sweeps and a zero-overlap full-space polish route.",
            "The receipt runner SHA does not match the only Git-visible hcc_smoke_runner.py, so the exact exp058 runner is unavailable.",
        ),
    )
    gcb = _lane(
        name="GCB",
        family="R",
        action="gcb",
        summary_path="results/exp_059_gcb_stable_all_case_25seed/validation/runs/R1/seed_117/exp_059_gcb_stable_all_case_25seed-r1-seed117/rastrigin/run_summary.json",
        receipt_path="results/exp_059_gcb_stable_all_case_25seed/validation/runs/R1/seed_117/exp_059_gcb_stable_all_case_25seed-r1-seed117/rastrigin/exp059_execution_receipt.json",
        action_artifact_paths=(
            "results/exp_059_gcb_stable_all_case_25seed/validation/runs/R1/seed_117/exp_059_gcb_stable_all_case_25seed-r1-seed117/rastrigin/gcb_action.json",
            "results/exp_059_gcb_stable_all_case_25seed/validation/runs/R1/seed_117/exp_059_gcb_stable_all_case_25seed-r1-seed117/rastrigin/gcb_stable_authorization.json",
        ),
        source_objects=(
            (HISTORICAL_REF, "scripts/hcc_smoke_runner.py"),
            ("ccc890a2", "src/arac/actions/gcb.py"),
            (HISTORICAL_REF, "src/arac/backends/hcc_gcb.py"),
            (HISTORICAL_REF, "vendor/hcc/HCC/OPT/CMAES/sepcmaes.py"),
            (HISTORICAL_REF, "vendor/hcc/AOB/AOB.py"),
        ),
        required_fields=(
            "problem_id", "seed", "configured_max_fes", "fitness_evaluations",
            "runtime_action", "runtime_policy_action", "runtime_policy_protocol",
            "persistent_phase2_execution_mode", "group_optimizer_mode",
        ),
        blockers=("exact_exp059_runner_sha_unavailable",),
        notes=(
            "The action artifact binds a one-native-sweep burst at a phase boundary, then three native resume sweeps to FE 3,000,000.",
            "The receipt runner SHA does not match the only Git-visible hcc_smoke_runner.py; exact exp059 runner source is unavailable.",
        ),
    )
    smp = _lane(
        name="SMP",
        family="E",
        action="smp",
        summary_path="artifacts/frozen_actions/smp_v26/evidence/v26_summary.json",
        receipt_path="artifacts/frozen_actions/smp_v26/manifest.json",
        action_artifact_paths=("artifacts/frozen_actions/smp_v26/manifest.json",),
        source_objects=(
            ("acd4d84b", "src/arac/actions/smp.py"),
            (HISTORICAL_REF, "src/arac/backends/hcc_a_actions.py"),
            (HISTORICAL_REF, "vendor/hcc/HCC/NDAs/MMES/mmes.py"),
            (HISTORICAL_REF, "vendor/hcc/AOB/AOB.py"),
        ),
        required_fields=("action_name", "max_fes", "completed", "failed", "all_terminal_fes_exact"),
        blockers=("historical_25_seed_smp_lane_absent",),
        notes=(
            "The available SMP artifact is a development candidate with five seeds, not the historical 25-seed lane.",
            "It cannot bind the historical per-seed checkpoint and lifecycle protocol needed for replay.",
        ),
    )
    lanes = [aor, ctp, gcb, smp]
    return {
        "schema_version": "arac-historical-protocol-reconstruction-v1",
        "historical_ref": HISTORICAL_REF,
        "seed_schedule": list(SEEDS),
        "max_fes": MAX_FES,
        "source_policy": "git_show_only_no_checkout",
        "replay_policy": "authorize only reconstructable lanes",
        "recovery_interpretation": "golden_reference_only_independent_semantic_parity",
        "lanes": lanes,
        "replay_authorized_lanes": [lane["lane"] for lane in lanes if lane["replay_authorized"]],
        "decision": "hold_replay_selector_and_end_to_end",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Historical action protocol reconstruction",
        "",
        "- Generated by Codex from read-only result artifacts and `git show`.",
        f"- Historical Git reference checked: `{report['historical_ref']}`.",
        f"- Common schedule: `{len(report['seed_schedule'])}` seeds ({report['seed_schedule'][0]}..{report['seed_schedule'][-1]}), `{report['max_fes']:,}` FE.",
        "",
        "## Replay gate",
        "",
        "No lane is authorized for replay unless the exact execution-critical source is present and its receipt runner hash matches.",
        f"Authorized lanes: **{', '.join(report['replay_authorized_lanes']) or 'none'}**.",
        "Historical lanes remain numerical golden references even when exact replay is blocked.",
        "Production ARAC must not restore an HCC runtime dependency.",
        "",
        "## Lane verdicts",
        "",
        "| Lane | Action | Verdict | Replay | Missing source/fields |",
        "|---|---|---|---|---|",
    ]
    for lane in report["lanes"]:
        missing = ", ".join(
            lane["blockers"] + lane["missing_sources"] + lane["missing_fields"]
        ) or "none"
        lines.append(
            f"| {lane['lane']} | `{lane['action']}` | **{lane['verdict']}** | "
            f"{'yes' if lane['replay_authorized'] else 'no'} | {missing} |"
        )
    lines.extend(["", "## Findings", ""])
    for lane in report["lanes"]:
        lines.append(f"### {lane['lane']}")
        lines.append("")
        for note in lane["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend([
        "## Decision",
        "",
        "The historical results remain provenance-bound numerical golden references, but they do not authorize a new HCC replay campaign. Keep the production runtime independent, recover the effective AOR/CTP/GCB/SMP semantics behind the ARAC action interface, pass fixed-checkpoint action-level parity gates, and only then evaluate selector correctness.",
        "",
    ])
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: Path = DEFAULT_OUTPUT_JSON, md_path: Path = DEFAULT_OUTPUT_MD) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="build and validate the report")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)
    report = build_report()
    write_report(report, args.json, args.markdown)
    if args.check:
        if not report["lanes"]:
            raise SystemExit("no historical lanes were audited")
        print(f"Wrote {args.json}")
        print(f"Wrote {args.markdown}")
        print("Replay-authorized lanes:", ", ".join(report["replay_authorized_lanes"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
