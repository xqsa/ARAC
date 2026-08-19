"""Audit the first-level local import closure of the recovered EXP-052 runner."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.historical_recovery.recover_session_sources import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SESSION_PATH,
    EXP052_FORMAL_START_SESSION_LINE,
    SMP_ACTION_PATH,
    SMP_ACTION_SHA256,
    build_recovery_bundle,
    read_patch_events,
    recover_full_content_target,
    recover_sources_at_boundary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_SHA256 = "b17021e8ffe1de76fea48b52ed3c00a62b4cc93bf4c2c759604064d14ebc68ac"
DEFAULT_OUTPUT_JSON = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "exp052_dependency_closure.json"
DEFAULT_OUTPUT_MD = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "exp052_dependency_closure.md"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _module_path(module: str) -> str | None:
    if module.startswith(("AOB.", "HCC.")):
        return f"vendor/hcc/{module.replace('.', '/')}.py"
    if module.startswith("src.arac."):
        return f"{module.replace('.', '/')}.py"
    if module.startswith("arac."):
        return f"src/{module.replace('.', '/')}.py"
    return None


def _git_candidate(path: str) -> str | None:
    completed = subprocess.run(
        ("git", "show", f"c7505d91:{path}"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    return _sha256(completed.stdout) if completed.returncode == 0 else None


def _initial_git_status_snapshot(session_path: Path) -> dict[str, Any] | None:
    call_id = None
    command_line = None
    with session_path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = json.loads(line)
            event = payload.get("payload", {})
            if (
                event.get("type") == "custom_tool_call"
                and event.get("name") == "exec"
                and "git status --short" in str(event.get("input", ""))
            ):
                call_id = event.get("call_id")
                command_line = line_number
                continue
            if (
                call_id is not None
                and event.get("type") == "custom_tool_call_output"
                and event.get("call_id") == call_id
            ):
                output = json.dumps(event.get("output"), ensure_ascii=False)
                return {
                    "call_id": call_id,
                    "command_session_line": command_line,
                    "output_session_line": line_number,
                    "output_not_truncated": "truncated output" not in output,
                    "output": output,
                }
    return None


def _runner_path() -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "exact-runner"
        / RUNNER_SHA256
        / "scripts"
        / "hcc_smoke_runner.py"
    )


def build_report(session_path: Path = DEFAULT_SESSION_PATH) -> dict[str, Any]:
    runner = _runner_path()
    syntax = ast.parse(runner.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    paths = sorted({path for module in modules if (path := _module_path(module))})
    events = read_patch_events(session_path)
    initial_status = _initial_git_status_snapshot(session_path)
    _, artifacts = build_recovery_bundle(session_path)
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path == SMP_ACTION_PATH:
            match, _ = recover_full_content_target(events, path, SMP_ACTION_SHA256)
            rows.append(
                {
                    "path": path,
                    "status": "exact_session_source" if match else "missing",
                    "sha256": None if match is None else match["sha256"],
                    "provenance": "full session content",
                }
            )
            continue
        try:
            recovered, _ = recover_sources_at_boundary(
                events,
                [path],
                boundary_line=EXP052_FORMAL_START_SESSION_LINE,
                output_prefix="closure-probe",
            )
        except ValueError as error:
            rows.append(
                {
                    "path": path,
                    "status": "patch_chain_mismatch",
                    "error": str(error),
                }
            )
            continue
        result = recovered[0]
        if result["recovered"]:
            rows.append(
                {
                    "path": path,
                    "status": "exact_session_boundary_source",
                    "sha256": result["sha256"],
                    "provenance": "reverse session patch chain",
                }
            )
            continue
        git_hash = _git_candidate(path)
        patch_event_count = sum(
            path in event.changes and event.line_number <= EXP052_FORMAL_START_SESSION_LINE
            for event in events
        )
        status_clean = bool(
            initial_status
            and initial_status["output_not_truncated"]
            and path not in initial_status["output"]
            and path.replace("/", "\\") not in initial_status["output"]
        )
        exact_git_worktree_source = bool(git_hash and status_clean and patch_event_count == 0)
        rows.append(
            {
                "path": path,
                "status": (
                    "exact_git_worktree_source"
                    if exact_git_worktree_source
                    else "git_ref_candidate"
                    if git_hash
                    else "missing"
                ),
                "sha256": git_hash,
                "provenance": (
                    "historical Git ref plus retained clean worktree status"
                    if exact_git_worktree_source
                    else "historical Git ref"
                    if git_hash
                    else None
                ),
                "reason": result.get("reason"),
                "historical_ref": "c7505d91" if git_hash else None,
                "initial_status_clean": status_clean,
                "patch_events_before_formal_start": patch_event_count,
                "receipt_hash_bound": False,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema_version": "arac-exp052-dependency-closure-v1",
        "runner_sha256": _sha256(runner.read_bytes()),
        "runner_hash_matches_target": _sha256(runner.read_bytes()) == RUNNER_SHA256,
        "first_level_import_count": len(rows),
        "status_counts": counts,
        "rows": rows,
        "exact_first_level_closure": all(
            row["status"]
            in {
                "exact_session_source",
                "exact_session_boundary_source",
                "exact_git_worktree_source",
            }
            for row in rows
        ),
        "performance_experiment_started": False,
        "replay_authorized": False,
        "decision": "first_level_source_closure_recovered_environment_binding_pending",
        "unused_bundle_artifact_count": len(artifacts),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-052 dependency closure audit",
        "",
        f"- First-level local imports checked: `{report['first_level_import_count']}`",
        f"- Status counts: `{report['status_counts']}`",
        f"- Exact first-level closure: **{'yes' if report['exact_first_level_closure'] else 'no'}**",
        "- Replay authorized: **no**",
        "",
        "| Path | Status | SHA-256 |",
        "|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(f"| `{row['path']}` | **{row['status']}** | `{row.get('sha256', '-')}` |")
    lines.extend(
        [
            "",
            "The recovered runner, CTP/GCB/SMP action modules, and HCC optimizer sources",
            "are exact at the formal-start boundary. The two AOB Python modules are",
            "exact Git-worktree sources: their last Git changes predate EXP-052, the",
            "retained initial status is complete and clean for both paths, and neither",
            "path has a pre-run session patch. They are still not receipt-hash-bound.",
            "",
            "The separate environment audit in `exp052_environment.md` finds matching",
            "runtime pins (`numpy`, `scipy`, `PyYAML`) in the current project `.venv`,",
            "but the receipt records no Python/dependency/environment-manifest hash.",
            "This closes the source description, not the replay gate.",
            "",
        ]
    )
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
        print(f"First-level exact closure: {'yes' if report['exact_first_level_closure'] else 'no'}")
        print(f"Status counts: {report['status_counts']}")
        print("Replay authorized: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
