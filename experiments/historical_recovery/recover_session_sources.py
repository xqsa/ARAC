"""Recover exact historical sources from the retained Codex session archive.

Recovered files are written only below the task-local ``raw`` directory.  The
production runner and frozen result directories are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_PATH = Path(
    r"C:\Users\83718\.codex\sessions\2026\07\26"
    + r"\rollout-2026-07-26T17-04-21-019f9dab-0f42-70f2-8f45-5cf51411e668.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / ".codex-tasks"
    / "historical-level-recovery"
    / "raw"
    / "session-source-recovery"
)
DEFAULT_OUTPUT_JSON = (
    REPOSITORY_ROOT
    / "experiments"
    / "historical_recovery"
    / "session_source_recovery.json"
)
DEFAULT_OUTPUT_MD = (
    REPOSITORY_ROOT
    / "experiments"
    / "historical_recovery"
    / "session_source_recovery.md"
)

RUNNER_PATH = "scripts/hcc_smoke_runner.py"
RUNNER_TARGETS = {
    "CTP/GCB": "9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594",
    "SMP": "b17021e8ffe1de76fea48b52ed3c00a62b4cc93bf4c2c759604064d14ebc68ac",
}
AOR_WORKER_PATH = "experiments/pilots/exp_057_a_series_aor_25seed/_worker.py"
AOR_WORKER_SHA256 = "2d870d14fa536dee488d45a69abea19e50e86dc20748b026d5fc4a16afcb4165"
SMP_ACTION_PATH = "src/arac/actions/smp.py"
SMP_ACTION_SHA256 = "5130b18a39c8f76e0713a30026fd9a997d34e982c7d7f8287b6d1047161efc16"
EXP052_FORMAL_START_SESSION_LINE = 2628
EXP052_OPTIMIZER_PATHS = (
    "vendor/hcc/HCC/OPT/CMAES/cmaes.py",
    "vendor/hcc/HCC/OPT/CMAES/es.py",
    "vendor/hcc/HCC/OPT/CMAES/optimizer.py",
    "vendor/hcc/HCC/NDAs/MMES/es.py",
    "vendor/hcc/HCC/NDAs/MMES/mmes.py",
    "vendor/hcc/HCC/NDAs/MMES/optimizer.py",
    "vendor/hcc/HCC/NDAs/MMES/state.py",
)
CAMPAIGN_PREFIXES = tuple(
    f"experiments/pilots/exp_{experiment:03d}_"
    for experiment in (52, 57, 58, 59)
)
CAMPAIGN_TEST_PREFIXES = tuple(
    f"tests/test_exp_{experiment:03d}_"
    for experiment in (52, 57, 58, 59)
)
HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


@dataclass(frozen=True)
class PatchEvent:
    timestamp: str
    line_number: int
    call_id: str
    changes: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    body: tuple[str, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/ARAC/"
    marker_index = normalized.upper().find(marker.upper())
    if marker_index >= 0:
        return normalized[marker_index + len(marker) :]
    return normalized.removeprefix("./").lstrip("/")


def read_patch_events(session_path: Path = DEFAULT_SESSION_PATH) -> list[PatchEvent]:
    events: list[PatchEvent] = []
    with Path(session_path).open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if "patch_apply_end" not in line:
                continue
            payload = json.loads(line)
            event = payload.get("payload", {})
            if event.get("type") != "patch_apply_end" or not event.get("success", True):
                continue
            raw_changes = event.get("changes")
            if not isinstance(raw_changes, dict):
                continue
            changes = {
                _normalized_repo_path(str(path)): metadata
                for path, metadata in raw_changes.items()
                if isinstance(metadata, dict)
            }
            events.append(
                PatchEvent(
                    timestamp=str(payload.get("timestamp", "")),
                    line_number=line_number,
                    call_id=str(event.get("call_id", "")),
                    changes=changes,
                )
            )
    return events


def parse_unified_diff(unified_diff: str) -> tuple[Hunk, ...]:
    lines = unified_diff.splitlines()
    hunks: list[Hunk] = []
    index = 0
    while index < len(lines):
        match = HUNK_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            index += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            if not line or line[0] not in " +-":
                raise ValueError(f"unsupported unified diff line: {line!r}")
            body.append(line)
        hunks.append(
            Hunk(
                old_start=int(match.group(1)),
                old_count=int(match.group(2) or 1),
                new_start=int(match.group(3)),
                new_count=int(match.group(4) or 1),
                body=tuple(body),
            )
        )
    if not hunks:
        raise ValueError("unified diff contains no hunks")
    return tuple(hunks)


def reverse_apply_unified_diff(content: str, unified_diff: str) -> str:
    """Apply one recorded update in reverse with exact line validation."""

    trailing_newline = content.endswith("\n")
    lines = content.splitlines()
    for hunk in reversed(parse_unified_diff(unified_diff)):
        source = [line[1:] for line in hunk.body if line[0] in " +"]
        target = [line[1:] for line in hunk.body if line[0] in " -"]
        if len(source) != hunk.new_count or len(target) != hunk.old_count:
            raise ValueError(
                "unified diff hunk counts do not match its body: "
                f"new={len(source)}/{hunk.new_count}, old={len(target)}/{hunk.old_count}"
            )
        position = hunk.new_start - 1
        observed = lines[position : position + len(source)]
        if observed != source:
            raise ValueError(
                "recorded patch cannot be reversed at its declared position: "
                f"line {hunk.new_start}"
            )
        lines[position : position + len(source)] = target
    return "\n".join(lines) + ("\n" if trailing_newline else "")


def _runner_events(events: Iterable[PatchEvent]) -> list[tuple[PatchEvent, dict[str, Any]]]:
    return [
        (event, event.changes[RUNNER_PATH])
        for event in events
        if RUNNER_PATH in event.changes
    ]


def reconstruct_runner_versions(
    events: Iterable[PatchEvent],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    runner_events = _runner_events(events)
    anchors = [
        (event, change)
        for event, change in runner_events
        if change.get("type") == "delete"
        and isinstance(change.get("content"), str)
        and len(change["content"]) > 100_000
    ]
    if not anchors:
        raise ValueError("no full-content runner deletion anchor was retained")
    anchor_event, anchor_change = anchors[-1]
    current = str(anchor_change["content"])
    target_by_sha = {digest: lane for lane, digest in RUNNER_TARGETS.items()}
    matches: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, bytes] = {}
    reversed_count = 0

    preceding_updates = [
        (event, change)
        for event, change in runner_events
        if event.line_number < anchor_event.line_number and change.get("type") == "update"
    ]
    for event, change in reversed(preceding_updates):
        unified_diff = change.get("unified_diff")
        if not isinstance(unified_diff, str):
            raise ValueError(f"runner update at session line {event.line_number} has no diff")
        current = reverse_apply_unified_diff(current, unified_diff)
        reversed_count += 1
        encoded = current.encode("utf-8")
        digest = _sha256(encoded)
        lane = target_by_sha.get(digest)
        if lane is None or lane in matches:
            continue
        output_path = f"exact-runner/{digest}/{RUNNER_PATH}"
        artifacts[output_path] = encoded
        matches[lane] = {
            "sha256": digest,
            "bytes": len(encoded),
            "characters": len(current),
            "lines": len(current.splitlines()),
            "state": "immediately_before_recorded_update",
            "boundary_event": {
                "timestamp": event.timestamp,
                "session_line": event.line_number,
                "call_id": event.call_id,
            },
            "reversed_patch_count_from_anchor": reversed_count,
            "output_path": output_path,
        }
        if len(matches) == len(RUNNER_TARGETS):
            break

    anchor_bytes = str(anchor_change["content"]).encode("utf-8")
    anchor = {
        "timestamp": anchor_event.timestamp,
        "session_line": anchor_event.line_number,
        "call_id": anchor_event.call_id,
        "sha256": _sha256(anchor_bytes),
        "bytes": len(anchor_bytes),
    }
    return matches, artifacts, anchor


def recover_full_content_target(
    events: Iterable[PatchEvent], path: str, target_sha256: str
) -> tuple[dict[str, Any] | None, bytes | None]:
    for event in reversed(list(events)):
        change = event.changes.get(path)
        if change is None or not isinstance(change.get("content"), str):
            continue
        encoded = change["content"].encode("utf-8")
        if _sha256(encoded) != target_sha256:
            continue
        output_path = f"exact-content/{target_sha256}/{path}"
        return (
            {
                "sha256": target_sha256,
                "bytes": len(encoded),
                "characters": len(change["content"]),
                "source_event": {
                    "timestamp": event.timestamp,
                    "session_line": event.line_number,
                    "call_id": event.call_id,
                    "change_type": change.get("type"),
                },
                "output_path": output_path,
            },
            encoded,
        )
    return None, None


def recover_retained_campaign_sources(
    events: Iterable[PatchEvent],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    retained: dict[str, tuple[PatchEvent, dict[str, Any]]] = {}
    for event in events:
        for path, change in event.changes.items():
            if not path.startswith((*CAMPAIGN_PREFIXES, *CAMPAIGN_TEST_PREFIXES)):
                continue
            if isinstance(change.get("content"), str):
                retained[path] = (event, change)

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, bytes] = {}
    for path, (event, change) in sorted(retained.items()):
        encoded = change["content"].encode("utf-8")
        output_path = f"retained-campaign-sources/{path}"
        artifacts[output_path] = encoded
        rows.append(
            {
                "path": path,
                "sha256": _sha256(encoded),
                "bytes": len(encoded),
                "source_event": {
                    "timestamp": event.timestamp,
                    "session_line": event.line_number,
                    "call_id": event.call_id,
                    "change_type": change.get("type"),
                },
                "output_path": output_path,
            }
        )
    return rows, artifacts


def recover_sources_at_boundary(
    events: Iterable[PatchEvent],
    paths: Iterable[str],
    *,
    boundary_line: int,
    output_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    event_list = list(events)
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, bytes] = {}
    for path in paths:
        path_events = [
            (event, event.changes[path])
            for event in event_list
            if event.line_number > boundary_line and path in event.changes
        ]
        anchors = [
            (event, change)
            for event, change in path_events
            if isinstance(change.get("content"), str)
        ]
        if not anchors:
            rows.append({"path": path, "recovered": False, "reason": "no_content_anchor"})
            continue
        # A later replacement can contain a tiny ``add`` snapshot after the
        # full-content deletion that is needed to reverse the prior patch
        # chain. Prefer the largest retained content as the stable anchor.
        anchor_event, anchor_change = max(
            anchors,
            key=lambda item: len(str(item[1]["content"])),
        )
        current = str(anchor_change["content"])
        reversed_patch_count = 0
        intervening_updates = [
            (event, change)
            for event, change in path_events
            if event.line_number < anchor_event.line_number and change.get("type") == "update"
        ]
        for event, change in reversed(intervening_updates):
            unified_diff = change.get("unified_diff")
            if not isinstance(unified_diff, str):
                raise ValueError(f"source update at session line {event.line_number} has no diff")
            current = reverse_apply_unified_diff(current, unified_diff)
            reversed_patch_count += 1
        encoded = current.encode("utf-8")
        digest = _sha256(encoded)
        output_path = f"{output_prefix}/{digest}/{path}"
        artifacts[output_path] = encoded
        rows.append(
            {
                "path": path,
                "recovered": True,
                "sha256": digest,
                "bytes": len(encoded),
                "boundary_session_line": boundary_line,
                "content_anchor_session_line": anchor_event.line_number,
                "reversed_patch_count": reversed_patch_count,
                "receipt_hash_bound": False,
                "output_path": output_path,
            }
        )
    return rows, artifacts


def build_recovery_bundle(
    session_path: Path = DEFAULT_SESSION_PATH,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    events = read_patch_events(session_path)
    runner_matches, runner_artifacts, anchor = reconstruct_runner_versions(events)
    aor_match, aor_content = recover_full_content_target(
        events, AOR_WORKER_PATH, AOR_WORKER_SHA256
    )
    smp_action_match, smp_action_content = recover_full_content_target(
        events, SMP_ACTION_PATH, SMP_ACTION_SHA256
    )
    campaign_sources, campaign_artifacts = recover_retained_campaign_sources(events)
    optimizer_sources, optimizer_artifacts = recover_sources_at_boundary(
        events,
        EXP052_OPTIMIZER_PATHS,
        boundary_line=EXP052_FORMAL_START_SESSION_LINE,
        output_prefix="exp052-formal-start-optimizer-sources",
    )
    artifacts = {**runner_artifacts, **campaign_artifacts, **optimizer_artifacts}
    if aor_match is not None and aor_content is not None:
        artifacts[aor_match["output_path"]] = aor_content
    if smp_action_match is not None and smp_action_content is not None:
        artifacts[smp_action_match["output_path"]] = smp_action_content

    exact_sources = {
        "AOR": aor_match,
        "CTP": runner_matches.get("CTP/GCB"),
        "GCB": runner_matches.get("CTP/GCB"),
        "SMP": runner_matches.get("SMP"),
    }
    report = {
        "schema_version": "arac-session-source-recovery-v1",
        "session_path": str(session_path),
        "source_policy": "session_patch_reconstruction_isolated_from_production",
        "patch_event_count": len(events),
        "runner_deletion_anchor": anchor,
        "runner_versions": runner_matches,
        "exact_sources": exact_sources,
        "all_exact_sources_recovered": all(exact_sources.values()),
        "supporting_sources": {"SMP action module": smp_action_match},
        "exp052_optimizer_sources": optimizer_sources,
        "all_exp052_optimizer_sources_recovered": all(
            row["recovered"] for row in optimizer_sources
        ),
        "retained_campaign_sources": campaign_sources,
        "replay_authorized": False,
        "remaining_blocker": "historical_optimizer_dependency_closure_and_environment_binding_not_recovered",
        "decision": "exact_runner_sources_recovered_hold_performance_experiments",
    }
    return report, artifacts


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Session source recovery",
        "",
        "The retained Codex session contains enough patch evidence to recover the",
        "execution-critical historical runner bytes without restoring them into the",
        "production runtime.",
        "",
        f"- Session: `{report['session_path']}`",
        f"- Patch events parsed: `{report['patch_event_count']}`",
        f"- Retained campaign files: `{len(report['retained_campaign_sources'])}`",
        "- Performance experiments started: **no**",
        "",
        "| Lane | Exact execution source | SHA-256 | Recovery method |",
        "|---|---|---|---|",
    ]
    for lane, source in report["exact_sources"].items():
        if source is None:
            lines.append(f"| {lane} | no | - | - |")
            continue
        method = "full session content" if lane == "AOR" else "reverse patch chain"
        lines.append(f"| {lane} | **yes** | `{source['sha256']}` | {method} |")
    lines.extend(
        [
            "",
            "## Gate status",
            "",
            "All four execution-source hashes are recovered exactly. EXP-052 starts",
            "from a fresh seeded run rather than an external checkpoint. Replay remains",
            "blocked because the historical dependency closure, optimizer lifecycle,",
            "and numerical environment have not yet been proven complete.",
            "",
        ]
    )
    return "\n".join(lines)


def write_recovery_bundle(
    report: dict[str, Any],
    artifacts: dict[str, bytes],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    json_path: Path = DEFAULT_OUTPUT_JSON,
    markdown_path: Path = DEFAULT_OUTPUT_MD,
) -> None:
    for relative_path, content in artifacts.items():
        destination = Path(output_root) / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    report, artifacts = build_recovery_bundle(args.session)
    write_recovery_bundle(report, artifacts, args.output_root, args.json, args.markdown)
    if args.check:
        if not report["all_exact_sources_recovered"]:
            raise SystemExit("one or more historical execution sources were not recovered")
        for source in report["exact_sources"].values():
            recovered = args.output_root / source["output_path"]
            if _sha256(recovered.read_bytes()) != source["sha256"]:
                raise SystemExit(f"materialized source hash mismatch: {recovered}")
        for source in report["supporting_sources"].values():
            if source is None:
                raise SystemExit("one or more historical supporting sources were not recovered")
            recovered = args.output_root / source["output_path"]
            if _sha256(recovered.read_bytes()) != source["sha256"]:
                raise SystemExit(f"materialized source hash mismatch: {recovered}")
        if not report["all_exp052_optimizer_sources_recovered"]:
            raise SystemExit("one or more EXP-052 optimizer sources were not recovered")
        for source in report["exp052_optimizer_sources"]:
            recovered = args.output_root / source["output_path"]
            if _sha256(recovered.read_bytes()) != source["sha256"]:
                raise SystemExit(f"materialized source hash mismatch: {recovered}")
        print("Exact historical execution sources recovered: AOR, CTP, GCB, SMP")
        print(f"Wrote {args.json}")
        print(f"Wrote {args.markdown}")
        print(f"Isolated source root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
