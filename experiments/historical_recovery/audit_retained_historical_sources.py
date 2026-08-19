"""Audit retained evidence for exact historical action source recovery.

This is a read-only provenance check.  It inspects Git history/reflog objects,
the local session archive, and source-like files under the repository without
checking out or restoring anything.  A result receipt is not treated as a
replayable source unless the byte-level runner hash is recovered and the other
execution-critical bindings are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from experiments.historical_recovery.recover_session_sources import (
    build_recovery_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "retained_source_recovery.json"
DEFAULT_OUTPUT_MD = REPOSITORY_ROOT / "experiments" / "historical_recovery" / "retained_source_recovery.md"
HISTORICAL_REF = "c7505d91"
SESSION_ROOT = Path(r"C:\Users\83718\.codex\sessions")
MAX_SESSION_FILE_BYTES = 64 * 1024 * 1024

LANE_TARGETS: dict[str, dict[str, Any]] = {
    "AOR": {
        "runner_sha256": "2d870d14fa536dee488d45a69abea19e50e86dc20748b026d5fc4a16afcb4165",
        "receipt": "results/exp_057_a_series_aor_25seed/a1-a6-25seed-v1/runs/A1/seed_117/execution_receipt.json",
        "source_paths": (
            "experiments/pilots/exp_057_a_series_aor_25seed/_worker.py",
            "experiments/pilots/exp_057_a_series_aor_25seed/run.py",
        ),
        "checkpoint_binding": False,
    },
    "CTP": {
        "runner_sha256": "9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594",
        "receipt": "results/exp_058_ctp_stable_v2_25seed/validation/runs/S1/seed_117/exp_058_ctp_stable_v2_25seed-s1-seed117/schwefel/exp058_execution_receipt.json",
        "source_paths": ("scripts/hcc_smoke_runner.py",),
        "checkpoint_binding": False,
    },
    "GCB": {
        "runner_sha256": "9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594",
        "receipt": "results/exp_059_gcb_stable_all_case_25seed/validation/runs/R1/seed_117/exp_059_gcb_stable_all_case_25seed-r1-seed117/rastrigin/exp059_execution_receipt.json",
        "source_paths": ("scripts/hcc_smoke_runner.py",),
        "checkpoint_binding": False,
    },
    "SMP": {
        "runner_sha256": "b17021e8ffe1de76fea48b52ed3c00a62b4cc93bf4c2c759604064d14ebc68ac",
        "receipt": "results/exp_052_e_series_smp_paired_gate/validation/runs/E1/candidate_smp/seed_117/exp_052_e_series_smp_paired_gate-e1-candidate_smp-seed117/elliptic/exp052_execution_receipt.json",
        "source_paths": ("scripts/hcc_smoke_runner.py",),
        "checkpoint_binding": False,
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )


def _git_commits() -> tuple[list[str], list[str]]:
    reflog = _git("reflog", "--all", "--format=%H")
    reflog_commits = list(dict.fromkeys(reflog.stdout.decode("utf-8", "replace").split()))
    fsck = _git("fsck", "--full", "--no-reflogs", "--unreachable")
    unreachable = []
    for line in fsck.stdout.decode("utf-8", "replace").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "commit":
            unreachable.append(fields[2])
    return reflog_commits, list(dict.fromkeys(unreachable))


def _tree_files(ref: str) -> Iterable[tuple[str, str]]:
    completed = _git("ls-tree", "-r", "--full-tree", ref)
    if completed.returncode != 0:
        return ()
    rows: list[tuple[str, str]] = []
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        header, separator, path = line.partition("\t")
        if not separator:
            continue
        fields = header.split()
        if len(fields) >= 3 and fields[1] == "blob":
            rows.append((fields[2], path))
    return rows


def _candidate_sources(refs: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {
        path
        for lane in LANE_TARGETS.values()
        for path in lane["source_paths"]
    }
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        for blob_id, path in _tree_files(ref):
            if path not in wanted or (ref, path) in seen:
                continue
            seen.add((ref, path))
            content = _git("show", f"{ref}:{path}")
            if content.returncode != 0:
                continue
            candidates.append(
                {
                    "ref": ref,
                    "path": path,
                    "git_blob_sha1": blob_id,
                    "sha256": _sha256(content.stdout),
                    "bytes": len(content.stdout),
                }
            )
    return candidates


def _all_blob_sha256_matches(targets: set[str]) -> list[dict[str, Any]]:
    """Find exact SHA-256 matches among every retained Git blob."""

    process = subprocess.Popen(
        ("git", "cat-file", "--batch-all-objects", "--batch"),
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    matches: list[dict[str, Any]] = []
    while True:
        header = process.stdout.readline()
        if not header:
            break
        fields = header.rstrip(b"\n").split()
        if len(fields) < 3:
            continue
        object_id, object_type, size_text = fields[:3]
        size = int(size_text)
        payload = process.stdout.read(size)
        process.stdout.read(1)  # batch records end with a newline
        if object_type == b"blob":
            digest = _sha256(payload)
            if digest in targets:
                matches.append(
                    {
                        "git_object": object_id.decode("ascii"),
                        "sha256": digest,
                        "bytes": size,
                    }
                )
    process.wait()
    return matches


def _session_hits(targets: set[str]) -> list[dict[str, Any]]:
    if not SESSION_ROOT.is_dir():
        return []
    hits: list[dict[str, Any]] = []
    for path in SESSION_ROOT.rglob("*.jsonl"):
        try:
            if path.stat().st_size > MAX_SESSION_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matched = sorted(target for target in targets if target in text)
        if matched:
            hits.append({"path": str(path), "matched_sha256": matched})
    return hits


def _receipt_info(relative_path: str) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": relative_path, "present": False, "payload": None}
    return {
        "path": relative_path,
        "present": isinstance(payload, dict),
        "payload": payload if isinstance(payload, dict) else None,
        "sha256": _sha256(path.read_bytes()),
    }


def _session_source_recovery() -> dict[str, Any]:
    try:
        recovery, _ = build_recovery_bundle()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
            "exact_sources": {},
        }
    return {
        "available": True,
        "session_path": recovery["session_path"],
        "source_policy": recovery["source_policy"],
        "patch_event_count": recovery["patch_event_count"],
        "all_exact_sources_recovered": recovery["all_exact_sources_recovered"],
        "exact_sources": recovery["exact_sources"],
        "supporting_sources": recovery["supporting_sources"],
        "runner_deletion_anchor": recovery["runner_deletion_anchor"],
    }


def _lane_verdict(*, receipt_present: bool, exact_source_recovered: bool, checkpoint_bound: bool) -> str:
    if receipt_present and exact_source_recovered and checkpoint_bound:
        return "recovered"
    if receipt_present or exact_source_recovered:
        return "partial"
    return "missing"


def build_report() -> dict[str, Any]:
    reflog_commits, unreachable_commits = _git_commits()
    refs = list(dict.fromkeys([HISTORICAL_REF, *reflog_commits, *unreachable_commits]))
    candidates = _candidate_sources(refs)
    targets = {item["runner_sha256"] for item in LANE_TARGETS.values()}
    blob_matches = _all_blob_sha256_matches(targets)
    session_hits = _session_hits(targets)
    session_recovery = _session_source_recovery()

    lane_rows: list[dict[str, Any]] = []
    for lane, target in LANE_TARGETS.items():
        receipt = _receipt_info(target["receipt"])
        source_candidates = [
            item
            for item in candidates
            if item["path"] in target["source_paths"]
        ]
        exact_candidates = [
            item for item in source_candidates if item["sha256"] == target["runner_sha256"]
        ]
        exact_blob_matches = [
            item for item in blob_matches if item["sha256"] == target["runner_sha256"]
        ]
        exact_session_match = session_recovery["exact_sources"].get(lane)
        exact_source_recovered = bool(
            exact_candidates or exact_blob_matches or exact_session_match
        )
        lane_rows.append(
            {
                "lane": lane,
                "runner_sha256": target["runner_sha256"],
                "receipt": receipt,
                "candidate_sources": source_candidates,
                "exact_source_candidates": exact_candidates,
                "exact_git_blob_matches": exact_blob_matches,
                "exact_session_source_match": exact_session_match,
                "checkpoint_binding_recovered": bool(target["checkpoint_binding"]),
                "verdict": _lane_verdict(
                    receipt_present=bool(receipt["present"]),
                    exact_source_recovered=exact_source_recovered,
                    checkpoint_bound=bool(target["checkpoint_binding"]),
                ),
                "replay_authorized": bool(
                    receipt["present"]
                    and exact_source_recovered
                    and target["checkpoint_binding"]
                ),
            }
        )

    return {
        "schema_version": "arac-retained-historical-source-recovery-v1",
        "historical_ref": HISTORICAL_REF,
        "source_policy": "git_scan_plus_isolated_session_patch_reconstruction_no_checkout",
        "refs_checked": refs,
        "reflog_commit_count": len(reflog_commits),
        "unreachable_commit_count": len(unreachable_commits),
        "git_candidate_source_count": len(candidates),
        "git_blob_sha256_matches": blob_matches,
        "session_hits": session_hits,
        "session_source_recovery": session_recovery,
        "lanes": lane_rows,
        "replay_authorized_lanes": [
            row["lane"] for row in lane_rows if row["replay_authorized"]
        ],
        "decision": "exact_sources_recovered_dependency_bindings_missing_hold_all_experiments",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Retained historical source recovery",
        "",
        "This report is generated by a read-only scan of Git history/reflog objects,",
        "the local Codex session archive, and result provenance. No checkout or", 
        "source restoration is performed.",
        "",
        f"- Historical ref: `{report['historical_ref']}`",
        f"- Reflog commits checked: `{report['reflog_commit_count']}`",
        f"- Unreachable commits checked: `{report['unreachable_commit_count']}`",
        f"- Candidate source entries: `{report['git_candidate_source_count']}`",
        f"- Exact Git blob matches: `{len(report['git_blob_sha256_matches'])}`",
        "- Exact session-reconstructed source lanes: "
        f"`{sum(bool(row['exact_session_source_match']) for row in report['lanes'])}/4`",
        f"- Exact replay-authorized lanes: **{', '.join(report['replay_authorized_lanes']) or 'none'}**",
        "",
        "## Lane status",
        "",
        "| Lane | Receipt | Exact runner source | Replay binding | Verdict | Replay |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["lanes"]:
        source = bool(
            row["exact_source_candidates"]
            or row["exact_git_blob_matches"]
            or row["exact_session_source_match"]
        )
        lines.append(
            f"| {row['lane']} | {'yes' if row['receipt']['present'] else 'no'} | "
            f"{'yes' if source else 'no'} | "
            f"{'yes' if row['checkpoint_binding_recovered'] else 'no'} | "
            f"**{row['verdict']}** | {'yes' if row['replay_authorized'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The exact execution-source hashes for AOR, CTP, GCB, and SMP were recovered",
            "from retained session content or a strictly validated reverse patch chain.",
            "EXP-052 starts from a fresh seeded run and has no external checkpoint file.",
            "Replay remains blocked because the historical optimizer dependency closure",
            "and numerical environment have not yet been bound and verified.",
            "",
            "Historical HCC replay, historical-level fairness claims, and selector claims",
            "remain blocked by the missing dependency and environment binding.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: Path = DEFAULT_OUTPUT_JSON, md_path: Path = DEFAULT_OUTPUT_MD) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)
    report = build_report()
    write_report(report, args.json, args.markdown)
    if args.check:
        print(f"Wrote {args.json}")
        print(f"Wrote {args.markdown}")
        print("Replay-authorized lanes:", ", ".join(report["replay_authorized_lanes"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
