from __future__ import annotations

import hashlib

import pytest

from experiments.historical_recovery.audit_exp052_smp_binding import (
    build_report as build_exp052_binding_report,
)
from experiments.historical_recovery.audit_exp052_dependency_closure import (
    build_report as build_exp052_dependency_report,
)
from experiments.historical_recovery.recover_session_sources import (
    AOR_WORKER_SHA256,
    DEFAULT_SESSION_PATH,
    PatchEvent,
    RUNNER_TARGETS,
    SMP_ACTION_SHA256,
    build_recovery_bundle,
    recover_sources_at_boundary,
    reverse_apply_unified_diff,
)


def test_reverse_apply_unified_diff_restores_multiple_hunks() -> None:
    original = "alpha\nkeep\nremove\nmiddle\nlast\n"
    updated = "alpha\ninsert\nkeep\nmiddle\nchanged\nlast\n"
    unified_diff = """@@ -1,3 +1,3 @@
 alpha
+insert
 keep
-remove
@@ -4,2 +4,3 @@
 middle
+changed
 last
"""

    assert reverse_apply_unified_diff(updated, unified_diff) == original


def test_boundary_recovery_prefers_full_content_anchor_over_later_add() -> None:
    events = [
        PatchEvent(
            timestamp="t1",
            line_number=10,
            call_id="delete",
            changes={"src/example.py": {"type": "delete", "content": "full\n"}},
        ),
        PatchEvent(
            timestamp="t2",
            line_number=11,
            call_id="add",
            changes={"src/example.py": {"type": "add", "content": "x\n"}},
        ),
    ]

    rows, artifacts = recover_sources_at_boundary(
        events,
        ["src/example.py"],
        boundary_line=1,
        output_prefix="test-boundary",
    )

    assert rows[0]["content_anchor_session_line"] == 10
    assert artifacts[rows[0]["output_path"]] == b"full\n"


@pytest.mark.skipif(
    not DEFAULT_SESSION_PATH.is_file(),
    reason="retained local Codex session is unavailable",
)
def test_retained_session_recovers_all_execution_source_hashes() -> None:
    report, artifacts = build_recovery_bundle()

    assert report["all_exact_sources_recovered"] is True
    assert report["replay_authorized"] is False
    assert len(report["retained_campaign_sources"]) == 21
    assert report["exact_sources"]["AOR"]["sha256"] == AOR_WORKER_SHA256
    assert report["supporting_sources"]["SMP action module"]["sha256"] == SMP_ACTION_SHA256
    assert report["all_exp052_optimizer_sources_recovered"] is True
    assert len(report["exp052_optimizer_sources"]) == 7
    assert all(
        source["receipt_hash_bound"] is False
        for source in report["exp052_optimizer_sources"]
    )
    for lane, target in RUNNER_TARGETS.items():
        assert report["runner_versions"][lane]["sha256"] == target
    for source in report["exact_sources"].values():
        content = artifacts[source["output_path"]]
        assert hashlib.sha256(content).hexdigest() == source["sha256"]


@pytest.mark.skipif(
    not DEFAULT_SESSION_PATH.is_file(),
    reason="retained local Codex session is unavailable",
)
def test_exp052_binding_has_no_external_checkpoint_gap() -> None:
    report = build_exp052_binding_report()

    assert report["exact_binding_complete"] is True
    assert report["external_checkpoint"]["required"] is False
    assert report["input_binding_complete"] is True
    assert {row["matching_representation"] for row in report["input_binding"]} == {
        "git_blob",
        "windows_checkout_crlf",
    }
    assert (
        report["optimizer_dependency_closure"]
        == "sources_recovered_not_receipt_hash_bound"
    )
    assert report["replay_authorized"] is False


@pytest.mark.skipif(
    not DEFAULT_SESSION_PATH.is_file(),
    reason="retained local Codex session is unavailable",
)
def test_exp052_dependency_closure_recovers_git_clean_aob_sources() -> None:
    report = build_exp052_dependency_report()

    assert report["runner_hash_matches_target"] is True
    assert report["first_level_import_count"] == 30
    assert report["status_counts"]["exact_session_boundary_source"] == 27
    assert report["status_counts"]["exact_session_source"] == 1
    assert "patch_chain_mismatch" not in report["status_counts"]
    assert report["status_counts"]["exact_git_worktree_source"] == 2
    assert "git_ref_candidate" not in report["status_counts"]
    assert report["exact_first_level_closure"] is True
    assert report["replay_authorized"] is False
