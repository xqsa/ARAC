from __future__ import annotations

from experiments.analyze_aob_preservation import summarize_v2, summarize_v3


def test_summarize_v2_exposes_same_action_and_probe_tax() -> None:
    rows = [
        {
            "context_id": "aob_A1_s1",
            "case_id": "A1",
            "run_seed": 1,
            "phase1_fes": 2500,
            "structural_inference_complete": 0.0,
            "method": "probe_commit_v2",
            "selected_action": "aor",
            "final_error": 11.0,
            "selected_ledger_fes": 38464,
            "selected_action_fes": 35964,
            "selection_reason": "probe_cap_insufficient_margin",
            "branch_probe_fes": 512,
            "global_max_fes": 40000,
        },
        {
            "context_id": "aob_A1_s1",
            "case_id": "A1",
            "run_seed": 1,
            "phase1_fes": 2500,
            "structural_inference_complete": 0.0,
            "method": "mechanism_score_v1",
            "selected_action": "aor",
            "final_error": 10.0,
            "selected_ledger_fes": 40000,
            "selected_action_fes": 37500,
            "selection_reason": "incomplete_structure",
            "branch_probe_fes": 0,
            "global_max_fes": 40000,
        },
    ]
    summary = summarize_v2(rows)
    assert summary["same_action_count"] == 1
    assert summary["probe_tax_fes"] == [1536]
    assert summary["structural_complete_counts"] == {"0.0": 1.0}
    assert summary["action_pair_counts"] == {"aor->aor": 1}


def test_summarize_v3_reports_restored_phase1_boundary() -> None:
    rows = [
        {
            "case_id": "A1",
            "run_seed": 7,
            "global_max_fes": 3000000,
            "phase1_fes": 180000,
            "structural_inference_complete": 1.0,
            "selected_action": "smp",
        },
        {
            "case_id": "E1",
            "run_seed": 7,
            "global_max_fes": 3000000,
            "phase1_fes": 180000,
            "structural_inference_complete": 1.0,
            "selected_action": "ctp",
        },
    ]
    summary = summarize_v3(rows)
    assert summary["global_max_fes"] == [3000000]
    assert summary["phase1_fes"] == [180000]
    assert summary["structural_complete_counts"] == {"1.0": 2}
    assert summary["action_counts"]["smp"] == 1
