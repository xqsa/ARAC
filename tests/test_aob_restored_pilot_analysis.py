from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.analyze_aob_restored_pilot as analysis
import experiments.phase2_v2_aob_restored_pilot as pilot
from arac.runtime.contracts import canonical_sha256


CONFIG = Path("experiments/phase2_v2_aob_restored_pilot_config.json")


def _fake_run_one(
    context: dict[str, object],
    method: str,
    config: dict[str, object],
) -> tuple[dict[str, object], int]:
    total = int(config["global_max_fes"])
    probe_tax = 4 * int(config["branch_probe_fes"])
    objective_fes = total - probe_tax if method == "mechanism_score_matched_v1" else total
    checkpoint = f"{context['case']}-{context['run_seed']}"
    result = {
        "method": method,
        "phase1": {
            "checkpoint_sha256": checkpoint,
            "phase1_fes": int(config["phase1_fes"]),
            "structural_inference_complete": 1.0,
        },
        "selected_action": "aor",
        "final_error": float(pilot.METHODS.index(method) + 1),
        "objective_fes": objective_fes,
        "global_total_fes": total,
        "reserved_probe_tax_fes": 0 if method == "mechanism_score_full_v1" else probe_tax,
        "terminal_complete": method != "mechanism_score_matched_v1",
    }
    return result, objective_fes


def _frozen_failed_campaign(monkeypatch, tmp_path) -> Path:
    root = tmp_path / "campaign"
    config_path = CONFIG.resolve()
    config = pilot.load_config(config_path)
    manifest = pilot._manifest(config_path, config, max_workers=8)
    receipt_root = pilot._prepare_campaign_root(
        root,
        config_path,
        manifest,
        resume=False,
    )
    monkeypatch.setattr(pilot, "_run_one", _fake_run_one)
    for context_index, context in enumerate(pilot._contexts(config)):
        for method in pilot.METHODS:
            pilot._write_method_receipt(
                receipt_root,
                context_index=context_index,
                context=context,
                method=method,
                config=config,
                manifest_sha256=manifest["manifest_sha256"],
            )
    context = pilot._contexts(config)[3]
    method = "probe_commit_v2"
    payload = {
        "schema_version": pilot.RECEIPT_SCHEMA,
        "status": "failed",
        "run_index": pilot._run_index(3, method),
        "context_id": pilot._context_id(context),
        "benchmark": context,
        "method": method,
        "manifest_sha256": manifest["manifest_sha256"],
        "error_type": "RuntimeError",
        "error": "optimizer sigma became non-finite or negative",
        "traceback": "synthetic traceback",
    }
    failed = {**payload, "receipt_sha256": canonical_sha256(payload)}
    pilot._write_json_atomic(
        pilot._receipt_path(receipt_root, 3, context, method),
        failed,
    )
    return root


def test_failure_audit_blocks_promotion_and_keeps_complete_triplets(monkeypatch, tmp_path) -> None:
    root = _frozen_failed_campaign(monkeypatch, tmp_path)
    output = tmp_path / "analysis"

    summary = analysis.run_analysis(input_root=root, output_root=output)

    assert summary["planned_receipts"] == 24
    assert summary["completed_receipts"] == 23
    assert summary["failed_receipts"] == 1
    assert summary["complete_context_triplets"] == 7
    assert summary["incomplete_contexts"] == ["aob_E1_s20261212"]
    assert summary["promotion_gate_passed"] is False
    assert summary["downstream_blocked"] is True
    assert summary["failure_evidence"][0]["error"] == (
        "optimizer sigma became non-finite or negative"
    )
    assert (output / "comparison_complete_contexts.csv").is_file()
    assert (output / "failures.json").is_file()


def test_failure_audit_rejects_receipt_hash_drift(monkeypatch, tmp_path) -> None:
    root = _frozen_failed_campaign(monkeypatch, tmp_path)
    receipt = root / "receipts" / "009_aob_E1_s20261212_probe_commit_v2.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["error"] = "tampered"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt hash drifted"):
        analysis.run_analysis(input_root=root, output_root=tmp_path / "analysis")
