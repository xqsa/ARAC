from __future__ import annotations

import csv
from pathlib import Path

from arac.backends.hcc import HccAobExecutionRequest, HccAobExecutionResult


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fake_result(request: HccAobExecutionRequest) -> HccAobExecutionResult:
    trace_path = request.output_dir / "action_trace.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "problem_id,seed,outer_iter,group_index,selected_action_name,"
        "relation_id,group_left,group_right,shared_vars_hash,action_family,"
        "canonical_action_name,relation_policy_source,overlap_size,previous_delta,"
        "current_delta,owner_selected,semantic_surface,state_mutated,"
        "action_value_delta_norm,downstream_consumed,downstream_consumption_scope,"
        "optimizer_consumed\n"
        f"{request.problem_id},{request.seed},0,1,{request.arac_action},O0_0_1,0,1,"
        f"hash,fallback,{request.arac_action},fixed_lane_final_protocol_pilot,"
        "1,1.000000e+00,1.000000e+00,test,test,1,0.000000e+00,0,,1\n",
        encoding="utf-8",
    )
    (request.output_dir / f"{request.problem_id}_aob_input_manifest.csv").write_text(
        "problem_id,file,path,sha256_before,sha256_after,unchanged\n"
        f"{request.problem_id},F{request.problem_id[1]}-info.txt,"
        f"{request.aob_data_root / ('F' + request.problem_id[1] + '-info.txt')},"
        "test-hash,test-hash,1\n",
        encoding="utf-8",
    )
    return HccAobExecutionResult(
        problem_id=request.problem_id,
        seed=request.seed,
        max_fes=request.max_fes,
        final_error=100.0 + request.seed,
        fe_used=request.max_fes,
        time_seconds=0.1,
        output_root=request.output_dir,
        fresh_optimizer_execution=True,
        status="ok",
        result_source="test",
        action_trace_path=trace_path,
        action_trace_rows=1,
    )


def test_exp_005_defaults_to_3m_fe_3_seed_canonical_single_lane(tmp_path: Path) -> None:
    from experiments.exp_005_hcc_final_protocol_pilot.run import (
        DEFAULT_MAX_FES,
        DEFAULT_PROBLEMS,
        DEFAULT_SEEDS,
        run_hcc_final_protocol_pilot,
    )

    output = run_hcc_final_protocol_pilot(
        output_dir=tmp_path / "exp005",
        execution_runner=_fake_result,
        jobs=1,
    )

    assert DEFAULT_MAX_FES == 3_000_000
    assert DEFAULT_SEEDS == (1, 2, 3)
    assert DEFAULT_PROBLEMS == (
        "E1",
        "E2",
        "E3",
        "E4",
        "E6",
        "S2",
        "S3",
        "S6",
        "R1",
        "R2",
        "R3",
        "A4",
        "A5",
    )

    result_rows = _read_csv(output / "our_result_by_case.csv")
    assert len(result_rows) == 39
    assert {row["lane_id"] for row in result_rows} == {
        "canonical_evidence_controller_v1"
    }
    assert {row["hcc_smoke_fe_used"] for row in result_rows} == {"3000000"}

    ledger_rows = _read_csv(output / "same_budget_ledger.csv")
    assert len(ledger_rows) == 39
    assert {row["budget_limit"] for row in ledger_rows} == {"3000000"}
    assert {row["same_budget_violation"] for row in ledger_rows} == {"0"}

    manifest = (output / "run_manifest.md").read_text(encoding="utf-8")
    assert "Final protocol pilot wrapper: exp_005_hcc_final_protocol_pilot" in manifest
    assert "Lane profile: canonical_evidence_controller_v1" in manifest
    assert "Budget: 3000000 FE per lane/case" in manifest
    assert "AOB data root:" in manifest
    assert "Actual cwd:" in manifest
    assert "Wrapper Python executable:" in manifest
    assert "Backend Python executable:" in manifest
    assert "Python version:" in manifest
    assert "NumPy version:" in manifest
    assert "SciPy version:" in manifest
    assert "Torch version:" in manifest
    assert "BLAS:" in manifest
    assert "Thread environment:" in manifest
    assert "MMES optimizer sha256:" in manifest
    assert "CMAES optimizer sha256:" in manifest
    assert "AOB input hashes: aob_input_manifest.csv" in manifest

    input_rows = _read_csv(output / "aob_input_manifest.csv")
    assert len(input_rows) == 39
    assert {row["unchanged"] for row in input_rows} == {"1"}


def test_canonical_protocol_gate_rejects_input_fe_leakage_and_no_harm_failures() -> None:
    from experiments.exp_005_hcc_final_protocol_pilot.run import (
        _canonical_protocol_gate_failures,
    )

    failures = _canonical_protocol_gate_failures(
        aob_input_rows=[{"unchanged": "0"}],
        ledger_rows=[{"same_budget_violation": "1"}],
        anti_leakage_rows=[{"audit_status": "fail"}],
        action_trace_rows=[{"best_before": "10", "best_after": "11"}],
    )

    assert failures == [
        "aob_input_changed_or_missing",
        "same_budget_violation",
        "anti_leakage_violation",
        "no_harm_violation",
    ]


def test_exp_005_writes_aob_protocol_audit(tmp_path: Path) -> None:
    from experiments.exp_005_hcc_final_protocol_pilot.run import run_hcc_final_protocol_pilot

    output = run_hcc_final_protocol_pilot(
        output_dir=tmp_path / "exp005",
        execution_runner=_fake_result,
        jobs=1,
    )

    audit_rows = _read_csv(output / "aob_protocol_audit.csv")
    assert {row["file"] for row in audit_rows} == {
        "Benchmarks.py",
        "elliptic.py",
        "schwefel.py",
        "rastrigin.py",
        "ackley.py",
    }
    assert {row["runtime_matches_canonical"] for row in audit_rows} == {"0"}
    assert "0" in {row["runtime_matches_mutable_hcc_src"] for row in audit_rows}

    manifest = (output / "run_manifest.md").read_text(encoding="utf-8")
    assert "AOB protocol audit: runtime_matches_canonical=0" in manifest
    assert "Canonical protocol gate: pass" in manifest
    assert "aob_protocol_audit.csv" in manifest
