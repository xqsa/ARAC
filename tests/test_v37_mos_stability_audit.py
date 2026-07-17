from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_v37_mos_stability.py"
)
SPEC = importlib.util.spec_from_file_location("audit_v37_mos_stability", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite(path: Path, predicate, field: str, value) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    for row in rows:
        if predicate(row):
            row[field] = str(value(row) if callable(value) else value)
    _write_csv(path, rows, fields)


def _environment(root: Path) -> str:
    payload = {
        "expected": {"python": "3.12.13", "numpy": "2.3.5"},
        "observed": {"python": "3.12.13", "numpy": "2.3.5"},
        "status": "pass",
    }
    path = root / "runtime_environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return AUDIT._file_sha256(path)


def _aob_info_path(root: Path, case: str) -> Path:
    del root
    return (AUDIT.AOB_DATA_ROOT / f"F{case[1:]}-info.txt").resolve()


def _paired_dataset(
    root: Path,
    *,
    stage: str = "development",
    smoke_name: str = "cli_smoke",
    include_cma: bool = True,
    include_sampling: bool = True,
) -> Path:
    config = AUDIT._load_config()
    matrix = AUDIT._matrix(config, stage, smoke_name=smoke_name)
    paper = AUDIT._paper_best(config)
    environment_hash = _environment(root)
    results: list[dict[str, object]] = []
    sampling: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    ledgers: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    aob: list[dict[str, object]] = []
    target = int(matrix["terminal_fe"])
    actual = target - 1 if include_cma else target
    phase_i_fe = (
        min(800_000, max(0, actual - 1_000)) if include_cma else actual
    )
    source_files = {
        "experiment_runner": "1" * 64,
        "hcc_smoke_runner": "2" * 64,
        "src/arac/backends/hcc_mos_cma.py": "4" * 64,
        "vendor/hcc/OPT/CMAES/cmaes.py": "3" * 64,
    }
    source_bundle_sha256 = AUDIT._canonical_sha256(source_files)
    for case in matrix["cases"]:
        for seed in matrix["seeds"]:
            for arm_index, arm in enumerate(matrix["arms"]):
                sampling_mode = "iid" if arm_index == 0 else "mirrored_orthogonal"
                run_id = f"run-{case}-seed{seed}-{arm}"
                error = paper[case] if arm_index == 0 else 0.8 * paper[case]
                results.append(
                    {
                        "lane_id": arm,
                        "problem_id": case,
                        "seed": seed,
                        "selected_action_name": "arac_evidence_action_controller_v37",
                        "cma_sampling_mode": sampling_mode,
                        "hcc_smoke_final_error": format(error, ".17e"),
                        "hcc_smoke_fe_used": actual,
                        "hcc_smoke_status": "completed",
                        "fresh_optimizer_execution": 1,
                        "result_source": "hcc_subprocess_smoke_execution",
                        "action_trace_sha256": _sha(f"trace|{case}|{seed}|{arm}"),
                    }
                )
                ledgers.append(
                    {
                        "lane_id": arm,
                        "problem_id": case,
                        "seed": seed,
                        "phase_i_fe": phase_i_fe,
                        "total_fe": actual,
                        "budget_limit": target,
                        "configured_budget_limit": target,
                        "budget_aligned_fe_used": actual,
                        "actual_fe_used": actual,
                        "same_budget_violation": 0,
                        "fresh_execution": 1,
                    }
                )
                aob_info = _aob_info_path(root, case)
                aob.append(
                    {
                        "lane_id": arm,
                        "problem_id": case,
                        "seed": seed,
                        "path": str(aob_info),
                        "sha256_before": AUDIT._file_sha256(aob_info),
                        "sha256_after": AUDIT._file_sha256(aob_info),
                        "unchanged": 1,
                    }
                )
                if include_cma:
                    traces.append(
                        {
                            "lane_id": arm,
                            "problem_id": case,
                            "seed": seed,
                            "outer_iter": 0,
                            "group_index": 0,
                            "selected_action_name": "arac_evidence_action_controller_v37",
                            "trace_event": "group_block",
                            "best_before": "1.00000000000000000e+00",
                            "sigma_before": "5.00000000000000000e-01",
                            "population_before": 19,
                            "population_after": 19,
                            "optimizer_seed": seed * 1000 + 7,
                            "state_fingerprint_before": _sha(f"state|{case}|{seed}"),
                            "pre_hold_phase_i_tail_utility": "1.00000000000000000e-02",
                            "pre_hold_group_count": 20,
                            "pre_hold_mean_group_size": 50,
                            "pre_hold_overlap_edge_count": 10,
                            "pre_hold_shared_variable_count": 100,
                        }
                    )
                has_sampling = arm_index == 1 and include_cma and include_sampling
                if has_sampling:
                    sampling.append(
                        {
                            "run_id": run_id,
                            "lane_id": arm,
                            "sampling_mode": "mirrored_orthogonal",
                            "problem_id": case,
                            "seed": seed,
                            "outer_iter": 0,
                            "group_index": 0,
                            "cma_scope": "v37_primary_group_cma",
                            "candidate_index": 0,
                            "optimizer_seed": seed * 1000 + 7,
                            "optimizer_restart_index": 0,
                            "generation": 0,
                            "population": 20,
                            "dimension": 10,
                            "pair_count": 10,
                            "block_count": 1,
                            "raw_draw_sha256": _sha(f"raw|{case}|{seed}"),
                            "sample_sha256": _sha(f"sample|{case}|{seed}"),
                            "max_orthogonality_error": "1.0e-14",
                            "rng_draw_count": 200,
                            "evaluated_count": 20,
                            "complete_population": 1,
                        }
                    )
                provenance.append(
                    {
                        "protocol_version": AUDIT.PROTOCOL_VERSION,
                        "run_id": run_id,
                        "lane_id": arm,
                        "sampling_mode": sampling_mode,
                        "problem_id": case,
                        "seed": seed,
                        "status": "completed",
                        "terminal_target_fe": target,
                        "terminal_completion_tolerance_fe": 19,
                        "phase_i_fe": phase_i_fe,
                        "phase_i_record_sha256": _sha(f"phase|{case}|{seed}"),
                        "phase_i_candidate_sha256": _sha(
                            f"phase-candidate|{case}|{seed}"
                        ),
                        "first_cma_prestate_sha256": _sha(
                            f"prestate|{case}|{seed}"
                            if include_cma
                            else f"prestate-not-reached|{case}|{seed}"
                        ),
                        "first_cma_prestate_status": (
                            "observed" if include_cma else "not_reached"
                        ),
                        "rng_descriptor_sha256": _sha(f"rng|{case}|{seed}"),
                        "terminal_record_sha256": _sha(
                            f"terminal|{case}|{seed}|{arm}"
                        ),
                        "mos_generation_rows": int(has_sampling),
                        "mos_primary_generation_rows": int(has_sampling),
                        "mos_rescue_generation_rows": 0,
                        "source_git_commit": "a" * 40,
                        "source_bundle_sha256": source_bundle_sha256,
                        "config_sha256": AUDIT.CONFIG_SHA256,
                        "runtime_environment_sha256": environment_hash,
                    }
                )
    _write_csv(root / "our_result_by_case.csv", results, AUDIT.RESULT_COLUMNS)
    _write_csv(root / "mos_sampling_audit.csv", sampling, AUDIT.SAMPLING_COLUMNS)
    _write_csv(
        root / "mos_branch_provenance.csv", provenance, AUDIT.PROVENANCE_COLUMNS
    )
    _write_csv(root / "same_budget_ledger.csv", ledgers, AUDIT.LEDGER_COLUMNS)
    _write_csv(root / "action_trace.csv", traces, AUDIT.TRACE_COLUMNS)
    _write_csv(root / "aob_input_manifest.csv", aob, AUDIT.AOB_COLUMNS)
    _write_csv(
        root / "anti_leakage_audit.csv",
        [
            {
                "forbidden_field": "paper_best",
                "found_in_runtime_payload": 0,
                "audit_status": "pass",
            }
        ],
        AUDIT.LEAKAGE_COLUMNS,
    )
    (root / "mos_source_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": source_files,
                "bundle_sha256": source_bundle_sha256,
            }
        ),
        encoding="utf-8",
    )
    (root / "run_manifest.md").write_text(
        "\n".join(
            (
                "Parallel jobs: 24",
                "Lanes: a0_v37_iid, a1_v37_mos",
                "CMA sampling modes: a0_v37_iid=iid, a1_v37_mos=mirrored_orthogonal",
                "- git commit: " + "a" * 40,
                "- experiment runner sha256: " + "1" * 64,
                "- HCC smoke runner sha256: " + "2" * 64,
                "- CMAES optimizer sha256: " + "3" * 64,
                "- MOS sampler sha256: " + "4" * 64,
                "- MOS source bundle sha256: " + source_bundle_sha256,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _baseline_dataset(
    root: Path,
    *,
    cases: list[str],
    seeds: list[int],
    lane: str,
) -> Path:
    _environment(root)
    results: list[dict[str, object]] = []
    ledgers: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    aob: list[dict[str, object]] = []
    for case in cases:
        for seed in seeds:
            trace_hash = _sha(f"trace|{case}|{seed}")
            results.append(
                {
                    "lane_id": lane,
                    "problem_id": case,
                    "seed": seed,
                    "selected_action_name": "arac_evidence_action_controller_v37",
                    "cma_sampling_mode": "iid",
                    "hcc_smoke_final_error": "1.000000e+00",
                    "hcc_smoke_fe_used": 2_999_990,
                    "hcc_smoke_status": "completed",
                    "fresh_optimizer_execution": 1,
                    "result_source": "hcc_subprocess_smoke_execution",
                    "action_trace_sha256": trace_hash,
                }
            )
            ledgers.append(
                {
                    "lane_id": lane,
                    "problem_id": case,
                    "seed": seed,
                    "phase_i_fe": 800_000,
                    "total_fe": 2_999_990,
                    "budget_limit": 3_000_000,
                    "configured_budget_limit": 3_000_000,
                    "budget_aligned_fe_used": 2_999_990,
                    "actual_fe_used": 2_999_990,
                    "same_budget_violation": 0,
                    "fresh_execution": 1,
                }
            )
            traces.append(
                {
                    "lane_id": lane,
                    "problem_id": case,
                    "seed": seed,
                    "population_before": 19,
                    "population_after": 19,
                }
            )
            aob_info = _aob_info_path(root, case)
            aob.append(
                {
                    "lane_id": lane,
                    "problem_id": case,
                    "seed": seed,
                    "path": str(aob_info),
                    "sha256_before": AUDIT._file_sha256(aob_info),
                    "sha256_after": AUDIT._file_sha256(aob_info),
                    "unchanged": 1,
                }
            )
    _write_csv(root / "our_result_by_case.csv", results, AUDIT.RESULT_COLUMNS)
    _write_csv(root / "same_budget_ledger.csv", ledgers, AUDIT.LEDGER_COLUMNS)
    _write_csv(root / "action_trace.csv", traces, AUDIT.TRACE_COLUMNS)
    _write_csv(root / "aob_input_manifest.csv", aob, AUDIT.AOB_COLUMNS)
    _write_csv(
        root / "anti_leakage_audit.csv",
        [{"forbidden_field": "paper_best", "found_in_runtime_payload": 0, "audit_status": "pass"}],
        AUDIT.LEAKAGE_COLUMNS,
    )
    (root / "run_manifest.md").write_text(
        "Parallel jobs: 24\n", encoding="utf-8"
    )
    return root


def test_frozen_config_and_exact_binomial_boundaries() -> None:
    config = AUDIT._load_config()

    assert AUDIT._file_sha256(AUDIT.CONFIG_PATH) == AUDIT.CONFIG_SHA256
    assert config["objective"]["primary"].startswith("every_preregistered_fresh_seed")
    assert AUDIT._clopper_pearson_upper(0, 192, 0.95) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 192)
    )
    assert AUDIT._clopper_pearson_lower(8, 8, 0.95) == pytest.approx(
        0.05 ** (1.0 / 8)
    )


def test_development_pass_writes_all_six_csvs_and_jsons(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "development")

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_pass"
    assert gate["metrics"]["arms"]["a1_v37_mos"]["minimum_seed_win_count"] == 24
    assert gate["metrics"]["arms"]["a1_v37_mos"]["seed_success_cp_ucb_95"] == 1.0
    assert gate["metrics"]["paired"]["paired_catastrophic_count"] == 0
    assert gate["metrics"]["paired"]["worst_ten_percent_count"] == 12
    assert {
        "mos_branch_provenance.csv",
        "mos_source_bundle.json",
    } <= set(gate["input_artifact_sha256"])
    source_provenance = {
        (row["lane_id"], row["problem_id"], row["seed"]): row
        for row in _rows(root / "mos_branch_provenance.csv")
    }
    for derived_branch in _rows(root / "mos_branch_manifest.csv"):
        key = (
            derived_branch["lane_id"],
            derived_branch["problem_id"],
            derived_branch["seed"],
        )
        assert derived_branch["terminal_record_sha256"] == source_provenance[key][
            "terminal_record_sha256"
        ]
        assert derived_branch["terminal_completion_tolerance_fe"] == "19"
    assert all(
        row["first_cma_prestate_status_match"] == "1"
        for row in _rows(root / "mos_pair_outcomes.csv")
    )
    for name in AUDIT._load_config()["artifacts"]:
        assert (root / name).is_file()


def test_sampling_restart_identity_and_partial_generation_semantics(
    tmp_path: Path,
) -> None:
    root = _paired_dataset(tmp_path / "restart")
    rows = _rows(root / "mos_sampling_audit.csv")
    first = rows[0]
    run_id = first["run_id"]
    first["evaluated_count"] = "7"
    first["complete_population"] = "False"
    restarted = dict(first)
    restarted.update(
        {
            "optimizer_restart_index": "1",
            "evaluated_count": restarted["population"],
            "complete_population": "True",
            "raw_draw_sha256": _sha("restart-raw"),
            "sample_sha256": _sha("restart-sample"),
        }
    )
    rows.append(restarted)
    _write_csv(root / "mos_sampling_audit.csv", rows, AUDIT.SAMPLING_COLUMNS)
    _rewrite(
        root / "mos_branch_provenance.csv",
        lambda row: row["run_id"] == run_id,
        "mos_generation_rows",
        2,
    )
    _rewrite(
        root / "mos_branch_provenance.csv",
        lambda row: row["run_id"] == run_id,
        "mos_primary_generation_rows",
        2,
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_pass"


def test_formal_sampling_rejects_partial_only_run(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "partial-only")
    first_run = _rows(root / "mos_sampling_audit.csv")[0]["run_id"]
    _rewrite(
        root / "mos_sampling_audit.csv",
        lambda row: row["run_id"] == first_run,
        "evaluated_count",
        7,
    )
    _rewrite(
        root / "mos_sampling_audit.csv",
        lambda row: row["run_id"] == first_run,
        "complete_population",
        "False",
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_no_go"
    assert "no complete distribution generation" in gate["integrity"]["blockers"][0]


def test_q13_is_strict_and_a_tie_is_a_loss(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "q13_tie")
    paper = AUDIT._paper_best(AUDIT._load_config())
    cases = list(AUDIT._load_config()["cases"])
    losing_cases = set(cases[12:])
    _rewrite(
        root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos"
        and row["seed"] == "96"
        and row["problem_id"] in losing_cases,
        "hcc_smoke_final_error",
        lambda row: format(paper[row["problem_id"]], ".17e"),
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_no_go"
    assert gate["checks"]["all_seed_q13_strictly_below_one"] is False
    row = next(
        row
        for row in _rows(root / "single_seed_summary.csv")
        if row["arm"] == "a1_v37_mos" and row["seed"] == "96"
    )
    assert float(row["q13"]) == pytest.approx(1.0)
    assert row["strict_win_count"] == "12"


def test_case_mean_is_arithmetic_and_equal_to_paper_loses(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "mean_boundary")
    paper = AUDIT._paper_best(AUDIT._load_config())
    factors = {96: 0.5, 97: 0.5, 98: 1.0, 99: 1.5, 100: 1.5}
    _rewrite(
        root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos" and row["problem_id"] == "E1",
        "hcc_smoke_final_error",
        lambda row: format(paper["E1"] * factors[int(row["seed"])], ".17e"),
    )

    AUDIT.audit_v37_mos_stability(root, stage="development")
    row = next(
        row
        for row in _rows(root / "single_seed_case_summary.csv")
        if row["arm"] == "a1_v37_mos" and row["problem_id"] == "E1"
    )
    assert float(row["arithmetic_mean_error"]) == pytest.approx(paper["E1"])
    assert row["arithmetic_mean_win"] == "0"


def test_stable_core_uses_four_of_five_development_seeds(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "stable_core")
    paper = AUDIT._paper_best(AUDIT._load_config())
    _rewrite(
        root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos"
        and row["problem_id"] == "E1"
        and row["seed"] == "96",
        "hcc_smoke_final_error",
        format(1.01 * paper["E1"], ".17e"),
    )
    AUDIT.audit_v37_mos_stability(root, stage="development")
    four_of_five = next(
        row
        for row in _rows(root / "single_seed_case_summary.csv")
        if row["arm"] == "a1_v37_mos" and row["problem_id"] == "E1"
    )
    assert four_of_five["seed_win_count"] == "4"
    assert four_of_five["stable_core_case"] == "1"

    _rewrite(
        root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos"
        and row["problem_id"] == "E1"
        and row["seed"] == "97",
        "hcc_smoke_final_error",
        format(1.01 * paper["E1"], ".17e"),
    )
    AUDIT.audit_v37_mos_stability(root, stage="development")
    three_of_five = next(
        row
        for row in _rows(root / "single_seed_case_summary.csv")
        if row["arm"] == "a1_v37_mos" and row["problem_id"] == "E1"
    )
    assert three_of_five["seed_win_count"] == "3"
    assert three_of_five["stable_core_case"] == "0"


def test_common_wins_jaccard_and_paper_catastrophic_are_reported(
    tmp_path: Path,
) -> None:
    root = _paired_dataset(tmp_path / "set-stability")
    paper = AUDIT._paper_best(AUDIT._load_config())
    _rewrite(
        root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos"
        and row["problem_id"] == "E1"
        and row["seed"] == "96",
        "hcc_smoke_final_error",
        format(1.2 * paper["E1"], ".17e"),
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")
    treatment = gate["metrics"]["arms"]["a1_v37_mos"]
    e1 = next(
        row
        for row in _rows(root / "single_seed_case_summary.csv")
        if row["arm"] == "a1_v37_mos" and row["problem_id"] == "E1"
    )

    assert treatment["strict_common_win_count"] == 23
    assert treatment["pairwise_jaccard_minimum"] == pytest.approx(23 / 24)
    assert treatment["pairwise_jaccard_mean"] == pytest.approx(
        (4 * (23 / 24) + 6) / 10
    )
    assert treatment["paper_relative_catastrophic_count"] == 1
    assert e1["strict_common_win"] == "0"
    assert e1["paper_relative_catastrophic_count"] == "1"


def test_catastrophic_equality_and_zero_cvar_fail_strictly(tmp_path: Path) -> None:
    catastrophic_root = _paired_dataset(tmp_path / "catastrophic")
    paper = AUDIT._paper_best(AUDIT._load_config())
    _rewrite(
        catastrophic_root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos"
        and row["problem_id"] == "E1"
        and row["seed"] == "96",
        "hcc_smoke_final_error",
        format(1.2 * paper["E1"], ".17e"),
    )
    catastrophic = AUDIT.audit_v37_mos_stability(
        catastrophic_root, stage="development"
    )
    assert catastrophic["checks"]["maximum_paired_catastrophic_events"] is False
    pair = next(
        row
        for row in _rows(catastrophic_root / "mos_pair_outcomes.csv")
        if row["problem_id"] == "E1" and row["seed"] == "96"
    )
    assert pair["paired_catastrophic"] == "1"

    cvar_root = _paired_dataset(tmp_path / "zero_cvar")
    first_twelve = set(AUDIT._load_config()["cases"][:12])
    _rewrite(
        cvar_root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a1_v37_mos"
        and row["seed"] == "96"
        and row["problem_id"] in first_twelve,
        "hcc_smoke_final_error",
        lambda row: format(paper[row["problem_id"]], ".17e"),
    )
    zero_cvar = AUDIT.audit_v37_mos_stability(cvar_root, stage="development")
    assert zero_cvar["metrics"]["paired"]["worst_ten_percent_cvar"] == pytest.approx(0.0)
    assert zero_cvar["checks"]["worst_ten_percent_cvar_strictly_positive"] is False


def test_upper_tail_preserves_count_and_mean_log_best(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "upper_tail")
    paper = AUDIT._paper_best(AUDIT._load_config())
    _rewrite(
        root / "our_result_by_case.csv",
        lambda row: row["lane_id"] == "a0_v37_iid",
        "hcc_smoke_final_error",
        lambda row: format(0.7 * paper[row["problem_id"]], ".17e"),
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_no_go"
    assert gate["checks"]["upper_tail_paper_win_count_not_below_v37"] is True
    assert gate["checks"]["upper_tail_mean_log_normalized_best_not_above_v37"] is False


def test_raw_artifact_hashes_are_frozen_before_paper_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _paired_dataset(tmp_path / "join-order")
    events: list[str] = []
    real_raw_hashes = AUDIT._raw_hashes
    real_paper_best = AUDIT._paper_best

    def observed_raw_hashes(path: Path, names: tuple[str, ...]):
        events.append("raw_hashes")
        return real_raw_hashes(path, names)

    def observed_paper_best(config):
        events.append("paper_join")
        return real_paper_best(config)

    monkeypatch.setattr(AUDIT, "_raw_hashes", observed_raw_hashes)
    monkeypatch.setattr(AUDIT, "_paper_best", observed_paper_best)

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_pass"
    assert events.index("raw_hashes") < events.index("paper_join")
    manifest = json.loads(
        (root / "single_seed_stability_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["paper_best_joined_after_raw_hash_freeze"] is True


@pytest.mark.parametrize(
    ("artifact", "field", "value", "blocker"),
    [
        ("same_budget_ledger.csv", "same_budget_violation", "1", "ledger integrity failed"),
        ("aob_input_manifest.csv", "unchanged", "0", "AOB input changed"),
        ("mos_sampling_audit.csv", "sampling_mode", "iid", "iid arm must have zero"),
        (
            "our_result_by_case.csv",
            "cma_sampling_mode",
            "iid",
            "branch not fresh/completed",
        ),
        ("our_result_by_case.csv", "fresh_optimizer_execution", "0", "branch not fresh/completed"),
        (
            "mos_branch_provenance.csv",
            "phase_i_record_sha256",
            "invalid",
            "invalid phase_i_record_sha256",
        ),
        (
            "mos_branch_provenance.csv",
            "terminal_completion_tolerance_fe",
            "20",
            "FE provenance mismatch",
        ),
    ],
)
def test_integrity_failures_are_explicit_and_fail_closed(
    tmp_path: Path, artifact: str, field: str, value: str, blocker: str
) -> None:
    root = _paired_dataset(tmp_path / field)
    _rewrite(root / artifact, lambda row: True, field, value)

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_no_go"
    assert gate["checks"] == {"integrity_fraction": False}
    assert blocker in gate["integrity"]["blockers"][0]
    assert AUDIT.main([str(root), "--stage", "development"]) == 1


def test_aob_info_must_still_match_the_runtime_hash(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "aob-current-hash")
    _rewrite(
        root / "aob_input_manifest.csv",
        lambda row: row["path"].endswith("F1-info.txt"),
        "sha256_before",
        "a" * 64,
    )
    _rewrite(
        root / "aob_input_manifest.csv",
        lambda row: row["path"].endswith("F1-info.txt"),
        "sha256_after",
        "a" * 64,
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_no_go"
    assert "no longer matches frozen hash" in gate["integrity"]["blockers"][0]


def test_aob_info_requires_canonical_path_and_every_branch(
    tmp_path: Path,
) -> None:
    missing_root = _paired_dataset(tmp_path / "aob-info-missing")
    replacement = (AUDIT.AOB_DATA_ROOT / "F1-p.txt").resolve()
    replacement_hash = AUDIT._file_sha256(replacement)
    missing_predicate = lambda row: (
        row["lane_id"] == "a0_v37_iid"
        and row["problem_id"] == "E1"
        and row["seed"] == "96"
    )
    for field, value in (
        ("path", str(replacement)),
        ("sha256_before", replacement_hash),
        ("sha256_after", replacement_hash),
    ):
        _rewrite(
            missing_root / "aob_input_manifest.csv",
            missing_predicate,
            field,
            value,
        )

    missing = AUDIT.audit_v37_mos_stability(
        missing_root, stage="development"
    )
    assert missing["status"] == "development_no_go"
    assert "info branch coverage" in missing["integrity"]["blockers"][0]

    escaped_root = _paired_dataset(tmp_path / "aob-info-escaped")
    escaped_info = escaped_root / "F1-info.txt"
    escaped_info.write_text(
        (AUDIT.AOB_DATA_ROOT / "F1-info.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    escaped_hash = AUDIT._file_sha256(escaped_info)
    for field, value in (
        ("path", str(escaped_info.resolve())),
        ("sha256_before", escaped_hash),
        ("sha256_after", escaped_hash),
    ):
        _rewrite(
            escaped_root / "aob_input_manifest.csv",
            lambda row: row["lane_id"] == "a0_v37_iid"
            and row["problem_id"] == "E1"
            and row["seed"] == "96"
            and row["path"].endswith("F1-info.txt"),
            field,
            value,
        )
    escaped = AUDIT.audit_v37_mos_stability(
        escaped_root, stage="development"
    )
    assert escaped["status"] == "development_no_go"
    assert "escaped canonical root" in escaped["integrity"]["blockers"][0]


def test_action_trace_population_is_not_terminal_tolerance_source(
    tmp_path: Path,
) -> None:
    root = _paired_dataset(tmp_path / "trace-without-population")
    for field in ("population_before", "population_after"):
        _rewrite(root / "action_trace.csv", lambda row: True, field, "")

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_pass"


def test_confirmation_requires_unchanged_passing_development_gate(tmp_path: Path) -> None:
    development_root = _paired_dataset(tmp_path / "development")
    development = AUDIT.audit_v37_mos_stability(
        development_root, stage="development"
    )
    assert development["status"] == "development_pass"
    development_gate = development_root / "single_seed_stability_gate.json"

    confirmation_root = _paired_dataset(tmp_path / "confirmation", stage="confirmation")
    missing = AUDIT.audit_v37_mos_stability(
        confirmation_root, stage="confirmation"
    )
    assert missing["status"] == "confirmation_no_go"
    assert "--development-gate" in missing["integrity"]["blockers"][0]

    passed = AUDIT.audit_v37_mos_stability(
        confirmation_root,
        stage="confirmation",
        development_gate_path=development_gate,
    )
    assert passed["status"] == "confirmation_pass"
    assert passed["checks"]["prior_development_gate_bound"] is True
    assert passed["metrics"]["paired"]["worst_ten_percent_count"] == 20
    assert passed["runtime_registration_authorized"] is True

    with (development_root / "mos_sampling_audit.csv").open("a", encoding="utf-8") as handle:
        handle.write("changed\n")
    changed = AUDIT.audit_v37_mos_stability(
        confirmation_root,
        stage="confirmation",
        development_gate_path=development_gate,
    )
    assert changed["status"] == "confirmation_no_go"
    assert "development artifact changed" in changed["integrity"]["blockers"][0]


def test_smoke_accepts_exact_cli_matrix_and_rejects_extra_rows(tmp_path: Path) -> None:
    root = _paired_dataset(tmp_path / "smoke", stage="smoke")
    gate = AUDIT.audit_v37_mos_stability(root, stage="smoke")
    assert gate["status"] == "smoke_pass"
    assert all(row["q13"] == "" for row in _rows(root / "single_seed_summary.csv"))

    with (root / "our_result_by_case.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    extra = dict(rows[0])
    extra["seed"] = "999"
    rows.append(extra)
    _write_csv(root / "our_result_by_case.csv", rows, fields)
    failed = AUDIT.audit_v37_mos_stability(root, stage="smoke")
    assert failed["status"] == "smoke_no_go"
    assert "smoke matrix" in failed["integrity"]["blockers"][0]


def test_trace_smoke_accepts_not_reached_prestate_and_zero_sampling_rows(
    tmp_path: Path,
) -> None:
    root = _paired_dataset(
        tmp_path / "trace-smoke",
        stage="smoke",
        smoke_name="trace_smoke",
        include_cma=False,
        include_sampling=False,
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="smoke")

    assert gate["status"] == "smoke_pass"
    assert _rows(root / "mos_sampling_audit.csv") == []
    assert {
        row["first_cma_prestate_status"]
        for row in _rows(root / "mos_branch_manifest.csv")
    } == {"not_reached"}


def test_paired_manifest_requires_frozen_jobs_lane_and_sampling_order(
    tmp_path: Path,
) -> None:
    root = _paired_dataset(tmp_path / "manifest-order")
    manifest = root / "run_manifest.md"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "Parallel jobs: 24", "Parallel jobs: 23"
        ),
        encoding="utf-8",
    )

    gate = AUDIT.audit_v37_mos_stability(root, stage="development")

    assert gate["status"] == "development_no_go"
    assert "jobs/lane/sampling order" in gate["integrity"]["blockers"][0]


def test_baseline_combines_disjoint_40_and_80_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AUDIT._load_config()
    existing_matrix = config["matrices"]["baseline_existing"]
    complement_matrix = config["matrices"]["baseline_complement"]
    existing = _baseline_dataset(
        tmp_path / "existing",
        cases=list(existing_matrix["cases"]),
        seeds=list(existing_matrix["seeds"]),
        lane="hypergraph_v37_observer",
    )
    manifest = {
        "status": "pass",
        "source_git_commit": config["preimplementation_anchor"]["git_commit"],
        "source_manifest_count": 40,
        "observer_calls": {"objective": 0, "rng": 0, "optimizer": 0, "fe": 0},
        "source_bundle": {"bundle_sha256": "b" * 64},
    }
    manifest_path = existing / "hypergraph_trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    complement = _baseline_dataset(
        tmp_path / "complement",
        cases=list(complement_matrix["cases"]),
        seeds=list(complement_matrix["seeds"]),
        lane="arac_evidence_action_controller_v37",
    )
    anchor = config["preimplementation_anchor"]
    (complement / "run_manifest.md").write_text(
        "\n".join(
            ["Parallel jobs: 24"]
            + [
                str(anchor[field])
                for field in (
                    "git_commit",
                    "hcc_runner_sha256",
                    "experiment_runner_sha256",
                    "vendor_cmaes_sha256",
                )
            ]
        ),
        encoding="utf-8",
    )
    real_sha = AUDIT._file_sha256

    def frozen_manifest_sha(path: Path) -> str:
        if path.resolve() == manifest_path.resolve():
            return str(existing_matrix["manifest_sha256"])
        return real_sha(path)

    monkeypatch.setattr(AUDIT, "_file_sha256", frozen_manifest_sha)
    output = tmp_path / "baseline_audit"

    gate = AUDIT.audit_v37_mos_stability(
        output,
        stage="baseline",
        existing_root=existing,
        complement_root=complement,
    )

    assert gate["status"] == "baseline_pass"
    assert len(_rows(output / "mos_branch_manifest.csv")) == 120
    assert len(_rows(output / "single_seed_summary.csv")) == 5
    assert len(_rows(output / "single_seed_case_summary.csv")) == 24
    assert _rows(output / "mos_pair_outcomes.csv") == []

    for field in ("budget_limit", "configured_budget_limit"):
        _rewrite(
            complement / "same_budget_ledger.csv",
            lambda row: True,
            field,
            "2999990",
        )
    tampered = AUDIT.audit_v37_mos_stability(
        tmp_path / "baseline_tampered_target",
        stage="baseline",
        existing_root=existing,
        complement_root=complement,
    )
    assert tampered["status"] == "baseline_no_go"
    assert "baseline FE integrity failed" in tampered["integrity"]["blockers"][0]
