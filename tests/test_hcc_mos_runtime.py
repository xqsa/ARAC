from __future__ import annotations

import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arac.backends.hcc import HccAobExecutionRequest, build_hcc_aob_smoke_command
from arac.backends.hcc import HccAobExecutionResult, resolve_hcc_vendor_paths
from arac.backends.hcc_mos_cma import create_hcc_cmaes


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "hcc_smoke_runner.py"
V37_ACTION = "arac_evidence_action_controller_v37"


def _load_runner():
    vendor_root = ROOT / "vendor" / "hcc"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    module_name = "hcc_smoke_runner_for_mos_runtime_test"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _mos_cli_args() -> list[str]:
    return [
        "--functions",
        "elliptic",
        "--ids",
        "2",
        "--output-root",
        "out",
        "--seed",
        "1",
        "--max-fes",
        "5000",
        "--lane-profile",
        "v37_mos_sampling",
        "--arac-action",
        V37_ACTION,
        "--enable-relation-dispatch",
        "--relation-policy",
        "controller_v31",
        "--cma-sampling-mode",
        "mirrored_orthogonal",
        "--skip-plots",
    ]


def _request(tmp_path: Path, **changes: object) -> HccAobExecutionRequest:
    request = HccAobExecutionRequest(
        problem_id="E2",
        seed=1,
        max_fes=5_000,
        output_dir=tmp_path,
        config_name="v37_mos_sampling",
        arac_action=V37_ACTION,
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
        cma_sampling_mode="mirrored_orthogonal",
        skip_plots=True,
    )
    return replace(request, **changes)


def test_mos_cli_accepts_only_the_frozen_profile() -> None:
    runner = _load_runner()

    args = runner.parse_args(_mos_cli_args())

    assert args.lane_profile == "v37_mos_sampling"
    assert args.cma_sampling_mode == "mirrored_orthogonal"
    assert args.arac_action == V37_ACTION
    assert args.seed == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--budget-accounting", "source"],
        ["--no-cmaes-restart"],
        ["--no-mmes-restart"],
        ["--search-state-backend", "diagonal_cma"],
        ["--precision-response-arm", "a1_probe_only"],
        ["--hypergraph-trace-mode", "observer"],
        ["--relation-policy", "adaptive_v26"],
        ["--early-stopping-evaluations", "999"],
    ],
)
def test_mos_cli_fails_closed_for_frozen_boundary_changes(extra: list[str]) -> None:
    runner = _load_runner()

    with pytest.raises(SystemExit):
        runner.parse_args([*_mos_cli_args(), *extra])


def test_mos_cli_requires_explicit_seed_and_v37_action() -> None:
    runner = _load_runner()
    without_seed = _mos_cli_args()
    seed_index = without_seed.index("--seed")
    del without_seed[seed_index : seed_index + 2]
    wrong_action = _mos_cli_args()
    wrong_action[wrong_action.index(V37_ACTION)] = "conservative_no_action"
    without_dispatch = _mos_cli_args()
    without_dispatch.remove("--enable-relation-dispatch")

    with pytest.raises(SystemExit):
        runner.parse_args(without_seed)
    with pytest.raises(SystemExit):
        runner.parse_args(wrong_action)
    with pytest.raises(SystemExit):
        runner.parse_args(without_dispatch)


def test_backend_propagates_mos_mode_and_profile(tmp_path: Path) -> None:
    command = build_hcc_aob_smoke_command(_request(tmp_path))

    mode_index = command.argv.index("--cma-sampling-mode")
    profile_index = command.argv.index("--lane-profile")
    assert command.argv[mode_index + 1] == "mirrored_orthogonal"
    assert command.argv[profile_index + 1] == "v37_mos_sampling"
    early_index = command.argv.index("--early-stopping-evaluations")
    assert command.argv[early_index + 1] == "1000"


def test_backend_accepts_paired_iid_arm_but_rejects_unscoped_mos(
    tmp_path: Path,
) -> None:
    iid = build_hcc_aob_smoke_command(
        _request(tmp_path / "iid", cma_sampling_mode="iid")
    )
    assert iid.argv[iid.argv.index("--cma-sampling-mode") + 1] == "iid"

    with pytest.raises(ValueError, match="v37_mos_sampling"):
        build_hcc_aob_smoke_command(
            _request(tmp_path / "unscoped", config_name="quick_smoke")
        )
    with pytest.raises(ValueError, match="frozen v37"):
        build_hcc_aob_smoke_command(
            _request(
                tmp_path / "wrong-action",
                cma_sampling_mode="iid",
                arac_action="conservative_no_action",
            )
        )


def test_backend_rejects_mos_profile_identity_drift(tmp_path: Path) -> None:
    for changes in (
        {"enable_relation_dispatch": False},
        {"relation_policy_mode": "adaptive_v26"},
        {"early_stopping_evaluations": 999},
    ):
        with pytest.raises(ValueError, match="v37_mos_sampling"):
            build_hcc_aob_smoke_command(_request(tmp_path, **changes))


def test_legacy_backend_command_has_no_mos_flags(tmp_path: Path) -> None:
    command = build_hcc_aob_smoke_command(
        HccAobExecutionRequest(
            problem_id="E2",
            seed=1,
            max_fes=5_000,
            output_dir=tmp_path,
        )
    )

    assert "--cma-sampling-mode" not in command.argv
    assert "--lane-profile" not in command.argv
    assert "--early-stopping-evaluations" not in command.argv


def test_runner_source_injects_mos_only_into_primary_and_phase_rescue() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert source.count("options_cc[CMA_SAMPLING_MODE_OPTION]") == 1
    assert source.count("rescue_options[CMA_SAMPLING_MODE_OPTION]") == 1
    assert "restart_options[CMA_SAMPLING_MODE_OPTION]" not in source
    assert source.count("create_hcc_cmaes(problem_cc, options_cc)") == 1
    assert source.count("create_hcc_cmaes(problem_cc, rescue_options)") == 1
    assert '"v37_primary_group_cma"' in source
    assert '"v37_phase_rescue_multistart_cma"' in source
    assert "MMES(problem, options)" in source


@pytest.mark.parametrize("is_restart", [False, True])
def test_iid_factory_is_numerically_bit_equivalent_to_vendor(
    is_restart: bool,
) -> None:
    from HCC.OPT.CMAES.cmaes import CMAES as VendorCMAES

    dimension = 4

    def sphere(values: np.ndarray) -> np.ndarray:
        array = np.atleast_2d(values)
        return np.sum(array * array, axis=1)

    problem = {
        "fitness_function": sphere,
        "ndim_problem": dimension,
        "lower_boundary": -5.0 * np.ones(dimension),
        "upper_boundary": 5.0 * np.ones(dimension),
    }
    options = {
        "max_function_evaluations": 31,
        "mean": (np.zeros(dimension),),
        "sigma": 0.5,
        "n_individuals": 6,
        "is_restart": is_restart,
        "stagnation": 1,
        "fitness_diff": 1e-12,
        "verbose": 0,
        "seed_rng": 73,
    }

    expected = VendorCMAES(dict(problem), dict(options)).optimize()
    actual = create_hcc_cmaes(problem, options).optimize()

    for key in (
        "best_so_far_y",
        "n_function_evaluations",
        "_n_generations",
        "_n_restart",
        "sigma",
    ):
        assert actual[key] == expected[key]
    for key in ("best_so_far_x", "mean", "p_s", "p_c"):
        assert np.array_equal(actual[key], expected[key])


@pytest.mark.parametrize(
    "changes",
    [
        {"enable_relation_dispatch": False},
        {"relation_policy_mode": "adaptive_v26"},
        {"early_stopping_evaluations": 999},
        {"sigma": 0.4},
    ],
)
def test_runtime_rejects_mos_profile_identity_drift_before_benchmark(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    runner = _load_runner()
    config = runner.SmokeConfig(
        max_fes=5_000,
        seed=1,
        arac_action=V37_ACTION,
        enable_relation_dispatch=True,
        relation_policy_mode="controller_v31",
        lane_profile="v37_mos_sampling",
        cma_sampling_mode="mirrored_orthogonal",
    )

    with pytest.raises(ValueError, match="v37_mos_sampling"):
        runner.run_problem(
            "elliptic",
            2,
            tmp_path,
            replace(config, **changes),
        )


def test_zero_row_sampling_audit_and_branch_provenance_have_stable_schemas(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    sampling_path = tmp_path / "mos_sampling_audit.csv"
    provenance_path = tmp_path / "mos_branch_provenance.csv"
    provenance = {
        field: ""
        for field in runner.MOS_BRANCH_PROVENANCE_FIELDS
    }

    runner._write_mos_sampling_audit(sampling_path, [])
    runner._write_mos_branch_provenance(provenance_path, provenance)

    with sampling_path.open(newline="", encoding="utf-8") as handle:
        sampling_rows = list(csv.DictReader(handle))
        assert tuple(handle.seek(0) or next(csv.reader(handle))) == (
            runner.MOS_SAMPLING_AUDIT_FIELDS
        )
    with provenance_path.open(newline="", encoding="utf-8") as handle:
        provenance_rows = list(csv.DictReader(handle))
    assert sampling_rows == []
    assert len(provenance_rows) == 1
    assert tuple(provenance_rows[0]) == runner.MOS_BRANCH_PROVENANCE_FIELDS


def _write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _offline_result(
    root: Path,
    *,
    sampling_path: Path | None = None,
    sampling_rows: int = 0,
) -> HccAobExecutionResult:
    return HccAobExecutionResult(
        problem_id="E2",
        seed=1,
        max_fes=5_000,
        final_error=1.0,
        fe_used=5_000,
        time_seconds=0.1,
        output_root=root,
        fresh_optimizer_execution=True,
        status="completed",
        result_source="hcc_subprocess_smoke_execution",
        mos_sampling_audit_path=sampling_path,
        mos_sampling_audit_rows=sampling_rows,
    )


def _raw_sampling_row(exp, *, restart: int) -> dict[str, object]:
    return {
        "run_id": "exp_003_hcc_runtime_consumer_smoke-E2-seed1-a1_v37_mos",
        "sampling_mode": "mirrored_orthogonal",
        "problem_id": "E2",
        "seed": 1,
        "outer_iter": 0,
        "group_index": 0,
        "cma_scope": "v37_primary_group_cma",
        "candidate_index": 0,
        "optimizer_seed": 1007,
        "optimizer_restart_index": restart,
        "generation": 0,
        "population": 4 * (2**restart),
        "dimension": 3,
        "pair_count": 2 * (2**restart),
        "block_count": 1 if restart == 0 else 2,
        "raw_draw_sha256": f"{restart + 1}" * 64,
        "sample_sha256": f"{restart + 3}" * 64,
        "max_orthogonality_error": 1e-14,
        "rng_draw_count": 12 * (2**restart),
        "evaluated_count": 4 * (2**restart),
        "complete_population": True,
    }


def test_exp003_mos_profile_and_source_bundle_are_frozen() -> None:
    from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp

    lanes = exp.lanes_for_profile("v37_mos_sampling")
    bundle = exp._mos_execution_source_bundle(
        resolve_hcc_vendor_paths(exp.HCC_VENDOR_ROOT)
    )

    assert tuple((lane.lane_id, lane.cma_sampling_mode) for lane in lanes) == (
        ("a0_v37_iid", "iid"),
        ("a1_v37_mos", "mirrored_orthogonal"),
    )
    assert "src/arac/backends/hcc_mos_cma.py" in bundle["files"]
    assert len(str(bundle["bundle_sha256"])) == 64


@pytest.mark.parametrize(
    ("stage", "cases", "seeds", "max_fes"),
    [
        ("cli_smoke", ("E1", "E2"), (1, 2), 5_000),
        ("trace_smoke", ("A4",), (1,), 100_000),
        (
            "development",
            tuple(
                f"{family}{index}"
                for family in ("E", "S", "R", "A")
                for index in range(1, 7)
            ),
            (96, 97, 98, 99, 100),
            3_000_000,
        ),
        (
            "confirmation",
            tuple(
                f"{family}{index}"
                for family in ("E", "S", "R", "A")
                for index in range(1, 7)
            ),
            (101, 102, 103, 104, 105, 106, 107, 108),
            3_000_000,
        ),
    ],
)
def test_exp003_mos_stage_matrices_are_preregistered(
    stage: str,
    cases: tuple[str, ...],
    seeds: tuple[int, ...],
    max_fes: int,
) -> None:
    from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp

    config = exp._validate_mos_stage_matrix(
        stage=stage,
        problem_ids=cases,
        seeds=seeds,
        max_fes=max_fes,
        worker_count=24,
        budget_accounting="strict",
        cmaes_restart=True,
        mmes_restart=True,
        search_state_backend="phase_i_mmes",
    )

    assert config["protocol_version"] == exp.MOS_STABILITY_PROTOCOL_VERSION


def test_exp003_mos_stage_requires_jobs_24() -> None:
    from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp

    with pytest.raises(ValueError, match="jobs=24"):
        exp._validate_mos_stage_matrix(
            stage="cli_smoke",
            problem_ids=("E1", "E2"),
            seeds=(1, 2),
            max_fes=5_000,
            worker_count=1,
            budget_accounting="strict",
            cmaes_restart=True,
            mmes_restart=True,
            search_state_backend="phase_i_mmes",
        )


def test_exp003_rechecks_mos_sources_after_manifest_write() -> None:
    from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp

    source = Path(exp.__file__).read_text(encoding="utf-8")
    manifest_call = source.rindex("    _write_manifest(\n")
    final_source_check = source.rindex("        _require_mos_source_unchanged(\n")

    assert final_source_check > manifest_call


def test_exp003_sampling_aggregation_preserves_restart_rows_and_checks_count(
    tmp_path: Path,
) -> None:
    from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp

    iid_lane, mos_lane = exp.lanes_for_profile("v37_mos_sampling")
    source_path = tmp_path / "mos" / "mos_sampling_audit.csv"
    source_rows = [
        _raw_sampling_row(exp, restart=0),
        _raw_sampling_row(exp, restart=1),
    ]
    _write_rows(source_path, exp.MOS_SAMPLING_AUDIT_SOURCE_FIELDS, source_rows)
    records = [
        {"lane": iid_lane, "result": _offline_result(tmp_path / "iid")},
        {
            "lane": mos_lane,
            "result": _offline_result(
                tmp_path / "mos",
                sampling_path=source_path,
                sampling_rows=2,
            ),
        },
    ]

    aggregated = exp._mos_sampling_audit_rows(records)

    assert len(aggregated) == 2
    assert [int(row["optimizer_restart_index"]) for row in aggregated] == [0, 1]
    assert [int(row["generation"]) for row in aggregated] == [0, 0]
    records[1]["result"] = replace(records[1]["result"], mos_sampling_audit_rows=1)
    with pytest.raises(ValueError, match="row count mismatch"):
        exp._mos_sampling_audit_rows(records)


def _raw_branch_row(exp, lane_id: str, mode: str, generations: int) -> dict[str, object]:
    return {
        "protocol_version": exp.MOS_STABILITY_PROTOCOL_VERSION,
        "run_id": f"{exp.RUN_ID}-E2-seed1-{lane_id}",
        "sampling_mode": mode,
        "problem_id": "E2",
        "seed": 1,
        "status": "completed",
        "terminal_target_fe": 5_000,
        "terminal_completion_tolerance_fe": 20,
        "phase_i_fe": 4_000,
        "phase_i_record_sha256": "1" * 64,
        "phase_i_candidate_sha256": "2" * 64,
        "first_cma_prestate_status": "observed",
        "first_cma_prestate_sha256": "3" * 64,
        "rng_descriptor_sha256": "4" * 64,
        "terminal_record_sha256": ("5" if mode == "iid" else "6") * 64,
        "mos_generation_rows": generations,
        "mos_primary_generation_rows": generations,
        "mos_rescue_generation_rows": 0,
    }


def test_exp003_branch_provenance_rejects_prefix_mismatch(tmp_path: Path) -> None:
    from experiments.pilots.exp_003_hcc_runtime_consumer_smoke import run as exp

    lanes = exp.lanes_for_profile("v37_mos_sampling")
    records: list[dict[str, object]] = []
    for lane, generation_count in zip(lanes, (0, 2)):
        root = tmp_path / lane.lane_id
        source = root / "mos_branch_provenance.csv"
        _write_rows(
            source,
            exp.MOS_BRANCH_PROVENANCE_SOURCE_FIELDS,
            [_raw_branch_row(exp, lane.lane_id, lane.cma_sampling_mode, generation_count)],
        )
        records.append({"lane": lane, "result": _offline_result(root)})
    branches = exp._mos_branch_provenance_rows(
        records,
        source_git_commit="c" * 40,
        source_bundle_sha256="a" * 64,
        runtime_environment_sha256="b" * 64,
    )
    sampling = [
        {**_raw_sampling_row(exp, restart=0), "lane_id": "a1_v37_mos"},
        {**_raw_sampling_row(exp, restart=1), "lane_id": "a1_v37_mos"},
    ]

    exp._validate_mos_pair_provenance(branches, sampling)
    branches[1]["first_cma_prestate_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="prefix mismatch"):
        exp._validate_mos_pair_provenance(branches, sampling)
