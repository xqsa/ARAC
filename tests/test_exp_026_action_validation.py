from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from experiments.pilots.exp_026_arac_vs_hcc_paired import run as exp026
from scripts import hcc_smoke_runner


def _config() -> dict[str, object]:
    return exp026.load_config(exp026.DEFAULT_CONFIG_PATH)


def _result(arm: str, case: str, seed: int, error: float) -> dict[str, object]:
    return {
        "arm": arm,
        "case": case,
        "seed": seed,
        "ok": True,
        "final_error": error,
    }


def _paired_results(*, action_multiplier: float = 0.9) -> list[dict[str, object]]:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    arm_a = execution["arm_a"]
    arm_b = execution["arm_b"]
    assert isinstance(arm_a, dict) and isinstance(arm_b, dict)
    rows: list[dict[str, object]] = []
    for case in exp026.SUPPORTED_CASES:
        for seed in exp026.VALIDATION_SEEDS:
            rows.append(_result(str(arm_a["label"]), case, seed, 100.0))
            rows.append(
                _result(str(arm_b["label"]), case, seed, 100.0 * action_multiplier)
            )
    return rows


def _analyze(results: list[dict[str, object]], *, replicates: int = 100) -> dict[str, object]:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    arm_a = execution["arm_a"]
    arm_b = execution["arm_b"]
    assert isinstance(arm_a, dict) and isinstance(arm_b, dict)
    return exp026.build_paired_analysis(
        results,
        native_label=str(arm_a["label"]),
        action_label=str(arm_b["label"]),
        expected_cases=exp026.SUPPORTED_CASES,
        expected_seeds=exp026.VALIDATION_SEEDS,
        bootstrap_replicates=replicates,
        bootstrap_seed=2026071901,
        material_positive_multiplier=1.01,
        catastrophic_multiplier=1.20,
    )


def test_config_is_fixed_action_validation_without_selector() -> None:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)

    assert tuple(execution["cases"]) == exp026.SUPPORTED_CASES
    assert tuple(execution["seeds"]) == exp026.VALIDATION_SEEDS
    assert execution["max_fes"] >= 300_000
    assert execution["arm_a"]["group_optimizer_mode"] == "full_cmaes"
    assert execution["arm_b"]["group_optimizer_mode"] == "diagonal_covariance"
    assert execution["arm_a"]["enable_relation_dispatch"] is False
    assert execution["arm_b"]["enable_relation_dispatch"] is False
    assert "full action library" not in config["description"].lower()
    assert "action_bandit" in config["forbidden_runtime_inputs"]


@pytest.mark.parametrize("arm_key", ["arm_a", "arm_b"])
@pytest.mark.parametrize("case", exp026.SUPPORTED_CASES)
def test_all_arm_commands_pass_the_real_runner_parser(
    arm_key: str,
    case: str,
    tmp_path: Path,
) -> None:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    arm = execution[arm_key]
    assert isinstance(arm, dict)
    command = exp026.build_command(
        str(arm["label"]),
        arm,
        case,
        117,
        config,
        tmp_path,
        "python",
    )

    parsed = hcc_smoke_runner.parse_args(list(command[2:]))

    assert parsed.group_optimizer_mode == arm["group_optimizer_mode"]
    assert parsed.enable_relation_dispatch is False
    assert parsed.evidence_overlay_mode == "off"
    assert parsed.runtime_probe_repair_mode == "hard_repair"


def test_config_rejects_case_outside_runner_contract(tmp_path: Path) -> None:
    payload = deepcopy(_config())
    payload["execution"]["cases"] = ["E2", *exp026.SUPPORTED_CASES[1:]]
    path = tmp_path / "bad-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported AOB cases"):
        exp026.load_config(path)


def test_summary_is_read_only_from_exact_runner_path(tmp_path: Path) -> None:
    config = _config()
    execution = config["execution"]
    assert isinstance(execution, dict)
    arm = execution["arm_b"]
    assert isinstance(arm, dict)
    exact_path = exp026.expected_summary_path(
        tmp_path, config, str(arm["label"]), "S5", 117
    )
    wrong_path = exp026.run_directory(tmp_path, str(arm["label"]), "S5", 117)
    wrong_path.mkdir(parents=True)
    (wrong_path / "run_summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="exact path"):
        exp026.read_run_summary(
            exact_path,
            expected_case="S5",
            expected_seed=117,
            expected_max_fes=300_000,
            expected_optimizer_mode="diagonal_covariance",
        )

    exact_path.parent.mkdir(parents=True)
    exact_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        exp026.read_run_summary(
            exact_path,
            expected_case="S5",
            expected_seed=117,
            expected_max_fes=300_000,
            expected_optimizer_mode="diagonal_covariance",
        )

    exact_path.write_text(
        json.dumps(
            {
                "protocol_version": "hcc-run-summary-v1",
                "problem_id": "S5",
                "seed": 117,
                "configured_max_fes": 300_000,
                "fitness_evaluations": 300_000,
                "final_error": 12.5,
                "group_optimizer_mode": "diagonal_covariance",
            }
        ),
        encoding="utf-8",
    )

    assert exp026.read_run_summary(
        exact_path,
        expected_case="S5",
        expected_seed=117,
        expected_max_fes=300_000,
        expected_optimizer_mode="diagonal_covariance",
    )["final_error"] == 12.5


def test_paired_delta_is_positive_when_action_improves() -> None:
    assert exp026.paired_delta(100.0, 50.0) == pytest.approx(math.log(2.0))
    assert exp026.paired_delta(50.0, 100.0) == pytest.approx(-math.log(2.0))


def test_case_seed_bootstrap_is_reproducible() -> None:
    first = _analyze(_paired_results(), replicates=200)
    second = _analyze(_paired_results(), replicates=200)

    assert first["case_macro_mean_delta"] == pytest.approx(math.log(1.0 / 0.9))
    assert first["case_macro_mean_delta_lcb"] == second["case_macro_mean_delta_lcb"]
    assert first["case_macro_mean_delta_ucb"] == second["case_macro_mean_delta_ucb"]
    assert first["decision"] == "candidate_for_broader_action_validation"


def test_catastrophic_boundary_is_a_hard_rejection() -> None:
    analysis = _analyze(_paired_results(action_multiplier=1.20))

    assert analysis["catastrophic_count"] == 25
    assert analysis["catastrophic_rate"] == 1.0
    assert analysis["decision"] == "reject_action_catastrophic_loss"


def test_incomplete_pair_set_fails_closed() -> None:
    results = _paired_results()
    results.pop()

    with pytest.raises(ValueError, match="incomplete"):
        _analyze(results)
