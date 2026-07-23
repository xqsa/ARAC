from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.pilots.exp_028_wloc_baseline_suite.protocol import (
    CASE_IDS,
    METHODS,
    WLOCBaselineTask,
    build_task,
    build_task_matrix,
    load_protocol,
)
from experiments.pilots.exp_028_wloc_baseline_suite.runner import (
    run_mechanical_smoke,
    validate_artifact,
    write_task_matrix,
)


def test_frozen_protocol_expands_to_a_stable_18_by_9_matrix() -> None:
    config = load_protocol()
    first = build_task_matrix(config)
    second = build_task_matrix(config)

    assert len(first) == 18 * 9 == 162
    assert tuple(Counter(task.case_id for task in first).values()) == (9,) * 18
    assert Counter(task.method for task in first) == {method: 18 for method in METHODS}
    assert {task.dimension for task in first} == {1000}
    assert [task.task_hash for task in first] == [task.task_hash for task in second]
    assert len({task.task_hash for task in first}) == len(first)
    assert all(task.instance_seed != task.optimizer_seed for task in first)
    assert first[0] == WLOCBaselineTask.from_dict(first[0].to_dict())


def test_protocol_rejects_stale_schema_and_matrix_writer_is_stable(tmp_path: Path) -> None:
    config = load_protocol()
    tasks = build_task_matrix(config)
    first_path = write_task_matrix(tmp_path / "first.json", config, tasks)
    second_path = write_task_matrix(tmp_path / "second.json", config, tasks)

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first == second
    assert first["task_count"] == 162
    assert len(first["matrix_hash"]) == 64

    stale = json.loads(
        Path(
            "experiments/pilots/exp_028_wloc_baseline_suite/config.json"
        ).read_text(encoding="utf-8")
    )
    stale["schema_version"] = "wloc-baseline-protocol-v0"
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported WLOC baseline protocol"):
        load_protocol(stale_path)


def test_mechanical_task_marks_only_expensive_decomposers_as_supplied_topology() -> None:
    config = load_protocol()
    modes = {
        method: build_task(config, "WLOC01", method, mechanical_smoke=True).decomposition_mode
        for method in METHODS
    }

    assert modes["DG2-CMAES"] == "provided_catalog_topology_smoke"
    assert modes["RDG3-CMAES"] == "provided_catalog_topology_smoke"
    assert modes["Random-CMAES"] == "generated_random"
    assert modes["RDDSM-CMAES"] == modes["HCC-ES"] == "design_matrix"
    assert all(modes[method] == "none" for method in METHODS[4:8])


def test_one_1000d_case_runs_all_nine_methods_and_emits_valid_artifacts(
    tmp_path: Path,
) -> None:
    config = load_protocol()
    manifests = run_mechanical_smoke(config, tmp_path, case_id="WLOC01")

    assert len(manifests) == 9
    payloads = [validate_artifact(path) for path in manifests]
    assert {payload["task"]["method"] for payload in payloads} == set(METHODS)
    assert all(payload["task"]["dimension"] == 1000 for payload in payloads)
    assert all(payload["result"]["optimization_fes"] == 31 for payload in payloads)
    assert all(payload["synthetic_only"] is True for payload in payloads)
    assert all(payload["real_aob_action_gate_eligible"] is False for payload in payloads)
    assert {
        payload["grouping"]["origin"]
        for payload in payloads
        if payload["task"]["method"] in {"DG2-CMAES", "RDG3-CMAES"}
    } == {"provided_catalog_topology_smoke"}

    trace_path = manifests[0].parent / "best_so_far.csv"
    trace_path.write_text(
        trace_path.read_text(encoding="utf-8") + "32,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trace file hash mismatch"):
        validate_artifact(manifests[0])


def test_case_order_and_method_order_are_frozen() -> None:
    assert CASE_IDS == tuple(f"WLOC{index:02d}" for index in range(1, 19))
    assert METHODS == (
        "DG2-CMAES",
        "Random-CMAES",
        "RDG3-CMAES",
        "RDDSM-CMAES",
        "Sep-CMAES",
        "LM-MA-ES",
        "LMCMA",
        "MM-ES",
        "HCC-ES",
    )
