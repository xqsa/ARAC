from __future__ import annotations

from pathlib import Path

import pytest

from arac.backends import hcc as hcc_backend
from arac.backends.hcc import load_hcc_aob_topology


def test_explicit_vendor_root_resolves_hcc_source_boundary(tmp_path: Path) -> None:
    vendor_root = tmp_path / "repo" / "vendor" / "hcc"
    (vendor_root / "AOB").mkdir(parents=True)
    (vendor_root / "HCC").mkdir()
    runner = tmp_path / "repo" / "scripts" / "hcc_smoke_runner.py"
    runner.parent.mkdir()
    runner.write_text("# test runner\n", encoding="utf-8")

    paths = hcc_backend.resolve_hcc_vendor_paths(
        vendor_root,
        repo_root=tmp_path / "repo",
    )

    assert paths.vendor_root == vendor_root.resolve()
    assert paths.aob_root == vendor_root.resolve() / "AOB"
    assert paths.hcc_root == vendor_root.resolve() / "HCC"
    assert paths.aob_data_root == vendor_root.resolve() / "AOB" / "AOBG" / "datafile"
    assert paths.runner == runner.resolve()


def test_vendor_root_requires_aob_hcc_and_explicit_runner_context(tmp_path: Path) -> None:
    vendor_root = tmp_path / "vendor-copy"
    vendor_root.mkdir()

    with pytest.raises(FileNotFoundError, match="valid HCC vendor root.*AOB"):
        hcc_backend.resolve_hcc_vendor_paths(
            vendor_root,
            runner_path=tmp_path / "runner.py",
        )
    (vendor_root / "AOB").mkdir()
    with pytest.raises(FileNotFoundError, match="valid HCC vendor root.*HCC"):
        hcc_backend.resolve_hcc_vendor_paths(
            vendor_root,
            runner_path=tmp_path / "runner.py",
        )
    (vendor_root / "HCC").mkdir()
    with pytest.raises(ValueError, match="repo_root or runner_path"):
        hcc_backend.resolve_hcc_vendor_paths(vendor_root)


def test_external_hcc_main_root_is_rejected_as_offline_only(tmp_path: Path) -> None:
    external_root = tmp_path / "HCC-main"
    (external_root / "AOB").mkdir(parents=True)
    (external_root / "HCC").mkdir()

    with pytest.raises(ValueError, match="offline-only.*vendor root"):
        hcc_backend.resolve_hcc_vendor_paths(
            external_root,
            runner_path=tmp_path / "hcc_smoke_runner.py",
        )


@pytest.mark.parametrize(
    (
        "problem_id",
        "function_name",
        "function_id",
        "dimension_real",
        "overlap_gamma",
        "overlapping_elements",
        "global_fes",
    ),
    (
        ("E1", "elliptic", 1, 1000, 0, 0, 0),
        ("E3", "elliptic", 3, 1057, 3, 57, 736_800),
        ("A4", "ackley", 4, 1095, 5, 95, 828_000),
        ("S5", "schwefel", 5, 1133, 7, 133, 919_200),
    ),
)
def test_load_hcc_aob_topology_covers_only_exp_018_cases(
    problem_id: str,
    function_name: str,
    function_id: int,
    dimension_real: int,
    overlap_gamma: int,
    overlapping_elements: int,
    global_fes: int,
) -> None:
    topology = load_hcc_aob_topology(problem_id)

    assert topology.problem_id == problem_id
    assert topology.function_name == function_name
    assert topology.function_id == function_id
    assert topology.dimension == 1000
    assert topology.dimension_real == dimension_real
    assert topology.overlap_gamma == overlap_gamma
    assert topology.group_count == 20
    assert topology.overlapping_element_count == overlapping_elements
    assert topology.degree_of_overlap == pytest.approx(overlapping_elements / 1000)
    assert topology.global_fes == global_fes
    assert topology.source_level == "hcc_source_topology"
    assert topology.fresh_optimizer_execution is False


@pytest.mark.parametrize("problem_id", ("R3", "E2", "S6", "A1"))
def test_hcc_aob_topology_rejects_cases_outside_exp_018(problem_id: str) -> None:
    with pytest.raises(ValueError, match="exp_018"):
        load_hcc_aob_topology(problem_id)
