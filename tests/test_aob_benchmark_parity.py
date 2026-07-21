from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts import hcc_smoke_runner as runner
from AOB.Benchmarks import Benchmarks


HCC_MAIN_CASE_FINGERPRINTS = {
    "E1": ("elliptic", 1, 133500215.80674617),
    "E3": ("elliptic", 3, 293025394.10295475),
    "A4": ("ackley", 4, 4865.544687570594),
    "R4": ("rastrigin", 4, 462974.34727924026),
    "S5": ("schwefel", 5, 25792.95204593285),
}

HCC_MAIN_RS_FINGERPRINTS = {
    ("rastrigin", 1): 3.356099730342505e17,
    ("rastrigin", 2): 1.148300408008746e15,
    ("rastrigin", 3): 2.061415793652355e16,
    ("rastrigin", 4): 9.594792291325266e14,
    ("rastrigin", 5): 7.723101323197032e16,
    ("rastrigin", 6): 9.2860052496582e14,
    ("schwefel", 1): 1.148975106224345e18,
    ("schwefel", 2): 2.4028204918621548e16,
    ("schwefel", 3): 1.5944258454298547e17,
    ("schwefel", 4): 2.557602568449081e16,
    ("schwefel", 5): 1.9853533540159702e17,
    ("schwefel", 6): 2.003510794368549e16,
}

HCC_MAIN_DATASET_DIGESTS = {
    2: "d7a82e4ba522455984cc20c3f7c873cac06bc525fb8fec770fb2bfcce32005e7",
    6: "3417429edf5d99ddb28eee8a6aa51d77864cc3d864fd77c40db0cfee08fc4eb2",
}

HCC_MAIN_ROTATION_DIMS = {
    2: (26, 50, 51, 101),
    6: (35, 50, 60, 110),
}


def _dataset_digest(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("case_id", "function_name", "function_id", "expected"),
    [
        (case_id, function_name, function_id, expected)
        for case_id, (function_name, function_id, expected)
        in HCC_MAIN_CASE_FINGERPRINTS.items()
    ],
)
def test_vendor_aob_matches_hcc_main_1000d_contract(
    case_id: str,
    function_name: str,
    function_id: int,
    expected: float,
) -> None:
    function = runner.Benchmark(None, data_dir=runner.DATA_DIR).get_function(
        function_name,
        function_id,
    )

    assert case_id in HCC_MAIN_CASE_FINGERPRINTS
    assert function.info()["dimension"] == 1000
    assert function.dimension == 1000
    assert len(function.Ovector) == 1000
    assert len(function.Pvector) == 1000
    assert sorted(function.Pvector.tolist()) == list(range(1000))

    candidate = np.clip(
        function.Ovector + np.linspace(-0.25, 0.25, 1000),
        function.minX,
        function.maxX,
    )
    assert float(function(candidate)[0]) == pytest.approx(expected, rel=1e-13)

    with pytest.raises(ValueError, match="candidate must have shape"):
        function(np.zeros(999))


@pytest.mark.parametrize(
    ("function_name", "function_id", "expected"),
    [
        (function_name, function_id, expected)
        for (function_name, function_id), expected in HCC_MAIN_RS_FINGERPRINTS.items()
    ],
)
def test_vendor_r_and_s_ids_match_hcc_main_fixed_input(
    function_name: str,
    function_id: int,
    expected: float,
) -> None:
    function = runner.Benchmark(None, data_dir=runner.DATA_DIR).get_function(
        function_name,
        function_id,
    )
    candidate = np.linspace(-0.25, 0.25, 1000)

    assert function.info()["dimension"] == 1000
    assert function.dimension == 1000
    assert candidate.shape == (1000,)
    assert function.Ovector.shape == (1000,)
    assert function.Pvector.shape == (1000,)
    assert sorted(function.Pvector.tolist()) == list(range(1000))
    assert function(candidate).shape == (1,)
    assert float(function(candidate)[0]) == pytest.approx(expected, rel=1e-13)

    with pytest.raises(ValueError, match="candidate must have shape"):
        function(np.zeros((1, 999)))


@pytest.mark.parametrize(("function_id", "gamma"), [(2, 1), (6, 10)])
def test_vendor_f2_f6_match_hcc_main_data_and_reference_blind_topology(
    function_id: int,
    gamma: int,
) -> None:
    rotation_dims = HCC_MAIN_ROTATION_DIMS[function_id]
    expected_names = {
        f"F{function_id}-{suffix}.txt"
        for suffix in ("design", "info", "p", "s", "w", "xopt")
    } | {f"F{function_id}-R{dimension}.txt" for dimension in rotation_dims}
    paths = sorted(runner.DATA_DIR.glob(f"F{function_id}-*"), key=lambda path: path.name)

    assert {path.name for path in paths} == expected_names
    assert _dataset_digest(paths) == HCC_MAIN_DATASET_DIGESTS[function_id]

    metadata = runner.load_aob_metadata(function_id, runner.DATA_DIR)
    design = runner.load_reference_blind_design_matrix(function_id, runner.DATA_DIR)
    grouping = runner.load_runtime_grouping(
        function_id,
        runner.DATA_DIR,
        evidence_overlay_mode="native_audit",
    )

    assert metadata["dimension"] == 1000
    assert metadata["sub_num"] == 20
    assert metadata["overlap_degree"] == gamma
    assert tuple(sorted(metadata["subgroups_type"])) == rotation_dims
    assert design.shape == (1000, 1000)
    assert len(grouping) == 20
    assert tuple(map(len, grouping)) == tuple(metadata["subgroups"])
    assert all(
        len(set(left) & set(right)) == gamma
        for left, right in zip(grouping, grouping[1:])
    )


def test_aob_vector_inputs_fail_on_wrong_shape_or_invalid_permutation(
    tmp_path,
) -> None:
    benchmark = Benchmarks(None, data_dir=tmp_path)
    benchmark.ID = 9
    benchmark.dimension = 3

    (tmp_path / "F9-xopt.txt").write_text("1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 3 values"):
        benchmark.readOvector()

    (tmp_path / "F9-p.txt").write_text("1,2,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="permutation of 1..3"):
        benchmark.readPermVector()
