from __future__ import annotations

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
