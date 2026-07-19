from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.pilots.exp_019_conflict_resolution_pilot.benchmark import (
    CASE_SPECS,
    CSV_FIELDS,
    FROZEN_RHO,
    MANIFEST_NAME,
    SYNTHETIC_DATA_DIR,
    VENDOR_DATA_DIR,
    ConflictBenchmarkFactory,
    generate_synthetic_bundle,
    validate_synthetic_bundle,
)

from AOB.AOB import Benchmark as VendorBenchmark


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "data"
    shutil.copytree(SYNTHETIC_DATA_DIR, destination)
    return destination


def _load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(data_dir: Path, manifest: dict) -> None:
    (data_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rebind_csv_hash(data_dir: Path, variant_id: str) -> None:
    manifest = _load_manifest(data_dir)
    csv_name = manifest["variants"][variant_id]["synthetic_csv"]["path"]
    manifest["variants"][variant_id]["synthetic_csv"]["sha256"] = _sha256(
        data_dir / csv_name
    )
    _write_manifest(data_dir, manifest)


def test_committed_bundle_has_frozen_cases_and_counts() -> None:
    loaded = validate_synthetic_bundle()
    manifest = _load_manifest(SYNTHETIC_DATA_DIR)

    assert set(loaded) == {case.variant_id for case in CASE_SPECS}
    assert manifest["rho"] == FROZEN_RHO
    assert {
        variant_id: (
            entry["synthetic_csv"]["row_count"],
            entry["synthetic_csv"]["shared_global_variable_count"],
        )
        for variant_id, entry in manifest["variants"].items()
    } == {
        "E3_conflict_variant_synthetic": (1057, 57),
        "A4_conflict_variant_synthetic": (1095, 95),
        "S5_conflict_variant_synthetic": (1133, 133),
    }


def test_diagnostic_config_freezes_stage_one_without_runtime_authority() -> None:
    config = json.loads(
        (SYNTHETIC_DATA_DIR.parent / "diagnostic_config.json").read_text(encoding="utf-8")
    )

    assert config["observer_only"] is True
    assert config["runtime_authorized"] is False
    assert config["synthetic_benchmark"]["rho"] == FROZEN_RHO
    assert config["smoke"] == {
        "cohort": "real_aob",
        "cases": ["A4"],
        "seeds": [1],
        "max_fes": 100000,
        "jobs": 1,
    }
    assert config["pilot"]["real_aob"]["cases"] == ["E1", "E3", "A4", "R4", "S5"]
    assert config["pilot"]["synthetic_conflict"]["cases"] == ["E3", "A4", "S5"]
    assert config["statistics"]["epsilon"] == 1e-300


def test_generator_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_synthetic_bundle(first)
    generate_synthetic_bundle(second)

    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for first_path in first.iterdir():
        assert first_path.read_bytes() == (second / first_path.name).read_bytes()


def test_only_shared_local_optima_change_and_remain_in_bounds() -> None:
    for case in CASE_SPECS:
        csv_path = SYNTHETIC_DATA_DIR / f"{case.variant_id}.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == CSV_FIELDS
        shared_by_global: dict[int, list[float]] = {}
        for row in rows:
            base = float(row["base_optimum"])
            conflict = float(row["conflict_optimum"])
            is_shared = row["is_shared"] == "true"
            assert -100.0 <= conflict <= 100.0
            assert (conflict != base) is is_shared
            if is_shared:
                shared_by_global.setdefault(
                    int(row["global_variable_index"]), []
                ).append(conflict)
        for local_optima in shared_by_global.values():
            assert len(local_optima) == 2
            assert abs(local_optima[1] - local_optima[0]) == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("function_name", "function_id"),
    [("elliptic", 3), ("ackley", 4), ("schwefel", 5)],
)
def test_zero_offset_conflict_is_numerically_equal_to_conform(
    tmp_path: Path,
    function_name: str,
    function_id: int,
) -> None:
    zero_data = tmp_path / "rho-zero"
    generate_synthetic_bundle(zero_data, rho=0.0)
    conflict = ConflictBenchmarkFactory(
        data_dir=VENDOR_DATA_DIR,
        synthetic_data_dir=zero_data,
    ).get_function(function_name, function_id)
    conform = VendorBenchmark(None, data_dir=VENDOR_DATA_DIR).get_function(
        function_name, function_id
    )
    rng = np.random.default_rng(1900 + function_id)
    batch = np.clip(
        conform.Ovector + rng.normal(0.0, 0.1, size=(3, conform.dimension)),
        conform.minX,
        conform.maxX,
    )

    np.testing.assert_allclose(conflict(batch), conform(batch), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(conflict(batch[0]), conform(batch[0]), rtol=1e-13, atol=0.0)


def test_objective_supports_1d_and_batch_and_records_every_value() -> None:
    benchmark = ConflictBenchmarkFactory().get_function("ackley", 4)
    first = np.asarray(benchmark.Ovector, dtype=float)
    batch = np.vstack([first, np.clip(first + 0.01, benchmark.minX, benchmark.maxX)])

    one_value = benchmark(first)
    batch_values = benchmark(batch)

    assert one_value.shape == (1,)
    assert batch_values.shape == (2,)
    np.testing.assert_allclose(
        benchmark.fitness_record,
        np.concatenate([one_value, batch_values]),
    )


def test_missing_csv_fails_closed(tmp_path: Path) -> None:
    data_dir = _copy_bundle(tmp_path)
    (data_dir / "E3_conflict_variant_synthetic.csv").unlink()

    with pytest.raises(FileNotFoundError, match="synthetic CSV is missing"):
        validate_synthetic_bundle(data_dir)


def test_tampered_csv_fails_closed_before_loading(tmp_path: Path) -> None:
    data_dir = _copy_bundle(tmp_path)
    csv_path = data_dir / "A4_conflict_variant_synthetic.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace(",false\n", ",true\n", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic CSV hash mismatch"):
        validate_synthetic_bundle(data_dir)


def test_duplicate_local_row_fails_closed_even_with_rebound_hash(tmp_path: Path) -> None:
    data_dir = _copy_bundle(tmp_path)
    variant_id = "S5_conflict_variant_synthetic"
    csv_path = data_dir / f"{variant_id}.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[1]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rebind_csv_hash(data_dir, variant_id)

    with pytest.raises(ValueError, match="duplicates local row"):
        validate_synthetic_bundle(data_dir)


def test_vendor_hash_tampering_fails_closed(tmp_path: Path) -> None:
    data_dir = _copy_bundle(tmp_path)
    manifest = _load_manifest(data_dir)
    manifest["variants"]["E3_conflict_variant_synthetic"]["vendor_files"][
        "F3-info.txt"
    ] = "0" * 64
    _write_manifest(data_dir, manifest)

    with pytest.raises(ValueError, match="vendor input hash mismatch"):
        validate_synthetic_bundle(data_dir)


def test_unsupported_case_and_wrong_candidate_shape_fail_closed() -> None:
    factory = ConflictBenchmarkFactory()
    with pytest.raises(ValueError, match="unsupported synthetic conflict case"):
        factory.get_function("elliptic", 4)
    benchmark = factory.get_function("elliptic", 3)
    with pytest.raises(ValueError, match="candidate must have shape"):
        benchmark(np.zeros(999))
