"""Deterministic, experiment-local conflict twins for AOB E3/A4/S5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "vendor" / "hcc"
VENDOR_DATA_DIR = VENDOR_ROOT / "AOB" / "AOBG" / "datafile"
SYNTHETIC_DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_NAME = "conflict_variants_manifest.json"
GENERATOR_VERSION = "exp019-conflict-generator-v1"
MANIFEST_SCHEMA_VERSION = "exp019-conflict-benchmark-manifest-v1"
FROZEN_RHO = 0.10
CSV_FIELDS = (
    "variant_id",
    "group_index",
    "local_index",
    "global_variable_index",
    "base_optimum",
    "conflict_optimum",
    "is_shared",
)

if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from AOB.ackley import ackley as VendorAckley  # noqa: E402
from AOB.elliptic import elliptic as VendorElliptic  # noqa: E402
from AOB.schwefel import schwefel as VendorSchwefel  # noqa: E402


@dataclass(frozen=True)
class _CaseSpec:
    label: str
    function_name: str
    function_id: int
    variant_id: str


CASE_SPECS = (
    _CaseSpec("E3", "elliptic", 3, "E3_conflict_variant_synthetic"),
    _CaseSpec("A4", "ackley", 4, "A4_conflict_variant_synthetic"),
    _CaseSpec("S5", "schwefel", 5, "S5_conflict_variant_synthetic"),
)
_CASE_BY_PAIR = {(case.function_name, case.function_id): case for case in CASE_SPECS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_numeric_file(path: Path, *, integer: bool = False) -> np.ndarray:
    values: list[float | int] = []
    for token in path.read_text(encoding="utf-8").replace(",", " ").split():
        values.append(int(float(token)) if integer else float(token))
    dtype = int if integer else float
    return np.asarray(values, dtype=dtype)


def _vendor_files(data_dir: Path, function_id: int) -> list[Path]:
    files = sorted(data_dir.glob(f"F{function_id}-*"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"no vendor inputs found for F{function_id}: {data_dir}")
    return files


def _case_inputs(case: _CaseSpec, data_dir: Path) -> dict[str, Any]:
    info_path = data_dir / f"F{case.function_id}-info.txt"
    with info_path.open("r", encoding="utf-8") as handle:
        info = yaml.safe_load(handle)
    sizes = _read_numeric_file(
        data_dir / f"F{case.function_id}-s.txt", integer=True
    )
    permutation = _read_numeric_file(
        data_dir / f"F{case.function_id}-p.txt", integer=True
    ) - 1
    optimum = _read_numeric_file(data_dir / f"F{case.function_id}-xopt.txt")

    dimension = int(info["dimension"])
    subgroup_count = int(info["sub_num"])
    if len(sizes) != subgroup_count:
        raise ValueError(f"F{case.function_id} subgroup count does not match info")
    if len(permutation) != dimension or sorted(permutation.tolist()) != list(range(dimension)):
        raise ValueError(f"F{case.function_id} permutation is not a complete zero-based mapping")
    if len(optimum) != dimension:
        raise ValueError(f"F{case.function_id} optimum length does not match dimension")

    return {
        "dimension": dimension,
        "sizes": sizes,
        "permutation": permutation,
        "optimum": optimum,
        "overlap": int(info["overlap_degree"]),
        "lower": float(info["lower_bound"]),
        "upper": float(info["upper_bound"]),
    }


def _group_indices(inputs: dict[str, Any]) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    consumed = 0
    overlap = inputs["overlap"]
    for group_index, subgroup_size in enumerate(inputs["sizes"]):
        start = consumed - group_index * overlap
        end = start + int(subgroup_size)
        indices = inputs["permutation"][start:end]
        if len(indices) != int(subgroup_size):
            raise ValueError(f"group {group_index} extends beyond the permutation")
        groups.append(indices)
        consumed += int(subgroup_size)
    if consumed - (len(groups) - 1) * overlap != inputs["dimension"]:
        raise ValueError("overlap topology does not reconstruct the global dimension")
    return groups


def _expected_rows(case: _CaseSpec, data_dir: Path, rho: float) -> list[dict[str, Any]]:
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be finite and within [0, 1]")
    inputs = _case_inputs(case, data_dir)
    groups = _group_indices(inputs)
    owners: dict[int, list[int]] = {}
    for group_index, indices in enumerate(groups):
        for global_index in indices:
            owners.setdefault(int(global_index), []).append(group_index)
    invalid_owners = {index: value for index, value in owners.items() if len(value) not in (1, 2)}
    if invalid_owners:
        raise ValueError(f"unexpected shared-variable owner multiplicity: {invalid_owners}")

    rows: list[dict[str, Any]] = []
    for group_index, indices in enumerate(groups):
        for local_index, global_index_value in enumerate(indices):
            global_index = int(global_index_value)
            base = float(inputs["optimum"][global_index])
            owner_groups = owners[global_index]
            is_shared = len(owner_groups) == 2
            conflict = base
            if is_shared:
                if group_index == min(owner_groups):
                    conflict = base + rho * (inputs["lower"] - base)
                else:
                    conflict = base + rho * (inputs["upper"] - base)
            rows.append(
                {
                    "variant_id": case.variant_id,
                    "group_index": group_index,
                    "local_index": local_index,
                    "global_variable_index": global_index,
                    "base_optimum": base,
                    "conflict_optimum": float(conflict),
                    "is_shared": is_shared,
                }
            )
    return rows


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in CSV_FIELDS})


def generate_synthetic_bundle(
    output_dir: Path = SYNTHETIC_DATA_DIR,
    *,
    vendor_data_dir: Path = VENDOR_DATA_DIR,
    rho: float = FROZEN_RHO,
) -> Path:
    """Generate a deterministic synthetic bundle; committed data uses frozen rho=0.10."""

    output_dir = Path(output_dir)
    vendor_data_dir = Path(vendor_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    for case in CASE_SPECS:
        rows = _expected_rows(case, vendor_data_dir, rho)
        csv_name = f"{case.variant_id}.csv"
        csv_path = output_dir / csv_name
        _write_rows(csv_path, rows)
        shared_rows = [row for row in rows if row["is_shared"]]
        variants[case.variant_id] = {
            "base_case": case.label,
            "function_name": case.function_name,
            "function_id": case.function_id,
            "synthetic_csv": {
                "path": csv_name,
                "sha256": _sha256(csv_path),
                "row_count": len(rows),
                "shared_local_row_count": len(shared_rows),
                "shared_global_variable_count": len(
                    {row["global_variable_index"] for row in shared_rows}
                ),
            },
            "vendor_files": {
                path.name: _sha256(path)
                for path in _vendor_files(vendor_data_dir, case.function_id)
            },
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rho": rho,
        "vendor_data_root": "vendor/hcc/AOB/AOBG/datafile",
        "variants": variants,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys do not match the frozen schema")


def _parse_and_validate_rows(
    csv_path: Path,
    case: _CaseSpec,
    expected_rows: list[dict[str, Any]],
) -> list[np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{csv_path.name} header does not match the frozen schema")
        raw_rows = list(reader)
    if len(raw_rows) != len(expected_rows):
        raise ValueError(f"{csv_path.name} row count does not match vendor topology")

    seen: set[tuple[int, int]] = set()
    vectors: dict[int, list[float]] = {}
    for row_number, (raw, expected) in enumerate(zip(raw_rows, expected_rows, strict=True), 2):
        if set(raw) != set(CSV_FIELDS):
            raise ValueError(f"{csv_path.name}:{row_number} has malformed columns")
        try:
            group_index = int(raw["group_index"])
            local_index = int(raw["local_index"])
            global_index = int(raw["global_variable_index"])
            base = float(raw["base_optimum"])
            conflict = float(raw["conflict_optimum"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{csv_path.name}:{row_number} has an invalid scalar") from error
        key = (group_index, local_index)
        if key in seen:
            raise ValueError(f"{csv_path.name}:{row_number} duplicates local row {key}")
        seen.add(key)
        actual = {
            "variant_id": raw["variant_id"],
            "group_index": group_index,
            "local_index": local_index,
            "global_variable_index": global_index,
            "base_optimum": base,
            "conflict_optimum": conflict,
            "is_shared": raw["is_shared"] == "true",
        }
        if raw["is_shared"] not in ("true", "false"):
            raise ValueError(f"{csv_path.name}:{row_number} has an invalid is_shared value")
        if actual != expected:
            raise ValueError(f"{csv_path.name}:{row_number} differs from frozen construction")
        if not np.isfinite(conflict):
            raise ValueError(f"{csv_path.name}:{row_number} has a non-finite optimum")
        vectors.setdefault(group_index, []).append(conflict)
    return [np.asarray(vectors[index], dtype=float) for index in range(len(vectors))]


def validate_synthetic_bundle(
    synthetic_data_dir: Path = SYNTHETIC_DATA_DIR,
    *,
    vendor_data_dir: Path = VENDOR_DATA_DIR,
) -> dict[str, list[np.ndarray]]:
    """Validate every bound input and return OvectorVec arrays for each variant."""

    synthetic_data_dir = Path(synthetic_data_dir)
    vendor_data_dir = Path(vendor_data_dir)
    manifest_path = synthetic_data_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"synthetic manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_exact_keys(
        manifest,
        {"schema_version", "generator_version", "rho", "vendor_data_root", "variants"},
        "manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("synthetic manifest schema version is not supported")
    if manifest["generator_version"] != GENERATOR_VERSION:
        raise ValueError("synthetic generator version is not supported")
    if manifest["vendor_data_root"] != "vendor/hcc/AOB/AOBG/datafile":
        raise ValueError("synthetic manifest vendor root label is not supported")
    rho = float(manifest["rho"])
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("manifest rho must be finite and within [0, 1]")
    expected_variant_ids = {case.variant_id for case in CASE_SPECS}
    if set(manifest["variants"]) != expected_variant_ids:
        raise ValueError("manifest variant set does not match the frozen cases")

    loaded: dict[str, list[np.ndarray]] = {}
    for case in CASE_SPECS:
        entry = manifest["variants"][case.variant_id]
        _require_exact_keys(
            entry,
            {"base_case", "function_name", "function_id", "synthetic_csv", "vendor_files"},
            case.variant_id,
        )
        if (
            entry["base_case"] != case.label
            or entry["function_name"] != case.function_name
            or entry["function_id"] != case.function_id
        ):
            raise ValueError(f"{case.variant_id} metadata does not match its frozen case")
        actual_vendor_hashes = {
            path.name: _sha256(path)
            for path in _vendor_files(vendor_data_dir, case.function_id)
        }
        if entry["vendor_files"] != actual_vendor_hashes:
            raise ValueError(f"{case.variant_id} vendor input hash mismatch")

        csv_meta = entry["synthetic_csv"]
        _require_exact_keys(
            csv_meta,
            {
                "path",
                "sha256",
                "row_count",
                "shared_local_row_count",
                "shared_global_variable_count",
            },
            f"{case.variant_id}.synthetic_csv",
        )
        expected_csv_name = f"{case.variant_id}.csv"
        if csv_meta["path"] != expected_csv_name:
            raise ValueError(f"{case.variant_id} synthetic CSV path is not frozen")
        csv_path = synthetic_data_dir / expected_csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"synthetic CSV is missing: {csv_path}")
        if _sha256(csv_path) != csv_meta["sha256"]:
            raise ValueError(f"{case.variant_id} synthetic CSV hash mismatch")
        expected_rows = _expected_rows(case, vendor_data_dir, rho)
        shared_rows = [row for row in expected_rows if row["is_shared"]]
        expected_counts = (
            len(expected_rows),
            len(shared_rows),
            len({row["global_variable_index"] for row in shared_rows}),
        )
        manifest_counts = (
            csv_meta["row_count"],
            csv_meta["shared_local_row_count"],
            csv_meta["shared_global_variable_count"],
        )
        if manifest_counts != expected_counts:
            raise ValueError(f"{case.variant_id} manifest counts do not match construction")
        loaded[case.variant_id] = _parse_and_validate_rows(csv_path, case, expected_rows)
    return loaded


class _ConflictComputeMixin:
    OvectorVec: list[np.ndarray]

    def _transform_conflict(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rotateVectorConflict(
        self,
        group_index: int,
        consumed: int,
        candidate: np.ndarray,
    ) -> np.ndarray:
        subgroup_size = int(self.s[group_index])
        start = consumed - group_index * self.overlap
        indices = self.Pvector[start : start + subgroup_size]
        centered = candidate[:, indices] - self.OvectorVec[group_index]
        return self.multiply(centered.astype(float), self.cache_Rotation[subgroup_size])

    def compute(self, x: np.ndarray) -> np.ndarray:
        candidate = np.asarray(x, dtype=float)
        if candidate.ndim == 1:
            candidate = np.expand_dims(candidate, axis=0)
        if candidate.ndim != 2 or candidate.shape[1] != self.dimension:
            raise ValueError(f"candidate must have shape (n, {self.dimension})")
        result = np.zeros(candidate.shape[0])
        consumed = 0
        for group_index in range(self.s_size):
            rotated = self.rotateVectorConflict(group_index, consumed, candidate)
            result += self.w[group_index] * self._objective(
                self._transform_conflict(rotated)
            )
            consumed += self.s[group_index]
        self.fitness_record.extend(result.tolist())
        return result


class _ConflictElliptic(_ConflictComputeMixin, VendorElliptic):
    _objective = VendorElliptic.elliptic

    def _transform_conflict(self, values: np.ndarray) -> np.ndarray:
        values = self.transform_osz(values)
        return self.transform_asy(values, 0.2)


class _ConflictAckley(_ConflictComputeMixin, VendorAckley):
    _objective = VendorAckley.ackley

    def _transform_conflict(self, values: np.ndarray) -> np.ndarray:
        values = self.transform_osz(values)
        return self.transform_asy(values, 0.2)


class _ConflictSchwefel(_ConflictComputeMixin, VendorSchwefel):
    _objective = VendorSchwefel.schwefel

    def _transform_conflict(self, values: np.ndarray) -> np.ndarray:
        values = self.transform_osz(values)
        return self.transform_asy(values, 0.2)


_BENCHMARK_CLASSES = {
    ("elliptic", 3): _ConflictElliptic,
    ("ackley", 4): _ConflictAckley,
    ("schwefel", 5): _ConflictSchwefel,
}


class ConflictBenchmarkFactory:
    """Fail-closed factory for the three experiment-local synthetic variants."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        data_dir: str | Path = VENDOR_DATA_DIR,
        synthetic_data_dir: str | Path = SYNTHETIC_DATA_DIR,
    ) -> None:
        self.output_path = output_path
        self.data_dir = Path(data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self._vectors = validate_synthetic_bundle(
            self.synthetic_data_dir,
            vendor_data_dir=self.data_dir,
        )

    def get_function(self, function_name: str, function_id: int):
        pair = (function_name, function_id)
        case = _CASE_BY_PAIR.get(pair)
        benchmark_class = _BENCHMARK_CLASSES.get(pair)
        if case is None or benchmark_class is None:
            raise ValueError(f"unsupported synthetic conflict case: {function_name}/{function_id}")
        benchmark = benchmark_class(function_id, self.output_path, data_dir=self.data_dir)
        benchmark.OvectorVec = [values.copy() for values in self._vectors[case.variant_id]]
        benchmark.variant_id = case.variant_id
        benchmark.synthetic_conflict = True
        return benchmark

    def get_info(self, function_name: str, function_id: int) -> dict[str, Any]:
        return self.get_function(function_name, function_id).info()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate exp_019 conflict twins.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.generate:
        generate_synthetic_bundle()
    validate_synthetic_bundle()
    print(f"validated {len(CASE_SPECS)} synthetic conflict variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
