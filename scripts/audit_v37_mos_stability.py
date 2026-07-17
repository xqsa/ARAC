"""Audit the frozen v37 mirrored-orthogonal-sampling stability protocol.

Paper-best values are deliberately joined only after all runtime artifacts pass
their integrity checks and their hashes have been frozen.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v37_mos_single_seed_stability_v1.json"
PAPER_BEST_PATH = ROOT / "references" / "paper_reported_table2_best_by_case.csv"
AOB_DATA_ROOT = (ROOT / "vendor" / "hcc" / "AOB" / "AOBG" / "datafile").resolve()
PROTOCOL_VERSION = "v37-mos-single-seed-stability-v1"
CONFIG_SHA256 = "9778cd77d9c1d6f0a881c67e18a6a98a9ac5e507cc1e96ff7ff1debf115edda0"
STAGES = ("baseline", "smoke", "development", "confirmation")

BRANCH_COLUMNS = (
    "protocol_version",
    "stage",
    "problem_id",
    "seed",
    "arm",
    "lane_id",
    "status",
    "fresh_optimizer_execution",
    "terminal_error",
    "terminal_target_fe",
    "terminal_observed_fe",
    "terminal_completion_tolerance_fe",
    "same_budget_violation",
    "source_git_commit",
    "source_bundle_sha256",
    "config_sha256",
    "runtime_environment_sha256",
    "phase_i_record_sha256",
    "first_cma_prestate_status",
    "first_cma_prestate_sha256",
    "rng_descriptor_sha256",
    "terminal_record_sha256",
)
SAMPLING_COLUMNS = (
    "run_id",
    "lane_id",
    "sampling_mode",
    "problem_id",
    "seed",
    "outer_iter",
    "group_index",
    "cma_scope",
    "candidate_index",
    "optimizer_seed",
    "optimizer_restart_index",
    "generation",
    "population",
    "dimension",
    "pair_count",
    "block_count",
    "raw_draw_sha256",
    "sample_sha256",
    "max_orthogonality_error",
    "rng_draw_count",
    "evaluated_count",
    "complete_population",
)
PROVENANCE_COLUMNS = (
    "protocol_version",
    "run_id",
    "lane_id",
    "sampling_mode",
    "problem_id",
    "seed",
    "status",
    "terminal_target_fe",
    "terminal_completion_tolerance_fe",
    "phase_i_fe",
    "phase_i_record_sha256",
    "phase_i_candidate_sha256",
    "first_cma_prestate_sha256",
    "first_cma_prestate_status",
    "rng_descriptor_sha256",
    "terminal_record_sha256",
    "mos_generation_rows",
    "mos_primary_generation_rows",
    "mos_rescue_generation_rows",
    "source_git_commit",
    "source_bundle_sha256",
    "config_sha256",
    "runtime_environment_sha256",
)
RESULT_COLUMNS = (
    "lane_id",
    "problem_id",
    "seed",
    "selected_action_name",
    "cma_sampling_mode",
    "hcc_smoke_final_error",
    "hcc_smoke_fe_used",
    "hcc_smoke_status",
    "fresh_optimizer_execution",
    "result_source",
    "action_trace_sha256",
)
BASELINE_RESULT_COLUMNS = tuple(
    field for field in RESULT_COLUMNS if field != "cma_sampling_mode"
)
LEDGER_COLUMNS = (
    "lane_id",
    "problem_id",
    "seed",
    "phase_i_fe",
    "total_fe",
    "budget_limit",
    "configured_budget_limit",
    "budget_aligned_fe_used",
    "actual_fe_used",
    "same_budget_violation",
    "fresh_execution",
)
AOB_COLUMNS = (
    "lane_id",
    "problem_id",
    "seed",
    "path",
    "sha256_before",
    "sha256_after",
    "unchanged",
)
LEAKAGE_COLUMNS = (
    "forbidden_field",
    "found_in_runtime_payload",
    "audit_status",
)
TRACE_COLUMNS = (
    "lane_id",
    "problem_id",
    "seed",
    "outer_iter",
    "group_index",
    "selected_action_name",
    "trace_event",
    "best_before",
    "sigma_before",
    "population_before",
    "population_after",
    "optimizer_seed",
    "state_fingerprint_before",
    "pre_hold_phase_i_tail_utility",
    "pre_hold_group_count",
    "pre_hold_mean_group_size",
    "pre_hold_overlap_edge_count",
    "pre_hold_shared_variable_count",
)

OUTCOME_COLUMNS = (
    "protocol_version",
    "stage",
    "problem_id",
    "seed",
    "arm",
    "terminal_error",
    "paper_best",
    "normalized_error",
    "strict_paper_win",
    "paper_relative_catastrophic",
    "runtime_dispatch_used",
)
SEED_SUMMARY_COLUMNS = (
    "protocol_version",
    "stage",
    "arm",
    "seed",
    "case_count",
    "strict_win_count",
    "q13",
    "single_seed_13_win",
)
CASE_SUMMARY_COLUMNS = (
    "protocol_version",
    "stage",
    "arm",
    "problem_id",
    "seed_count",
    "paper_best",
    "arithmetic_mean_error",
    "arithmetic_mean_win",
    "seed_win_count",
    "seed_win_fraction",
    "stable_core_case",
    "strict_common_win",
    "best_error",
    "best_normalized_error",
    "paper_relative_catastrophic_count",
)
PAIR_COLUMNS = (
    "protocol_version",
    "stage",
    "problem_id",
    "seed",
    "a0_terminal_error",
    "a1_terminal_error",
    "tau",
    "paired_catastrophic",
    "phase_i_hash_match",
    "first_cma_prestate_status_match",
    "first_cma_prestate_hash_match",
    "pair_integrity",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(set(required) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")
        return list(reader)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered: dict[str, object] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    value = format(value, ".17e")
                rendered[field] = value
            writer.writerow(rendered)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_hex(value: object, length: int = 64) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def _flag(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field, "")
    if value not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1")
    return value == "1"


def _boolean(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise ValueError(f"{field} must be a boolean")


def _integer(row: Mapping[str, str], field: str, *, minimum: int = 0) -> int:
    try:
        value = int(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _number(row: Mapping[str, str], field: str, *, minimum: float = 0.0) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return value


def _load_config() -> dict[str, object]:
    if _file_sha256(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("frozen config hash mismatch")
    config = _read_json(CONFIG_PATH)
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("frozen config protocol mismatch")
    if config.get("status") != "frozen_before_new_optimizer_fe":
        raise ValueError("frozen config status mismatch")
    expected_paper_hash = config["preimplementation_anchor"]["paper_best_sha256"]
    if _file_sha256(PAPER_BEST_PATH) != expected_paper_hash:
        raise ValueError("canonical paper-best hash mismatch")
    return config


def _matrix(config: Mapping[str, object], stage: str, *, smoke_name: str | None = None) -> dict[str, object]:
    matrices = config["matrices"]
    assert isinstance(matrices, Mapping)
    if stage == "development":
        raw = matrices["development"]
    elif stage == "confirmation":
        raw = matrices["confirmation"]
    elif stage == "smoke":
        raw = matrices[smoke_name or "cli_smoke"]
    else:
        raise ValueError(f"stage has no paired matrix: {stage}")
    assert isinstance(raw, Mapping)
    cases = raw.get("cases", config["cases"])
    return {
        "cases": list(cases),
        "seeds": list(raw["seeds"]),
        "arms": list(raw["arms"]),
        "terminal_fe": int(raw["terminal_fe"]),
    }


def _matrix_sha256(matrix: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "cases": list(matrix["cases"]),
            "seeds": [int(seed) for seed in matrix["seeds"]],
            "arms": list(matrix["arms"]),
            "terminal_fe": int(matrix["terminal_fe"]),
        }
    )


def _paper_best(config: Mapping[str, object]) -> dict[str, float]:
    rows = _read_csv(
        PAPER_BEST_PATH,
        ("case", "paper_best", "comparison_role", "runtime_dispatch_allowed"),
    )
    expected_cases = list(config["cases"])
    values: dict[str, float] = {}
    for row in rows:
        case = row["case"].strip()
        if case in values:
            raise ValueError(f"duplicate paper-best case: {case}")
        value = _number(row, "paper_best", minimum=0.0)
        if value <= 0.0:
            raise ValueError(f"paper-best must be positive: {case}")
        if row["runtime_dispatch_allowed"] != "0":
            raise ValueError("paper-best cannot be runtime dispatch input")
        values[case] = value
    if list(values) != expected_cases:
        raise ValueError("paper-best case order or coverage mismatch")
    return values


def _raw_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: _file_sha256(root / name) for name in names}


def _runtime_environment(root: Path) -> tuple[dict[str, object], str]:
    path = root / "runtime_environment.json"
    payload = _read_json(path)
    if payload.get("status") != "pass" or payload.get("expected") != payload.get("observed"):
        raise ValueError("runtime environment is not pinned or does not match")
    return payload, _file_sha256(path)


def _indexed_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    key_fields: Sequence[str],
    artifact: str,
) -> dict[tuple[str, ...], Mapping[str, str]]:
    indexed: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if not all(key):
            raise ValueError(f"{artifact} has an empty identity")
        if key in indexed:
            raise ValueError(f"{artifact} duplicate identity: {key}")
        indexed[key] = row
    return indexed


def _validate_leakage(root: Path) -> None:
    rows = _read_csv(root / "anti_leakage_audit.csv", LEAKAGE_COLUMNS)
    if not rows or any(
        row["audit_status"] != "pass" or row["found_in_runtime_payload"] != "0"
        for row in rows
    ):
        raise ValueError("anti-leakage audit failed")


def _validate_aob(
    root: Path,
    expected_keys: set[tuple[str, str, str]],
) -> dict[str, int]:
    rows = _read_csv(root / "aob_input_manifest.csv", AOB_COLUMNS)
    seen: set[tuple[str, str, str]] = set()
    path_hashes: dict[str, str] = {}
    current_hashes: dict[Path, str] = {}
    info_paths: dict[str, Path] = {}
    info_seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["lane_id"], row["problem_id"], row["seed"])
        if key not in expected_keys:
            raise ValueError(f"unexpected AOB row: {key}")
        if row["unchanged"] != "1" or row["sha256_before"] != row["sha256_after"]:
            raise ValueError(f"AOB input changed: {key}")
        if not _is_hex(row["sha256_before"]):
            raise ValueError(f"invalid AOB hash: {key}")
        prior = path_hashes.setdefault(row["path"], row["sha256_before"])
        if prior != row["sha256_before"]:
            raise ValueError(f"inconsistent AOB path hash: {row['path']}")
        source_path = Path(row["path"]).resolve()
        try:
            source_path.relative_to(AOB_DATA_ROOT)
        except ValueError as exc:
            raise ValueError(f"AOB input escaped canonical root: {source_path}") from exc
        if not source_path.is_file():
            raise ValueError(f"AOB input no longer exists: {source_path}")
        if source_path not in current_hashes:
            current_hashes[source_path] = _file_sha256(source_path)
        if current_hashes[source_path] != row["sha256_before"]:
            raise ValueError(f"AOB input no longer matches frozen hash: {source_path}")
        case = row["problem_id"]
        if source_path.name == f"F{case[1:]}-info.txt":
            expected_info_path = AOB_DATA_ROOT / source_path.name
            if source_path != expected_info_path:
                raise ValueError(f"AOB info path is not canonical: {case}")
            prior_info = info_paths.setdefault(case, source_path)
            if prior_info != source_path:
                raise ValueError(f"inconsistent AOB info path: {case}")
            info_seen_keys.add(key)
        seen.add(key)
    if seen != expected_keys:
        raise ValueError("AOB matrix coverage mismatch")
    if info_seen_keys != expected_keys:
        raise ValueError("AOB info branch coverage mismatch")
    expected_cases = {case for _, case, _ in expected_keys}
    if set(info_paths) != expected_cases:
        raise ValueError("AOB info coverage mismatch")
    return {
        case: _terminal_tolerance_from_aob_info(path)
        for case, path in info_paths.items()
    }


def _terminal_tolerance_from_aob_info(path: Path) -> int:
    subgroups: object | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "subgroups":
            try:
                subgroups = ast.literal_eval(value.strip())
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"invalid AOB subgroup metadata: {path}") from exc
            break
    if (
        not isinstance(subgroups, list)
        or not subgroups
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in subgroups
        )
    ):
        raise ValueError(f"invalid AOB subgroup metadata: {path}")
    return max(1, max(4 + 3 * math.ceil(math.log(size)) for size in subgroups))


def _run_source_binding(root: Path) -> tuple[str, str]:
    text = (root / "run_manifest.md").read_text(encoding="utf-8-sig")
    if (
        not re.search(r"^Parallel jobs: 24$", text, re.MULTILINE)
        or not re.search(
            r"^Lanes: a0_v37_iid, a1_v37_mos$", text, re.MULTILINE
        )
        or not re.search(
            r"^CMA sampling modes: "
            r"a0_v37_iid=iid, a1_v37_mos=mirrored_orthogonal$",
            text,
            re.MULTILINE,
        )
    ):
        raise ValueError("run manifest jobs/lane/sampling order is not frozen")
    commit_match = re.search(r"^- git commit: ([0-9a-f]{40})$", text, re.MULTILINE)
    if commit_match is None:
        raise ValueError("run manifest source git commit is missing")
    hashes = {
        label.strip(): value
        for label, value in re.findall(
            r"^- ([^\r\n:]+) sha256: ([0-9a-f]{64})$", text, re.MULTILINE
        )
    }
    required_labels = {
        "experiment runner",
        "HCC smoke runner",
        "CMAES optimizer",
        "MOS sampler",
        "MOS source bundle",
    }
    if not required_labels <= set(hashes):
        raise ValueError("run manifest source bundle hashes are incomplete")
    bundle = _read_json(root / "mos_source_bundle.json")
    files = bundle.get("files")
    if (
        bundle.get("schema_version") != 1
        or not isinstance(files, Mapping)
        or not files
        or any(not _is_hex(value) for value in files.values())
    ):
        raise ValueError("MOS source bundle schema is invalid")
    bundle_sha256 = str(bundle.get("bundle_sha256", ""))
    if (
        not _is_hex(bundle_sha256)
        or bundle_sha256 != _canonical_sha256(dict(files))
        or hashes["MOS source bundle"] != bundle_sha256
        or files.get("experiment_runner") != hashes["experiment runner"]
        or files.get("hcc_smoke_runner") != hashes["HCC smoke runner"]
        or files.get("src/arac/backends/hcc_mos_cma.py") != hashes["MOS sampler"]
        or hashes["CMAES optimizer"] not in files.values()
    ):
        raise ValueError("MOS source bundle does not match run manifest")
    return commit_match.group(1), bundle_sha256


def _validate_manifest_jobs(root: Path) -> None:
    text = (root / "run_manifest.md").read_text(encoding="utf-8-sig")
    if not re.search(r"^Parallel jobs: 24$", text, re.MULTILINE):
        raise ValueError("run manifest must bind Parallel jobs: 24")


def _paired_branch_rows(
    root: Path,
    *,
    stage: str,
    matrix: Mapping[str, object],
    environment_sha256: str,
    source_git_commit: str,
    source_bundle_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    results = _read_csv(root / "our_result_by_case.csv", RESULT_COLUMNS)
    expected = {
        (str(arm), str(case), str(int(seed)))
        for arm in matrix["arms"]
        for case in matrix["cases"]
        for seed in matrix["seeds"]
    }
    result_index = _indexed_rows(
        results,
        key_fields=("lane_id", "problem_id", "seed"),
        artifact="our_result_by_case.csv",
    )
    if set(result_index) != expected:
        raise ValueError("paired result matrix has duplicates, missing, or extra rows")

    ledgers = _read_csv(root / "same_budget_ledger.csv", LEDGER_COLUMNS)
    ledger_index = _indexed_rows(
        ledgers,
        key_fields=("lane_id", "problem_id", "seed"),
        artifact="same_budget_ledger.csv",
    )
    if set(ledger_index) != expected:
        raise ValueError("same-budget ledger matrix mismatch")

    provenance_rows = _read_csv(
        root / "mos_branch_provenance.csv", PROVENANCE_COLUMNS
    )
    provenance_index = _indexed_rows(
        provenance_rows,
        key_fields=("lane_id", "problem_id", "seed"),
        artifact="mos_branch_provenance.csv",
    )
    if set(provenance_index) != expected:
        raise ValueError("MOS branch provenance matrix mismatch")
    run_ids = [row["run_id"] for row in provenance_rows]
    if any(not run_id for run_id in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("MOS branch provenance run ids must be nonempty and unique")

    traces = _read_csv(root / "action_trace.csv", TRACE_COLUMNS)
    trace_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in traces:
        key = (row["lane_id"], row["problem_id"], row["seed"])
        if key not in expected:
            raise ValueError(f"unexpected action trace row: {key}")
        trace_by_key[key].append(row)

    aob_tolerances = _validate_aob(root, expected)
    _validate_leakage(root)
    target = int(matrix["terminal_fe"])
    branches: list[dict[str, object]] = []
    paired_prefix: dict[
        tuple[str, str], dict[str, tuple[object, ...]]
    ] = defaultdict(dict)
    expected_modes = {
        str(config_arm): mode
        for config_arm, mode in zip(
            ("a0_v37_iid", "a1_v37_mos"),
            ("iid", "mirrored_orthogonal"),
            strict=True,
        )
    }
    for key in sorted(expected):
        arm, case, seed_text = key
        result = result_index[key]
        ledger = ledger_index[key]
        provenance = provenance_index[key]
        prefix = f"{case}:seed{seed_text}:{arm}"
        if (
            result["selected_action_name"] != "arac_evidence_action_controller_v37"
            or result["cma_sampling_mode"] != expected_modes.get(arm)
            or result["hcc_smoke_status"] != "completed"
            or result["fresh_optimizer_execution"] != "1"
            or result["result_source"] != "hcc_subprocess_smoke_execution"
        ):
            raise ValueError(f"branch not fresh/completed: {prefix}")
        error = _number(result, "hcc_smoke_final_error")
        actual = _integer(ledger, "actual_fe_used", minimum=1)
        if (
            _flag(ledger, "same_budget_violation")
            or not _flag(ledger, "fresh_execution")
            or _integer(ledger, "total_fe", minimum=1) != actual
            or _integer(ledger, "budget_aligned_fe_used", minimum=1) != actual
            or _integer(ledger, "budget_limit", minimum=1) != target
            or _integer(ledger, "configured_budget_limit", minimum=1) != target
            or result["hcc_smoke_fe_used"] != str(actual)
        ):
            raise ValueError(f"ledger integrity failed: {key}")

        expected_mode = expected_modes.get(arm)
        if expected_mode is None:
            raise ValueError(f"unexpected MOS lane: {arm}")
        if (
            provenance["protocol_version"] != PROTOCOL_VERSION
            or provenance["lane_id"] != arm
            or provenance["problem_id"] != case
            or provenance["seed"] != seed_text
            or provenance["sampling_mode"] != expected_mode
            or provenance["status"] != "completed"
            or provenance["source_git_commit"] != source_git_commit
            or provenance["source_bundle_sha256"] != source_bundle_sha256
            or provenance["config_sha256"] != CONFIG_SHA256
            or provenance["runtime_environment_sha256"] != environment_sha256
        ):
            raise ValueError(f"MOS branch provenance binding failed: {prefix}")
        provenance_target = _integer(
            provenance, "terminal_target_fe", minimum=1
        )
        tolerance = _integer(
            provenance, "terminal_completion_tolerance_fe", minimum=1
        )
        phase_i_fe = _integer(provenance, "phase_i_fe")
        if (
            provenance_target != target
            or phase_i_fe != _integer(ledger, "phase_i_fe")
            or phase_i_fe > actual
            or tolerance > target
            or tolerance != aob_tolerances[case]
        ):
            raise ValueError(f"MOS branch FE provenance mismatch: {prefix}")
        for field in (
            "phase_i_record_sha256",
            "phase_i_candidate_sha256",
            "rng_descriptor_sha256",
            "terminal_record_sha256",
        ):
            if not _is_hex(provenance[field]):
                raise ValueError(f"invalid {field}: {prefix}")
        first_cma_prestate = provenance["first_cma_prestate_sha256"]
        first_cma_status = provenance["first_cma_prestate_status"]
        if first_cma_status not in {"observed", "not_reached"}:
            raise ValueError(f"invalid first_cma_prestate_status: {prefix}")
        if not _is_hex(first_cma_prestate):
            raise ValueError(f"invalid first_cma_prestate_sha256: {prefix}")
        if (
            stage in {"development", "confirmation"}
            and first_cma_status != "observed"
        ):
            raise ValueError(f"first CMA prestate is missing: {prefix}")
        generation_rows = _integer(provenance, "mos_generation_rows")
        primary_rows = _integer(provenance, "mos_primary_generation_rows")
        rescue_rows = _integer(provenance, "mos_rescue_generation_rows")
        if primary_rows + rescue_rows > generation_rows:
            raise ValueError(f"invalid MOS generation counts: {prefix}")
        if arm == "a0_v37_iid" and any(
            value != 0 for value in (generation_rows, primary_rows, rescue_rows)
        ):
            raise ValueError(f"iid branch reports MOS generations: {prefix}")

        populations = [
            int(float(row[field]))
            for row in trace_by_key.get(key, [])
            for field in ("population_before", "population_after")
            if row.get(field, "")
        ]
        if any(population <= 0 for population in populations):
            raise ValueError(f"invalid action-trace population: {prefix}")
        if not target - tolerance <= actual <= target:
            raise ValueError(f"terminal FE outside native population tolerance: {prefix}")
        paired_prefix[(case, seed_text)][arm] = (
            provenance_target,
            tolerance,
            phase_i_fe,
            provenance["phase_i_record_sha256"],
            provenance["phase_i_candidate_sha256"],
            first_cma_status,
            first_cma_prestate,
            provenance["rng_descriptor_sha256"],
        )
        branches.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "stage": stage,
                "problem_id": case,
                "seed": int(seed_text),
                "arm": arm,
                "lane_id": arm,
                "status": "completed",
                "fresh_optimizer_execution": 1,
                "terminal_error": error,
                "terminal_target_fe": target,
                "terminal_observed_fe": actual,
                "terminal_completion_tolerance_fe": tolerance,
                "same_budget_violation": 0,
                "source_git_commit": source_git_commit,
                "source_bundle_sha256": source_bundle_sha256,
                "config_sha256": CONFIG_SHA256,
                "runtime_environment_sha256": environment_sha256,
                "phase_i_record_sha256": provenance["phase_i_record_sha256"],
                "first_cma_prestate_status": first_cma_status,
                "first_cma_prestate_sha256": first_cma_prestate,
                "rng_descriptor_sha256": provenance["rng_descriptor_sha256"],
                "terminal_record_sha256": provenance["terminal_record_sha256"],
            }
        )
    for pair_id, prefix_by_arm in paired_prefix.items():
        if set(prefix_by_arm) != set(matrix["arms"]):
            raise ValueError(f"incomplete paired provenance: {pair_id}")
        if len(set(prefix_by_arm.values())) != 1:
            raise ValueError(f"paired prefix/provenance mismatch: {pair_id}")
    return branches, provenance_rows, traces


def _validate_sampling_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    stage: str,
    matrix: Mapping[str, object],
    config: Mapping[str, object],
    provenance_rows: Sequence[Mapping[str, str]],
    branches: Sequence[Mapping[str, object]],
) -> None:
    del matrix
    provenance_by_run = _indexed_rows(
        provenance_rows,
        key_fields=("run_id",),
        artifact="mos_branch_provenance.csv",
    )
    branch_by_key = {
        (str(row["lane_id"]), str(row["problem_id"]), str(row["seed"])): row
        for row in branches
    }
    row_counts: dict[str, int] = defaultdict(int)
    primary_counts: dict[str, int] = defaultdict(int)
    rescue_counts: dict[str, int] = defaultdict(int)
    complete_counts: dict[str, int] = defaultdict(int)
    identities: set[tuple[str, ...]] = set()
    allowed_scopes = set(config["treatment"]["allowed_scope"])
    for row in rows:
        run_id = row["run_id"]
        provenance = provenance_by_run.get((run_id,))
        key = (row["lane_id"], row["problem_id"], row["seed"])
        if provenance is None or key not in branch_by_key:
            raise ValueError(f"MOS sampling row has no matching branch: {run_id}")
        if (
            row["lane_id"] != "a1_v37_mos"
            or row["sampling_mode"] != "mirrored_orthogonal"
        ):
            raise ValueError("iid arm must have zero MOS sampling rows")
        if any(
            row[field] != provenance[field]
            for field in ("lane_id", "sampling_mode", "problem_id", "seed")
        ):
            raise ValueError(f"MOS sampling/provenance identity mismatch: {run_id}")
        if row["cma_scope"] not in allowed_scopes:
            raise ValueError(f"MOS sampling scope is forbidden: {row['cma_scope']}")
        population = _integer(row, "population", minimum=1)
        dimension = _integer(row, "dimension", minimum=1)
        pair_count = _integer(row, "pair_count")
        block_count = _integer(row, "block_count", minimum=1)
        draw_count = _integer(row, "rng_draw_count", minimum=1)
        evaluated_count = _integer(row, "evaluated_count", minimum=1)
        complete_population = _boolean(row, "complete_population")
        base_count = math.ceil(population / 2)
        if (
            pair_count != population // 2
            or block_count != math.ceil(base_count / dimension)
            or draw_count != population * dimension
            or evaluated_count > population
            or complete_population != (evaluated_count == population)
        ):
            raise ValueError(f"MOS sampling geometry mismatch: {key}")
        for field in (
            "outer_iter",
            "group_index",
            "candidate_index",
            "optimizer_seed",
            "optimizer_restart_index",
            "generation",
        ):
            _integer(row, field)
        for field in ("raw_draw_sha256", "sample_sha256"):
            if not _is_hex(row[field]):
                raise ValueError(f"invalid MOS sampling hash: {key}")
        _number(row, "max_orthogonality_error")
        identity = tuple(
            row[field]
            for field in (
                "run_id",
                "lane_id",
                "problem_id",
                "seed",
                "outer_iter",
                "group_index",
                "cma_scope",
                "candidate_index",
                "optimizer_seed",
                "optimizer_restart_index",
                "generation",
            )
        )
        if identity in identities:
            raise ValueError(f"duplicate MOS generation audit: {identity}")
        identities.add(identity)
        row_counts[run_id] += 1
        primary_counts[run_id] += int(row["cma_scope"] == "v37_primary_group_cma")
        rescue_counts[run_id] += int(
            row["cma_scope"] == "v37_phase_rescue_multistart_cma"
        )
        complete_counts[run_id] += int(complete_population)

    for provenance in provenance_rows:
        run_id = provenance["run_id"]
        lane = provenance["lane_id"]
        reported_total = _integer(provenance, "mos_generation_rows")
        reported_primary = _integer(provenance, "mos_primary_generation_rows")
        reported_rescue = _integer(provenance, "mos_rescue_generation_rows")
        if (
            row_counts[run_id] != reported_total
            or primary_counts[run_id] != reported_primary
            or rescue_counts[run_id] != reported_rescue
        ):
            raise ValueError(f"MOS sampling/provenance row mismatch: {run_id}")
        if lane == "a0_v37_iid":
            if reported_total != 0:
                raise ValueError(f"iid branch contains MOS generations: {run_id}")
            continue
        branch = branch_by_key[(lane, provenance["problem_id"], provenance["seed"])]
        cc_fe_used = int(branch["terminal_observed_fe"]) - _integer(
            provenance, "phase_i_fe"
        )
        if cc_fe_used > 0 and reported_total == 0:
            raise ValueError(f"MOS CC execution has no sampling audit: {run_id}")
        if stage in {"development", "confirmation"} and complete_counts[run_id] < 1:
            raise ValueError(
                f"formal MOS run has no complete distribution generation: {run_id}"
            )


def _validate_paired_root(
    root: Path,
    *,
    stage: str,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, str]], dict[str, object], dict[str, str]]:
    results = _read_csv(root / "our_result_by_case.csv", RESULT_COLUMNS)
    if stage == "smoke":
        identities = {(row["problem_id"], int(row["seed"])) for row in results}
        cli = _matrix(config, stage, smoke_name="cli_smoke")
        trace = _matrix(config, stage, smoke_name="trace_smoke")
        cli_ids = {(case, int(seed)) for case in cli["cases"] for seed in cli["seeds"]}
        trace_ids = {(case, int(seed)) for case in trace["cases"] for seed in trace["seeds"]}
        if identities == cli_ids:
            matrix = cli
        elif identities == trace_ids:
            matrix = trace
            ledger_rows = _read_csv(
                root / "same_budget_ledger.csv", LEDGER_COLUMNS
            )
            observed_targets = {
                _integer(row, "budget_limit", minimum=1) for row in ledger_rows
            }
            allowed_targets = {
                int(trace["terminal_fe"]),
                int(config["matrices"]["trace_smoke"]["single_escalation_terminal_fe"]),
            }
            if len(observed_targets) != 1 or not observed_targets <= allowed_targets:
                raise ValueError("trace smoke terminal target mismatch")
            matrix = dict(trace)
            matrix["terminal_fe"] = next(iter(observed_targets))
        else:
            raise ValueError("smoke matrix does not match cli or trace smoke")
    else:
        matrix = _matrix(config, stage)
    environment, environment_sha256 = _runtime_environment(root)
    source_git_commit, source_bundle_sha256 = _run_source_binding(root)
    normalized, provenance, _ = _paired_branch_rows(
        root,
        stage=stage,
        matrix=matrix,
        environment_sha256=environment_sha256,
        source_git_commit=source_git_commit,
        source_bundle_sha256=source_bundle_sha256,
    )
    sampling = _read_csv(root / "mos_sampling_audit.csv", SAMPLING_COLUMNS)
    _validate_sampling_rows(
        sampling,
        stage=stage,
        matrix=matrix,
        config=config,
        provenance_rows=provenance,
        branches=normalized,
    )

    raw_names = (
        "mos_sampling_audit.csv",
        "mos_branch_provenance.csv",
        "mos_source_bundle.json",
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "action_trace.csv",
        "aob_input_manifest.csv",
        "anti_leakage_audit.csv",
        "runtime_environment.json",
        "run_manifest.md",
    )
    return normalized, sampling, {
        "matrix": matrix,
        "matrix_sha256": _matrix_sha256(matrix),
        "source_git_commit": source_git_commit,
        "source_bundle_sha256": source_bundle_sha256,
        "runtime_environment": environment,
    }, _raw_hashes(root, raw_names)


def _baseline_population(
    root: Path,
    *,
    cases: Sequence[str],
    seeds: Sequence[int],
    allowed_lanes: set[str],
    expected_terminal_fe: int,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    results = _read_csv(root / "our_result_by_case.csv", BASELINE_RESULT_COLUMNS)
    selected = [row for row in results if row["lane_id"] in allowed_lanes]
    expected = {(case, str(seed)) for case in cases for seed in seeds}
    indexed = _indexed_rows(
        selected,
        key_fields=("problem_id", "seed"),
        artifact="our_result_by_case.csv",
    )
    if set(indexed) != expected:
        raise ValueError("baseline result matrix mismatch")
    environment, environment_hash = _runtime_environment(root)
    _validate_manifest_jobs(root)
    ledgers = _read_csv(root / "same_budget_ledger.csv", LEDGER_COLUMNS)
    ledger_selected = [row for row in ledgers if row["lane_id"] in allowed_lanes]
    ledger_index = _indexed_rows(
        ledger_selected,
        key_fields=("problem_id", "seed"),
        artifact="same_budget_ledger.csv",
    )
    if set(ledger_index) != expected:
        raise ValueError("baseline ledger matrix mismatch")
    _read_csv(root / "action_trace.csv", TRACE_COLUMNS)
    lane_keys = {
        (indexed[key]["lane_id"], key[0], key[1])
        for key in expected
    }
    aob_tolerances = _validate_aob(root, lane_keys)
    normalized: list[dict[str, object]] = []
    for key in sorted(expected):
        result = indexed[key]
        ledger = ledger_index[key]
        lane = result["lane_id"]
        prefix = f"{key[0]}:seed{key[1]}"
        if (
            result["selected_action_name"] != "arac_evidence_action_controller_v37"
            or result["hcc_smoke_status"] != "completed"
            or result["fresh_optimizer_execution"] != "1"
            or result["result_source"] != "hcc_subprocess_smoke_execution"
        ):
            raise ValueError(f"baseline result integrity failed: {prefix}")
        error = _number(result, "hcc_smoke_final_error")
        observed = _integer(ledger, "actual_fe_used", minimum=1)
        target = _integer(ledger, "budget_limit", minimum=1)
        tolerance = aob_tolerances[key[0]]
        if (
            result["hcc_smoke_fe_used"] != str(observed)
            or _integer(ledger, "total_fe", minimum=1) != observed
            or _integer(ledger, "budget_aligned_fe_used", minimum=1) != observed
            or target != expected_terminal_fe
            or _integer(ledger, "configured_budget_limit", minimum=1)
            != expected_terminal_fe
            or not target - tolerance <= observed <= target
            or _flag(ledger, "same_budget_violation")
            or not _flag(ledger, "fresh_execution")
        ):
            raise ValueError(f"baseline FE integrity failed: {prefix}")
        normalized.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "stage": "baseline",
                "problem_id": key[0],
                "seed": int(key[1]),
                "arm": "v37_observer_bit_equivalent",
                "lane_id": lane,
                "status": "completed",
                "fresh_optimizer_execution": 1,
                "terminal_error": error,
                "terminal_target_fe": target,
                "terminal_observed_fe": observed,
                "terminal_completion_tolerance_fe": tolerance,
                "same_budget_violation": 0,
                "source_git_commit": "",
                "source_bundle_sha256": "",
                "config_sha256": CONFIG_SHA256,
                "runtime_environment_sha256": environment_hash,
                "phase_i_record_sha256": "",
                "first_cma_prestate_sha256": "",
                "rng_descriptor_sha256": "",
                "terminal_record_sha256": result["action_trace_sha256"],
            }
        )
    _validate_leakage(root)
    raw_names = (
        "our_result_by_case.csv",
        "same_budget_ledger.csv",
        "action_trace.csv",
        "aob_input_manifest.csv",
        "anti_leakage_audit.csv",
        "runtime_environment.json",
        "run_manifest.md",
    )
    return normalized, _raw_hashes(root, raw_names), environment


def _validate_baseline_roots(
    existing_root: Path,
    complement_root: Path,
    *,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, str]]:
    matrices = config["matrices"]
    existing_matrix = matrices["baseline_existing"]
    complement_matrix = matrices["baseline_complement"]
    manifest_path = existing_root / "hypergraph_trace_manifest.json"
    if _file_sha256(manifest_path) != existing_matrix["manifest_sha256"]:
        raise ValueError("frozen existing baseline manifest hash mismatch")
    manifest = _read_json(manifest_path)
    anchor = config["preimplementation_anchor"]
    if (
        manifest.get("status") != "pass"
        or manifest.get("source_git_commit") != anchor["git_commit"]
        or manifest.get("source_manifest_count") != existing_matrix["run_count"]
        or manifest.get("observer_calls")
        != {"objective": 0, "rng": 0, "optimizer": 0, "fe": 0}
    ):
        raise ValueError("existing baseline observer manifest failed")
    existing, existing_hashes, existing_env = _baseline_population(
        existing_root,
        cases=existing_matrix["cases"],
        seeds=existing_matrix["seeds"],
        allowed_lanes={"hypergraph_v37_observer"},
        expected_terminal_fe=int(existing_matrix["terminal_fe"]),
    )
    complement, complement_hashes, complement_env = _baseline_population(
        complement_root,
        cases=complement_matrix["cases"],
        seeds=complement_matrix["seeds"],
        allowed_lanes={"arac_evidence_action_controller_v37", "hypergraph_v37_observer"},
        expected_terminal_fe=int(complement_matrix["terminal_fe"]),
    )
    if existing_env != complement_env:
        raise ValueError("baseline runtime environments differ")
    run_manifest = (complement_root / "run_manifest.md").read_text(encoding="utf-8-sig")
    required_anchor_values = (
        anchor["git_commit"],
        anchor["hcc_runner_sha256"],
        anchor["experiment_runner_sha256"],
        anchor["vendor_cmaes_sha256"],
    )
    if not all(str(value) in run_manifest for value in required_anchor_values):
        raise ValueError("baseline complement is not bound to the frozen v37 anchor")
    branches = existing + complement
    expected = {
        (case, int(seed)) for case in config["cases"] for seed in existing_matrix["seeds"]
    }
    observed = {(str(row["problem_id"]), int(row["seed"])) for row in branches}
    if observed != expected or len(branches) != len(expected):
        raise ValueError("combined baseline is not a unique 24-case by 5-seed matrix")
    hashes = {f"existing/{name}": value for name, value in existing_hashes.items()}
    hashes.update({f"complement/{name}": value for name, value in complement_hashes.items()})
    hashes["existing/hypergraph_trace_manifest.json"] = _file_sha256(manifest_path)
    return branches, {
        "matrix": {
            "cases": list(config["cases"]),
            "seeds": list(existing_matrix["seeds"]),
            "arms": ["v37_observer_bit_equivalent"],
            "terminal_fe": int(existing_matrix["terminal_fe"]),
        },
        "matrix_sha256": _canonical_sha256(
            {
                "cases": list(config["cases"]),
                "seeds": list(existing_matrix["seeds"]),
                "arms": ["v37_observer_bit_equivalent"],
                "terminal_fe": int(existing_matrix["terminal_fe"]),
            }
        ),
        "source_git_commit": anchor["git_commit"],
        "source_bundle_sha256": manifest["source_bundle"]["bundle_sha256"],
    }, hashes


def _clopper_pearson_upper(events: int, total: int, confidence: float) -> float:
    if total <= 0 or not 0 <= events <= total:
        return 1.0
    alpha = 1.0 - confidence
    if events == total:
        return 1.0
    if events == 0:
        return 1.0 - alpha ** (1.0 / total)

    def cdf(probability: float) -> float:
        return sum(
            math.comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(events + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _clopper_pearson_lower(successes: int, total: int, confidence: float) -> float:
    if total <= 0 or not 0 <= successes <= total or successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    if successes == total:
        return alpha ** (1.0 / total)

    def upper_tail(probability: float) -> float:
        return sum(
            math.comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(successes, total + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if upper_tail(middle) > alpha:
            high = middle
        else:
            low = middle
    return low


def _build_outputs(
    branches: Sequence[Mapping[str, object]],
    *,
    stage: str,
    config: Mapping[str, object],
    paper: Mapping[str, float],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    floor = float(config["numeric"]["error_floor"])
    multiplier = float(config["numeric"]["catastrophic_multiplier"])
    core_fraction = float(config["objective"]["stable_core_minimum_seed_fraction"])
    by_arm_case_seed: dict[tuple[str, str, int], float] = {}
    outcomes: list[dict[str, object]] = []
    for row in branches:
        arm = str(row["arm"])
        case = str(row["problem_id"])
        seed = int(row["seed"])
        error = float(row["terminal_error"])
        reference = paper[case]
        normalized = error / reference
        by_arm_case_seed[(arm, case, seed)] = error
        outcomes.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "stage": stage,
                "problem_id": case,
                "seed": seed,
                "arm": arm,
                "terminal_error": error,
                "paper_best": reference,
                "normalized_error": normalized,
                "strict_paper_win": int(error < reference),
                "paper_relative_catastrophic": int(
                    error >= multiplier * reference
                ),
                "runtime_dispatch_used": 0,
            }
        )
    outcomes.sort(key=lambda row: (str(row["arm"]), int(row["seed"]), str(row["problem_id"])))
    arms = sorted({str(row["arm"]) for row in branches})
    cases = [case for case in config["cases"] if any(key[1] == case for key in by_arm_case_seed)]
    seeds = sorted({int(row["seed"]) for row in branches})

    seed_summaries: list[dict[str, object]] = []
    win_sets: dict[tuple[str, int], set[str]] = {}
    for arm in arms:
        for seed in seeds:
            ratios = sorted(
                by_arm_case_seed[(arm, case, seed)] / paper[case]
                for case in cases
                if (arm, case, seed) in by_arm_case_seed
            )
            wins = {
                case
                for case in cases
                if (arm, case, seed) in by_arm_case_seed
                and by_arm_case_seed[(arm, case, seed)] < paper[case]
            }
            win_sets[(arm, seed)] = wins
            q13: float | str = ratios[12] if len(ratios) >= 13 else ""
            seed_summaries.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": stage,
                    "arm": arm,
                    "seed": seed,
                    "case_count": len(ratios),
                    "strict_win_count": len(wins),
                    "q13": q13,
                    "single_seed_13_win": int(len(ratios) >= 13 and len(wins) >= 13),
                }
            )

    case_summaries: list[dict[str, object]] = []
    for arm in arms:
        for case in cases:
            errors = [
                by_arm_case_seed[(arm, case, seed)]
                for seed in seeds
                if (arm, case, seed) in by_arm_case_seed
            ]
            reference = paper[case]
            win_count = sum(error < reference for error in errors)
            required_core_wins = math.ceil(core_fraction * len(errors))
            case_summaries.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "stage": stage,
                    "arm": arm,
                    "problem_id": case,
                    "seed_count": len(errors),
                    "paper_best": reference,
                    "arithmetic_mean_error": statistics.fmean(errors),
                    "arithmetic_mean_win": int(statistics.fmean(errors) < reference),
                    "seed_win_count": win_count,
                    "seed_win_fraction": win_count / len(errors),
                    "stable_core_case": int(win_count >= required_core_wins),
                    "strict_common_win": int(win_count == len(errors)),
                    "best_error": min(errors),
                    "best_normalized_error": min(errors) / reference,
                    "paper_relative_catastrophic_count": sum(
                        error >= multiplier * reference for error in errors
                    ),
                }
            )

    pair_rows: list[dict[str, object]] = []
    paired = set(config["treatment"]["arms"]) <= set(arms)
    if paired:
        a0, a1 = config["treatment"]["arms"]
        branch_index = {
            (str(row["arm"]), str(row["problem_id"]), int(row["seed"])): row
            for row in branches
        }
        for case in cases:
            for seed in seeds:
                baseline = by_arm_case_seed[(a0, case, seed)]
                treatment = by_arm_case_seed[(a1, case, seed)]
                a0_branch = branch_index[(a0, case, seed)]
                a1_branch = branch_index[(a1, case, seed)]
                phase_match = a0_branch["phase_i_record_sha256"] == a1_branch["phase_i_record_sha256"]
                prestate_status_match = (
                    a0_branch["first_cma_prestate_status"]
                    == a1_branch["first_cma_prestate_status"]
                )
                prestate_match = a0_branch["first_cma_prestate_sha256"] == a1_branch["first_cma_prestate_sha256"]
                pair_rows.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "stage": stage,
                        "problem_id": case,
                        "seed": seed,
                        "a0_terminal_error": baseline,
                        "a1_terminal_error": treatment,
                        "tau": math.log(max(baseline, floor) / max(treatment, floor)),
                        "paired_catastrophic": int(
                            treatment >= multiplier * max(baseline, floor)
                        ),
                        "phase_i_hash_match": int(phase_match),
                        "first_cma_prestate_status_match": int(
                            prestate_status_match
                        ),
                        "first_cma_prestate_hash_match": int(prestate_match),
                        "pair_integrity": int(
                            phase_match and prestate_status_match and prestate_match
                        ),
                    }
                )

    arm_metrics: dict[str, dict[str, object]] = {}
    for arm in arms:
        arm_seed_rows = [row for row in seed_summaries if row["arm"] == arm]
        arm_case_rows = [row for row in case_summaries if row["arm"] == arm]
        similarities: list[float] = []
        for left, right in combinations(seeds, 2):
            left_set = win_sets[(arm, left)]
            right_set = win_sets[(arm, right)]
            union = left_set | right_set
            similarities.append(len(left_set & right_set) / len(union) if union else 1.0)
        seed_successes = sum(int(row["single_seed_13_win"]) for row in arm_seed_rows)
        arm_metrics[arm] = {
            "all_seed_q13_strictly_below_one": bool(
                arm_seed_rows
                and all(row["q13"] != "" and float(row["q13"]) < 1.0 for row in arm_seed_rows)
            ),
            "minimum_seed_win_count": min(
                (int(row["strict_win_count"]) for row in arm_seed_rows), default=0
            ),
            "case_arithmetic_mean_win_count": sum(
                int(row["arithmetic_mean_win"]) for row in arm_case_rows
            ),
            "stable_core_case_count": sum(int(row["stable_core_case"]) for row in arm_case_rows),
            "strict_common_win_count": sum(int(row["strict_common_win"]) for row in arm_case_rows),
            "pairwise_jaccard_minimum": min(similarities) if similarities else 1.0,
            "pairwise_jaccard_mean": statistics.fmean(similarities) if similarities else 1.0,
            "upper_tail_paper_win_count": sum(
                float(row["best_normalized_error"]) < 1.0 for row in arm_case_rows
            ),
            "upper_tail_mean_log_normalized_best": statistics.fmean(
                math.log(max(float(row["best_error"]), floor) / float(row["paper_best"]))
                for row in arm_case_rows
            ),
            "paper_relative_catastrophic_count": sum(
                int(row["paper_relative_catastrophic_count"]) for row in arm_case_rows
            ),
            "seed_success_count": seed_successes,
            "seed_success_total": len(arm_seed_rows),
            "seed_success_cp_lcb_95": _clopper_pearson_lower(
                seed_successes,
                len(arm_seed_rows),
                float(config["numeric"]["confidence_level"]),
            ),
            "seed_success_cp_ucb_95": _clopper_pearson_upper(
                seed_successes,
                len(arm_seed_rows),
                float(config["numeric"]["confidence_level"]),
            ),
        }

    paired_metrics: dict[str, object] = {}
    if pair_rows:
        tau_values = sorted(float(row["tau"]) for row in pair_rows)
        if stage == "confirmation":
            tail_count = int(config["confirmation_gate"]["worst_ten_percent_count"])
        else:
            tail_count = max(
                1,
                math.ceil(float(config["numeric"]["cvar_fraction"]) * len(pair_rows)),
            )
        catastrophic_count = sum(int(row["paired_catastrophic"]) for row in pair_rows)
        paired_metrics = {
            "pair_count": len(pair_rows),
            "paired_catastrophic_count": catastrophic_count,
            "paired_catastrophic_cp_ucb_95": _clopper_pearson_upper(
                catastrophic_count,
                len(pair_rows),
                float(config["numeric"]["confidence_level"]),
            ),
            "worst_ten_percent_count": tail_count,
            "worst_ten_percent_cvar": statistics.fmean(tau_values[:tail_count]),
        }
    return outcomes, seed_summaries, case_summaries, pair_rows, {
        "arms": arm_metrics,
        "paired": paired_metrics,
    }


def _development_gate_binding(
    path: Path | None,
    *,
    config: Mapping[str, object],
    source_bundle_sha256: str,
) -> tuple[dict[str, object] | None, str | None, list[str]]:
    if path is None:
        return None, None, ["confirmation requires --development-gate"]
    try:
        payload = _read_json(path.resolve())
        checks = payload.get("checks")
        blockers = payload.get("integrity", {}).get("blockers")
        expected_matrix_hash = _matrix_sha256(_matrix(config, "development"))
        if (
            payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("stage") != "development"
            or payload.get("status") != "development_pass"
            or payload.get("config_sha256") != CONFIG_SHA256
            or payload.get("matrix_sha256") != expected_matrix_hash
            or payload.get("source_bundle_sha256") != source_bundle_sha256
            or not isinstance(checks, Mapping)
            or not checks
            or not all(value is True for value in checks.values())
            or blockers != []
        ):
            raise ValueError("development gate protocol/status/hash/check binding failed")
        source_root = Path(str(payload.get("source_root", ""))).resolve()
        input_hashes = payload.get("input_artifact_sha256")
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            raise ValueError("development gate input hashes missing")
        for name, expected_hash in input_hashes.items():
            artifact = source_root / str(name)
            if not artifact.is_file() or _file_sha256(artifact) != expected_hash:
                raise ValueError(f"development artifact changed: {name}")
        return payload, _file_sha256(path.resolve()), []
    except (OSError, TypeError, ValueError) as exc:
        return None, None, [str(exc)]


def _gate_checks(
    *,
    stage: str,
    metrics: Mapping[str, object],
    config: Mapping[str, object],
    development_binding_ok: bool,
) -> dict[str, bool]:
    if stage == "baseline":
        return {"integrity_fraction": True}
    if stage == "smoke":
        return {"integrity_fraction": True, "sampling_audit_complete": True}
    gate_config = config[f"{stage}_gate"]
    treatment_arm = config["treatment"]["arms"][1]
    baseline_arm = config["treatment"]["arms"][0]
    treatment = metrics["arms"][treatment_arm]
    baseline = metrics["arms"][baseline_arm]
    paired = metrics["paired"]
    checks = {
        "integrity_fraction": True,
        "all_seed_q13_strictly_below_one": bool(
            treatment["all_seed_q13_strictly_below_one"]
        ),
        "minimum_case_arithmetic_mean_wins": int(
            treatment["case_arithmetic_mean_win_count"]
        )
        >= int(gate_config["minimum_case_arithmetic_mean_wins"]),
        "minimum_stable_core_cases": int(treatment["stable_core_case_count"])
        >= int(gate_config["minimum_stable_core_cases"]),
        "maximum_paired_catastrophic_events": int(
            paired["paired_catastrophic_count"]
        )
        <= int(gate_config["maximum_paired_catastrophic_events"]),
        "worst_ten_percent_cvar_strictly_positive": float(
            paired["worst_ten_percent_cvar"]
        )
        > 0.0,
        "upper_tail_paper_win_count_not_below_v37": int(
            treatment["upper_tail_paper_win_count"]
        )
        >= int(baseline["upper_tail_paper_win_count"]),
        "upper_tail_mean_log_normalized_best_not_above_v37": float(
            treatment["upper_tail_mean_log_normalized_best"]
        )
        <= float(baseline["upper_tail_mean_log_normalized_best"]),
    }
    if stage == "confirmation":
        checks["prior_development_gate_bound"] = development_binding_ok
    return checks


def _artifact_names(config: Mapping[str, object]) -> tuple[str, ...]:
    artifacts = tuple(str(name) for name in config["artifacts"])
    if len(artifacts) != 8 or len(set(artifacts)) != 8:
        raise ValueError("frozen artifact list mismatch")
    return artifacts


def _finish(
    output_root: Path,
    *,
    gate: dict[str, object],
    manifest_base: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    names = _artifact_names(config)
    gate_path = output_root / names[6]
    manifest_path = output_root / names[7]
    _write_json(gate_path, gate)
    csv_hashes = {
        name: _file_sha256(output_root / name)
        for name in names[:6]
        if (output_root / name).is_file()
    }
    manifest = {
        **manifest_base,
        "status": gate["status"],
        "gate_sha256": _file_sha256(gate_path),
        "output_csv_sha256": csv_hashes,
    }
    _write_json(manifest_path, manifest)
    return gate


def audit_v37_mos_stability(
    output_root: Path | str,
    *,
    stage: str,
    existing_root: Path | str | None = None,
    complement_root: Path | str | None = None,
    development_gate_path: Path | str | None = None,
) -> dict[str, object]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _load_config()
    input_hashes: dict[str, str] = {}
    context: dict[str, object] = {}
    branches: list[dict[str, object]] = []
    sampling: list[dict[str, str]] = []
    blockers: list[str] = []
    development_gate_sha256: str | None = None
    try:
        if stage == "baseline":
            if existing_root is None or complement_root is None:
                raise ValueError("baseline requires existing and complement roots")
            branches, context, input_hashes = _validate_baseline_roots(
                Path(existing_root).resolve(),
                Path(complement_root).resolve(),
                config=config,
            )
            _write_csv(output / "mos_sampling_audit.csv", (), SAMPLING_COLUMNS)
            _write_csv(output / "mos_branch_manifest.csv", branches, BRANCH_COLUMNS)
        else:
            if existing_root is not None or complement_root is not None:
                raise ValueError("paired stages accept only their two-arm output root")
            branches, sampling, context, input_hashes = _validate_paired_root(
                output, stage=stage, config=config
            )
            _write_csv(output / "mos_branch_manifest.csv", branches, BRANCH_COLUMNS)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        blockers.append(str(exc))

    development_gate: dict[str, object] | None = None
    if not blockers and stage == "confirmation":
        development_gate, development_gate_sha256, binding_blockers = _development_gate_binding(
            Path(development_gate_path).resolve() if development_gate_path else None,
            config=config,
            source_bundle_sha256=str(context["source_bundle_sha256"]),
        )
        blockers.extend(binding_blockers)

    names = _artifact_names(config)
    manifest_base = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "config_sha256": CONFIG_SHA256,
        "paper_best_sha256": _file_sha256(PAPER_BEST_PATH),
        "source_root": str(output),
        "input_artifact_sha256": input_hashes,
        "matrix_sha256": context.get("matrix_sha256"),
        "source_git_commit": context.get("source_git_commit"),
        "source_bundle_sha256": context.get("source_bundle_sha256"),
        "development_gate_sha256": development_gate_sha256,
        "paper_best_joined_after_raw_hash_freeze": not blockers,
    }
    if blockers:
        gate = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "status": f"{stage}_no_go",
            "config_sha256": CONFIG_SHA256,
            "matrix_sha256": context.get("matrix_sha256"),
            "source_root": str(output),
            "source_bundle_sha256": context.get("source_bundle_sha256"),
            "input_artifact_sha256": input_hashes,
            "checks": {"integrity_fraction": False},
            "integrity": {"blockers": blockers},
            "runtime_registration_authorized": False,
        }
        return _finish(output, gate=gate, manifest_base=manifest_base, config=config)

    try:
        paper = _paper_best(config)
        outcomes, seed_rows, case_rows, pair_rows, metrics = _build_outputs(
            branches, stage=stage, config=config, paper=paper
        )
        _write_csv(output / names[2], outcomes, OUTCOME_COLUMNS)
        _write_csv(output / names[3], seed_rows, SEED_SUMMARY_COLUMNS)
        _write_csv(output / names[4], case_rows, CASE_SUMMARY_COLUMNS)
        _write_csv(output / names[5], pair_rows, PAIR_COLUMNS)
        checks = _gate_checks(
            stage=stage,
            metrics=metrics,
            config=config,
            development_binding_ok=(stage != "confirmation" or development_gate is not None),
        )
        passed = bool(checks) and all(checks.values())
        gate = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "status": f"{stage}_pass" if passed else f"{stage}_no_go",
            "config_sha256": CONFIG_SHA256,
            "matrix_sha256": context["matrix_sha256"],
            "source_root": str(output),
            "source_git_commit": context["source_git_commit"],
            "source_bundle_sha256": context["source_bundle_sha256"],
            "input_artifact_sha256": input_hashes,
            "paper_best_sha256": _file_sha256(PAPER_BEST_PATH),
            "checks": checks,
            "metrics": metrics,
            "integrity": {"blockers": []},
            "development_gate_sha256": development_gate_sha256,
            "runtime_registration_authorized": bool(stage == "confirmation" and passed),
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        gate = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "status": f"{stage}_no_go",
            "config_sha256": CONFIG_SHA256,
            "matrix_sha256": context.get("matrix_sha256"),
            "source_root": str(output),
            "source_bundle_sha256": context.get("source_bundle_sha256"),
            "input_artifact_sha256": input_hashes,
            "checks": {"integrity_fraction": False},
            "integrity": {"blockers": [str(exc)]},
            "runtime_registration_authorized": False,
        }
    return _finish(output, gate=gate, manifest_base=manifest_base, config=config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--existing-root", type=Path)
    parser.add_argument("--complement-root", type=Path)
    parser.add_argument("--development-gate", type=Path)
    args = parser.parse_args(argv)
    gate = audit_v37_mos_stability(
        args.output_root,
        stage=args.stage,
        existing_root=args.existing_root,
        complement_root=args.complement_root,
        development_gate_path=args.development_gate,
    )
    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=False))
    return 0 if gate["status"] == f"{args.stage}_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
