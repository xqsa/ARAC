"""G-A checkpoint gate for the soft-RDDSM mainline v10 candidate.

Produces the composed Phase-I checkpoints on AOB-24 x seeds 117-123 (the
G-B matrix seeds, paired with the frozen v9 four-arm matrix) and verifies
the preregistered contract:

1. coverage: 24 cases x 7 seeds, every run at the exact 180,000 FE boundary;
2. landscape parity: every probe-derived landscape feature is BITWISE equal
   to the frozen v9 checkpoint of the same (case, seed) - the v10 probe
   block replicates the v9 rng namespaces and evaluation order verbatim;
3. discovery precision: every case with a non-empty recovered shared set
   has precision 1.0 against the AOB construction truth, and every
   zero-overlap case (function id 1: A1/E1/R1/S1) recovers an empty set
   (zero false positives) - the C1 convention;
4. determinism: the preregistered replay case (S3/117) reproduces its
   checkpoint hash bit-identically on a second fresh ledger;
5. recorded (not gated): shared recall per case against the C1 frozen
   range, and the dispatch-rule features (tail_log10_gain,
   structural_relation_density) v9-vs-v10 for transfer reference.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import canonical_sha256
from experiments.soft_rddsm_aob_baseline_v2 import _truth_groups
from experiments.upgrade.soft_rddsm_mainline_v10.phase1_v10 import (
    V10_DISCOVERY_WINDOW,
    V10_PROTOCOL,
    landscape_feature_names,
    run_phase1_v10,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("ga_protocol_v1.json")
RECEIPT_SCHEMA = "arac-upgrade-soft-rddsm-mainline-v10-ga-receipt-v1"
FAILURE_SCHEMA = "arac-upgrade-soft-rddsm-mainline-v10-ga-failure-v1"
V9_CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/historical_recovery_fixed_expert_v1/checkpoints"
CASES = tuple(
    f"{family}{index}"
    for family in ("A", "E", "R", "S")
    for index in range(1, 7)
)
SEEDS = (117, 118, 119, 120, 121, 122, 123)
REPLAY_CASE = "S3"
REPLAY_SEED = 117
PHASE1_FES = 180_000


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": __import__("sys").executable, "numpy_version": np.__version__, "threadpools": pools}


def _v9_features(case_id: str, seed: int) -> dict[str, float]:
    wrapper = _load_json(V9_CHECKPOINT_ROOT / case_id / f"seed_{seed}" / "checkpoint.json")
    names = wrapper["checkpoint"]["feature_names"]
    values = wrapper["checkpoint"]["feature_values"]
    return dict(zip(names, values, strict=True))


def _truth_audit(case_id: str, recovered: frozenset[int]) -> dict[str, Any]:
    problem_dimension = 1000
    truth_groups = _truth_groups(AobBenchmark().data_root, int(case_id[1]))
    membership = np.zeros(problem_dimension, dtype=int)
    for group in truth_groups:
        membership[sorted(group)] += 1
    truth_shared = frozenset(int(v) for v in np.nonzero(membership > 1)[0])
    recall = len(recovered & truth_shared) / len(truth_shared) if truth_shared else None
    precision = len(recovered & truth_shared) / len(recovered) if recovered else None
    return {
        "truth_shared_count": len(truth_shared),
        "recovered_count": len(recovered),
        "shared_recall": recall,
        "shared_precision": precision,
    }


@dataclass(frozen=True)
class GaJob:
    case_id: str
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "checkpoints" / self.case_id / f"seed_{self.run_seed}" / "receipt.json"

    @property
    def key(self) -> str:
        return f"ga:{self.case_id}:seed-{self.run_seed}"


def _run_job(job: GaJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            problem = AobBenchmark().load(job.case_id)
            result = run_phase1_v10(problem, run_seed=job.run_seed)
            checkpoint = result.checkpoint
            if checkpoint.phase1_fes != PHASE1_FES:
                raise RuntimeError(f"{job.key} boundary drifted: {checkpoint.phase1_fes}")
            replay = run_phase1_v10(problem, run_seed=job.run_seed)
            determinism = replay.checkpoint.checkpoint_hash == checkpoint.checkpoint_hash

            v9_features = _v9_features(job.case_id, job.run_seed)
            v10_features = dict(zip(checkpoint.feature_names, checkpoint.feature_values, strict=True))
            landscape_names = landscape_feature_names()
            parity = {
                name: {
                    "v9": v9_features[name],
                    "v10": v10_features[name],
                    "bitwise_equal": v9_features[name] == v10_features[name],
                }
                for name in landscape_names
            }
            dispatch_features = {
                "tail_log10_gain": {
                    "v9": v9_features["tail_log10_gain"],
                    "v10": v10_features["tail_log10_gain"],
                },
                "structural_relation_density": {
                    "v9": v9_features["structural_relation_density"],
                    "v10": v10_features["structural_relation_density"],
                },
            }
            audit = _truth_audit(job.case_id, frozenset(result.shared_candidates))
            body = {
                "schema_version": RECEIPT_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "protocol": V10_PROTOCOL,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "checkpoint": checkpoint.payload(),
                "discovery_fes": result.discovery_fes,
                "topup_fes": result.topup_fes,
                "discovery_window": V10_DISCOVERY_WINDOW,
                "incumbent_error": checkpoint.incumbent_error,
                "landscape_parity": parity,
                "landscape_parity_all": all(row["bitwise_equal"] for row in parity.values()),
                "dispatch_features": dispatch_features,
                "truth_audit": audit,
                "determinism_replay": determinism,
                "runtime": runtime,
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            body["receipt_hash"] = canonical_sha256(body)
            _write_json(job.receipt_path, body)
            return body
    except BaseException as exc:
        _write_json(
            job.output_root / "failures" / f"{job.key.replace(':', '_')}.json",
            {
                "schema_version": FAILURE_SCHEMA,
                "key": job.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    case_rows = []
    for case_id in protocol["cases"]:
        seed_rows = []
        for seed in protocol["seeds"]:
            receipt = _load_json(output_root / "checkpoints" / case_id / f"seed_{seed}" / "receipt.json")
            if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
                raise ValueError(f"G-A receipt hash drifted: {case_id}/{seed}")
            audit = receipt["truth_audit"]
            seed_rows.append(
                {
                    "seed": seed,
                    "landscape_parity_all": receipt["landscape_parity_all"],
                    "determinism_replay": receipt["determinism_replay"],
                    "phase1_fes": receipt["checkpoint"]["phase1_fes"],
                    "recall": audit["shared_recall"],
                    "precision": audit["shared_precision"],
                    "recovered_count": audit["recovered_count"],
                    "truth_shared_count": audit["truth_shared_count"],
                    "seed_passed": bool(
                        receipt["landscape_parity_all"]
                        and receipt["determinism_replay"]
                        and receipt["checkpoint"]["phase1_fes"] == PHASE1_FES
                        and (audit["shared_precision"] is None or audit["shared_precision"] == 1.0)
                    ),
                }
            )
        case_rows.append(
            {
                "case_id": case_id,
                "all_seeds_passed": all(row["seed_passed"] for row in seed_rows),
                "seed_rows": seed_rows,
            }
        )
    zero_overlap_cases = {f"{family}1" for family in ("A", "E", "R", "S")}
    checks = {
        "coverage_complete": len(case_rows) == 24 and all(len(row["seed_rows"]) == len(SEEDS) for row in case_rows),
        "landscape_parity_all": all(row["all_seeds_passed"] for row in case_rows),
        "precision_perfect": all(
            row["seed_passed"] for row in case_rows
        ),
        "zero_overlap_no_false_positives": all(
            all(s_row["recovered_count"] == 0 for s_row in row["seed_rows"])
            for row in case_rows
            if row["case_id"] in zero_overlap_cases
        ),
        "replay_determinism": all(
            s_row["determinism_replay"] for row in case_rows for s_row in row["seed_rows"]
        ),
    }
    recalls = [
        s_row["recall"]
        for row in case_rows
        for s_row in row["seed_rows"]
        if s_row["recall"] is not None
    ]
    body = {
        "schema_version": "arac-upgrade-soft-rddsm-mainline-v10-ga-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "case_rows": case_rows,
        "recall_reference": {
            "observed_range": [min(recalls) if recalls else None, max(recalls) if recalls else None],
            "c1_frozen_range": [0.789, 1.0],
            "note": "recorded reference, not gated",
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
        "gb_authorized": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = _load_json(resolved)
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-soft-rddsm-mainline-v10-ga-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"G-A output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("G-A manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = [
        GaJob(case_id=case, run_seed=int(seed), output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for case in protocol["cases"]
        for seed in protocol["seeds"]
    ]
    pending = [job for job in jobs if not job.receipt_path.is_file()]
    failures = 0
    if pending:
        with ProcessPoolExecutor(max_workers=protocol.get("max_workers", 8)) as pool:
            futures = {pool.submit(_run_job, job): job.key for job in pending}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    receipt = future.result()
                    print(f"[ga] done {key} parity={receipt['landscape_parity_all']} precision={receipt['truth_audit']['shared_precision']}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[ga] FAILED {key}", flush=True)
    if failures:
        raise RuntimeError(f"G-A had {failures} failed jobs; inspect {output_root / 'failures'}")
    summary = summarize(protocol, output_root)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        protocol = _load_json(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, resume=args.resume)
    print(json.dumps({"stage": "ga", "gate_passed": summary["gate_passed"], "checks": summary["checks"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["run_stage", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
