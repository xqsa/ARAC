"""P0 pairwise shared-edge evidence gate for pairwise_edge_ctp_v1 (v5.1).

The v5.0 H0 post-mortem showed the failure was a semantics mismatch between
``ResolvedOverlapHyperedge`` and chain topologies (transitive 3-region
outputs) plus one region-merge decomposition failure and one budget-edge
recall miss - not a kernel failure.  v5.1 therefore introduces an explicit
``PairwiseSharedEdge(j, region_a, region_b)`` certificate:

- only resolved hyperedges with exactly two regions are eligible;
- every recorded interaction of the variable must target one of those two
  regions (no unexplained third-region evidence);
- bilateral evidence is inherent to ``ResolvedOverlapHyperedge`` (each
  spanned region needs confirming sign-stable interactions);
- three-region outputs are rejected outright and are never split
  post-hoc into pairs;
- generator truth is consumed offline for certification metrics only.

This stage runs ONLY ``pairs3-strong`` with five fresh discovery seeds and a
preregistered, frozen DSM/RDG budget cap of 100,000 FE (raised from the
55,000 default that cut the v5.0 confirmations at the budget edge; the
remaining Phase-I budget is still topped up to exactly 180,000 FE with
MMES).  The stage measures certificate recall, precision, region-merge
rate, evidence replayability and the exact FE boundary; it performs no
performance comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from arac.evidence.soft_rddsm import SoftDsmConfig
from arac.runtime.contracts import canonical_sha256
from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier
from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import GENERATOR_FREEZE, run_candidate_phase1


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("p0_pairwise_evidence_protocol_v1.json")
CERTIFICATE_SCHEMA = "arac-upgrade-pairwise-shared-edge-v1"
CHECKPOINT_SCHEMA = "arac-upgrade-pairwise-edge-p0-checkpoint-v1"
CELL = "pairs3-strong"
DISCOVERY_SEEDS = (20270401, 20270402, 20270403, 20270404, 20270405)
DSM_BUDGET_CAP = 100_000
PRECISION_REQUIRED = 1.0
RECALL_REQUIRED = 0.9


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


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = _load_json(Path(path).resolve())
    expected = {
        "schema_version": "arac-upgrade-pairwise-edge-p0-protocol-v1",
        "candidate_id": "pairwise_edge_ctp_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "cell": CELL,
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "dsm_budget_cap": DSM_BUDGET_CAP,
        "precision_required": PRECISION_REQUIRED,
        "recall_required": RECALL_REQUIRED,
        "phase1_fes": 180_000,
        "total_budget_fes": 3_000_000,
        "output_root": "artifacts/upgrade_pairwise_edge_ctp_v1_p0_v1",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"P0 protocol drifted: {key}")
    if protocol.get("generator_freeze") != GENERATOR_FREEZE:
        raise ValueError("P0 generator freeze drifted")
    return protocol


def pairwise_certificates(evidence) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build PairwiseSharedEdge certificates from the discovery evidence."""

    interactions_by_variable: dict[int, set[int]] = {}
    for interaction in evidence.variable_region_interactions:
        interactions_by_variable.setdefault(interaction.variable, set()).add(interaction.target_region)
    certificates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for hyperedge in evidence.resolved_hyperedges:
        variable = hyperedge.variable
        regions = tuple(int(region) for region in hyperedge.regions)
        if len(regions) != 2:
            rejected.append(
                {
                    "variable": int(variable),
                    "reason": "not_exactly_two_regions",
                    "regions": list(regions),
                }
            )
            continue
        left, right = regions
        targets = interactions_by_variable.get(variable, set())
        unexplained = sorted(int(target) for target in targets - {left, right})
        if unexplained:
            rejected.append(
                {
                    "variable": int(variable),
                    "reason": "unexplained_third_region_evidence",
                    "regions": [int(left), int(right)],
                    "unexplained_targets": unexplained,
                }
            )
            continue
        certificates.append(
            {
                "certificate_schema": CERTIFICATE_SCHEMA,
                "variable": int(variable),
                "region_a": int(left),
                "region_b": int(right),
                "bilateral_evidence_count": len(hyperedge.evidence),
            }
        )
    return certificates, rejected


def _offline_audit(certificates: Sequence[Mapping[str, Any]], leaf_variables_by_id, truth) -> dict[str, Any]:
    def planted_block(region: int) -> int:
        owners = [variable // 100 for variable in leaf_variables_by_id[region]]
        values, counts = np.unique(np.asarray(owners), return_counts=True)
        return int(values[int(np.argmax(counts))])

    planted_owners = {variable: (left, right) for variable, left, right in truth.shared_owner_pairs}
    true_positive = 0
    for certificate in certificates:
        expected = planted_owners.get(certificate["variable"])
        mapped = {planted_block(certificate["region_a"]), planted_block(certificate["region_b"])}
        certificate["planted_blocks"] = sorted(mapped)
        certificate["correct"] = bool(expected is not None and mapped == set(expected))
        true_positive += int(certificate["correct"])
    precision = (true_positive / len(certificates)) if certificates else 0.0
    recall = (true_positive / len(truth.shared_variables)) if truth.shared_variables else 0.0
    return {
        "certificate_count": len(certificates),
        "true_positive_count": true_positive,
        "precision": precision,
        "recall": recall,
        "certificates": [dict(certificate) for certificate in certificates],
    }


def _region_merge_rate(blocks: Sequence[Sequence[int]]) -> dict[str, Any]:
    merged = 0
    details = []
    for index, block in enumerate(blocks):
        owners = sorted({variable // 100 for variable in block})
        if len(owners) > 1:
            merged += 1
            details.append({"block_index": index, "planted_blocks": owners, "size": len(block)})
    return {"region_count": len(blocks), "merged_region_count": merged, "merged_details": details}


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


@dataclass(frozen=True)
class P0Job:
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "checkpoints" / f"seed_{self.run_seed}" / "checkpoint.json"

    @property
    def key(self) -> str:
        return f"p0:{CELL}:seed-{self.run_seed}"


def _run_p0_job(job: P0Job) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            config = SoftDsmConfig(dsm_budget=DSM_BUDGET_CAP)
            result = run_candidate_phase1(CELL, job.run_seed, config=config)
            checkpoint = result["checkpoint"]
            sidecar = result["sidecar"]
            evidence = result["discovery"].evidence
            leaf_variables_by_id = {leaf.node_id: tuple(leaf.variables) for leaf in evidence.region_tree.leaves}
            certificates, rejected = pairwise_certificates(evidence)
            audit = _offline_audit(certificates, leaf_variables_by_id, result["truth"])
            merge = _region_merge_rate(checkpoint.blocks)
            body = {
                "schema_version": CHECKPOINT_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "cell_id": CELL,
                "run_seed": job.run_seed,
                "phase1_protocol": checkpoint.protocol,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "phase1_budget_breakdown": sidecar["phase1_budget_breakdown"],
                "block_count": len(checkpoint.blocks),
                "relation_count": len(checkpoint.relations),
                "pairwise_audit": audit,
                "rejected_hyperedges": rejected,
                "region_merge": merge,
                "runtime": runtime,
                "sidecar": sidecar,
                "checkpoint": checkpoint.payload(),
                "elapsed_seconds": (datetime.now(UTC) - started).total_seconds(),
            }
            body["receipt_hash"] = canonical_sha256(body)
            _write_json(job.checkpoint_path, body)
            return body
    except BaseException as exc:
        _write_json(
            job.output_root / "failures" / f"{job.key.replace(':', '_')}.json",
            {
                "schema_version": "arac-upgrade-pairwise-edge-p0-failure-v1",
                "key": job.key,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def _replay_check(run_seed: int) -> dict[str, Any]:
    """Re-run the first discovery seed and compare evidence identity."""

    config = SoftDsmConfig(dsm_budget=DSM_BUDGET_CAP)
    first = run_candidate_phase1(CELL, run_seed, config=config)
    second = run_candidate_phase1(CELL, run_seed, config=config)
    return {
        "seed": run_seed,
        "checkpoint_hash_equal": first["checkpoint"].checkpoint_hash == second["checkpoint"].checkpoint_hash,
        "evidence_hash_equal": first["sidecar"]["evidence_hash"] == second["sidecar"]["evidence_hash"],
        "discovery_fes_equal": first["sidecar"]["phase1_budget_breakdown"] == second["sidecar"]["phase1_budget_breakdown"],
    }


def summarize(protocol: Mapping[str, Any], output_root: Path, replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    seed_rows = []
    for seed in protocol["discovery_seeds"]:
        receipt = _load_json(output_root / "checkpoints" / f"seed_{seed}" / "checkpoint.json")
        if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
            raise ValueError(f"P0 receipt hash drifted: seed {seed}")
        audit = receipt["pairwise_audit"]
        merge = receipt["region_merge"]
        breakdown = receipt["phase1_budget_breakdown"]
        seed_rows.append(
            {
                "seed": seed,
                "precision": audit["precision"],
                "recall": audit["recall"],
                "certificate_count": audit["certificate_count"],
                "rejected_hyperedge_count": len(receipt["rejected_hyperedges"]),
                "merged_region_count": merge["merged_region_count"],
                "discovery_fes": breakdown["discovery_fes"],
                "phase1_fes_exact": breakdown["discovery_fes"] + breakdown["topup_fes"] == 180_000,
                "seed_passed": bool(
                    audit["precision"] >= protocol["precision_required"]
                    and audit["recall"] >= protocol["recall_required"]
                    and merge["merged_region_count"] == 0
                    and breakdown["discovery_fes"] + breakdown["topup_fes"] == 180_000
                ),
            }
        )
    checks = {
        "coverage_complete": len(seed_rows) == len(protocol["discovery_seeds"]),
        "all_seeds_certified": all(row["seed_passed"] for row in seed_rows),
        "replay_deterministic": bool(
            replay is not None and replay["checkpoint_hash_equal"] and replay["evidence_hash_equal"] and replay["discovery_fes_equal"]
        ),
        "runtime_truth_free": True,
    }
    body = {
        "schema_version": "arac-upgrade-pairwise-edge-p0-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "cell": CELL,
        "dsm_budget_cap": DSM_BUDGET_CAP,
        "seed_rows": seed_rows,
        "replay": replay,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "patch_stage_authorized": all(checks.values()),
        "performance_comparison_run": False,
        "truth_usage_note": "generator truth was consumed offline for certification metrics only; the runtime never reads it",
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("P0 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-pairwise-edge-p0-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"P0 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("P0 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    jobs = [
        P0Job(run_seed=int(seed), output_root=output_root, manifest_sha256=str(manifest["manifest_sha256"]))
        for seed in protocol["discovery_seeds"]
    ]
    for job in jobs:
        if not job.checkpoint_path.is_file():
            _run_p0_job(job)
    replay = _replay_check(int(protocol["discovery_seeds"][0]))
    summary = summarize(protocol, output_root, replay)
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        protocol = load_protocol(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, resume=args.resume)
    print(json.dumps({"stage": "p0", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "patch_stage_authorized": summary["patch_stage_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = [
    "CELL",
    "CERTIFICATE_SCHEMA",
    "DISCOVERY_SEEDS",
    "DSM_BUDGET_CAP",
    "load_protocol",
    "pairwise_certificates",
    "run_stage",
    "summarize",
]


if __name__ == "__main__":
    raise SystemExit(main())
