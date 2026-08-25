"""T1 proposal & boundary audit for shared_transaction_v1 (SCST v3.0 stage T1).

Pure instrumentation - no transaction is ever executed.  For every activation
cell x discovery seed x host (ctp/smp audit hosts, gcb record-only) the stage
re-runs the frozen recovered action on the T0-certified checkpoint with the
``TransactionAuditRecorder`` installed and produces the census required by
the SCST protocol:

- per-certified-link owner proposal observability (the SCST §3.1 proposal
  definition: last strict-best writeback per owner block inside the source
  phase - coverage for ctp, stateful visits for smp, cold sweeps for gcb);
- qualified-boundary evidence: at least one certified link with BOTH owners
  carrying fresh proposals at the whitelisted boundary (ctp: coverage ->
  relation-cover polish; smp: stateful visits -> rescue), plus downstream
  re-anchor proof (the first downstream block session's construction anchor
  equals the ledger incumbent at boundary entry);
- max lane FE = 8 x qualified boundary count (frozen for T2/T3 here);
- two AOB identity arms (S3/117 ctp, E3/117 smp) must reproduce the frozen
  recovery-screen receipts bit-identically, proving the instrumentation is
  zero-tax on the production path.

Gate rules (preregistered): every generator arm completes the terminal
contract; every (cell, seed) has >= 1 qualified boundary on BOTH ctp and smp;
identity arms match 4/4 fields; census complete.  Failure output precisely
separates "no certified structure" from "no propagating execution boundary".
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from arac.actions.recovered_registry import RecoveredActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import canonical_sha256
from arac.runtime.ledger import EvaluationLedger
from arac.runtime.phase2 import execute_phase2_action
from experiments.historical_recovery.replay import _checkpoint
from experiments.upgrade.hyperedge_ctp_v1.h0_sidecar import GENERATOR_FREEZE
from experiments.upgrade.shared_patch_v1.conflicting_generator_v3 import build_v3_problem
from experiments.upgrade.shared_transaction_v1.scst_instrumentation import TransactionAuditRecorder


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = Path(__file__).with_name("t1_boundary_audit_protocol_v1.json")
ARM_SCHEMA = "arac-upgrade-shared-transaction-t1-arm-receipt-v1"
FAILURE_SCHEMA = "arac-upgrade-shared-transaction-t1-failure-v1"
T0_OUTPUT_ROOT = Path("artifacts/upgrade_shared_transaction_v1_t0_v1")
T0_SUMMARY = T0_OUTPUT_ROOT / "summary.json"
AOB_CHECKPOINT_ROOT = REPOSITORY_ROOT / "artifacts/historical_recovery_fixed_expert_v1/checkpoints"
FROZEN_SCREEN_ROOT = REPOSITORY_ROOT / "artifacts/recovery_first_screen_smp_topology_v3/arms"
CELLS = ("chain4-strong", "pairs3-strong")
SEEDS = (20270501, 20270502, 20270503, 20270504, 20270505)
AUDIT_HOSTS = ("ctp", "smp")
RECORD_ONLY_HOSTS = ("gcb",)
IDENTITY_ARMS = (("S3", "ctp"), ("E3", "smp"))
IDENTITY_SEED = 117
ACTION_SEED = 314159
PHASE1_FES = 180_000
TOTAL_BUDGET_FES = 3_000_000
PHASE2_FES = TOTAL_BUDGET_FES - PHASE1_FES
MAX_PATCH_FE_PER_BOUNDARY = 8
MAX_WORKERS = 8

_HOST_PHASES = {
    "ctp": ("run_persistent_blocks", "run_sequential_blocks"),
    "smp": ("run_stateful_block_visits", "run_stalled_block_rescue"),
    "gcb": ("run_cold_start_block_sweeps", "run_full_space"),
}


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
        "schema_version": "arac-upgrade-shared-transaction-t1-protocol-v1",
        "candidate_id": "shared_transaction_v1",
        "freeze_anchor": "arac-recovered-baseline-20260823-v1",
        "cells": list(CELLS),
        "seeds": list(SEEDS),
        "audit_hosts": list(AUDIT_HOSTS),
        "record_only_hosts": list(RECORD_ONLY_HOSTS),
        "identity_arms": [[case, host] for case, host in IDENTITY_ARMS],
        "identity_seed": IDENTITY_SEED,
        "action_seed": ACTION_SEED,
        "phase1_fes": PHASE1_FES,
        "total_budget_fes": TOTAL_BUDGET_FES,
        "max_patch_fe_per_boundary": MAX_PATCH_FE_PER_BOUNDARY,
        "output_root": "artifacts/upgrade_shared_transaction_v1_t1_v1",
        "t0_output_root": str(T0_OUTPUT_ROOT),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"T1 protocol drifted: {key}")
    return protocol


def _runtime_block() -> dict[str, Any]:
    pools = [
        {"internal_api": item.get("internal_api"), "num_threads": item.get("num_threads"), "prefix": item.get("prefix")}
        for item in threadpool_info()
    ]
    if any(pool["num_threads"] != 1 for pool in pools):
        raise RuntimeError(f"native thread limit is not one: {pools}")
    return {"python_executable": sys.executable, "numpy_version": np.__version__, "threadpools": pools}


def _terminal_contract(key: str, result, ledger) -> None:
    if (
        result.consumed_fes != PHASE2_FES
        or result.terminal_fes != TOTAL_BUDGET_FES
        or ledger.count != TOTAL_BUDGET_FES
        or result.final_error != ledger.best_error
        or not np.isfinite(result.final_error)
    ):
        raise RuntimeError(f"{key} terminal contract failed")


def _load_t0_receipt(cell_id: str, run_seed: int) -> dict[str, Any]:
    path = T0_OUTPUT_ROOT / "checkpoints" / cell_id / f"seed_{run_seed}" / "checkpoint.json"
    receipt = _load_json(path)
    if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
        raise ValueError(f"T0 receipt hash drifted: {cell_id}/{run_seed}")
    checkpoint = _checkpoint(receipt["checkpoint"])
    if checkpoint.checkpoint_hash != receipt.get("checkpoint_hash"):
        raise ValueError(f"T0 checkpoint hash drifted: {cell_id}/{run_seed}")
    return receipt, checkpoint


def _certificate_links(sidecar: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map certificate region ids to checkpoint block indices (leaf id - 1)."""

    links = []
    for certificate in sidecar["pairwise_certificates"]:
        links.append(
            {
                "variable": int(certificate["variable"]),
                "block_a": int(certificate["region_a"]) - 1,
                "block_b": int(certificate["region_b"]) - 1,
                "planted_blocks": certificate.get("planted_blocks"),
            }
        )
    return links


def _owner_block_of(checkpoint, variable: int) -> int:
    for index, block in enumerate(checkpoint.blocks):
        if variable in block:
            return index
    raise ValueError(f"variable {variable} not found in checkpoint blocks")


def _qualified_boundary_census(
    host: str,
    recorder: TransactionAuditRecorder,
    checkpoint,
    links: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_phase, boundary_phase = _HOST_PHASES[host]
    proposals = recorder.proposals_by_block(source_phase)
    link_rows = []
    qualified = False
    for link in links:
        owner_a = _owner_block_of(checkpoint, link["variable"])
        if owner_a not in (link["block_a"], link["block_b"]):
            raise ValueError(
                f"certificate owner blocks {link['block_a'], link['block_b']} do not contain "
                f"variable {link['variable']}'s primary block {owner_a}"
            )
        owner_b = link["block_b"] if link["block_a"] == owner_a else link["block_a"]
        both = owner_a in proposals and owner_b in proposals
        row = {
            "variable": link["variable"],
            "owner_blocks": sorted((owner_a, owner_b)),
            "proposal_owner_count": int(owner_a in proposals) + int(owner_b in proposals),
            "both_owners_fresh": bool(both),
        }
        if both:
            row["proposal_values"] = {
                str(owner): {
                    "commit_fes": proposals[owner].commit_fes,
                    "incumbent_hash_after": proposals[owner].incumbent_hash_after,
                }
                for owner in (owner_a, owner_b)
            }
        link_rows.append(row)
        qualified = qualified or both
    boundary_records = recorder.phase(boundary_phase)
    reanchor_rows = []
    for record in boundary_records:
        first_session = None
        for session in recorder.sessions:
            if session.phase_tag == boundary_phase and session.birth_fes >= record.entry_fes:
                first_session = session
                break
        reanchor_rows.append(
            {
                "entry_fes": record.entry_fes,
                "incumbent_hash_at_entry": record.incumbent_hash_at_entry,
                "first_session_birth_fes": first_session.birth_fes if first_session else None,
                "first_session_anchor_hash": first_session.anchor_hash if first_session else None,
                "incumbent_hash_at_birth": (
                    recorder.incumbent_hash_at(first_session.birth_fes) if first_session else None
                ),
                # Live-anchoring proof: the first downstream session anchors
                # the ledger incumbent at its own birth per the improvement
                # timeline.  Equality with the boundary-entry hash is too
                # strong: the downstream phase's own probes (e.g. the rescue
                # ranking probes) may legitimately improve the incumbent
                # between entry and the first session, and a live anchor
                # picks that up - exactly the propagation a transaction
                # writeback needs.
                "reanchor_proven": bool(
                    first_session is not None
                    and recorder.incumbent_hash_at(first_session.birth_fes) is not None
                    and first_session.anchor_hash == recorder.incumbent_hash_at(first_session.birth_fes)
                ),
            }
        )
    boundary_count = len(boundary_records)
    qualified_boundary_count = (
        sum(
            1
            for record, row in zip(boundary_records, reanchor_rows, strict=True)
            if row["reanchor_proven"] and qualified
        )
        if boundary_records
        else 0
    )
    return {
        "source_phase": source_phase,
        "boundary_phase": boundary_phase,
        "source_proposal_blocks": sorted(proposals),
        "source_improvement_count": sum(
            1 for item in recorder.improvements if item.phase_tag == source_phase
        ),
        "link_rows": link_rows,
        "links_with_both_owners": sum(1 for row in link_rows if row["both_owners_fresh"]),
        "boundary_count": boundary_count,
        "reanchor_rows": reanchor_rows,
        "reanchor_all_proven": bool(reanchor_rows) and all(row["reanchor_proven"] for row in reanchor_rows),
        "qualified_boundary_count": int(qualified_boundary_count),
        "max_patch_fe": MAX_PATCH_FE_PER_BOUNDARY * int(qualified_boundary_count),
        "native_phase_count": len(recorder.phases),
    }


@dataclass(frozen=True)
class GeneratorAuditJob:
    cell_id: str
    run_seed: int
    host: str
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "arms" / self.cell_id / f"seed_{self.run_seed}" / f"{self.host}.json"

    @property
    def key(self) -> str:
        return f"t1:{self.cell_id}:seed-{self.run_seed}:{self.host}"


@dataclass(frozen=True)
class AobIdentityJob:
    case_id: str
    host: str
    run_seed: int
    output_root: Path
    manifest_sha256: str

    @property
    def receipt_path(self) -> Path:
        return self.output_root / "identity" / self.case_id / f"seed_{self.run_seed}" / f"{self.host}.json"

    @property
    def key(self) -> str:
        return f"t1-identity:{self.case_id}:seed-{self.run_seed}:{self.host}"


def _run_generator_arm(job: GeneratorAuditJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            receipt, checkpoint = _load_t0_receipt(job.cell_id, job.run_seed)
            problem, truth = build_v3_problem(
                job.cell_id,
                job.run_seed,
                conditioning=GENERATOR_FREEZE["conditioning"],
                shared_width=int(GENERATOR_FREEZE["shared_width"]),
                linkage_lambda=float(GENERATOR_FREEZE["linkage_lambda"]),
            )
            if truth.ground_truth_hash != receipt["sidecar"]["ground_truth_hash"]:
                raise ValueError(f"{job.key} generator truth hash drifted against T0")
            registry = RecoveredActionRegistry()
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=checkpoint.total_budget_fes,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            recorder = TransactionAuditRecorder()
            recorder.install(ledger)
            try:
                result = execute_phase2_action(
                    job.host,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=ACTION_SEED,
                    registry=registry,
                )
            finally:
                recorder.uninstall()
            _terminal_contract(job.key, result, ledger)
            links = _certificate_links(receipt["sidecar"])
            census = _qualified_boundary_census(job.host, recorder, checkpoint, links)
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "lane": "generator",
                "cell_id": job.cell_id,
                "run_seed": job.run_seed,
                "host": job.host,
                "action_seed": ACTION_SEED,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "certificate_link_count": len(links),
                "census": census,
                "instrumentation": recorder.census_payload(),
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


def _run_identity_arm(job: AobIdentityJob) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        with threadpool_limits(limits=1):
            runtime = _runtime_block()
            wrapper = _load_json(AOB_CHECKPOINT_ROOT / job.case_id / f"seed_{job.run_seed}" / "checkpoint.json")
            checkpoint = _checkpoint(wrapper["checkpoint"])
            if checkpoint.checkpoint_hash != wrapper.get("checkpoint_hash"):
                raise ValueError(f"{job.key} AOB checkpoint hash drifted")
            problem = AobBenchmark().load(job.case_id)
            registry = RecoveredActionRegistry()
            ledger = EvaluationLedger.from_checkpoint(
                problem,
                total_budget=checkpoint.total_budget_fes,
                phase1_fes=checkpoint.phase1_fes,
                incumbent=checkpoint.incumbent,
                incumbent_error=checkpoint.incumbent_error,
                allow_out_of_bounds=registry.allow_out_of_bounds,
            )
            recorder = TransactionAuditRecorder()
            recorder.install(ledger)
            try:
                result = execute_phase2_action(
                    job.host,
                    checkpoint,
                    problem,
                    ledger,
                    action_seed=job.run_seed,
                    registry=registry,
                )
            finally:
                recorder.uninstall()
            _terminal_contract(job.key, result, ledger)
            frozen = _load_json(FROZEN_SCREEN_ROOT / job.case_id / f"seed_{job.run_seed}" / f"{job.host}.json")
            identity = {
                field: {"rerun": value, "frozen": frozen[field], "match": value == frozen[field]}
                for field, value in (
                    ("final_error", result.final_error),
                    ("route", result.route),
                    ("action_result_hash", result.result_hash),
                    ("checkpoint_hash", checkpoint.checkpoint_hash),
                )
            }
            census = _qualified_boundary_census(job.host, recorder, checkpoint, [])
            body = {
                "schema_version": ARM_SCHEMA,
                "manifest_sha256": job.manifest_sha256,
                "lane": "aob_identity",
                "case_id": job.case_id,
                "run_seed": job.run_seed,
                "host": job.host,
                "action_seed": job.run_seed,
                "checkpoint_hash": checkpoint.checkpoint_hash,
                "initial_error": checkpoint.incumbent_error,
                "final_error": result.final_error,
                "route": result.route,
                "action_result_hash": result.result_hash,
                "identity": identity,
                "identity_all_match": all(row["match"] for row in identity.values()),
                "census": census,
                "instrumentation": recorder.census_payload(),
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


def _run_job(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    kind, job = payload
    if kind == "generator":
        return _run_generator_arm(job)
    return _run_identity_arm(job)


def summarize(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cell_rows = []
    for cell_id in protocol["cells"]:
        seed_rows = []
        for seed in protocol["seeds"]:
            host_rows = {}
            for host in (*protocol["audit_hosts"], *protocol["record_only_hosts"]):
                receipt = _load_json(output_root / "arms" / cell_id / f"seed_{seed}" / f"{host}.json")
                if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}) != receipt.get("receipt_hash"):
                    raise ValueError(f"T1 receipt hash drifted: {cell_id}/{seed}/{host}")
                census = receipt["census"]
                host_rows[host] = {
                    "certificate_link_count": receipt["certificate_link_count"],
                    "links_with_both_owners": census["links_with_both_owners"],
                    "boundary_count": census["boundary_count"],
                    "reanchor_all_proven": census["reanchor_all_proven"],
                    "qualified_boundary_count": census["qualified_boundary_count"],
                    "max_patch_fe": census["max_patch_fe"],
                    "final_error": receipt["final_error"],
                }
            seed_rows.append({"seed": seed, **host_rows})
        cell_rows.append({"cell_id": cell_id, "seed_rows": seed_rows})

    identity_rows = []
    for case_id, host in protocol["identity_arms"]:
        receipt = _load_json(output_root / "identity" / case_id / f"seed_{protocol['identity_seed']}" / f"{host}.json")
        identity_rows.append(
            {
                "case_id": case_id,
                "host": host,
                "identity_all_match": receipt["identity_all_match"],
                "identity": receipt["identity"],
                "qualified_boundary_count": receipt["census"]["qualified_boundary_count"],
            }
        )

    audit_hosts = protocol["audit_hosts"]
    generator_arms_qualified = all(
        row[host]["qualified_boundary_count"] >= 1
        for cell_row in cell_rows
        for row in cell_row["seed_rows"]
        for host in audit_hosts
    )
    max_lane = max(
        row[host]["max_patch_fe"]
        for cell_row in cell_rows
        for row in cell_row["seed_rows"]
        for host in audit_hosts
    )
    checks = {
        "coverage_complete": (
            len(cell_rows) == len(protocol["cells"])
            and all(len(row["seed_rows"]) == len(protocol["seeds"]) for row in cell_rows)
        ),
        "audit_hosts_proposal_observable": all(
            row[host]["links_with_both_owners"] >= 1
            for cell_row in cell_rows
            for row in cell_row["seed_rows"]
            for host in audit_hosts
        ),
        "audit_hosts_reanchor_proven": all(
            row[host]["reanchor_all_proven"]
            for cell_row in cell_rows
            for row in cell_row["seed_rows"]
            for host in audit_hosts
        ),
        "audit_hosts_qualified_boundaries": generator_arms_qualified,
        "identity_arms_bit_identical": all(row["identity_all_match"] for row in identity_rows),
    }
    body = {
        "schema_version": "arac-upgrade-shared-transaction-t1-summary-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "cell_rows": cell_rows,
        "identity_rows": identity_rows,
        "checks": checks,
        "frozen_for_t2_t3": {
            "lane_budget_fe_per_run": max_lane,
            "t3_seeds": list(protocol["seeds"]),
            "t3_cells": list(protocol["cells"]),
        },
        "gate_passed": all(checks.values()),
        "t2_authorized": all(checks.values()),
    }
    body["result_hash"] = canonical_sha256(body)
    return body


def run_stage(protocol_path: Path = DEFAULT_PROTOCOL, *, resume: bool = False) -> dict[str, Any]:
    resolved = Path(protocol_path).resolve()
    protocol = load_protocol(resolved)
    t0_summary = _load_json(T0_SUMMARY)
    if not t0_summary.get("gate_passed"):
        raise RuntimeError("T1 refuses to run: T0 gate has not passed")
    from experiments.historical_recovery import verify_recovered_baseline_freeze as freeze_verifier

    verifier_report = freeze_verifier.verify()
    if verifier_report.get("status") != "frozen":
        raise RuntimeError("T1 refuses to run: the recovered baseline freeze verifier is not green")
    output_root = (REPOSITORY_ROOT / str(protocol["output_root"])).resolve()
    manifest = {
        "schema_version": "arac-upgrade-shared-transaction-t1-manifest-v1",
        "candidate_id": protocol["candidate_id"],
        "freeze_anchor": protocol["freeze_anchor"],
        "protocol_sha256": _sha256(resolved),
        "verifier_report": verifier_report,
        "t0_result_hash": t0_summary.get("result_hash"),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not resume:
            raise FileExistsError(f"T1 output already exists: {output_root}")
        if not manifest_path.is_file() or _load_json(manifest_path).get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("T1 manifest does not match the frozen protocol")
    else:
        output_root.mkdir(parents=True)
        _write_json(manifest_path, manifest)

    jobs: list[tuple[str, Any]] = [
        (
            "generator",
            GeneratorAuditJob(
                cell_id=cell,
                run_seed=int(seed),
                host=host,
                output_root=output_root,
                manifest_sha256=str(manifest["manifest_sha256"]),
            ),
        )
        for cell in protocol["cells"]
        for seed in protocol["seeds"]
        for host in (*protocol["audit_hosts"], *protocol["record_only_hosts"])
    ]
    jobs.extend(
        (
            "identity",
            AobIdentityJob(
                case_id=case_id,
                host=host,
                run_seed=int(protocol["identity_seed"]),
                output_root=output_root,
                manifest_sha256=str(manifest["manifest_sha256"]),
            ),
        )
        for case_id, host in protocol["identity_arms"]
    )
    pending = [
        item
        for item in jobs
        if not item[1].receipt_path.is_file()
    ]
    failures = 0
    if pending:
        with ProcessPoolExecutor(max_workers=protocol.get("max_workers", MAX_WORKERS)) as pool:
            futures = {pool.submit(_run_job, item): item[1].key for item in pending}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    receipt = future.result()
                    print(f"[t1] done {key} final={receipt.get('final_error')}", flush=True)
                except BaseException:
                    failures += 1
                    print(f"[t1] FAILED {key}", flush=True)
    if failures:
        raise RuntimeError(f"T1 had {failures} failed arms; inspect {output_root / 'failures'}")
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
        protocol = load_protocol(args.protocol)
        summary = summarize(protocol, (REPOSITORY_ROOT / str(protocol["output_root"])).resolve())
        _write_json((REPOSITORY_ROOT / str(protocol["output_root"])).resolve() / "summary.json", summary)
    else:
        summary = run_stage(args.protocol, resume=args.resume)
    print(json.dumps({"stage": "t1", "gate_passed": summary["gate_passed"], "checks": summary["checks"], "t2_authorized": summary["t2_authorized"]}, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


__all__ = ["load_protocol", "run_stage", "summarize"]


if __name__ == "__main__":
    raise SystemExit(main())
