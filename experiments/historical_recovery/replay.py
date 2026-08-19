"""Replay four frozen checkpoints with the current legacy action path."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any
import warnings

from arac.actions.registry import ActionRegistry
from arac.benchmarks.aob import AobBenchmark
from arac.runtime.contracts import (
    ACTION_NAMES,
    ActionContext,
    PhaseCheckpoint,
    RelationEvidence,
    canonical_sha256,
)
from arac.runtime.ledger import EvaluationLedger


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("current_replay_config.json")
RECEIPT_SCHEMA = "arac-current-action-replay-receipt-v1"
MANIFEST_SCHEMA = "arac-current-action-replay-manifest-v1"
SUMMARY_SCHEMA = "arac-current-action-replay-summary-v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if path.is_file() and "__pycache__" not in relative.parts and path.suffix != ".pyc":
            entries.append((relative.as_posix(), _sha256(path)))
    return {"file_count": len(entries), "tree_sha256": canonical_sha256(entries)}


def _checkpoint(payload: dict[str, Any]) -> PhaseCheckpoint:
    return PhaseCheckpoint(
        protocol=str(payload["protocol"]),
        run_seed=int(payload["run_seed"]),
        total_budget_fes=int(payload["total_budget_fes"]),
        phase1_fes=int(payload["phase1_fes"]),
        incumbent=tuple(float(value) for value in payload["incumbent"]),
        incumbent_error=float(payload["incumbent_error"]),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        feature_values=tuple(float(value) for value in payload["feature_values"]),
        blocks=tuple(tuple(int(value) for value in block) for block in payload["blocks"]),
        relations=tuple(
            RelationEvidence(
                int(item["left_block"]),
                int(item["right_block"]),
                float(item["strength"]),
                float(item["disagreement"]),
            )
            for item in payload["relations"]
        ),
    )


def _context_paths(source_root: Path, context: dict[str, Any]) -> tuple[Path, Path]:
    case_id = context["case"]
    seed = int(context["seed"])
    action = context["action"]
    checkpoint = source_root / "checkpoints" / case_id / f"seed_{seed}" / "checkpoint.json"
    arm = source_root / "arms" / case_id / f"seed_{seed}" / f"{action}.json"
    return checkpoint, arm


def load_replay_plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_json(config_path.resolve())
    if config.get("schema_version") != "arac-current-action-replay-config-v1":
        raise ValueError("replay config schema drifted")
    source_root = REPOSITORY_ROOT / config["source_matrix_root"]
    matrix_manifest = _load_json(source_root / "campaign_manifest.json")
    if matrix_manifest.get("manifest_sha256") != canonical_sha256(
        {key: value for key, value in matrix_manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("source matrix manifest hash drifted")
    current_vendor = _tree_sha256(REPOSITORY_ROOT / "vendor" / "aob")
    if current_vendor != matrix_manifest["vendor_trees"]["vendor/aob"]:
        raise ValueError("AOB vendor tree drifted from the source matrix")

    contexts = config["contexts"]
    if len(contexts) != 4 or {item["case"][0] for item in contexts} != set("AESR"):
        raise ValueError("replay must contain one representative from each AOB family")
    if any(item["action"] not in ACTION_NAMES for item in contexts):
        raise ValueError("replay config contains an unsupported action")
    for item in contexts:
        checkpoint_path, arm_path = _context_paths(source_root, item)
        checkpoint_receipt = _load_json(checkpoint_path)
        arm_receipt = _load_json(arm_path)
        checkpoint = _checkpoint(checkpoint_receipt["checkpoint"])
        if checkpoint.checkpoint_hash != item["checkpoint_hash"]:
            raise ValueError(f"checkpoint hash drifted: {item['case']}")
        expected = {
            "action_name": item["action"],
            "checkpoint_hash": item["checkpoint_hash"],
            "final_error": item["expected_final_error"],
            "action_result_hash": item["expected_result_hash"],
            "terminal_fes": item["expected_terminal_fes"],
        }
        for key, value in expected.items():
            if arm_receipt.get(key) != value:
                raise ValueError(f"source arm {key} drifted: {item['case']}/{item['action']}")

    audit_config = _load_json(REPOSITORY_ROOT / config["audit_config"])
    source_files = audit_config["frozen_independent_matrix"]["source_files"]
    current_hashes = {
        name: _sha256(REPOSITORY_ROOT / relative)
        for name, relative in sorted(source_files.items())
    }
    return {
        "config": config,
        "config_path": str(config_path.resolve()),
        "source_root": str(source_root.resolve()),
        "source_manifest_sha256": matrix_manifest["manifest_sha256"],
        "current_source_hashes": current_hashes,
        "source_files": source_files,
        "vendor_tree": current_vendor,
    }


def _run_one(
    context: dict[str, Any],
    source_root_text: str,
    output_root_text: str,
    rel_tol: float,
    abs_tol: float,
) -> dict[str, Any]:
    source_root = Path(source_root_text)
    output_root = Path(output_root_text)
    checkpoint_path, _ = _context_paths(source_root, context)
    checkpoint_receipt = _load_json(checkpoint_path)
    checkpoint = _checkpoint(checkpoint_receipt["checkpoint"])
    problem = AobBenchmark().load(context["case"])
    ledger = EvaluationLedger.from_checkpoint(
        problem,
        total_budget=checkpoint.total_budget_fes,
        phase1_fes=checkpoint.phase1_fes,
        incumbent=checkpoint.incumbent,
        incumbent_error=checkpoint.incumbent_error,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ActionRegistry().execute(
            ActionContext(
                context["action"],
                checkpoint,
                problem,
                ledger,
                action_seed=int(context["seed"]),
            )
        )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "case": context["case"],
        "action": context["action"],
        "seed": int(context["seed"]),
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "terminal_fes": result.terminal_fes,
        "final_error": result.final_error,
        "result_hash": result.result_hash,
        "expected_terminal_fes": int(context["expected_terminal_fes"]),
        "expected_final_error": float(context["expected_final_error"]),
        "expected_result_hash": context["expected_result_hash"],
        "terminal_fes_match": result.terminal_fes == int(context["expected_terminal_fes"]),
        "final_error_match": math.isclose(
            result.final_error,
            float(context["expected_final_error"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ),
        "exact_result_match": result.result_hash == context["expected_result_hash"],
        "runtime_warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ],
    }
    payload["replay_passed"] = all(
        payload[key]
        for key in ("terminal_fes_match", "final_error_match", "exact_result_match")
    )
    payload["receipt_hash"] = canonical_sha256(payload)
    destination = output_root / "receipts" / f"{context['case']}_{context['action']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _manifest(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan["config"]
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "config_sha256": _sha256(Path(plan["config_path"])),
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "current_source_hashes": plan["current_source_hashes"],
        "vendor_tree": plan["vendor_tree"],
        "contexts": config["contexts"],
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _prepare_output(plan: dict[str, Any], *, resume: bool) -> Path:
    config = plan["config"]
    output_root = (REPOSITORY_ROOT / config["output_root"]).resolve()
    manifest = _manifest(plan)
    manifest_path = output_root / "frozen_protocol" / "manifest.json"
    if resume:
        existing = _load_json(manifest_path)
        if existing != manifest:
            raise ValueError("replay manifest drifted")
        return output_root
    if output_root.exists():
        raise ValueError(f"replay output already exists: {output_root}")
    source_root = output_root / "frozen_protocol" / "sources"
    source_root.mkdir(parents=True)
    shutil.copy2(Path(plan["config_path"]), output_root / "frozen_protocol" / "config.json")
    for name, relative in plan["source_files"].items():
        source = REPOSITORY_ROOT / relative
        shutil.copy2(source, source_root / f"{name}{source.suffix}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_root


def run_replay(config_path: Path = DEFAULT_CONFIG, *, resume: bool = False) -> dict[str, Any]:
    plan = load_replay_plan(config_path)
    config = plan["config"]
    output_root = _prepare_output(plan, resume=resume)
    pending = []
    receipts = []
    for context in config["contexts"]:
        path = output_root / "receipts" / f"{context['case']}_{context['action']}.json"
        if path.is_file():
            receipt = _load_json(path)
            claimed = receipt.pop("receipt_hash", None)
            if claimed != canonical_sha256(receipt):
                raise ValueError(f"replay receipt hash drifted: {path}")
            receipt["receipt_hash"] = claimed
            receipts.append(receipt)
        else:
            pending.append(context)
    with ProcessPoolExecutor(max_workers=int(config["max_workers"])) as executor:
        futures = {
            executor.submit(
                _run_one,
                context,
                plan["source_root"],
                str(output_root),
                float(config["final_error_relative_tolerance"]),
                float(config["final_error_absolute_tolerance"]),
            ): context
            for context in pending
        }
        for future in as_completed(futures):
            receipts.append(future.result())
    receipts.sort(key=lambda row: row["case"])
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "context_count": len(receipts),
        "passed_count": sum(row["replay_passed"] for row in receipts),
        "failed_count": sum(not row["replay_passed"] for row in receipts),
        "all_replays_passed": all(row["replay_passed"] for row in receipts),
        "receipts": [
            {
                key: row[key]
                for key in (
                    "case",
                    "action",
                    "terminal_fes_match",
                    "final_error_match",
                    "exact_result_match",
                    "replay_passed",
                    "receipt_hash",
                )
            }
            for row in receipts
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-equivalent", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        plan = load_replay_plan(args.config)
        output_root = (REPOSITORY_ROOT / plan["config"]["output_root"]).resolve()
        if output_root.exists() and not args.resume:
            raise ValueError(f"replay output already exists: {output_root}")
        print(
            json.dumps(
                {
                    "context_count": len(plan["config"]["contexts"]),
                    "output_root": str(output_root),
                    "source_inputs_valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    summary = run_replay(args.config, resume=args.resume)
    print(json.dumps(summary, sort_keys=True))
    return int(args.require_equivalent and not summary["all_replays_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
