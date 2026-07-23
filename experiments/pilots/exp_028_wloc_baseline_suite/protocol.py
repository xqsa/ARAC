"""Frozen protocol and static 18-by-9 task expansion for exp_028."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from arac.baselines import derive_optimizer_seed
from arac.benchmarks import (
    WANG2025_CONTINUOUS_SCHEMA_VERSION,
    WANG2025_LOCAL_ESCAPE_CASES,
    WANG2025_LOCAL_ESCAPE_SUITE_VERSION,
    Wang2025ContinuousProblem,
    get_wang2025_local_escape_case,
)


PROTOCOL_SCHEMA_VERSION = "wloc-baseline-protocol-v1"
TASK_SCHEMA_VERSION = "wloc-baseline-task-v1"
CONFIG_PATH = Path(__file__).with_name("config.json")
METHODS = (
    "DG2-CMAES",
    "Random-CMAES",
    "RDG3-CMAES",
    "RDDSM-CMAES",
    "Sep-CMAES",
    "LM-MA-ES",
    "LMCMA",
    "MM-ES",
    "HCC-ES",
)
CASE_IDS = tuple(case.case_id for case in WANG2025_LOCAL_ESCAPE_CASES)
DECOMPOSITION_MODES = {
    "DG2-CMAES": "measured_objective",
    "Random-CMAES": "generated_random",
    "RDG3-CMAES": "measured_objective",
    "RDDSM-CMAES": "design_matrix",
    "Sep-CMAES": "none",
    "LM-MA-ES": "none",
    "LMCMA": "none",
    "MM-ES": "none",
    "HCC-ES": "design_matrix",
}


def stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProtocolConfig:
    schema_version: str
    suite_version: str
    objective_schema_version: str
    dimension: int
    cases: tuple[str, ...]
    methods: tuple[str, ...]
    optimizer_seed_base: int
    optimization_fes: int
    initial_mean: float
    sigma: float
    repair_policy: str
    random_group_count: int
    rdg3_nonseparable_threshold: int
    rdg3_separable_chunk_size: int
    mechanical_smoke_fes: int
    bootstrap_or_performance_claim_authorized: bool
    real_aob_action_gate_eligible: bool
    protocol_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported WLOC baseline protocol schema")
        if self.suite_version != WANG2025_LOCAL_ESCAPE_SUITE_VERSION:
            raise ValueError("protocol suite version does not match the benchmark catalog")
        if self.objective_schema_version != WANG2025_CONTINUOUS_SCHEMA_VERSION:
            raise ValueError("protocol objective schema does not match the implementation")
        if self.dimension != 1000:
            raise ValueError("the frozen WLOC baseline protocol must use 1000 dimensions")
        if self.cases != CASE_IDS or self.methods != METHODS:
            raise ValueError("protocol cases or methods do not match the frozen 18 x 9 matrix")
        if self.optimization_fes <= 0 or self.mechanical_smoke_fes <= 0:
            raise ValueError("optimization FE budgets must be positive")
        if not 0.0 <= self.initial_mean <= 1.0:
            raise ValueError("initial_mean must lie in [0, 1]")
        if self.sigma <= 0.0 or not math.isfinite(self.sigma):
            raise ValueError("sigma must be finite and positive")
        if self.repair_policy != "clip_to_bounds":
            raise ValueError("the frozen protocol requires clip_to_bounds")
        if self.random_group_count != 20:
            raise ValueError("Random-CMAES must use 20 subspaces")
        if self.rdg3_nonseparable_threshold != 50:
            raise ValueError("RDG3 nonseparable threshold must be 50")
        if self.rdg3_separable_chunk_size != 100:
            raise ValueError("RDG3 separable chunk size must be 100")
        if self.bootstrap_or_performance_claim_authorized:
            raise ValueError("this implementation-only protocol cannot authorize performance claims")
        if self.real_aob_action_gate_eligible:
            raise ValueError("synthetic WLOC tasks cannot enter the real AOB action gate")
        object.__setattr__(self, "protocol_hash", stable_hash(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {data_field.name: getattr(self, data_field.name) for data_field in fields(self) if data_field.init}


@dataclass(frozen=True)
class WLOCBaselineTask:
    schema_version: str
    protocol_hash: str
    case_id: str
    method: str
    dimension: int
    instance_seed: int
    optimizer_seed: int
    source_instance_hash: str
    objective_hash: str
    optimization_fes: int
    decomposition_mode: str
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != TASK_SCHEMA_VERSION:
            raise ValueError("unsupported WLOC baseline task schema")
        if self.case_id not in CASE_IDS or self.method not in METHODS:
            raise ValueError("task case or method is outside the frozen matrix")
        if self.dimension != 1000 or self.optimization_fes <= 0:
            raise ValueError("task dimension or FE budget is invalid")
        if self.decomposition_mode not in {
            *DECOMPOSITION_MODES.values(),
            "provided_catalog_topology_smoke",
        }:
            raise ValueError("unsupported decomposition mode")
        object.__setattr__(self, "task_hash", stable_hash(self.payload()))

    def payload(self) -> dict[str, Any]:
        return {data_field.name: getattr(self, data_field.name) for data_field in fields(self) if data_field.init}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WLOCBaselineTask:
        expected_fields = {
            "schema_version",
            "protocol_hash",
            "case_id",
            "method",
            "dimension",
            "instance_seed",
            "optimizer_seed",
            "source_instance_hash",
            "objective_hash",
            "optimization_fes",
            "decomposition_mode",
            "task_hash",
        }
        if set(payload) != expected_fields:
            raise ValueError("task fields do not match the frozen schema")
        expected_hash = str(payload["task_hash"])
        task = cls(**{key: value for key, value in payload.items() if key != "task_hash"})
        if task.task_hash != expected_hash:
            raise ValueError("task hash mismatch")
        return task


def load_protocol(path: Path = CONFIG_PATH) -> ProtocolConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {data_field.name for data_field in fields(ProtocolConfig) if data_field.init}
    if set(payload) != expected_fields:
        raise ValueError("config fields do not match the frozen protocol schema")
    payload["cases"] = tuple(payload["cases"])
    payload["methods"] = tuple(payload["methods"])
    return ProtocolConfig(**payload)


def build_task(
    config: ProtocolConfig,
    case_id: str,
    method: str,
    *,
    mechanical_smoke: bool = False,
) -> WLOCBaselineTask:
    case = get_wang2025_local_escape_case(case_id)
    source = case.generate()
    problem = Wang2025ContinuousProblem(source)
    decomposition_mode = DECOMPOSITION_MODES[method]
    if mechanical_smoke and method in {"DG2-CMAES", "RDG3-CMAES"}:
        decomposition_mode = "provided_catalog_topology_smoke"
    budget = config.mechanical_smoke_fes if mechanical_smoke else config.optimization_fes
    optimizer_seed = derive_optimizer_seed(config.optimizer_seed_base, case.case_id, method)
    return WLOCBaselineTask(
        schema_version=TASK_SCHEMA_VERSION,
        protocol_hash=config.protocol_hash,
        case_id=case.case_id,
        method=method,
        dimension=source.dimension,
        instance_seed=case.spec.seed,
        optimizer_seed=optimizer_seed,
        source_instance_hash=source.instance_hash,
        objective_hash=problem.objective_hash,
        optimization_fes=budget,
        decomposition_mode=decomposition_mode,
    )


def build_task_matrix(config: ProtocolConfig) -> tuple[WLOCBaselineTask, ...]:
    return tuple(
        build_task(config, case_id, method)
        for case_id in config.cases
        for method in config.methods
    )
