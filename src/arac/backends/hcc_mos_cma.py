"""ARAC-owned mirrored-orthogonal sampling adapter for vendor CMA-ES."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np


IID_CMA_SAMPLING = "iid"
MIRRORED_ORTHOGONAL_CMA_SAMPLING = "mirrored_orthogonal"
CMA_SAMPLING_MODES = frozenset(
    {IID_CMA_SAMPLING, MIRRORED_ORTHOGONAL_CMA_SAMPLING}
)
CMA_SAMPLING_MODE_OPTION = "_arac_cma_sampling_mode"
MOS_AUDIT_SINK_OPTION = "_arac_mos_audit_sink"
MOS_AUDIT_CONTEXT_OPTION = "_arac_mos_audit_context"
_MOS_AUDIT_FIELDS = frozenset(
    {
        "generation",
        "optimizer_restart_index",
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
    }
)


class MirroredOrthogonalSamplingError(RuntimeError):
    """Raised when mirrored-orthogonal sampling cannot be completed safely."""


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = f"{array.shape}|".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _validate_size(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _sample_with_audit(
    rng: np.random.Generator,
    population_size: int,
    dimension: int,
) -> tuple[np.ndarray, dict[str, object]]:
    population = _validate_size(
        population_size,
        name="population_size",
        minimum=2,
    )
    ndim = _validate_size(dimension, name="dimension", minimum=1)

    try:
        raw = np.asarray(rng.standard_normal((population, ndim)), dtype=float)
    except Exception as exc:
        raise MirroredOrthogonalSamplingError("standard-normal draw failed") from exc
    if raw.shape != (population, ndim):
        raise MirroredOrthogonalSamplingError(
            "standard-normal draw returned an unexpected shape"
        )
    if not np.all(np.isfinite(raw)):
        raise MirroredOrthogonalSamplingError(
            "standard-normal draw contained non-finite values"
        )

    base_count = (population + 1) // 2
    base_samples = np.empty((base_count, ndim), dtype=float)
    block_count = 0
    max_orthogonality_error = 0.0
    eps = np.finfo(float).eps

    for start in range(0, base_count, ndim):
        stop = min(base_count, start + ndim)
        raw_block = raw[start:stop]
        block_size = stop - start
        try:
            q, r = np.linalg.qr(raw_block.T, mode="reduced")
        except np.linalg.LinAlgError as exc:
            raise MirroredOrthogonalSamplingError("QR decomposition failed") from exc
        if q.shape != (ndim, block_size) or r.shape != (block_size, block_size):
            raise MirroredOrthogonalSamplingError("QR decomposition returned invalid shapes")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(r)):
            raise MirroredOrthogonalSamplingError(
                "QR decomposition returned non-finite values"
            )

        diagonal = np.diag(r)
        rank_scale = max(1.0, float(np.linalg.norm(r, ord=np.inf)))
        rank_tolerance = eps * max(ndim, block_size) * rank_scale
        if np.any(np.abs(diagonal) <= rank_tolerance):
            raise MirroredOrthogonalSamplingError(
                "Gaussian direction block is numerically rank deficient"
            )

        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        q = q * signs
        gram_error = float(
            np.linalg.norm(q.T @ q - np.eye(block_size), ord=np.inf)
        )
        orthogonality_tolerance = 64.0 * eps * max(ndim, block_size)
        if not np.isfinite(gram_error) or gram_error > orthogonality_tolerance:
            raise MirroredOrthogonalSamplingError(
                "QR directions failed the orthogonality check"
            )

        radii = np.linalg.norm(raw_block, axis=1)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise MirroredOrthogonalSamplingError("Gaussian direction radius is invalid")
        base_samples[start:stop] = q.T * radii[:, None]
        block_count += 1
        max_orthogonality_error = max(max_orthogonality_error, gram_error)

    samples = np.empty((population, ndim), dtype=float)
    pair_count = population // 2
    for pair_index in range(pair_count):
        output_index = 2 * pair_index
        samples[output_index] = base_samples[pair_index]
        samples[output_index + 1] = -base_samples[pair_index]
    if population % 2:
        samples[-1] = base_samples[-1]
    if not np.all(np.isfinite(samples)):
        raise MirroredOrthogonalSamplingError("generated samples are non-finite")

    return samples, {
        "population": population,
        "dimension": ndim,
        "pair_count": pair_count,
        "block_count": block_count,
        "raw_draw_sha256": _array_sha256(raw),
        "sample_sha256": _array_sha256(samples),
        "max_orthogonality_error": max_orthogonality_error,
        "rng_draw_count": population * ndim,
    }


def sample_mirrored_orthogonal(
    rng: np.random.Generator,
    population_size: int,
    dimension: int,
) -> np.ndarray:
    """Draw one mirrored-orthogonal population with standard-normal marginals."""

    samples, _ = _sample_with_audit(rng, population_size, dimension)
    return samples


class _MirroredOrthogonalCMAESMixin:
    _arac_mos_audit_sink: Any = None
    _arac_mos_audit_context: Mapping[str, object] = {}

    def iterate(
        self,
        x=None,
        mean=None,
        e_ve=None,
        e_va=None,
        y=None,
        d=None,
        args=None,
    ):
        if self._check_terminations():
            return x, y, d

        z, audit = _sample_with_audit(
            self.rng_optimization,
            self.n_individuals,
            self.ndim_problem,
        )
        d = np.dot(z, np.dot(np.diag(e_va), e_ve.T))
        x = mean + self.sigma * d
        evaluations_before = int(self.n_function_evaluations)
        y = self._evaluate_fitness(x, args)
        evaluated_count = int(self.n_function_evaluations) - evaluations_before

        sink = self._arac_mos_audit_sink
        if sink is not None:
            sink.append(
                {
                    **self._arac_mos_audit_context,
                    "generation": int(self._n_generations),
                    "optimizer_restart_index": int(self._n_restart),
                    **audit,
                    "evaluated_count": evaluated_count,
                    "complete_population": evaluated_count == self.n_individuals,
                }
            )
        return x, y, d


@functools.lru_cache(maxsize=None)
def _mos_cmaes_subclass(vendor_cmaes_type: type) -> type:
    return type(
        "ARACMirroredOrthogonalCMAES",
        (_MirroredOrthogonalCMAESMixin, vendor_cmaes_type),
        {"__module__": __name__},
    )


def _normalize_audit_context(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("MOS audit context must be a mapping")

    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("MOS audit context keys must be non-empty strings")
        if key in _MOS_AUDIT_FIELDS:
            raise ValueError(f"MOS audit context cannot replace reserved field: {key}")
        if isinstance(item, np.generic):
            item = item.item()
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("MOS audit context values must be scalar")
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError("MOS audit context floats must be finite")
        normalized[key] = item
    return normalized


def create_hcc_cmaes(
    problem: Mapping[str, object],
    options: Mapping[str, object],
):
    """Create an iid vendor CMA-ES or the opt-in ARAC MOS subclass."""

    clean_options = dict(options)
    mode = clean_options.pop(CMA_SAMPLING_MODE_OPTION, IID_CMA_SAMPLING)
    audit_sink = clean_options.pop(MOS_AUDIT_SINK_OPTION, None)
    audit_context_value = clean_options.pop(MOS_AUDIT_CONTEXT_OPTION, None)
    if mode not in CMA_SAMPLING_MODES:
        raise ValueError(f"unsupported CMA sampling mode: {mode}")

    from HCC.OPT.CMAES.cmaes import CMAES as VendorCMAES

    if mode == IID_CMA_SAMPLING:
        return VendorCMAES(dict(problem), clean_options)
    if audit_sink is not None and not callable(getattr(audit_sink, "append", None)):
        raise ValueError("MOS audit sink must provide append()")
    audit_context = _normalize_audit_context(audit_context_value)

    optimizer_type = _mos_cmaes_subclass(VendorCMAES)
    optimizer = optimizer_type(dict(problem), clean_options)
    optimizer._arac_mos_audit_sink = audit_sink
    optimizer._arac_mos_audit_context = audit_context
    return optimizer
