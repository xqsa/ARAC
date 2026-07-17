from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from arac.backends import hcc_mos_cma as mos


class RecordingGenerator:
    def __init__(self, seed: int = 7, values: np.ndarray | None = None):
        self.generator = np.random.default_rng(seed)
        self.values = values
        self.calls: list[tuple[int, ...]] = []
        self.draws: list[np.ndarray] = []

    def standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        self.calls.append(shape)
        values = (
            self.generator.standard_normal(shape)
            if self.values is None
            else np.asarray(self.values, dtype=float)
        )
        self.draws.append(np.asarray(values, dtype=float).copy())
        return values


def _base_samples(samples: np.ndarray) -> np.ndarray:
    pair_count = len(samples) // 2
    bases = [samples[2 * index] for index in range(pair_count)]
    if len(samples) % 2:
        bases.append(samples[-1])
    return np.asarray(bases)


def _assert_block_orthogonal(block: np.ndarray, atol: float = 1e-12) -> None:
    normalized = block / np.linalg.norm(block, axis=1)[:, None]
    assert np.allclose(normalized @ normalized.T, np.eye(len(block)), atol=atol)


def _vendor_cmaes_type():
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "hcc"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from HCC.OPT.CMAES.cmaes import CMAES

    return CMAES


def _sphere(values: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(values)
    return np.sum(array * array, axis=1)


def _problem(dimension: int) -> dict[str, object]:
    return {
        "fitness_function": _sphere,
        "ndim_problem": dimension,
        "lower_boundary": -5.0 * np.ones(dimension),
        "upper_boundary": 5.0 * np.ones(dimension),
    }


def _options(
    *,
    dimension: int,
    population: int,
    budget: int,
    seed: int = 13,
) -> dict[str, object]:
    return {
        "max_function_evaluations": budget,
        "mean": (np.zeros(dimension),),
        "sigma": 0.5,
        "n_individuals": population,
        "is_restart": False,
        "verbose": 0,
        "seed_rng": seed,
    }


def test_even_population_is_interleaved_mirrored_and_rng_draw_parity() -> None:
    rng = RecordingGenerator(seed=11)

    samples = mos.sample_mirrored_orthogonal(rng, 8, 5)

    assert rng.calls == [(8, 5)]
    assert rng.draws[0].size == 8 * 5
    for pair_index in range(4):
        assert np.array_equal(samples[2 * pair_index + 1], -samples[2 * pair_index])
    bases = _base_samples(samples)
    _assert_block_orthogonal(bases)
    assert np.allclose(
        np.linalg.norm(bases, axis=1),
        np.linalg.norm(rng.draws[0][:4], axis=1),
    )
    assert all(
        np.dot(rng.draws[0][index], bases[index]) > 0.0
        for index in range(4)
    )


def test_odd_population_has_one_final_unpaired_direction() -> None:
    rng = RecordingGenerator(seed=17)

    samples = mos.sample_mirrored_orthogonal(rng, 7, 3)

    assert samples.shape == (7, 3)
    for pair_index in range(3):
        assert np.array_equal(samples[2 * pair_index + 1], -samples[2 * pair_index])
    assert np.linalg.norm(samples[-1]) == pytest.approx(
        np.linalg.norm(rng.draws[0][3])
    )
    bases = _base_samples(samples)
    _assert_block_orthogonal(bases[:3])


def test_more_base_directions_than_dimensions_use_independent_blocks() -> None:
    rng = RecordingGenerator(seed=19)

    samples = mos.sample_mirrored_orthogonal(rng, 14, 2)

    bases = _base_samples(samples)
    assert len(bases) == 7
    for start in range(0, len(bases), 2):
        _assert_block_orthogonal(bases[start : start + 2])


def test_dimension_one_supports_multiple_mirror_blocks() -> None:
    samples = mos.sample_mirrored_orthogonal(np.random.default_rng(23), 5, 1)

    assert samples.shape == (5, 1)
    assert np.array_equal(samples[1], -samples[0])
    assert np.array_equal(samples[3], -samples[2])
    assert np.isfinite(samples).all()


def test_sampling_is_seed_reproducible() -> None:
    first = mos.sample_mirrored_orthogonal(np.random.default_rng(29), 9, 4)
    second = mos.sample_mirrored_orthogonal(np.random.default_rng(29), 9, 4)
    different = mos.sample_mirrored_orthogonal(np.random.default_rng(30), 9, 4)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize(
    ("population", "dimension", "message"),
    [
        (1, 2, "population_size"),
        (4, 0, "dimension"),
        (True, 2, "population_size"),
        (4, 1.5, "dimension"),
    ],
)
def test_invalid_sizes_are_rejected(population, dimension, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mos.sample_mirrored_orthogonal(
            np.random.default_rng(31),
            population,
            dimension,
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.full((4, 2), np.nan), "non-finite"),
        (np.ones((4, 2)), "rank deficient"),
        (np.ones((3, 2)), "unexpected shape"),
    ],
)
def test_bad_random_draws_fail_closed(values: np.ndarray, message: str) -> None:
    with pytest.raises(mos.MirroredOrthogonalSamplingError, match=message):
        mos.sample_mirrored_orthogonal(RecordingGenerator(values=values), 4, 2)


def test_qr_failure_is_wrapped_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_qr(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(mos.np.linalg, "qr", fail_qr)

    with pytest.raises(mos.MirroredOrthogonalSamplingError, match="QR decomposition"):
        mos.sample_mirrored_orthogonal(np.random.default_rng(37), 4, 2)


def test_iid_factory_returns_exact_vendor_type_and_strips_private_options() -> None:
    vendor_type = _vendor_cmaes_type()
    sink: list[dict[str, object]] = []
    options = _options(dimension=3, population=4, budget=4)
    options[mos.CMA_SAMPLING_MODE_OPTION] = mos.IID_CMA_SAMPLING
    options[mos.MOS_AUDIT_SINK_OPTION] = sink
    options[mos.MOS_AUDIT_CONTEXT_OPTION] = {"problem_id": "E1"}

    optimizer = mos.create_hcc_cmaes(_problem(3), options)

    assert type(optimizer) is vendor_type
    assert mos.CMA_SAMPLING_MODE_OPTION not in optimizer.options
    assert mos.MOS_AUDIT_SINK_OPTION not in optimizer.options
    assert mos.MOS_AUDIT_CONTEXT_OPTION not in optimizer.options
    assert options[mos.CMA_SAMPLING_MODE_OPTION] == mos.IID_CMA_SAMPLING
    assert sink == []


def test_mos_factory_returns_vendor_subclass_and_emits_safe_audit() -> None:
    vendor_type = _vendor_cmaes_type()
    sink: list[dict[str, object]] = []
    options = _options(dimension=3, population=5, budget=5)
    options[mos.CMA_SAMPLING_MODE_OPTION] = mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING
    options[mos.MOS_AUDIT_SINK_OPTION] = sink
    options[mos.MOS_AUDIT_CONTEXT_OPTION] = {
        "problem_id": "E2",
        "seed": np.int64(60),
        "outer_iter": 3,
        "group_index": 4,
        "cma_scope": "primary_cc",
        "optimizer_seed": 12345,
    }

    optimizer = mos.create_hcc_cmaes(_problem(3), options)
    result = optimizer.optimize()

    assert isinstance(optimizer, vendor_type)
    assert type(optimizer) is not vendor_type
    assert type(optimizer).__name__ == "ARACMirroredOrthogonalCMAES"
    assert result["n_function_evaluations"] == 5
    assert len(sink) == 1
    assert set(sink[0]) == {
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
        "problem_id",
        "seed",
        "outer_iter",
        "group_index",
        "cma_scope",
        "optimizer_seed",
    }
    assert sink[0]["generation"] == 0
    assert sink[0]["optimizer_restart_index"] == 0
    assert sink[0]["population"] == 5
    assert sink[0]["dimension"] == 3
    assert sink[0]["pair_count"] == 2
    assert sink[0]["block_count"] == 1
    assert sink[0]["rng_draw_count"] == 15
    assert sink[0]["evaluated_count"] == 5
    assert sink[0]["complete_population"] is True
    assert sink[0]["problem_id"] == "E2"
    assert sink[0]["seed"] == 60
    assert sink[0]["outer_iter"] == 3
    assert sink[0]["group_index"] == 4
    assert sink[0]["cma_scope"] == "primary_cc"
    assert sink[0]["optimizer_seed"] == 12345
    assert len(str(sink[0]["raw_draw_sha256"])) == 64
    assert len(str(sink[0]["sample_sha256"])) == 64
    assert not ({"raw_vectors", "objectives"} & set(sink[0]))


def test_audit_hashes_are_reproducible_for_matching_seed() -> None:
    audits: list[list[dict[str, object]]] = [[], []]
    for sink in audits:
        options = _options(dimension=4, population=6, budget=6, seed=41)
        options[mos.CMA_SAMPLING_MODE_OPTION] = mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING
        options[mos.MOS_AUDIT_SINK_OPTION] = sink
        mos.create_hcc_cmaes(_problem(4), options).optimize()

    assert audits[0] == audits[1]


def test_partial_final_generation_is_audited_but_not_distribution_updated() -> None:
    sink: list[dict[str, object]] = []
    options = _options(dimension=3, population=5, budget=7, seed=43)
    options[mos.CMA_SAMPLING_MODE_OPTION] = mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING
    options[mos.MOS_AUDIT_SINK_OPTION] = sink

    optimizer = mos.create_hcc_cmaes(_problem(3), options)
    result = optimizer.optimize()
    one_sample_tail_options = _options(
        dimension=3, population=5, budget=6, seed=43
    )
    one_sample_tail_options[mos.CMA_SAMPLING_MODE_OPTION] = (
        mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING
    )
    one_sample_tail_result = mos.create_hcc_cmaes(
        _problem(3), one_sample_tail_options
    ).optimize()

    assert result["n_function_evaluations"] == 7
    assert result["_n_generations"] == 1
    assert [row["generation"] for row in sink] == [0, 1]
    assert [row["population"] for row in sink] == [5, 5]
    assert [row["evaluated_count"] for row in sink] == [5, 2]
    assert [row["complete_population"] for row in sink] == [True, False]
    assert result["sigma"] == one_sample_tail_result["sigma"]
    for key in ("mean", "p_s", "p_c"):
        assert np.array_equal(result[key], one_sample_tail_result[key])


def test_vendor_restart_doubles_population_and_mos_uses_new_shape() -> None:
    sink: list[dict[str, object]] = []
    options = _options(dimension=3, population=4, budget=12, seed=47)
    options.update(
        {
            "is_restart": True,
            "stagnation": 1,
            "fitness_diff": 1e-12,
            mos.CMA_SAMPLING_MODE_OPTION: mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING,
            mos.MOS_AUDIT_SINK_OPTION: sink,
        }
    )

    optimizer = mos.create_hcc_cmaes(_problem(3), options)
    result = optimizer.optimize()

    assert result["n_function_evaluations"] == 12
    assert result["_n_restart"] == 1
    assert [row["population"] for row in sink] == [4, 8]
    assert [row["rng_draw_count"] for row in sink] == [12, 24]
    assert [row["generation"] for row in sink] == [0, 0]
    assert [row["optimizer_restart_index"] for row in sink] == [0, 1]
    assert [row["evaluated_count"] for row in sink] == [4, 8]
    assert [row["complete_population"] for row in sink] == [True, True]


def test_factory_rejects_unknown_mode_and_invalid_mos_sink() -> None:
    unknown = _options(dimension=2, population=4, budget=4)
    unknown[mos.CMA_SAMPLING_MODE_OPTION] = "unknown"
    with pytest.raises(ValueError, match="unsupported CMA sampling mode"):
        mos.create_hcc_cmaes(_problem(2), unknown)

    invalid_sink = _options(dimension=2, population=4, budget=4)
    invalid_sink[mos.CMA_SAMPLING_MODE_OPTION] = (
        mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING
    )
    invalid_sink[mos.MOS_AUDIT_SINK_OPTION] = object()
    with pytest.raises(ValueError, match="audit sink"):
        mos.create_hcc_cmaes(_problem(2), invalid_sink)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (["not", "a", "mapping"], "must be a mapping"),
        ({1: "value"}, "keys must be"),
        ({"nested": {"value": 1}}, "values must be scalar"),
        ({"score": float("inf")}, "floats must be finite"),
        ({"generation": 99}, "reserved field"),
    ],
)
def test_factory_rejects_unsafe_mos_audit_context(context, message: str) -> None:
    options = _options(dimension=2, population=4, budget=4)
    options[mos.CMA_SAMPLING_MODE_OPTION] = mos.MIRRORED_ORTHOGONAL_CMA_SAMPLING
    options[mos.MOS_AUDIT_CONTEXT_OPTION] = context

    with pytest.raises(ValueError, match=message):
        mos.create_hcc_cmaes(_problem(2), options)
