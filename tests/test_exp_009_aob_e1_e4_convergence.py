from __future__ import annotations

import csv

import numpy as np


def test_exp_009_defaults_to_e1_e4_single_seed_protocol() -> None:
    from experiments.exp_009_aob_e1_e4_convergence.run import (
        DEFAULT_CASES,
        DEFAULT_MAX_FES,
        METHOD_ORDER,
        parse_args,
    )

    args = parse_args([])

    assert tuple(args.cases) == DEFAULT_CASES
    assert args.seed == 1
    assert args.max_fes == DEFAULT_MAX_FES
    assert args.jobs == 4
    assert tuple(METHOD_ORDER) == ("ARAC-v33.8", "HCC_ES", "MMES", "RDDSM_CMAES")


def test_exp_009_downsample_preserves_monotonic_best_so_far_and_endpoint() -> None:
    from experiments.exp_009_aob_e1_e4_convergence.run import downsample_curve

    points = downsample_curve([9.0, 8.0, 8.5, 4.0, 4.0, 2.0], max_fes=6, checkpoints=4)

    assert points[-1] == (6, 2.0)
    assert [fe for fe, _value in points] == sorted(fe for fe, _value in points)
    assert np.all(np.diff([value for _fe, value in points]) <= 0)


def test_exp_009_curve_csv_is_explicitly_fe_and_best_so_far(tmp_path) -> None:
    from experiments.exp_009_aob_e1_e4_convergence.run import write_curve

    path = tmp_path / "convergence.csv"
    write_curve(path, [(1, 10.0), (4, 2.0)])

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"fe": "1", "best_so_far": "10.0"},
        {"fe": "4", "best_so_far": "2.0"},
    ]
