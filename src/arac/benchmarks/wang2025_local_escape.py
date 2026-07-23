"""Frozen 18-case Wang screening suite for deceptive local-optimum escape."""

from __future__ import annotations

from dataclasses import dataclass

from .wang2025_overlapping import Wang2025OverlappingProblem, Wang2025OverlappingSpec


WANG2025_LOCAL_ESCAPE_SUITE_VERSION = "wang2025-local-escape-screening-v1"


@dataclass(frozen=True)
class Wang2025LocalEscapeCase:
    """One paper-indexed synthetic case with a frozen generated instance hash."""

    case_id: str
    paper_function_id: str
    grouping_mode: str
    overlap_percent: int
    spec: Wang2025OverlappingSpec
    expected_instance_hash: str

    def generate(self) -> Wang2025OverlappingProblem:
        problem = Wang2025OverlappingProblem.generate(self.spec)
        if problem.instance_hash != self.expected_instance_hash:
            raise RuntimeError(
                f"{self.case_id} generated instance hash does not match the frozen catalog"
            )
        return problem


def _case(
    index: int,
    alpha: float,
    min_group_size: int,
    max_group_size: int,
    overlap_count: int,
    seed: int,
    instance_hash: str,
) -> Wang2025LocalEscapeCase:
    return Wang2025LocalEscapeCase(
        case_id=f"WLOC{index:02d}",
        paper_function_id=f"f{index}",
        grouping_mode="equal" if min_group_size == max_group_size else "unequal",
        overlap_percent=overlap_count // 10,
        spec=Wang2025OverlappingSpec(
            dimension=1000,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            alpha=alpha,
            overlap_count=overlap_count,
            beta=0.5,
            gamma=0.5,
            permuted=False,
            seed=seed,
            conflict_ratio=0.0,
        ),
        expected_instance_hash=instance_hash,
    )


# alpha, min/max group size, overlap count, paired seed, frozen instance hash.
_CASE_ROWS = (
    (
        0.1,
        5,
        5,
        100,
        2026072301,
        "a712c988c2de91fe32ddfc41b5fef6dae88031546eace61c1d665fc755edea47",
    ),
    (
        0.1,
        5,
        5,
        200,
        2026072302,
        "2c5fd5ddb52c43c50ad2c0c55ab2b5e1340480ae7643ff6476733f2b4ec4d9b5",
    ),
    (
        0.1,
        5,
        5,
        300,
        2026072303,
        "a9166488ded43ca1f367c17b02c84f3c93b35e3d662865f67618fa105fd97c7d",
    ),
    (
        0.1,
        2,
        5,
        100,
        2026072304,
        "1df156b578e6e04669a252cb53378a2cba6cfed383bb5d8c2ba9e461eacdae5c",
    ),
    (
        0.1,
        2,
        5,
        200,
        2026072305,
        "8741278b0778128db6ba56c033daf7efcc6b9f8f800e81a00a1e6ee341d1a2ca",
    ),
    (
        0.1,
        2,
        5,
        300,
        2026072306,
        "7ac44e6acb9bce8cae700e304913d45a86605d850665c8e8a69fbd0cc4993959",
    ),
    (
        0.5,
        5,
        5,
        100,
        2026072301,
        "eeb7d3232084f0f19a85a4c2a529286bdcb4b00369e83351c38d09a600a53849",
    ),
    (
        0.5,
        5,
        5,
        200,
        2026072302,
        "c7bc5af5c8dc80dcd1f085c47938318068bbc22a423adbebffe188bf05397b12",
    ),
    (
        0.5,
        5,
        5,
        300,
        2026072303,
        "d6ad164e3e0107300518d3040c9c42360ea831ae33ae53021f14a323cce99c27",
    ),
    (
        0.5,
        2,
        5,
        100,
        2026072304,
        "dd37ca60011bbb987b167472e0e13ea87281af46fdee8a2e045ce0a08f5e9377",
    ),
    (
        0.5,
        2,
        5,
        200,
        2026072305,
        "6fcf1e96c52dfab8be7cbd83e0d3b2fd7b45f3cc2fc30442dddf0b6483acbe88",
    ),
    (
        0.5,
        2,
        5,
        300,
        2026072306,
        "636a939545fa6fd2815dc8cdc213cbbe9ef262ff8bef1954f21e7ac6e44ae134",
    ),
    (
        0.8,
        5,
        5,
        100,
        2026072301,
        "fff1edab92004bf1a33d95083ea95d0b88897fcd0127545f9804a8c905a87c01",
    ),
    (
        0.8,
        5,
        5,
        200,
        2026072302,
        "a786447c1bf14f66f7b4fa463ec1cd134206838143c2c56c7b2aabf557641676",
    ),
    (
        0.8,
        5,
        5,
        300,
        2026072303,
        "c87736d294e319eaf6cd6d308556e623b19d0ceb9ad1b54af2fd532a6881d060",
    ),
    (
        0.8,
        2,
        5,
        100,
        2026072304,
        "17f55161c80f34b94b4f8851109d1924ab264e9c201425cdc74b4138a94fa2e8",
    ),
    (
        0.8,
        2,
        5,
        200,
        2026072305,
        "f8c8b54c0ef81fee1737c9eec5c5c185dca9353d4fb2286e282752d62dda24e7",
    ),
    (
        0.8,
        2,
        5,
        300,
        2026072306,
        "cfe872c00779b6faa09c0df9b8f849976e92e3fdae3083941e1d68850bd0c912",
    ),
)

WANG2025_LOCAL_ESCAPE_CASES = tuple(
    _case(index, *row) for index, row in enumerate(_CASE_ROWS, start=1)
)


def get_wang2025_local_escape_case(case_id: str) -> Wang2025LocalEscapeCase:
    normalized = str(case_id).strip().upper()
    for case in WANG2025_LOCAL_ESCAPE_CASES:
        if normalized in {case.case_id, case.paper_function_id.upper()}:
            return case
    raise KeyError(f"unknown Wang 2025 local-escape case: {case_id}")
