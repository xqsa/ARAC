"""Continuous WLOC baseline methods and reproducible execution contracts."""

from .contracts import (
    BASELINE_RESULT_SCHEMA_VERSION,
    GROUPING_SCHEMA_VERSION,
    BaselineResult,
    GroupingResult,
)
from .grouping import (
    design_matrix_from_groups,
    dg2_grouping,
    random_grouping,
    rddsm_grouping,
    rdg3_grouping,
)
from .optimization import (
    cmaes_population_size,
    derive_optimizer_seed,
    run_cooperative_cmaes,
)

__all__ = [
    "BASELINE_RESULT_SCHEMA_VERSION",
    "GROUPING_SCHEMA_VERSION",
    "BaselineResult",
    "GroupingResult",
    "cmaes_population_size",
    "design_matrix_from_groups",
    "dg2_grouping",
    "derive_optimizer_seed",
    "random_grouping",
    "rddsm_grouping",
    "rdg3_grouping",
    "run_cooperative_cmaes",
]
