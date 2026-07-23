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
from .hcc_es import hcc_global_phase_fes, overlap_degree, run_hcc_es
from .optimization import (
    cmaes_population_size,
    derive_optimizer_seed,
    run_cooperative_cmaes,
)
from .pypop_adapters import PYPOP7_VERSION, PYPOP_METHODS, run_pypop_baseline

__all__ = [
    "BASELINE_RESULT_SCHEMA_VERSION",
    "GROUPING_SCHEMA_VERSION",
    "BaselineResult",
    "GroupingResult",
    "PYPOP7_VERSION",
    "PYPOP_METHODS",
    "cmaes_population_size",
    "design_matrix_from_groups",
    "dg2_grouping",
    "derive_optimizer_seed",
    "hcc_global_phase_fes",
    "overlap_degree",
    "random_grouping",
    "rddsm_grouping",
    "rdg3_grouping",
    "run_cooperative_cmaes",
    "run_hcc_es",
    "run_pypop_baseline",
]
