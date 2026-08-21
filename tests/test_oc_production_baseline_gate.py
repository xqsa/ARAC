from __future__ import annotations

import numpy as np

from experiments.oc_lagged_coupling_shadow import ARMS
from experiments.oc_production_baseline_gate import (
    ACTION_ARM_MAP,
    BEST_FIXED_ARM,
    SCENARIOS,
    _bootstrap_ci_mean,
    _decide,
)


def test_action_arm_map_is_complete_and_pre_registered() -> None:
    from arac.coordination.contract import (
        OC_ACTION_AOR,
        OC_ACTION_ARBITRATION,
        OC_ACTION_CTP_RESTRICTED,
        OC_ACTION_CTP_SHARED_CORE,
        OC_ACTION_SMP,
    )

    assert set(ACTION_ARM_MAP) == {
        OC_ACTION_ARBITRATION,
        OC_ACTION_SMP,
        OC_ACTION_CTP_RESTRICTED,
        OC_ACTION_CTP_SHARED_CORE,
        OC_ACTION_AOR,
    }
    for arm in ACTION_ARM_MAP.values():
        assert arm is None or arm in ARMS
    assert BEST_FIXED_ARM in ARMS
    assert SCENARIOS == {"fresh": 0, "persistent": 2, "escalated": 6}


def test_bootstrap_ci_mean_width() -> None:
    values = [1.0] * 5 + [0.0] * 5
    low, high = _bootstrap_ci_mean(values)
    assert low < 0.5 < high
    degenerate = _bootstrap_ci_mean([3.0, 3.0, 3.0])
    assert degenerate == (3.0, 3.0)


def test_single_context_decision_replay() -> None:
    decision = _decide("conforming", "chain", 6, 31501)
    assert set(decision["decisions"]) == set(SCENARIOS)
    for scenario in SCENARIOS:
        block = decision["decisions"][scenario]
        assert block["action"] in ACTION_ARM_MAP
        assert block["reason"]
        assert block["mapped_arm"] is None or block["mapped_arm"] in ARMS
        assert block["reserved_fes"] >= 0
    assert decision["selected_component"]
    assert np.isfinite(decision["arbitration_gain"])
    assert decision["reconstruction_fes"] > 0
