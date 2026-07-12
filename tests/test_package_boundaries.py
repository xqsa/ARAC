from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from arac.action_space import (
    ActionDecision as LegacyActionDecision,
    ActionFamily as LegacyActionFamily,
    ActionSpec as LegacyActionSpec,
    DEFAULT_ACTION_SPACE as LEGACY_ACTION_SPACE,
    action_by_name as legacy_action_by_name,
)
from arac.actions import (
    ActionDecision,
    ActionFamily,
    ActionSpec,
    DEFAULT_ACTION_SPACE,
    action_by_name,
)
from arac.audit import (
    active_action_has_effect as legacy_active_action_has_effect,
    claim_gate as legacy_claim_gate,
    find_forbidden_runtime_fields as legacy_find_forbidden_runtime_fields,
)
from arac.audits import (
    active_action_has_effect,
    claim_gate,
    find_forbidden_runtime_fields,
)
from arac.backend_adapter import (
    BackendAdapter as LegacyBackendAdapter,
    BackendSemanticsDiff as LegacyBackendSemanticsDiff,
    NullBackendAdapter as LegacyNullBackendAdapter,
    ToyBackendAdapter as LegacyToyBackendAdapter,
)
from arac.evaluation import (
    SameBudgetLedger as LegacySameBudgetLedger,
    classify_utility as legacy_classify_utility,
    relative_gain as legacy_relative_gain,
)
from arac.execution import (
    BackendAdapter,
    BackendSemanticsDiff,
    NullBackendAdapter,
    ToyBackendAdapter,
)
from arac.evaluation.ledger import SameBudgetLedger, classify_utility, relative_gain


ROOT = Path(__file__).resolve().parents[1]


def test_new_package_boundaries_export_legacy_objects_by_identity() -> None:
    assert LegacyActionFamily is ActionFamily
    assert LegacyActionDecision is ActionDecision
    assert LegacyActionSpec is ActionSpec
    assert LEGACY_ACTION_SPACE is DEFAULT_ACTION_SPACE
    assert legacy_action_by_name is action_by_name

    assert LegacyBackendAdapter is BackendAdapter
    assert LegacyBackendSemanticsDiff is BackendSemanticsDiff
    assert LegacyNullBackendAdapter is NullBackendAdapter
    assert LegacyToyBackendAdapter is ToyBackendAdapter

    assert LegacySameBudgetLedger is SameBudgetLedger
    assert legacy_relative_gain is relative_gain
    assert legacy_classify_utility is classify_utility

    assert legacy_find_forbidden_runtime_fields is find_forbidden_runtime_fields
    assert legacy_active_action_has_effect is active_action_has_effect
    assert legacy_claim_gate is claim_gate


def test_migrated_behavior_is_unchanged() -> None:
    decision = ActionDecision(
        ActionFamily.ISOLATE,
        "isolate_conflicting_relation",
        "allow",
        "boundary-test",
        0.4,
    )
    semantics = ToyBackendAdapter().apply(decision)
    ledger = SameBudgetLedger(
        phase_i_fe=10,
        phase_ii_fe=20,
        budget_limit=40,
        fresh_execution=True,
    )

    assert semantics == BackendSemanticsDiff(relation_handling_changed=True)
    assert semantics.changed is True
    assert relative_gain(10.0, 9.0) == legacy_relative_gain(10.0, 9.0)
    assert classify_utility(10.0, 9.0) == legacy_classify_utility(10.0, 9.0)
    assert find_forbidden_runtime_fields({"final_error": 1.0}) == ["final_error"]
    assert active_action_has_effect(decision, semantics) is True
    assert claim_gate(
        runtime_payload={"overlap_degree": 0.2},
        decision=decision,
        semantics_diff=semantics,
        ledger=ledger,
        utility_label="meaningful_win",
        negative_control_pass=True,
    ) == (True, [])


def test_compatibility_modules_contain_only_reexports() -> None:
    for relative_path in (
        "src/arac/action_space.py",
        "src/arac/backend_adapter.py",
        "src/arac/audit.py",
    ):
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert definitions == [], relative_path


def test_runtime_imports_do_not_load_vendor_or_read_offline_csv() -> None:
    script = r'''
import builtins
import sys

real_open = builtins.open

def guarded_open(file, *args, **kwargs):
    path = str(file).replace("\\", "/").lower()
    if path.endswith(".csv") and ("/paper/" in path or "/references/" in path):
        raise AssertionError(f"runtime opened offline CSV: {file}")
    return real_open(file, *args, **kwargs)

builtins.open = guarded_open
import arac.actions
import arac.audit
import arac.audits
import arac.backend_adapter
import arac.backends.hcc
import arac.evaluation
import arac.execution
import arac.policy
import arac.evidence
assert "vendor.hcc" not in sys.modules
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
