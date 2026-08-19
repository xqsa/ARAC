from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from arac.analysis.outcome_selector import (
    ActionOutcome,
    EVALUATION_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODEL_PARAMETERS,
    REGRET_TARGET_CAP,
    UNCERTAINTY_PENALTY_CANDIDATES,
    OutcomeRecord,
    OutcomeSelector,
    evaluate_training_selector,
    file_sha256,
    fit_outcome_selector,
)
from arac.evidence.phase1 import PHASE1_FEATURE_NAMES
from arac.runtime.contracts import ACTION_NAMES


def _record(case_id: str = "A1") -> OutcomeRecord:
    errors = {"ctp": 4.0, "smp": 3.0, "gcb": 2.0, "aor": 1.0}
    return OutcomeRecord(
        case_id=case_id,
        run_seed=9,
        checkpoint_hash="a" * 64,
        feature_names=PHASE1_FEATURE_NAMES,
        feature_values=(0.0,) * len(PHASE1_FEATURE_NAMES),
        outcomes=tuple(
            ActionOutcome(action, errors[action], str(index) * 64)
            for index, action in enumerate(ACTION_NAMES, start=1)
        ),
    )


def _selector_records(seed: int, *, separable: bool = True) -> tuple[OutcomeRecord, ...]:
    records = []
    for action_index, winner in enumerate(ACTION_NAMES):
        for repeat in range(6):
            errors = {action: 100.0 for action in ACTION_NAMES}
            errors[winner] = 1.0
            signal = float(action_index * 100 + repeat) if separable else 0.0
            records.append(
                OutcomeRecord(
                    case_id=f"case-{seed}-{action_index}-{repeat}",
                    run_seed=seed,
                    checkpoint_hash=f"{seed % 16:x}" * 64,
                    feature_names=PHASE1_FEATURE_NAMES,
                    feature_values=(signal,)
                    + (0.0,) * (len(PHASE1_FEATURE_NAMES) - 1),
                    outcomes=tuple(
                        ActionOutcome(action, errors[action], str(index) * 64)
                        for index, action in enumerate(ACTION_NAMES, start=1)
                    ),
                )
            )
    return tuple(records)


@pytest.fixture(autouse=True)
def _fast_random_forest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MODEL_PARAMETERS, "n_estimators", 64)


@pytest.fixture
def published_selector(tmp_path: Path) -> Path:
    destination = tmp_path / "selector"
    train = _selector_records(11) + _selector_records(13)
    holdout = _selector_records(17)
    result = fit_outcome_selector(train, holdout, output_directory=destination)
    assert result["holdout_passed"] is True
    return destination


def test_label_is_terminal_argmin_not_case_family() -> None:
    record = _record("S6")
    changed = replace(
        record,
        outcomes=tuple(
            replace(outcome, final_error=0.5 if outcome.action_name == "ctp" else 2.0)
            for outcome in record.outcomes
        ),
    )

    assert record.action_label == "aor"
    assert changed.action_label == "ctp"
    assert record.case_id == changed.case_id
    assert "case_id" not in PHASE1_FEATURE_NAMES


def test_outcome_record_round_trip_recomputes_and_checks_label() -> None:
    record = _record()

    assert OutcomeRecord.from_payload(record.payload()) == record
    corrupted = record.payload()
    corrupted["action_label"] = "ctp"
    body = dict(corrupted)
    body.pop("record_hash")
    from arac.analysis.outcome_selector import _canonical_sha256

    corrupted["record_hash"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="disagrees"):
        OutcomeRecord.from_payload(corrupted)


def test_frozen_selector_predicts_four_action_regrets(tmp_path: Path) -> None:
    train = _selector_records(11) + _selector_records(13)
    holdout = _selector_records(17)
    destination = tmp_path / "selector"

    result = fit_outcome_selector(train, holdout, output_directory=destination)
    selector = OutcomeSelector.load(destination)

    assert result["holdout_passed"] is True
    assert selector.metadata["model_type"] == "RandomForestRegressor"
    assert selector.metadata["predicts_action_log10_regret"] is True
    assert selector.metadata["regret_target_cap"] == REGRET_TARGET_CAP
    assert selector.metadata["uncertainty_penalty"] in UNCERTAINTY_PENALTY_CANDIDATES
    assert selector.metadata["training_seed_cv_passed"] is True
    assert selector.select(holdout[0].feature_names, holdout[0].feature_values) in ACTION_NAMES
    assert {path.name for path in destination.iterdir()} == {
        MODEL_FILENAME,
        METADATA_FILENAME,
        EVALUATION_FILENAME,
    }
    assert not tuple(tmp_path.glob(".selector.staging-*"))


def test_selector_penalizes_uncertain_action_regret() -> None:
    class Tree:
        def __init__(self, prediction):
            self.prediction = prediction

        def predict(self, features):
            return [self.prediction for _ in features]

    class Forest:
        estimators_ = (
            Tree((0.00, 0.15, 0.50, 0.50)),
            Tree((0.20, 0.15, 0.50, 0.50)),
        )

        def predict(self, features):
            return [[0.10, 0.15, 0.50, 0.50] for _ in features]

    selector = OutcomeSelector(
        model=Forest(),
        metadata={"uncertainty_penalty": 1.0},
    )

    selected = selector.select(
        PHASE1_FEATURE_NAMES,
        (0.0,) * len(PHASE1_FEATURE_NAMES),
    )

    assert selected == ACTION_NAMES[1]


def test_training_evaluation_reports_tail_risk_metrics() -> None:
    metrics = evaluate_training_selector(
        _selector_records(11) + _selector_records(13)
    )

    regret = metrics["terminal_regret"]
    assert regret["p95_log10_regret"] >= 0.0
    assert regret["cvar95_log10_regret"] >= regret["p95_log10_regret"]
    assert metrics["uncertainty_penalty"] in UNCERTAINTY_PENALTY_CANDIDATES


def test_failed_holdout_gate_does_not_publish_selector(tmp_path: Path) -> None:
    destination = tmp_path / "selector"
    train = _selector_records(11) + _selector_records(13)
    holdout = _selector_records(17, separable=False)

    with pytest.raises(RuntimeError, match="holdout gate failed"):
        fit_outcome_selector(train, holdout, output_directory=destination)

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".selector.staging-*"))


def test_failed_training_cv_gate_keeps_holdout_sealed(tmp_path: Path) -> None:
    destination = tmp_path / "selector"
    train = _selector_records(11, separable=False) + _selector_records(
        13, separable=False
    )

    with pytest.raises(RuntimeError, match="training CV gate failed"):
        fit_outcome_selector(
            train,
            _selector_records(17),
            output_directory=destination,
        )

    assert evaluate_training_selector(train)["passed"] is False
    assert not destination.exists()


def test_regression_target_uses_frozen_training_cap() -> None:
    from arac.analysis.outcome_selector import _regression_targets

    record = _record()
    record = replace(
        record,
        outcomes=(
            ActionOutcome("ctp", 1_000_000.0, "1" * 64),
            *record.outcomes[1:],
        ),
    )
    targets = _regression_targets((record,))

    assert targets.shape == (1, len(ACTION_NAMES))
    assert float(targets.max()) == REGRET_TARGET_CAP


@pytest.mark.parametrize(
    "filename",
    [MODEL_FILENAME, METADATA_FILENAME, EVALUATION_FILENAME],
)
def test_load_rejects_incomplete_artifact_set(
    published_selector: Path,
    filename: str,
) -> None:
    (published_selector / filename).unlink()

    with pytest.raises(ValueError, match="artifact set is incomplete"):
        OutcomeSelector.load(published_selector)


def test_load_rejects_tampered_model(published_selector: Path) -> None:
    model_path = published_selector / MODEL_FILENAME
    with model_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="model hash drifted"):
        OutcomeSelector.load(published_selector)


def test_load_rejects_tampered_evaluation(published_selector: Path) -> None:
    evaluation_path = published_selector / EVALUATION_FILENAME
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["holdout"]["accuracy"] = 0.0
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation hash drifted"):
        OutcomeSelector.load(published_selector)


def test_load_rejects_failed_holdout_even_when_hash_is_updated(
    published_selector: Path,
) -> None:
    evaluation_path = published_selector / EVALUATION_FILENAME
    metadata_path = published_selector / METADATA_FILENAME
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["holdout"]["passed"] = False
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["holdout_passed"] = False
    metadata["evaluation_sha256"] = file_sha256(evaluation_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="holdout gate did not pass"):
        OutcomeSelector.load(published_selector)


def test_load_rejects_regret_target_cap_drift(published_selector: Path) -> None:
    metadata_path = published_selector / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["regret_target_cap"] = REGRET_TARGET_CAP + 0.01
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="regret target cap drifted"):
        OutcomeSelector.load(published_selector)


def test_load_rejects_uncertainty_penalty_drift(published_selector: Path) -> None:
    metadata_path = published_selector / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["uncertainty_penalty"] = 999.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="uncertainty penalty drifted"):
        OutcomeSelector.load(published_selector)
