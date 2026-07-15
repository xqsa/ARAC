"""Identity-free causal utility inference and fail-closed action release.

The module is deliberately independent from HCC and training dependencies.
Offline code exports a small audited JSON tree bundle; runtime code validates
that bundle, evaluates only the fixed pre-action feature schema, and falls
back to the baseline whenever causal benefit or tail safety is uncertain.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Mapping, Sequence


PRE_ACTION_UTILITY_SCHEMA_VERSION = "arac.pre_action_utility.v1"
CAUSAL_RISK_MODEL_SCHEMA_VERSION = "arac.causal_risk_model.v1"
MAX_CATASTROPHIC_RISK = 0.05
BOOTSTRAP_TREE_COUNT = 1000
MAX_TREE_DEPTH = 2

UTILITY_FEATURE_NAMES = (
    "remaining_fe_ratio",
    "revisit_cap_remaining_ratio",
    "component_group_fraction",
    "component_shared_variable_ratio",
    "component_mean_overlap_ratio",
    "proposal_disagreement_mean_2",
    "candidate_dose_ratio",
    "phase_i_tail_progress_rate",
    "cc_progress_rate_last",
    "cc_progress_rate_slope_4",
    "cc_progress_rate_std_4",
    "cc_stagnation_streak",
    "terminal_sigma_ratio_last",
    "log_sigma_slope_3",
    "success_generation_ratio_last",
    "offspring_diversity_ratio_last",
)


class CausalRiskInvariantError(ValueError):
    """Raised when a causal-risk hard contract is invalid."""


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CausalRiskInvariantError("payload must be finite JSON data") from exc


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


FEATURE_SCHEMA_SHA256 = _sha256(
    {
        "schema_version": PRE_ACTION_UTILITY_SCHEMA_VERSION,
        "feature_names": list(UTILITY_FEATURE_NAMES),
    }
)


def compute_model_sha256(payload: Mapping[str, object]) -> str:
    """Hash a model payload while excluding its self-referential hash field."""

    if not isinstance(payload, Mapping):
        raise CausalRiskInvariantError("model payload must be a JSON object")
    material = dict(payload)
    material.pop("model_sha256", None)
    return _sha256(material)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise CausalRiskInvariantError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalRiskInvariantError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise CausalRiskInvariantError(f"{name} must be finite")
    return numeric


def _strict_mapping(
    payload: object,
    *,
    name: str,
    fields: Sequence[str],
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise CausalRiskInvariantError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise CausalRiskInvariantError(f"{name} keys must be strings")
    expected = set(fields)
    observed = set(payload)
    unknown = sorted(observed - expected)
    if unknown:
        raise CausalRiskInvariantError(
            f"unknown {name} field(s): " + ", ".join(unknown)
        )
    missing = sorted(expected - observed)
    if missing:
        raise CausalRiskInvariantError(
            f"missing {name} field(s): " + ", ".join(missing)
        )
    return payload


def _numeric_vector(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CausalRiskInvariantError(f"{name} must be a numeric array")
    values = tuple(_finite(f"{name}[{index}]", item) for index, item in enumerate(value))
    if len(values) != len(UTILITY_FEATURE_NAMES):
        raise CausalRiskInvariantError(
            f"{name} must contain exactly {len(UTILITY_FEATURE_NAMES)} values"
        )
    return values


@dataclass(frozen=True)
class PreActionUtilityState:
    """Immutable, identity-free state captured strictly before an action."""

    schema_version: str
    remaining_fe_ratio: float
    revisit_cap_remaining_ratio: float
    component_group_fraction: float
    component_shared_variable_ratio: float
    component_mean_overlap_ratio: float
    proposal_disagreement_mean_2: float
    candidate_dose_ratio: float
    phase_i_tail_progress_rate: float
    cc_progress_rate_last: float
    cc_progress_rate_slope_4: float
    cc_progress_rate_std_4: float
    cc_stagnation_streak: float
    terminal_sigma_ratio_last: float
    log_sigma_slope_3: float
    success_generation_ratio_last: float
    offspring_diversity_ratio_last: float

    _RUNTIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        *UTILITY_FEATURE_NAMES,
    )
    _FORBIDDEN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "action_error",
            "baseline_error",
            "case",
            "case_id",
            "case_label",
            "component_fingerprint",
            "component_gain",
            "component_id",
            "final_error",
            "final_outcome",
            "function_family",
            "function_name",
            "graph_fingerprint",
            "group_id",
            "group_index",
            "historical_best",
            "historical_outcome",
            "incumbent",
            "lane",
            "neighbor_gain",
            "objective",
            "overwrite",
            "paper_best",
            "problem",
            "problem_id",
            "raw_objective",
            "relation_fingerprint",
            "relative_gain",
            "resolution",
            "run",
            "run_id",
            "seed",
            "survival",
            "target_vector",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != PRE_ACTION_UTILITY_SCHEMA_VERSION:
            raise CausalRiskInvariantError(
                f"unsupported pre-action schema: {self.schema_version!r}"
            )
        for name in UTILITY_FEATURE_NAMES:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @classmethod
    def runtime_field_names(cls) -> tuple[str, ...]:
        return cls._RUNTIME_FIELDS

    @classmethod
    def forbidden_field_names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._FORBIDDEN_FIELDS))

    @classmethod
    def from_runtime_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "PreActionUtilityState":
        if not isinstance(payload, Mapping):
            raise CausalRiskInvariantError("runtime payload must be a mapping")
        if any(not isinstance(key, str) for key in payload):
            raise CausalRiskInvariantError("runtime payload keys must be strings")
        observed = set(payload)
        forbidden = sorted(observed & cls._FORBIDDEN_FIELDS)
        if forbidden:
            raise CausalRiskInvariantError(
                "forbidden runtime field(s): " + ", ".join(forbidden)
            )
        expected = set(cls._RUNTIME_FIELDS)
        unknown = sorted(observed - expected)
        if unknown:
            raise CausalRiskInvariantError(
                "unknown runtime field(s): " + ", ".join(unknown)
            )
        missing = sorted(expected - observed)
        if missing:
            raise CausalRiskInvariantError(
                "missing runtime field(s): " + ", ".join(missing)
            )
        return cls(**{name: payload[name] for name in cls._RUNTIME_FIELDS})

    @property
    def feature_values(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in UTILITY_FEATURE_NAMES)

    @property
    def feature_sha256(self) -> str:
        return _sha256(
            {
                "schema_version": self.schema_version,
                "features": [
                    [name, value]
                    for name, value in zip(
                        UTILITY_FEATURE_NAMES,
                        self.feature_values,
                        strict=True,
                    )
                ],
            }
        )


@dataclass(frozen=True)
class UtilityEstimate:
    """Calibrated causal utility and tail-risk estimate for one snapshot."""

    tau_hat: float
    tau_lcb: float
    catastrophic_risk_ucb: float
    in_distribution: bool
    ood_reasons: tuple[str, ...]
    model_hash: str
    feature_hash: str

    def __post_init__(self) -> None:
        tau_hat = _finite("tau_hat", self.tau_hat)
        tau_lcb = _finite("tau_lcb", self.tau_lcb)
        risk = _finite("catastrophic_risk_ucb", self.catastrophic_risk_ucb)
        if not 0.0 <= risk <= 1.0:
            raise CausalRiskInvariantError(
                "catastrophic_risk_ucb must be within [0, 1]"
            )
        if not isinstance(self.in_distribution, bool):
            raise CausalRiskInvariantError("in_distribution must be boolean")
        reasons = tuple(str(reason) for reason in self.ood_reasons)
        if any(not reason for reason in reasons):
            raise CausalRiskInvariantError("OOD reasons must be non-empty strings")
        if self.in_distribution == bool(reasons):
            raise CausalRiskInvariantError(
                "OOD reasons must be empty exactly when state is in-distribution"
            )
        for name, value in (
            ("model_hash", self.model_hash),
            ("feature_hash", self.feature_hash),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise CausalRiskInvariantError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "tau_hat", tau_hat)
        object.__setattr__(self, "tau_lcb", tau_lcb)
        object.__setattr__(self, "catastrophic_risk_ucb", risk)
        object.__setattr__(self, "ood_reasons", reasons)


@dataclass(frozen=True)
class _TreeNode:
    value: float | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    def predict(self, features: tuple[float, ...]) -> float:
        node = self
        while node.value is None:
            if (
                node.feature_index is None
                or node.threshold is None
                or node.left is None
                or node.right is None
            ):
                raise CausalRiskInvariantError("tree node is incomplete")
            node = (
                node.left
                if features[node.feature_index] <= node.threshold
                else node.right
            )
        return node.value


def _parse_tree(
    payload: object,
    *,
    name: str,
    depth: int = 0,
    risk_tree: bool = False,
) -> _TreeNode:
    if not isinstance(payload, Mapping):
        raise CausalRiskInvariantError(f"{name} node must be a JSON object")
    keys = set(payload)
    if keys == {"value"}:
        value = _finite(f"{name}.value", payload["value"])
        if risk_tree and not 0.0 <= value <= 1.0:
            raise CausalRiskInvariantError(f"{name}.value must be within [0, 1]")
        return _TreeNode(value=value)
    expected = {"feature", "threshold", "left", "right"}
    if keys != expected:
        raise CausalRiskInvariantError(
            f"{name} node must be exactly a value leaf or split node"
        )
    if depth >= MAX_TREE_DEPTH:
        raise CausalRiskInvariantError(f"{name} depth exceeds {MAX_TREE_DEPTH}")
    feature = payload["feature"]
    if feature not in UTILITY_FEATURE_NAMES:
        raise CausalRiskInvariantError(f"unknown tree feature: {feature!r}")
    threshold = _finite(f"{name}.threshold", payload["threshold"])
    return _TreeNode(
        feature_index=UTILITY_FEATURE_NAMES.index(str(feature)),
        threshold=threshold,
        left=_parse_tree(
            payload["left"],
            name=f"{name}.left",
            depth=depth + 1,
            risk_tree=risk_tree,
        ),
        right=_parse_tree(
            payload["right"],
            name=f"{name}.right",
            depth=depth + 1,
            risk_tree=risk_tree,
        ),
    )


def _parse_bootstrap_trees(
    payload: object,
    *,
    name: str,
    risk_tree: bool,
) -> tuple[_TreeNode, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise CausalRiskInvariantError(f"{name} must be an array")
    if len(payload) != BOOTSTRAP_TREE_COUNT:
        raise CausalRiskInvariantError(
            f"{name} must contain exactly {BOOTSTRAP_TREE_COUNT} trees"
        )
    return tuple(
        _parse_tree(
            tree,
            name=f"{name}[{index}]",
            risk_tree=risk_tree,
        )
        for index, tree in enumerate(payload)
    )


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise CausalRiskInvariantError("cannot aggregate an empty tree ensemble")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_json_object(text: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CausalRiskInvariantError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(text, object_pairs_hook=reject_duplicates)
    except CausalRiskInvariantError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CausalRiskInvariantError("model bundle is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise CausalRiskInvariantError("model bundle must be a JSON object")
    return payload


@dataclass(frozen=True)
class CausalRiskModelBundle:
    """Validated pure-Python representation of an audited JSON model."""

    model_sha256: str
    _median: tuple[float, ...]
    _iqr: tuple[float, ...]
    _minimum: tuple[float, ...]
    _maximum: tuple[float, ...]
    _reference_scaled: tuple[tuple[float, ...], ...]
    _knn_k: int
    _knn_distance_threshold: float
    _utility_trees: tuple[_TreeNode, ...]
    _lcb_quantile: float
    _conformal_margin: float
    _risk_trees: tuple[_TreeNode, ...]
    _risk_quantile: float
    _clopper_pearson_tree: _TreeNode

    @classmethod
    def from_json(cls, text: str | bytes | bytearray) -> "CausalRiskModelBundle":
        if isinstance(text, (bytes, bytearray)):
            try:
                decoded = bytes(text).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CausalRiskInvariantError("model bundle must be UTF-8 JSON") from exc
        elif isinstance(text, str):
            decoded = text
        else:
            raise CausalRiskInvariantError("model bundle must be JSON text")
        return cls.from_mapping(_load_json_object(decoded))

    @classmethod
    def load(cls, path: str | Path) -> "CausalRiskModelBundle":
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CausalRiskInvariantError(f"cannot read model bundle: {path}") from exc
        return cls.from_json(text)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
    ) -> "CausalRiskModelBundle":
        root = _strict_mapping(
            payload,
            name="model",
            fields=(
                "schema_version",
                "feature_schema",
                "ood",
                "utility",
                "catastrophic_risk",
                "model_sha256",
            ),
        )
        if root["schema_version"] != CAUSAL_RISK_MODEL_SCHEMA_VERSION:
            raise CausalRiskInvariantError(
                f"unsupported model schema: {root['schema_version']!r}"
            )
        observed_hash = root["model_sha256"]
        expected_hash = compute_model_sha256(root)
        if observed_hash != expected_hash:
            raise CausalRiskInvariantError("model_sha256 mismatch")

        feature_schema = _strict_mapping(
            root["feature_schema"],
            name="feature schema",
            fields=("schema_version", "feature_names", "sha256"),
        )
        if feature_schema["schema_version"] != PRE_ACTION_UTILITY_SCHEMA_VERSION:
            raise CausalRiskInvariantError("feature schema version mismatch")
        feature_names = feature_schema["feature_names"]
        if (
            not isinstance(feature_names, Sequence)
            or isinstance(feature_names, (str, bytes, bytearray))
            or tuple(feature_names) != UTILITY_FEATURE_NAMES
        ):
            raise CausalRiskInvariantError("feature name/order mismatch")
        if feature_schema["sha256"] != FEATURE_SCHEMA_SHA256:
            raise CausalRiskInvariantError("feature schema hash mismatch")

        ood = _strict_mapping(
            root["ood"],
            name="OOD",
            fields=(
                "median",
                "iqr",
                "minimum",
                "maximum",
                "reference_scaled",
                "knn_k",
                "knn_distance_threshold",
            ),
        )
        median = _numeric_vector(ood["median"], name="OOD median")
        iqr = _numeric_vector(ood["iqr"], name="OOD IQR")
        minimum = _numeric_vector(ood["minimum"], name="OOD minimum")
        maximum = _numeric_vector(ood["maximum"], name="OOD maximum")
        for index, name in enumerate(UTILITY_FEATURE_NAMES):
            if iqr[index] < 0.0:
                raise CausalRiskInvariantError(f"OOD IQR for {name} must be non-negative")
            if minimum[index] > maximum[index]:
                raise CausalRiskInvariantError(f"OOD range for {name} is reversed")
            if not minimum[index] <= median[index] <= maximum[index]:
                raise CausalRiskInvariantError(f"OOD median for {name} is outside its range")
        knn_k_raw = ood["knn_k"]
        if isinstance(knn_k_raw, bool) or knn_k_raw != 5:
            raise CausalRiskInvariantError("OOD knn_k must equal 5")
        knn_k = 5
        threshold = _finite(
            "OOD knn_distance_threshold",
            ood["knn_distance_threshold"],
        )
        if threshold < 0.0:
            raise CausalRiskInvariantError(
                "OOD knn_distance_threshold must be non-negative"
            )
        references_raw = ood["reference_scaled"]
        if (
            not isinstance(references_raw, Sequence)
            or isinstance(references_raw, (str, bytes, bytearray))
        ):
            raise CausalRiskInvariantError("OOD reference_scaled must be an array")
        references = tuple(
            _numeric_vector(row, name=f"OOD reference_scaled[{index}]")
            for index, row in enumerate(references_raw)
        )
        if len(references) < knn_k:
            raise CausalRiskInvariantError(
                "OOD reference_scaled must contain at least knn_k rows"
            )

        utility = _strict_mapping(
            root["utility"],
            name="utility",
            fields=("bootstrap_trees", "lcb_quantile", "conformal_margin"),
        )
        lcb_quantile = _finite("utility lcb_quantile", utility["lcb_quantile"])
        if lcb_quantile != 0.05:
            raise CausalRiskInvariantError("utility lcb_quantile must equal 0.05")
        conformal_margin = _finite(
            "utility conformal_margin",
            utility["conformal_margin"],
        )
        if conformal_margin < 0.0:
            raise CausalRiskInvariantError(
                "utility conformal_margin must be non-negative"
            )
        utility_trees = _parse_bootstrap_trees(
            utility["bootstrap_trees"],
            name="utility bootstrap_trees",
            risk_tree=False,
        )

        risk = _strict_mapping(
            root["catastrophic_risk"],
            name="catastrophic risk",
            fields=(
                "bootstrap_trees",
                "bootstrap_quantile",
                "clopper_pearson_tree",
            ),
        )
        risk_quantile = _finite(
            "catastrophic risk bootstrap_quantile",
            risk["bootstrap_quantile"],
        )
        if risk_quantile != 0.95:
            raise CausalRiskInvariantError(
                "catastrophic risk bootstrap_quantile must equal 0.95"
            )
        risk_trees = _parse_bootstrap_trees(
            risk["bootstrap_trees"],
            name="catastrophic risk bootstrap_trees",
            risk_tree=True,
        )
        clopper_pearson_tree = _parse_tree(
            risk["clopper_pearson_tree"],
            name="catastrophic risk clopper_pearson_tree",
            risk_tree=True,
        )
        return cls(
            model_sha256=str(observed_hash),
            _median=median,
            _iqr=iqr,
            _minimum=minimum,
            _maximum=maximum,
            _reference_scaled=references,
            _knn_k=knn_k,
            _knn_distance_threshold=threshold,
            _utility_trees=utility_trees,
            _lcb_quantile=lcb_quantile,
            _conformal_margin=conformal_margin,
            _risk_trees=risk_trees,
            _risk_quantile=risk_quantile,
            _clopper_pearson_tree=clopper_pearson_tree,
        )

    def estimate(self, state: PreActionUtilityState) -> UtilityEstimate:
        if not isinstance(state, PreActionUtilityState):
            raise CausalRiskInvariantError("estimate requires PreActionUtilityState")
        raw = state.feature_values
        scaled = tuple(
            (value - median) / (iqr if iqr > 0.0 else 1.0)
            for value, median, iqr in zip(raw, self._median, self._iqr, strict=True)
        )
        ood_reasons = [
            f"feature_out_of_range:{name}"
            for name, value, minimum, maximum in zip(
                UTILITY_FEATURE_NAMES,
                raw,
                self._minimum,
                self._maximum,
                strict=True,
            )
            if value < minimum or value > maximum
        ]
        distances = sorted(
            math.dist(scaled, reference) for reference in self._reference_scaled
        )
        if distances[self._knn_k - 1] > self._knn_distance_threshold:
            ood_reasons.append("knn_distance_exceeded")

        utility_predictions = tuple(
            tree.predict(scaled) for tree in self._utility_trees
        )
        tau_hat = math.fsum(utility_predictions) / len(utility_predictions)
        tau_lcb = (
            _quantile(utility_predictions, self._lcb_quantile)
            - self._conformal_margin
        )
        risk_predictions = tuple(tree.predict(scaled) for tree in self._risk_trees)
        bootstrap_risk_ucb = _quantile(risk_predictions, self._risk_quantile)
        cp_risk_ucb = self._clopper_pearson_tree.predict(scaled)
        return UtilityEstimate(
            tau_hat=tau_hat,
            tau_lcb=tau_lcb,
            catastrophic_risk_ucb=max(bootstrap_risk_ucb, cp_risk_ucb),
            in_distribution=not ood_reasons,
            ood_reasons=tuple(ood_reasons),
            model_hash=self.model_sha256,
            feature_hash=state.feature_sha256,
        )


@dataclass(frozen=True)
class SafeReleaseDecision:
    """One-action release decision; every rejected gate selects baseline."""

    released: bool
    selected_action: str
    reason: str
    estimate: UtilityEstimate | None = None

    def __post_init__(self) -> None:
        expected = "post_retirement_precision_reanchor" if self.released else "baseline"
        if self.selected_action != expected:
            raise CausalRiskInvariantError(
                "selected action is inconsistent with release decision"
            )
        if not self.reason:
            raise CausalRiskInvariantError("safe release reason is required")


def _abstain(
    reason: str,
    estimate: UtilityEstimate | None = None,
) -> SafeReleaseDecision:
    return SafeReleaseDecision(
        released=False,
        selected_action="baseline",
        reason=reason,
        estimate=estimate,
    )


def decide_safe_release(
    *,
    candidate_feasible: bool,
    component_unlocked: bool,
    release_already_consumed: bool,
    state: PreActionUtilityState | None,
    model_bundle: CausalRiskModelBundle | None,
) -> SafeReleaseDecision:
    """Release the first precision action only when every hard gate passes."""

    if candidate_feasible is not True:
        return _abstain("abstain_candidate_infeasible")
    if component_unlocked is not True:
        return _abstain("abstain_component_locked")
    if release_already_consumed is not False:
        return _abstain("abstain_release_already_consumed")
    if not isinstance(state, PreActionUtilityState):
        return _abstain("abstain_pre_action_state_missing")
    if not isinstance(model_bundle, CausalRiskModelBundle):
        return _abstain("abstain_model_unavailable")

    estimate = model_bundle.estimate(state)
    if not estimate.in_distribution:
        return _abstain("abstain_out_of_distribution", estimate)
    if estimate.tau_lcb <= 0.0:
        return _abstain("abstain_causal_lcb_not_positive", estimate)
    if estimate.catastrophic_risk_ucb > MAX_CATASTROPHIC_RISK:
        return _abstain("abstain_catastrophic_risk_above_limit", estimate)
    return SafeReleaseDecision(
        released=True,
        selected_action="post_retirement_precision_reanchor",
        reason="causal_risk_gate_passed",
        estimate=estimate,
    )
