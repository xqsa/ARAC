"""Metadata for deterministic ARAC action executors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    """Describe an action surface without embedding selection logic."""

    name: str
    semantic_surface: str
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.semantic_surface:
            raise ValueError("ActionSpec fields must be non-empty strings")
        if not self.parameter_names or any(
            not isinstance(name, str) or not name for name in self.parameter_names
        ):
            raise ValueError("ActionSpec parameter names must be non-empty strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("ActionSpec parameter names must be unique")
