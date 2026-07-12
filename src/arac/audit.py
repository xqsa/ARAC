"""Compatibility exports for the moved claim audits."""

from .audits.claims import active_action_has_effect, claim_gate, find_forbidden_runtime_fields

__all__ = [
    "active_action_has_effect",
    "claim_gate",
    "find_forbidden_runtime_fields",
]
