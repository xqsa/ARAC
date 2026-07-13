"""Compatibility exports for the moved backend execution contracts."""

from .execution.backend import (
    BackendAdapter,
    BackendSemanticsDiff,
    NullBackendAdapter,
    ToyBackendAdapter,
)

__all__ = [
    "BackendAdapter",
    "BackendSemanticsDiff",
    "NullBackendAdapter",
    "ToyBackendAdapter",
]
