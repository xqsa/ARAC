"""Stable backend execution contracts exposed by ARAC."""

from .backend import (
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
