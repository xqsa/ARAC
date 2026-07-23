"""Typed dispatch for deterministic runtime action executors."""

from __future__ import annotations

from typing import overload

from arac.actions.gcb import (
    GcbAction,
    GcbExecutionContext,
    GcbExecutionResult,
    execute_gcb_action,
)


class UnsupportedRuntimeActionError(LookupError):
    """Raised when no explicit executor exists for an action type."""


class RuntimeActionDispatcher:
    """Route a frozen action instance without selecting or rewriting it."""

    @overload
    def execute(
        self,
        action: GcbAction,
        context: GcbExecutionContext,
    ) -> GcbExecutionResult: ...

    def execute(self, action: object, context: object) -> object:
        if type(action) is GcbAction:
            if not isinstance(context, GcbExecutionContext):
                raise TypeError(
                    "GcbAction requires "
                    "GcbExecutionContext"
                )
            return execute_gcb_action(action, context)
        raise UnsupportedRuntimeActionError(
            f"no runtime executor registered for {type(action).__name__}"
        )


DEFAULT_RUNTIME_ACTION_DISPATCHER = RuntimeActionDispatcher()


__all__ = [
    "DEFAULT_RUNTIME_ACTION_DISPATCHER",
    "RuntimeActionDispatcher",
    "UnsupportedRuntimeActionError",
]
