"""Typed dispatch for deterministic runtime action executors."""

from __future__ import annotations

from typing import overload

from arac.actions.full_space_sep_cma import (
    FullSpaceSepCmaAction,
    FullSpaceSepCmaExecutionContext,
    FullSpaceSepCmaExecutionResult,
    execute_full_space_sep_cma_action,
)


class UnsupportedRuntimeActionError(LookupError):
    """Raised when no explicit executor exists for an action type."""


class RuntimeActionDispatcher:
    """Route a frozen action instance without selecting or rewriting it."""

    @overload
    def execute(
        self,
        action: FullSpaceSepCmaAction,
        context: FullSpaceSepCmaExecutionContext,
    ) -> FullSpaceSepCmaExecutionResult: ...

    def execute(self, action: object, context: object) -> object:
        if type(action) is FullSpaceSepCmaAction:
            if not isinstance(context, FullSpaceSepCmaExecutionContext):
                raise TypeError(
                    "FullSpaceSepCmaAction requires "
                    "FullSpaceSepCmaExecutionContext"
                )
            return execute_full_space_sep_cma_action(action, context)
        raise UnsupportedRuntimeActionError(
            f"no runtime executor registered for {type(action).__name__}"
        )


DEFAULT_RUNTIME_ACTION_DISPATCHER = RuntimeActionDispatcher()


__all__ = [
    "DEFAULT_RUNTIME_ACTION_DISPATCHER",
    "RuntimeActionDispatcher",
    "UnsupportedRuntimeActionError",
]
