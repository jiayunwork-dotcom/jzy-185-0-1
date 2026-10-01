"""Domain error types.

A :class:`ValidationError` carries a mapping ``fields`` whose keys are dotted
field paths (e.g. ``ingredients[3].price``) so API clients know exactly which
input field is invalid.
"""
from __future__ import annotations

from typing import Any, Mapping


class FeedOptError(Exception):
    """Base class for application errors."""


class NotFoundError(FeedOptError):
    def __init__(self, resource: str, identifier: Any):
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} not found: {identifier!r}")


class ValidationError(FeedOptError):
    def __init__(self, message: str, fields: Mapping[str, str] | None = None):
        super().__init__(message)
        self.message = message
        # dict: field path -> human readable reason
        self.fields: dict[str, str] = dict(fields or ())

    def add(self, field: str, reason: str) -> None:
        self.fields.setdefault(field, reason)


class InfeasibleError(FeedOptError):
    """Raised when a feed formula specification admits no feasible blend."""

    def __init__(self, conflicts: list[dict[str, Any]]):
        super().__init__("formula is infeasible")
        self.conflicts = conflicts
