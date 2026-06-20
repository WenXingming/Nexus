"""Model contracts."""

from __future__ import annotations

from typing import Protocol

from src.core.contracts import Message


class Model(Protocol):
    """A model that can complete from chat messages."""

    def complete(self, messages: list[Message]) -> str:
        ...
