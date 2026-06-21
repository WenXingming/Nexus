"""Model contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from src.core.contracts import Message


class Model(Protocol):
    """A model that can complete from chat messages."""

    def complete(self, messages: list[Message]) -> str:
        ...

    def stream(self, messages: list[Message]) -> Iterable[str]:
        ...
