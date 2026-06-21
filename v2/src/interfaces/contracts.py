"""Contracts for user-facing interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplStepResult:
    session_id: str
    output: str


class SessionNotFoundError(Exception):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")
