"""Contracts for user-facing interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplStepResult:
    session_id: str
    output: str
