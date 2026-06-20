"""Minimal CLI-facing helpers."""

from __future__ import annotations

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest


def run_once(text: str) -> str:
    runtime = create_runtime()
    result = runtime.run(AgentRequest(input=text))
    return result.output
