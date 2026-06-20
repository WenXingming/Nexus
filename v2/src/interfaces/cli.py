"""Minimal CLI-facing helpers."""

from __future__ import annotations

import sys

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest


def run_once(text: str) -> str:
    runtime = create_runtime()
    result = runtime.run(AgentRequest(input=text))
    return result.output


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python -m src.interfaces.cli <message>")
        return 1

    text = " ".join(argv)
    print(run_once(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
