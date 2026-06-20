"""Minimal CLI-facing helpers."""

from __future__ import annotations

import sys

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest
from src.runtime.agent_runtime import AgentRuntime


def run_once(text: str) -> str:
    runtime = create_runtime()
    result = runtime.run(AgentRequest(input=text))
    return result.output


def run_repl_step(
    runtime: AgentRuntime,
    session_id: str | None,
    text: str,
) -> tuple[str, str]:
    result = runtime.run(AgentRequest(input=text, session_id=session_id))
    return result.session_id, result.output


def run_repl(input_func, output_func) -> None:
    runtime = create_runtime()
    session_id: str | None = None

    while True:
        text = input_func("> ")
        if text == "/exit":
            return

        session_id, output = run_repl_step(runtime, session_id, text)
        output_func(output)


def main(argv: list[str], input_func=input, output_func=print) -> int:
    if argv == ["--repl"]:
        run_repl(input_func=input_func, output_func=output_func)
        return 0

    if not argv:
        output_func("Usage: python -m src.interfaces.cli <message>")
        return 1

    text = " ".join(argv)
    output_func(run_once(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
