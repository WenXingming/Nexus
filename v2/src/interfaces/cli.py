"""Minimal CLI-facing helpers."""

from __future__ import annotations

import sys

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest
from src.interfaces.contracts import ReplStepResult, SessionNotFoundError
from src.runtime.agent_runtime import AgentRuntime


def parse_session_args(argv: list[str]) -> tuple[str | None, list[str]]:
    if len(argv) >= 2 and argv[0] == "--session":
        return argv[1], argv[2:]
    return None, argv


def run_once(text: str, session_id: str | None = None) -> str:
    runtime = create_runtime()
    if session_id is not None and not runtime.has_session(session_id):
        raise SessionNotFoundError(session_id)
    result = runtime.run(AgentRequest(input=text, session_id=session_id))
    return result.output


def run_repl_step(
    runtime: AgentRuntime,
    session_id: str | None,
    text: str,
) -> ReplStepResult:
    result = runtime.run(AgentRequest(input=text, session_id=session_id))
    return ReplStepResult(session_id=result.session_id, output=result.output)


def run_repl(input_func, output_func, session_id: str | None = None) -> None:
    runtime = create_runtime()
    if session_id is not None:
        if not runtime.has_session(session_id):
            raise SessionNotFoundError(session_id)
        output_func(f"session: {session_id}")

    while True:
        text = input_func("> ")
        if text == "/exit":
            return

        is_new_session = session_id is None
        result = run_repl_step(runtime, session_id, text)
        session_id = result.session_id
        if is_new_session:
            output_func(f"session: {session_id}")
        output_func(result.output)


def main(argv: list[str], input_func=input, output_func=print) -> int:
    if argv == ["--repl"]:
        run_repl(input_func=input_func, output_func=output_func)
        return 0
    if len(argv) == 3 and argv[0] == "--repl" and argv[1] == "--session":
        try:
            run_repl(
                input_func=input_func,
                output_func=output_func,
                session_id=argv[2],
            )
        except SessionNotFoundError as error:
            output_func(str(error))
            return 1
        return 0
    if argv and argv[0] == "--repl":
        output_func("Usage: python -m src.interfaces.cli <message>")
        return 1

    if not argv:
        output_func("Usage: python -m src.interfaces.cli <message>")
        return 1

    session_id, message_args = parse_session_args(argv)
    if not message_args:
        output_func("Usage: python -m src.interfaces.cli <message>")
        return 1

    text = " ".join(message_args)
    try:
        output_func(run_once(text, session_id=session_id))
    except SessionNotFoundError as error:
        output_func(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
