"""Minimal CLI-facing helpers."""

from __future__ import annotations

import sys

from src.composition.runtime_factory import create_runtime
from src.core.contracts import AgentRequest
from src.interfaces.contracts import SessionNotFoundError
from src.memory.contracts import SessionFileFormatError


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


def run_once_stream(text: str, session_id: str | None, output_func) -> str:
    runtime = create_runtime()
    if session_id is not None and not runtime.has_session(session_id):
        raise SessionNotFoundError(session_id)

    final_session_id = session_id
    for chunk in runtime.stream(AgentRequest(input=text, session_id=session_id)):
        final_session_id = chunk.session_id
        write_stream_chunk(chunk.text, output_func)
    return final_session_id or ""


def run_repl(input_func, output_func, session_id: str | None = None) -> None:
    runtime = create_runtime()
    if session_id is not None:
        if not runtime.has_session(session_id):
            raise SessionNotFoundError(session_id)

    while True:
        text = input_func("> ")
        if text == "/exit":
            if session_id is not None:
                output_func(f"session: {session_id}")
            return

        chunks: list[str] = []
        for chunk in runtime.stream(AgentRequest(input=text, session_id=session_id)):
            session_id = chunk.session_id
            if output_func is print:
                write_stream_chunk(chunk.text, output_func)
            else:
                chunks.append(chunk.text)
        if output_func is print:
            print()
        else:
            output_func("".join(chunks))


def print_cli_error(error: Exception, output_func) -> int:
    output_func(str(error))
    return 1


def write_stream_chunk(text: str, output_func) -> None:
    if output_func is print:
        print(text, end="", flush=True)
    else:
        output_func(text)


def main(argv: list[str], input_func=input, output_func=print) -> int:
    if argv and argv[0] == "--stream":
        session_id, message_args = parse_session_args(argv[1:])
        if not message_args:
            output_func("Usage: python -m src.interfaces.cli <message>")
            return 1

        text = " ".join(message_args)
        try:
            run_once_stream(
                text,
                session_id=session_id,
                output_func=output_func,
            )
        except (SessionNotFoundError, SessionFileFormatError) as error:
            return print_cli_error(error, output_func)
        if output_func is print:
            print()
        return 0

    if argv == ["--repl"]:
        try:
            run_repl(input_func=input_func, output_func=output_func)
        except SessionFileFormatError as error:
            return print_cli_error(error, output_func)
        return 0
    if len(argv) == 3 and argv[0] == "--repl" and argv[1] == "--session":
        try:
            run_repl(
                input_func=input_func,
                output_func=output_func,
                session_id=argv[2],
            )
        except (SessionNotFoundError, SessionFileFormatError) as error:
            return print_cli_error(error, output_func)
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
    except (SessionNotFoundError, SessionFileFormatError) as error:
        return print_cli_error(error, output_func)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
