from uuid import UUID

import pytest
from pathlib import Path
from uuid import uuid4

from src.core.contracts import Message
import src.interfaces.cli as cli
from src.interfaces.cli import (
    main,
    parse_session_args,
    run_once,
    run_once_stream,
    run_repl,
    run_repl_step,
)
from src.interfaces.contracts import SessionNotFoundError
from src.memory.contracts import SessionFileFormatError
from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.fake_client import FakeClient
from src.runtime.agent_runtime import AgentRuntime


@pytest.fixture(autouse=True)
def isolate_default_session_root(monkeypatch) -> None:
    root = Path("v2/test/.tmp/cli") / uuid4().hex
    monkeypatch.setenv("NEXUS_SESSION_ROOT", str(root))


def assert_uuid_string(value: str) -> None:
    assert str(UUID(value)) == value


def test_run_once_returns_agent_output() -> None:
    assert run_once("hi") == "Echo: hi"


def test_parse_session_args_without_session() -> None:
    session_id, message_args = parse_session_args(["hi"])

    assert session_id is None
    assert message_args == ["hi"]


def test_parse_session_args_with_session() -> None:
    session_id, message_args = parse_session_args(["--session", "s1", "hi"])

    assert session_id == "s1"
    assert message_args == ["hi"]


def test_run_once_accepts_session_id(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)

    assert run_once("second", session_id=session_id) == "Echo: second"
    assert store.load(session_id) == [
        Message(role="user", content="second"),
        Message(role="assistant", content="Echo: second"),
    ]


def test_run_once_rejects_missing_session(monkeypatch) -> None:
    runtime = AgentRuntime(
        model=FakeClient(),
        session_store=InMemorySessionStore(),
    )
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)

    with pytest.raises(SessionNotFoundError, match="Session not found: missing-session"):
        run_once("hi", session_id="missing-session")


def test_run_once_stream_outputs_chunks_and_returns_session_id(monkeypatch) -> None:
    store = InMemorySessionStore()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)
    outputs: list[str] = []

    session_id = run_once_stream("hi", session_id=None, output_func=outputs.append)

    assert_uuid_string(session_id)
    assert outputs == ["Echo: ", "hi"]
    assert store.load(session_id) == [
        Message(role="user", content="hi"),
        Message(role="assistant", content="Echo: hi"),
    ]


def test_run_once_stream_accepts_session_id(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)
    outputs: list[str] = []

    result_session_id = run_once_stream(
        "hi",
        session_id=session_id,
        output_func=outputs.append,
    )

    assert result_session_id == session_id
    assert outputs == ["Echo: ", "hi"]


def test_run_once_stream_rejects_missing_session(monkeypatch) -> None:
    runtime = AgentRuntime(
        model=FakeClient(),
        session_store=InMemorySessionStore(),
    )
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)

    with pytest.raises(SessionNotFoundError, match="Session not found: missing-session"):
        run_once_stream("hi", session_id="missing-session", output_func=lambda text: None)


def test_run_repl_step_creates_session() -> None:
    runtime = AgentRuntime(
        model=FakeClient(),
        session_store=InMemorySessionStore(),
    )

    result = run_repl_step(runtime, session_id=None, text="hi")

    assert_uuid_string(result.session_id)
    assert result.output == "Echo: hi"


def test_run_repl_step_reuses_session() -> None:
    store = InMemorySessionStore()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)

    first_result = run_repl_step(runtime, session_id=None, text="first")
    second_result = run_repl_step(
        runtime,
        session_id=first_result.session_id,
        text="second",
    )

    assert second_result.session_id == first_result.session_id
    assert second_result.output == "Echo: second"
    assert store.load(first_result.session_id) == [
        Message(role="user", content="first"),
        Message(role="assistant", content="Echo: first"),
        Message(role="user", content="second"),
        Message(role="assistant", content="Echo: second"),
    ]


def test_run_repl_outputs_until_exit() -> None:
    inputs = iter(["hi", "second", "/exit"])
    outputs: list[str] = []

    run_repl(
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert outputs[0].startswith("session: ")
    assert_uuid_string(outputs[0].removeprefix("session: "))
    assert outputs[1:] == ["Echo: hi", "Echo: second"]


def test_run_repl_reuses_existing_session(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    store.save(session_id, [Message(role="user", content="first")])
    runtime = AgentRuntime(model=FakeClient(), session_store=store)
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)
    inputs = iter(["second", "/exit"])
    outputs: list[str] = []

    run_repl(
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        session_id=session_id,
    )

    assert outputs == [f"session: {session_id}", "Echo: second"]
    assert store.load(session_id) == [
        Message(role="user", content="first"),
        Message(role="user", content="second"),
        Message(role="assistant", content="Echo: second"),
    ]


def test_run_repl_rejects_missing_session(monkeypatch) -> None:
    runtime = AgentRuntime(
        model=FakeClient(),
        session_store=InMemorySessionStore(),
    )
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)

    with pytest.raises(SessionNotFoundError, match="Session not found: missing-session"):
        run_repl(
            input_func=lambda prompt: "/exit",
            output_func=lambda text: None,
            session_id="missing-session",
        )


def test_main_prints_output_for_single_word(capsys) -> None:
    exit_code = main(["hi"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Echo: hi\n"


def test_main_joins_multiple_words(capsys) -> None:
    exit_code = main(["hello", "world"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Echo: hello world\n"


def test_main_passes_session_id(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_once",
        lambda text, session_id=None: f"{session_id}: {text}",
    )

    exit_code = main(["--session", "s1", "hello"])

    assert exit_code == 0
    assert capsys.readouterr().out == "s1: hello\n"


def test_main_stream_outputs_chunks(monkeypatch) -> None:
    def stream(text, session_id=None, output_func=print) -> str:
        output_func("a")
        output_func("b")
        return "s1"

    monkeypatch.setattr(cli, "run_once_stream", stream)
    outputs: list[str] = []

    exit_code = main(["--stream", "hi"], output_func=outputs.append)

    assert exit_code == 0
    assert outputs == ["a", "b"]


def test_main_stream_passes_session_id(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def stream(text, session_id=None, output_func=print) -> str:
        calls.append((text, session_id))
        return session_id or "s1"

    monkeypatch.setattr(cli, "run_once_stream", stream)

    exit_code = main(["--stream", "--session", "s1", "hi"])

    assert exit_code == 0
    assert calls == [("hi", "s1")]


def test_main_stream_prints_usage_without_message(capsys) -> None:
    exit_code = main(["--stream"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Usage: python -m src.interfaces.cli <message>\n"


def test_main_stream_prints_error_for_missing_session(capsys) -> None:
    exit_code = main(["--stream", "--session", "missing-session", "hi"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Session not found: missing-session\n"


def test_main_prints_usage_without_message(capsys) -> None:
    exit_code = main([])

    assert exit_code == 1
    assert capsys.readouterr().out == "Usage: python -m src.interfaces.cli <message>\n"


def test_main_prints_usage_without_message_after_session(capsys) -> None:
    exit_code = main(["--session", "s1"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Usage: python -m src.interfaces.cli <message>\n"


def test_main_prints_error_for_missing_session(capsys) -> None:
    exit_code = main(["--session", "missing-session", "hi"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Session not found: missing-session\n"


def test_main_prints_error_for_invalid_session_file(monkeypatch, capsys) -> None:
    def raise_invalid_session_file(text, session_id=None) -> str:
        raise SessionFileFormatError("s1")

    monkeypatch.setattr(cli, "run_once", raise_invalid_session_file)

    exit_code = main(["--session", "s1", "hi"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Invalid session file: s1\n"


def test_main_repl_accepts_existing_session(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    runtime = AgentRuntime(model=FakeClient(), session_store=store)
    monkeypatch.setattr(cli, "create_runtime", lambda: runtime)
    inputs = iter(["hi", "/exit"])
    outputs: list[str] = []

    exit_code = main(
        ["--repl", "--session", session_id],
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert exit_code == 0
    assert outputs == [f"session: {session_id}", "Echo: hi"]


def test_main_repl_prints_error_for_missing_session(capsys) -> None:
    exit_code = main(["--repl", "--session", "missing-session"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Session not found: missing-session\n"


def test_main_repl_prints_error_for_invalid_session_file(monkeypatch) -> None:
    def raise_invalid_session_file(input_func, output_func, session_id=None) -> None:
        raise SessionFileFormatError("s1")

    monkeypatch.setattr(cli, "run_repl", raise_invalid_session_file)
    outputs: list[str] = []

    exit_code = main(
        ["--repl", "--session", "s1"],
        output_func=outputs.append,
    )

    assert exit_code == 1
    assert outputs == ["Invalid session file: s1"]


def test_main_repl_runs_until_exit() -> None:
    inputs = iter(["hi", "/exit"])
    outputs: list[str] = []

    exit_code = main(
        ["--repl"],
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert exit_code == 0
    assert outputs[0].startswith("session: ")
    assert_uuid_string(outputs[0].removeprefix("session: "))
    assert outputs[1:] == ["Echo: hi"]
