from uuid import UUID

from src.core.contracts import Message
import src.interfaces.cli as cli
from src.interfaces.cli import (
    main,
    parse_session_args,
    run_once,
    run_repl,
    run_repl_step,
)
from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.fake_client import FakeClient
from src.runtime.agent_runtime import AgentRuntime


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


def test_main_prints_usage_without_message(capsys) -> None:
    exit_code = main([])

    assert exit_code == 1
    assert capsys.readouterr().out == "Usage: python -m src.interfaces.cli <message>\n"


def test_main_prints_usage_without_message_after_session(capsys) -> None:
    exit_code = main(["--session", "s1"])

    assert exit_code == 1
    assert capsys.readouterr().out == "Usage: python -m src.interfaces.cli <message>\n"


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
