from src.core.contracts import Message
from src.interfaces.cli import main, run_once, run_repl, run_repl_step
from src.memory.in_memory_session_store import InMemorySessionStore
from src.model.fake_model import FakeModel
from src.runtime.agent_runtime import AgentRuntime


def test_run_once_returns_agent_output() -> None:
    assert run_once("hi") == "Echo: hi"


def test_run_repl_step_creates_session() -> None:
    runtime = AgentRuntime(
        model=FakeModel(),
        session_store=InMemorySessionStore(),
    )

    session_id, output = run_repl_step(runtime, session_id=None, text="hi")

    assert session_id == "s1"
    assert output == "Echo: hi"


def test_run_repl_step_reuses_session() -> None:
    store = InMemorySessionStore()
    runtime = AgentRuntime(model=FakeModel(), session_store=store)

    session_id, _ = run_repl_step(runtime, session_id=None, text="first")
    next_session_id, output = run_repl_step(runtime, session_id=session_id, text="second")

    assert next_session_id == session_id
    assert output == "Echo: second"
    assert store.load(session_id) == [
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

    assert outputs == ["Echo: hi", "Echo: second"]


def test_main_prints_output_for_single_word(capsys) -> None:
    exit_code = main(["hi"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Echo: hi\n"


def test_main_joins_multiple_words(capsys) -> None:
    exit_code = main(["hello", "world"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Echo: hello world\n"


def test_main_prints_usage_without_message(capsys) -> None:
    exit_code = main([])

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
    assert outputs == ["Echo: hi"]
