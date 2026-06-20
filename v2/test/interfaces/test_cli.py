from src.interfaces.cli import run_once


def test_run_once_returns_agent_output() -> None:
    assert run_once("hi") == "Echo: hi"
