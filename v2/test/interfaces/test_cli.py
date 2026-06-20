from src.interfaces.cli import main, run_once


def test_run_once_returns_agent_output() -> None:
    assert run_once("hi") == "Echo: hi"


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
