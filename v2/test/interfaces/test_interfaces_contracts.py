from src.interfaces.contracts import ReplStepResult


def test_repl_step_result_keeps_session_and_output() -> None:
    result = ReplStepResult(session_id="s1", output="Echo: hi")

    assert result.session_id == "s1"
    assert result.output == "Echo: hi"
