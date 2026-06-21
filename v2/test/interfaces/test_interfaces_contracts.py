from src.interfaces.contracts import SessionNotFoundError


def test_session_not_found_error_keeps_session_id() -> None:
    error = SessionNotFoundError("s1")

    assert error.session_id == "s1"
    assert str(error) == "Session not found: s1"
