from uuid import UUID

from src.memory.session_id import new_session_id


def test_new_session_id_returns_uuid_string() -> None:
    session_id = new_session_id()

    assert str(UUID(session_id)) == session_id


def test_new_session_id_returns_unique_values() -> None:
    first = new_session_id()
    second = new_session_id()

    assert first != second
