from src.core.contracts import Message
from src.memory.in_memory_session_store import InMemorySessionStore


def test_create_returns_incrementing_session_ids() -> None:
    store = InMemorySessionStore()

    assert store.create() == "s1"
    assert store.create() == "s2"


def test_new_session_starts_empty() -> None:
    store = InMemorySessionStore()
    session_id = store.create()

    assert store.load(session_id) == []


def test_save_and_load_messages() -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    messages = [Message(role="user", content="hi")]

    store.save(session_id, messages)

    assert store.load(session_id) == messages


def test_load_returns_copy() -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    store.save(session_id, [Message(role="user", content="hi")])

    loaded = store.load(session_id)
    loaded.append(Message(role="assistant", content="changed"))

    assert store.load(session_id) == [Message(role="user", content="hi")]


def test_save_stores_copy() -> None:
    store = InMemorySessionStore()
    session_id = store.create()
    messages = [Message(role="user", content="hi")]

    store.save(session_id, messages)
    messages.append(Message(role="assistant", content="changed"))

    assert store.load(session_id) == [Message(role="user", content="hi")]
