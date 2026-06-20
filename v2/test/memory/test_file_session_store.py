import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.core.contracts import Message
from src.memory.file_session_store import FileSessionStore


@pytest.fixture
def local_tmp_path() -> Path:
    root = Path("v2/test/.tmp/file_session_store") / uuid4().hex
    root.mkdir(parents=True)
    return root


def test_init_creates_root_directory(local_tmp_path) -> None:
    root = local_tmp_path / "sessions"

    FileSessionStore(root=root)

    assert root.is_dir()


def test_create_returns_first_session_id(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)

    assert store.create() == "s1"


def test_create_writes_empty_session_file(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)

    session_id = store.create()

    assert (local_tmp_path / f"{session_id}.json").read_text(encoding="utf-8") == "[]"


def test_create_returns_incrementing_session_ids(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)

    assert store.create() == "s1"
    assert store.create() == "s2"


def test_create_writes_file_for_each_session(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)

    first = store.create()
    second = store.create()

    assert (local_tmp_path / f"{first}.json").is_file()
    assert (local_tmp_path / f"{second}.json").is_file()


def test_save_writes_messages_to_session_file(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)
    session_id = store.create()

    store.save(session_id, [
        Message(role="user", content="hi"),
        Message(role="assistant", content="Echo: hi"),
    ])

    raw = (local_tmp_path / f"{session_id}.json").read_text(encoding="utf-8")
    assert json.loads(raw) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Echo: hi"},
    ]


def test_save_overwrites_existing_messages(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)
    session_id = store.create()

    store.save(session_id, [Message(role="user", content="first")])
    store.save(session_id, [Message(role="user", content="second")])

    raw = (local_tmp_path / f"{session_id}.json").read_text(encoding="utf-8")
    assert json.loads(raw) == [
        {"role": "user", "content": "second"},
    ]


def test_save_preserves_non_ascii_content(local_tmp_path) -> None:
    store = FileSessionStore(root=local_tmp_path)
    session_id = store.create()

    store.save(session_id, [Message(role="user", content="你好")])

    raw = (local_tmp_path / f"{session_id}.json").read_text(encoding="utf-8")
    assert json.loads(raw) == [
        {"role": "user", "content": "你好"},
    ]
