"""Snipper 单元测试。

覆盖 snip 操作的全部分支：system 前缀保留、尾部窗口保留、
各角色消息的可/不可剪裁判断、tombstone 结构校验、token 统计正确性。
"""

from __future__ import annotations

import pytest

from src.context.snipper import Snipper, _TOMBSTONE_MARKER
from src.context.token_estimator import TokenEstimator
from src.core_contracts.model_contracts import Message


# =============================================================================
# 测试夹具
# =============================================================================


@pytest.fixture
def snipper() -> Snipper:
    """返回使用默认参数的 Snipper 实例。"""
    return Snipper(token_estimator=TokenEstimator())


def make_long_assistant(chars: int = 400) -> Message:
    """构造超过 long_assistant_threshold（默认 300）的 assistant 消息。"""
    return Message(role="assistant", content="x" * chars)


def make_short_assistant(chars: int = 50) -> Message:
    """构造不超过阈值的短 assistant 消息。"""
    return Message(role="assistant", content="x" * chars)


# =============================================================================
# 空列表 / 无可剪裁消息
# =============================================================================


class TestSnipNoOp:
    def test_empty_list(self, snipper: Snipper) -> None:
        """空列表直接返回 snipped_count=0。"""
        msgs: list[Message] = []
        result = snipper.snip(msgs)
        assert result.snipped_count == 0
        assert result.tokens_removed == 0

    def test_only_user_messages_not_snipped(self, snipper: Snipper) -> None:
        """user 角色消息不可剪裁。"""
        msgs = [Message(role="user", content="hello")] * 5
        original = list(msgs)
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 0
        assert msgs == original

    def test_short_assistant_not_snipped(self, snipper: Snipper) -> None:
        """短 assistant 消息（未超阈值）不被剪裁。"""
        msgs = [make_short_assistant()] * 4
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 0


# =============================================================================
# tool 消息剪裁
# =============================================================================


class TestSnipToolMessage:
    def test_tool_message_snipped(self, snipper: Snipper) -> None:
        """tool 消息被 tombstone 化。"""
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="q"),
            Message(role="tool", content="long result", tool_call_id="c1", name="my_tool"),
        ]
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 1

    def test_tool_tombstone_preserves_role_and_fields(self, snipper: Snipper) -> None:
        """tool tombstone 保留 role、tool_call_id 和 name。"""
        msgs = [
            Message(role="user", content="q"),
            Message(role="tool", content="result", tool_call_id="c1", name="fn"),
        ]
        snipper.snip(msgs, preserve_messages=0)
        tombstone = msgs[1]
        assert tombstone.role == "tool"
        assert tombstone.tool_call_id == "c1"
        assert tombstone.name == "fn"
        assert _TOMBSTONE_MARKER in (tombstone.content or "")


# =============================================================================
# assistant 消息剪裁
# =============================================================================


class TestSnipAssistantMessage:
    def test_long_assistant_snipped(self, snipper: Snipper) -> None:
        """超过阈值的 assistant 消息被 tombstone 化。"""
        msgs = [Message(role="user", content="q"), make_long_assistant()]
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 1

    def test_assistant_with_tool_calls_snipped(self, snipper: Snipper) -> None:
        """携带 tool_calls 的 assistant 消息无论长短均被剪裁。"""
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="ok", tool_calls=tool_calls),
        ]
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 1

    def test_assistant_tool_calls_tombstone_keeps_tool_calls(self, snipper: Snipper) -> None:
        """携带 tool_calls 的 assistant tombstone 保留 tool_calls 字段。"""
        tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="calling", tool_calls=tool_calls),
        ]
        snipper.snip(msgs, preserve_messages=0)
        tombstone = msgs[1]
        assert tombstone.tool_calls == tool_calls


# =============================================================================
# system 前缀保留
# =============================================================================


class TestSnipSystemPrefix:
    def test_system_messages_never_snipped(self, snipper: Snipper) -> None:
        """头部 system 消息不参与剪裁。"""
        sys_msg = Message(role="system", content="instructions")
        msgs = [sys_msg, make_long_assistant(), Message(role="user", content="q")]
        snipper.snip(msgs, preserve_messages=0)
        assert msgs[0].role == "system"
        assert msgs[0].content == "instructions"


# =============================================================================
# 尾部保留窗口
# =============================================================================


class TestSnipPreserveTail:
    def test_preserve_messages_protects_tail(self, snipper: Snipper) -> None:
        """preserve_messages=2 时最后 2 条消息受保护，前面的仍被剪裁。"""
        msgs = [
            Message(role="user", content="q"),
            make_long_assistant(),
            Message(role="user", content="q2"),
            make_long_assistant(),  # 尾部保留窗口内（倒数第 2）
            make_long_assistant(),  # 尾部保留窗口内（倒数第 1）
        ]
        result = snipper.snip(msgs, preserve_messages=2)
        # prefix=0, tail=2, upper=5-2=3
        # range(0,3): idx 0 user、idx 1 long_assistant(✓ 剪裁)、idx 2 user
        assert result.snipped_count == 1
        assert msgs[-1].role == "assistant"  # 尾部 assistant 内容未被 tombstone
        assert _TOMBSTONE_MARKER not in (msgs[-1].content or "")

    def test_preserve_zero_snips_all_eligible(self, snipper: Snipper) -> None:
        """preserve_messages=0 时所有符合条件的消息均被剪裁。"""
        msgs = [make_long_assistant(), make_long_assistant()]
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 2


# =============================================================================
# tombstone 不被重复剪裁
# =============================================================================


class TestSnipTombstoneNotReSnipped:
    def test_already_tombstoned_message_skipped(self, snipper: Snipper) -> None:
        """已是 tombstone 的消息不会被再次剪裁。"""
        tombstone_content = f"{_TOMBSTONE_MARKER}assistant was snipped.\nPreview: ...\n</system-reminder>"
        msgs = [Message(role="assistant", content=tombstone_content)]
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 0


# =============================================================================
# token 统计
# =============================================================================


class TestSnipTokenStats:
    def test_tokens_removed_positive_when_snipped(self, snipper: Snipper) -> None:
        """成功剪裁后 tokens_removed 应为正数（tombstone < 原消息）。"""
        msgs = [Message(role="user", content="q"), make_long_assistant(1000)]
        result = snipper.snip(msgs, preserve_messages=0)
        assert result.snipped_count == 1
        assert result.tokens_removed > 0
