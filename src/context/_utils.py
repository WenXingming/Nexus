from __future__ import annotations

from src.core_contracts.model_contracts import Message


def count_system_prefix(messages: list[Message]) -> int:
    """返回头部连续 system 消息的数量。"""
    count = 0
    for message in messages:
        if message.role == "system":
            count += 1
        else:
            break
    return count
