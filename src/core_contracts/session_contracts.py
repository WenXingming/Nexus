"""
Session 模块跨边界契约定义。

本文件定义 session 模块对外暴露的全部 DTO 与运行态对象。
外部调用方只能依赖这些纯数据结构与 SessionGateway 交互，
不得感知任何内部存储或恢复实现细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message, TokenUsage

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonDict = dict[str, JsonValue]


@dataclass(frozen=True)
class SessionSnapshot:
    """会话持久化快照。

    该对象表示可以直接落盘和恢复的稳定视图。
    """

    session_id: str
    model_config: ModelConfig
    messages: tuple[Message, ...]
    transcript: tuple[Message, ...] = ()
    last_response: str = ""
    turns: int = 0
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """将快照对象转换为可 JSON 序列化的字典。

        Args:
            None
        Returns:
            JsonDict: 可直接落盘的 JSON 载荷。
        Raises:
            ValueError: metadata 中包含不可序列化值时抛出。
        """
        return {
            "session_id": self.session_id,
            "model_config": self._serialize_model_config(self.model_config),
            "messages": [self._serialize_message(message) for message in self.messages],
            "transcript": [self._serialize_message(message) for message in self.transcript],
            "last_response": self.last_response,
            "turns": self.turns,
            "usage": self._serialize_usage(self.usage),
            "metadata": self._serialize_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: JsonDict | dict[str, object]) -> "SessionSnapshot":
        """从 JSON 字典恢复快照对象。

        Args:
            payload (JsonDict | dict[str, object]): 从磁盘读取的原始字典。
        Returns:
            SessionSnapshot: 恢复后的快照对象。
        Raises:
            ValueError: 当关键字段缺失或格式错误时抛出。
        """
        if not isinstance(payload, dict):
            raise ValueError("会话快照顶层结构必须为 JSON 对象。")

        session_id = cls._require_non_empty_str(payload.get("session_id"), "session_id")
        model_config = cls._deserialize_model_config(payload.get("model_config"))
        messages = cls._deserialize_message_list(payload.get("messages"), "messages")
        transcript = cls._deserialize_message_list(payload.get("transcript", []), "transcript")
        last_response = cls._read_optional_str(payload.get("last_response"), "last_response")
        turns = cls._read_non_negative_int(payload.get("turns", 0), "turns")
        usage = cls._deserialize_usage(payload.get("usage", {}))
        metadata = cls._deserialize_metadata(payload.get("metadata", {}))
        return cls(
            session_id=session_id,
            model_config=model_config,
            messages=tuple(messages),
            transcript=tuple(transcript),
            last_response=last_response,
            turns=turns,
            usage=usage,
            metadata=metadata,
        )

    @staticmethod
    def _serialize_model_config(model_config: ModelConfig) -> JsonDict:
        """将 ModelConfig 转为字典。

        Args:
            model_config (ModelConfig): 模型配置对象。
        Returns:
            JsonDict: 序列化后的模型配置字典。
        Raises:
            None
        """
        return {
            "api_key": model_config.api_key,
            "base_url": model_config.base_url,
            "model_name": model_config.model_name,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
        }

    @staticmethod
    def _serialize_message(message: Message) -> JsonDict:
        """将 Message 转为字典。

        Args:
            message (Message): 待序列化的消息对象。
        Returns:
            JsonDict: 序列化后的消息字典。
        Raises:
            None
        """
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _serialize_usage(usage: TokenUsage) -> JsonDict:
        """将 TokenUsage 转为字典。

        Args:
            usage (TokenUsage): Token 统计对象。
        Returns:
            JsonDict: 序列化后的 Token 统计字典。
        Raises:
            None
        """
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    @classmethod
    def _serialize_metadata(cls, metadata: JsonDict) -> JsonDict:
        """校验并复制 metadata 字典。

        Args:
            metadata (JsonDict): 原始 metadata 字典。
        Returns:
            JsonDict: 校验后的 metadata 副本。
        Raises:
            ValueError: metadata 含非法值时抛出。
        """
        if not isinstance(metadata, dict):
            raise ValueError("metadata 必须为字典。")
        cls._validate_json_value(metadata, "metadata")
        return dict(metadata)

    @classmethod
    def _validate_json_value(cls, value: JsonValue | object, field_name: str) -> None:
        """递归校验值是否满足 JSON 兼容约束。

        Args:
            value (JsonValue | object): 待校验的值。
            field_name (str): 字段名。
        Returns:
            None
        Raises:
            ValueError: 值不满足 JSON 兼容约束时抛出。
        """
        if isinstance(value, (str, int, float, bool)) or value is None:
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                cls._validate_json_value(item, f"{field_name}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{field_name} 的键必须为字符串。")
                cls._validate_json_value(item, f"{field_name}.{key}")
            return
        raise ValueError(f"{field_name} 包含不可序列化值: {type(value).__name__}")

    @classmethod
    def _deserialize_model_config(cls, payload: object) -> ModelConfig:
        """从字典恢复 ModelConfig。

        Args:
            payload (object): 原始模型配置载荷。
        Returns:
            ModelConfig: 恢复后的模型配置对象。
        Raises:
            ValueError: 模型配置结构不合法时抛出。
        """
        if not isinstance(payload, dict):
            raise ValueError("model_config 必须为字典。")
        api_key = cls._require_non_empty_str(payload.get("api_key"), "model_config.api_key")
        base_url = cls._read_optional_str(payload.get("base_url"), "model_config.base_url") or None
        model_name = cls._require_non_empty_str(payload.get("model_name"), "model_config.model_name")
        temperature = cls._read_float(payload.get("temperature"), "model_config.temperature")
        max_tokens = cls._read_positive_int(payload.get("max_tokens"), "model_config.max_tokens")
        return ModelConfig(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def _deserialize_message_list(cls, payload: object, field_name: str) -> list[Message]:
        """从列表载荷恢复 Message 列表。

        Args:
            payload (object): 原始消息列表载荷。
            field_name (str): 字段名。
        Returns:
            list[Message]: 恢复后的消息对象列表。
        Raises:
            ValueError: 列表结构不合法时抛出。
        """
        if not isinstance(payload, list):
            raise ValueError(f"{field_name} 必须为列表。")
        result: list[Message] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"{field_name}[{index}] 必须为字典。")
            role = cls._require_non_empty_str(item.get("role"), f"{field_name}[{index}].role")
            content = cls._require_non_empty_str(item.get("content"), f"{field_name}[{index}].content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"{field_name}[{index}].role 非法: {role}")
            result.append(Message(role=role, content=content))
        return result

    @classmethod
    def _deserialize_usage(cls, payload: object) -> TokenUsage:
        """从字典恢复 TokenUsage。

        Args:
            payload (object): 原始 usage 载荷。
        Returns:
            TokenUsage: 恢复后的 Token 统计对象。
        Raises:
            ValueError: usage 结构不合法时抛出。
        """
        if not isinstance(payload, dict):
            raise ValueError("usage 必须为字典。")
        return TokenUsage(
            prompt_tokens=cls._read_non_negative_int(payload.get("prompt_tokens", 0), "usage.prompt_tokens"),
            completion_tokens=cls._read_non_negative_int(payload.get("completion_tokens", 0), "usage.completion_tokens"),
            total_tokens=cls._read_non_negative_int(payload.get("total_tokens", 0), "usage.total_tokens"),
        )

    @classmethod
    def _deserialize_metadata(cls, payload: object) -> JsonDict:
        """从字典恢复 metadata。

        Args:
            payload (object): 原始 metadata 载荷。
        Returns:
            JsonDict: 恢复后的 metadata 字典。
        Raises:
            ValueError: metadata 结构不合法时抛出。
        """
        if not isinstance(payload, dict):
            raise ValueError("metadata 必须为字典。")
        cls._validate_json_value(payload, "metadata")
        return dict(payload)

    @staticmethod
    def _require_non_empty_str(value: object, field_name: str) -> str:
        """读取并校验非空字符串。

        Args:
            value (object): 原始输入值。
            field_name (str): 字段名。
        Returns:
            str: 去除首尾空白后的字符串。
        Raises:
            ValueError: 值不是非空字符串时抛出。
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 必须为非空字符串。")
        return value.strip()

    @staticmethod
    def _read_optional_str(value: object, field_name: str) -> str:
        """读取可选字符串。

        Args:
            value (object): 原始输入值。
            field_name (str): 字段名。
        Returns:
            str: 规范化后的字符串；None 时返回空字符串。
        Raises:
            ValueError: 值不是字符串时抛出。
        """
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} 必须为字符串或 None。")
        return value

    @staticmethod
    def _read_non_negative_int(value: object, field_name: str) -> int:
        """读取非负整数。

        Args:
            value (object): 原始输入值。
            field_name (str): 字段名。
        Returns:
            int: 校验通过的非负整数。
        Raises:
            ValueError: 值不是非负整数时抛出。
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} 必须为非负整数。")
        return value

    @classmethod
    def _read_positive_int(cls, value: object, field_name: str) -> int:
        """读取正整数。

        Args:
            value (object): 原始输入值。
            field_name (str): 字段名。
        Returns:
            int: 校验通过的正整数。
        Raises:
            ValueError: 值不是正整数时抛出。
        """
        number = cls._read_non_negative_int(value, field_name)
        if number < 1:
            raise ValueError(f"{field_name} 必须为正整数。")
        return number

    @staticmethod
    def _read_float(value: object, field_name: str) -> float:
        """读取浮点数。

        Args:
            value (object): 原始输入值。
            field_name (str): 字段名。
        Returns:
            float: 校验通过的浮点值。
        Raises:
            ValueError: 值不是数字时抛出。
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} 必须为数字。")
        return float(value)


@dataclass
class SessionState:
    """代理运行时会话状态（纯数据结构）。"""

    session_id: str = ""
    messages: list[Message] = field(default_factory=list)
    transcript_entries: list[Message] = field(default_factory=list)