"""Application 类的测试。

覆盖：
1. Application 类：依赖构造、斜杠命令处理、Agent 循环。
2. Context 具体类型。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.main as main_module
from src.core_contracts.context_contracts import ContextRunState, PreModelBudgetGuard
from src.core_contracts.model_config import ModelConfig
from src.core_contracts.model_contracts import Message
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.tools_contracts import McpToolConfig


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(api_key="sk-test")


@pytest.fixture
def state() -> SessionState:
    return SessionState(session_id="session-1")


# ============================================================
# Application 依赖构造测试
# ============================================================

class TestApplicationDependencies:
    def test_create_dependencies_builds_all_members(self) -> None:
        config = ModelConfig(api_key="sk-test", max_tokens=2048)
        llm_client = MagicMock()
        context_gateway = MagicMock()
        session_gateway = MagicMock()
        tools_gateway = MagicMock()
        interaction_gateway = MagicMock()
        agent_gateway = MagicMock()
        conversation_orchestrator = MagicMock()
        rag_gateway = MagicMock()

        app = main_module.Application()
        app._config = config

        with patch("src.main.create_llm_gateway", return_value=llm_client), \
            patch("src.main.create_context_gateway", return_value=context_gateway), \
            patch("src.main.create_session_gateway", return_value=session_gateway), \
            patch("src.main.create_tools_gateway", return_value=tools_gateway), \
            patch("src.main.create_rag_gateway", return_value=rag_gateway), \
            patch("src.main.create_interaction_gateway", return_value=interaction_gateway), \
            patch("src.main.create_agent_gateway", return_value=agent_gateway), \
            patch("src.main.create_conversation_orchestrator", return_value=conversation_orchestrator), \
            patch("src.main.RagModelConfig.from_env", return_value=MagicMock()):
            app._create_dependencies()

        assert app._client is llm_client
        assert app._context_gateway is context_gateway
        assert app._session_gateway is session_gateway
        assert app._tools_gateway is tools_gateway
        assert app._rag_gateway is rag_gateway
        assert app._interaction_gateway is interaction_gateway
        assert app._agent_gateway is agent_gateway
        assert app._conversation_orchestrator is conversation_orchestrator
        assert app._budget_config.output_reserve_tokens == 2048
        assert app._budget_config.soft_buffer_tokens == 13_000
        assert app._context_policy.compact_preserve_messages == 4
        assert app._context_policy.auto_compact_threshold_tokens == 24_000
        assert isinstance(app._budget_guard, PreModelBudgetGuard)
        assert app._mcp_config == McpToolConfig()


# ============================================================
# Application 运行测试
# ============================================================

class TestApplicationRun:
    def test_run_delegates_to_conversation_orchestrator(
        self, config: ModelConfig, state: SessionState
    ) -> None:
        app = main_module.Application()
        app._conversation_orchestrator = MagicMock()

        with patch.object(app, '_load_config'), \
            patch.object(app, '_create_dependencies'), \
            patch.object(app, '_init_session', return_value=state), \
            patch.object(app, '_render_exit'):
            result = app.run()

        app._conversation_orchestrator.run.assert_called_once_with(state)
        assert result == 0

    def test_init_session_creates_empty_state(self, state: SessionState) -> None:
        app = main_module.Application()
        app._workspace_config = MagicMock(root=Path("D:/WorkSpace/Nexus"))
        app._interaction_gateway = MagicMock()
        app._session_gateway = MagicMock()
        app._session_gateway.create_empty_state.return_value = state

        result = app._init_session()

        assert result is state
        app._session_gateway.create_empty_state.assert_called_once_with()
        app._session_gateway.create_state.assert_not_called()


# ============================================================
# Context 具体类型测试
# ============================================================

class TestContextConcreteTypes:
    def test_context_run_state_defaults(self) -> None:
        result = ContextRunState(session_messages=[Message(role="user", content="hi")])

        assert result.turn_index == 0
        assert result.model_call_count == 0
        assert result.usage_delta.total_tokens == 0
        assert result.token_budget_snapshot is None

    def test_hard_limit_budget_guard_blocks_only_hard_over(self) -> None:
        guard = PreModelBudgetGuard()
        soft_ok_snapshot = MagicMock(is_hard_over=False)
        hard_over_snapshot = MagicMock(is_hard_over=True)

        assert guard.check_pre_model(
            turns_offset=0,
            turns_this_run=0,
            model_call_count=0,
            snapshot=soft_ok_snapshot,
            usage_delta=MagicMock(),
        ) is None
        assert guard.check_pre_model(
            turns_offset=0,
            turns_this_run=0,
            model_call_count=0,
            snapshot=hard_over_snapshot,
            usage_delta=MagicMock(),
        ) == "hard_input_budget_exceeded"
