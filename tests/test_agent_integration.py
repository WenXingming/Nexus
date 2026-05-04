"""Agent 模块集成测试。

验证重构后的 AgentGateway 和 create_gateway 工厂函数是否正常工作。
"""

from unittest.mock import Mock

import pytest

from src.agent import create_gateway
from src.core_contracts.client_contracts import LlmResult
from src.core_contracts.context_contracts import BudgetConfig, ContextPolicy
from src.core_contracts.model_contracts import TokenUsage
from src.core_contracts.session_contracts import SessionState


@pytest.fixture
def mock_dependencies():
    """创建所有 mock 依赖。"""
    client = Mock()
    context_gateway = Mock()
    session_gateway = Mock()
    tools_gateway = Mock()
    interaction_gateway = Mock()

    return {
        'client': client,
        'context_gateway': context_gateway,
        'session_gateway': session_gateway,
        'tools_gateway': tools_gateway,
        'interaction_gateway': interaction_gateway,
    }


def test_create_gateway(mock_dependencies):
    """测试 create_gateway 工厂函数。"""
    mock_dependencies['interaction_gateway'].register_slash_commands = Mock()

    gateway = create_gateway(
        client=mock_dependencies['client'],
        context_gateway=mock_dependencies['context_gateway'],
        session_gateway=mock_dependencies['session_gateway'],
        tools_gateway=mock_dependencies['tools_gateway'],
        interaction_gateway=mock_dependencies['interaction_gateway'],
        budget_config=BudgetConfig(),
        context_policy=ContextPolicy(),
        budget_guard=None,
        tools=[],
    )

    assert gateway is not None
    assert gateway.client == mock_dependencies['client']
    assert gateway.executor is not None
    mock_dependencies['interaction_gateway'].register_slash_commands.assert_called_once()


def test_agent_gateway_run_simple_response(mock_dependencies):
    """测试 AgentGateway 运行简单响应（无工具调用）。"""
    # 设置 mock 行为
    mock_pre_model = Mock()
    mock_pre_model.pre_model_stop = None
    mock_pre_model.events = ()
    mock_dependencies['context_gateway'].run_pre_model_cycle.return_value = mock_pre_model

    mock_result = LlmResult(
        content="Hello!",
        model="test-model",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        tool_calls=None,
    )
    mock_dependencies['client'].chat.return_value = mock_result
    mock_dependencies['interaction_gateway'].register_slash_commands = Mock()

    # 创建 gateway
    gateway = create_gateway(
        client=mock_dependencies['client'],
        context_gateway=mock_dependencies['context_gateway'],
        session_gateway=mock_dependencies['session_gateway'],
        tools_gateway=mock_dependencies['tools_gateway'],
        interaction_gateway=mock_dependencies['interaction_gateway'],
        budget_config=BudgetConfig(),
        context_policy=ContextPolicy(),
        budget_guard=None,
        tools=[],
    )

    # 执行
    state = SessionState(session_id="test-session")
    result_state = gateway.run(state)

    # 验证
    assert result_state is not None
    mock_dependencies['session_gateway'].append_assistant.assert_called_once()


def test_agent_gateway_run_with_pre_model_stop(mock_dependencies):
    """测试 AgentGateway 在 pre-model 阶段停止。"""
    # 设置 mock 行为
    mock_pre_model = Mock()
    mock_pre_model.pre_model_stop = "budget_exceeded"
    mock_pre_model.events = ()
    mock_dependencies['context_gateway'].run_pre_model_cycle.return_value = mock_pre_model
    mock_dependencies['interaction_gateway'].register_slash_commands = Mock()

    # 创建 gateway
    gateway = create_gateway(
        client=mock_dependencies['client'],
        context_gateway=mock_dependencies['context_gateway'],
        session_gateway=mock_dependencies['session_gateway'],
        tools_gateway=mock_dependencies['tools_gateway'],
        interaction_gateway=mock_dependencies['interaction_gateway'],
        budget_config=BudgetConfig(),
        context_policy=ContextPolicy(),
        budget_guard=None,
        tools=[],
    )

    # 执行
    state = SessionState(session_id="test-session")
    result_state = gateway.run(state)

    # 验证
    assert result_state is not None
    mock_dependencies['client'].chat.assert_not_called()
