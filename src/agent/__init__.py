"""应用层 Agent 入口。"""

from src.agent.agent_executor import AgentLoopExecutor
from src.agent.agent_gateway import AgentGateway
from src.agent.slash_commands import SlashCommandRegistry
from src.client import ClientGateway
from src.context import ContextGateway
from src.core_contracts.context_contracts import BudgetConfig, ContextPolicy, PreModelBudgetGuard
from src.interaction import InteractionGateway
from src.rag import RagGateway
from src.session import SessionGateway
from src.tools import ToolsGateway


def create_gateway(
    *,
    client: ClientGateway,
    context_gateway: ContextGateway,
    session_gateway: SessionGateway,
    tools_gateway: ToolsGateway,
    interaction_gateway: InteractionGateway,
    budget_config: BudgetConfig,
    context_policy: ContextPolicy,
    budget_guard: PreModelBudgetGuard,
    tools: list[dict],
    rag_gateway: RagGateway | None = None,
) -> AgentGateway:
    """创建 AgentGateway 的工厂函数。

    Args:
        client: 模型调用客户端
        context_gateway: 上下文治理网关
        session_gateway: 会话状态管理网关
        tools_gateway: 工具执行网关
        interaction_gateway: 交互观察网关
        budget_config: Token 预算配置
        context_policy: 上下文治理策略
        budget_guard: 预算守卫
        tools: OpenAI 格式的工具定义列表
        rag_gateway: RAG 网关，用于 slash 命令
    Returns:
        AgentGateway: 创建的网关实例
    """
    slash_command_specs = SlashCommandRegistry(
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        rag_gateway=rag_gateway,
    ).get_specs()

    interaction_gateway.register_slash_commands(slash_command_specs)

    executor = AgentLoopExecutor(
        client=client,
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        tools_gateway=tools_gateway,
        interaction_gateway=interaction_gateway,
        budget_config=budget_config,
        context_policy=context_policy,
        budget_guard=budget_guard,
        tools=tools,
    )

    return AgentGateway(
        client=client,
        context_gateway=context_gateway,
        session_gateway=session_gateway,
        tools_gateway=tools_gateway,
        interaction_gateway=interaction_gateway,
        budget_config=budget_config,
        context_policy=context_policy,
        budget_guard=budget_guard,
        tools=tools,
        executor=executor,
    )


__all__ = ['create_gateway']