"""最小闭环验证入口 (MVP)。

支持 Session 管理、Tools 工具调用、context 治理，以及命令式最小 RAG 接入的主循环。

斜杠命令通过 interaction 模块的 SlashCommandDispatcher 统一分发，覆盖：
/help, /context, /status, /permissions, /tools, /clear, /exit, /quit,
/new, /save, /load, /rag-index, /rag-ask。

Agent 主循环编排已下沉至 src.agent.agent_gateway，main.py 仅保留组合根与 REPL 控制流。
"""

import sys
from pathlib import Path

from src.agent import (
    AgentGateway,
    ConversationOrchestrator,
    build_default_slash_command_specs,
    create_conversation_orchestrator,
    create_gateway as create_agent_gateway,
)
from src.client import ClientGateway, create_gateway as create_llm_gateway
from src.context import create_gateway as create_context_gateway
from src.context.context_gateway import ContextGateway
from src.core_contracts.context_contracts import (
    BudgetConfig,
    ContextPolicy,
    PreModelBudgetGuard,
)
from src.core_contracts.model_config import ModelConfig, RagModelConfig
from src.core_contracts.session_contracts import SessionState
from src.core_contracts.tools_contracts import McpToolConfig
from src.core_contracts.workspace_config import WorkspaceConfig
from src.rag import RagGateway, build_rag_gateway
from src.session import SessionGateway, create_gateway as create_session_gateway
from src.tools import create_gateway as create_tools_gateway
from src.tools.tools_gateway import ToolsGateway

from src.interaction import InteractionGateway, create_interaction_gateway


# ============================================================
# Application — 单层编排 (Single Orchestrator + Atomic Steps)
# ============================================================

class Application:
    """REPL 应用主类：组装依赖并驱动主循环。

    run() 是唯一流程入口，所有私有方法均为其直接子步骤，
    私有方法间禁止互相调用，执行路径呈扁平 DAG 而非调用树。

    具体对话编排由 ConversationOrchestrator 负责；本类只保留组合根、
    启动与退出职责。
    """

    def __init__(self) -> None:
        """依赖注入入口，所有成员变量在此集中初始化。"""
        self._config: ModelConfig | None = None  # 模型配置 (ModelConfig.from_env)
        self._client: ClientGateway | None = None  # LLM 客户端门面
        self._context_gateway: ContextGateway | None = None  # context 治理门面 (预算/compact)
        self._session_gateway: SessionGateway | None = None  # 会话管理门面
        self._tools_gateway: ToolsGateway | None = None  # 工具注册与执行网关
        self._rag_gateway: RagGateway | None = None  # RAG 检索增强门面
        self._interaction_gateway: InteractionGateway | None = None  # interaction 门面 (slash/渲染/追踪)
        self._agent_gateway: AgentGateway | None = None  # Agent 主流程编排网关
        self._conversation_orchestrator: ConversationOrchestrator | None = None  # REPL 输入编排器
        self._budget_config: BudgetConfig | None = None  # token 预算配置
        self._context_policy: ContextPolicy | None = None  # 上下文治理策略
        self._budget_guard: PreModelBudgetGuard | None = None  # 预算守卫
        self._mcp_config = McpToolConfig()  # MCP 工具装配契约由组合根持有。
        self._workspace_config: WorkspaceConfig | None = None  # 工作空间配置

    # ---- 唯一流程入口 (Single Orchestrator) ----

    def run(self) -> int:
        """唯一流程入口：加载配置 → 构造依赖 → 初始化会话 → REPL 循环。

        Returns:
            int: 0 表示正常退出，1 表示配置错误。
        """
        try:
            self._load_config()
        except ValueError as e:
            print(f"配置错误: {e}", file=sys.stderr)
            print("请设置环境变量 OPENAI_API_KEY 后重试", file=sys.stderr)
            return 1

        self._create_dependencies()
        state = self._init_session()

        if self._conversation_orchestrator is None:
            raise RuntimeError("Conversation orchestrator 尚未初始化。")
        
        try:
            final_state = self._conversation_orchestrator.run(state)
        except KeyboardInterrupt:
            print("\n[中断] 用户取消操作", file=sys.stderr)
            final_state = state
        
        self._save_session(final_state)
        self._render_exit()
        return 0

    # ---- 原子步骤 (Atomic Steps) — 均为 run() 的直接子步骤，禁止互相调用 ----

    def _load_config(self) -> None:
        """从环境变量加载模型配置并写入 self._config。

        Raises:
            ValueError: 环境变量配置不合法时抛出。
        """
        self._config = ModelConfig.from_env()

    def _create_dependencies(self) -> None:
        """集中构造主循环所需的全部长期依赖与默认策略。

        Raises:
            RuntimeError: 网关创建失败时抛出。
        """
        self._workspace_config = WorkspaceConfig.from_cwd()
        self._client = create_llm_gateway(config=self._config)
        self._context_gateway = create_context_gateway(client=self._client)
        self._session_gateway = create_session_gateway()
        self._tools_gateway = create_tools_gateway(
            mcp_config=self._mcp_config,
            async_mode=True,
        )
        rag_config = RagModelConfig.from_env()
        self._rag_gateway = build_rag_gateway(
            model_config=self._config,
            rag_config=rag_config,
        )
        slash_specs = build_default_slash_command_specs(
            context_gateway=self._context_gateway,
            session_gateway=self._session_gateway,
            rag_gateway=self._rag_gateway,
            client_gateway=self._client,
            tools_gateway=self._tools_gateway,
        )
        self._interaction_gateway = create_interaction_gateway(slash_specs=slash_specs)

        self._budget_config = BudgetConfig(
            max_input_tokens=None,
            output_reserve_tokens=max(256, self._config.max_tokens),
        )
        self._context_policy = ContextPolicy()
        self._budget_guard = PreModelBudgetGuard()
        self._agent_gateway = create_agent_gateway(
            client=self._client,
            context_gateway=self._context_gateway,
            session_gateway=self._session_gateway,
            tools_gateway=self._tools_gateway,
            interaction_gateway=self._interaction_gateway,
            budget_config=self._budget_config,
            context_policy=self._context_policy,
            budget_guard=self._budget_guard,
        )
        self._conversation_orchestrator = create_conversation_orchestrator(
            interaction_gateway=self._interaction_gateway,
            agent_gateway=self._agent_gateway,
            session_gateway=self._session_gateway,
            tools_gateway=self._tools_gateway,
            budget_config=self._budget_config,
            context_policy=self._context_policy,
            model_config=self._config,
            workspace_config=self._workspace_config,
        )

    def _init_session(self) -> SessionState:
        """初始化 REPL 会话，渲染启动界面并创建新会话状态。

        Returns:
            SessionState: 新创建的会话状态。
        """
        self._interaction_gateway.render_startup()

        # 显示本地工具数量
        # local_tools_count = len(self._tools_gateway.list_tools())
        # print(f"已加载 {local_tools_count} 个本地工具")

        # # 提示 MCP 工具正在加载
        # if self._tools_gateway.is_mcp_loading():
        #     print("MCP 工具加载中... (输入 /tools 查看加载状态)")

        print(f"工作目录: {self._workspace_config.root}")
        state = self._session_gateway.create_state("新会话已创建")
        self._interaction_gateway.start_session_tracker(state.session_id)
        # print(f"[会话已创建] session_id={state.session_id}")
        return state

    def _save_session(self, state: SessionState) -> None:
        """保存当前会话状态。

        Args:
            state (SessionState): 待保存的会话状态。
        Returns:
            None
        Raises:
            None
        """
        if state is None or self._session_gateway is None or self._config is None:
            return
        try:
            session_id, session_path = self._session_gateway.save_state(
                state=state,
                model_config=self._config,
            )
            # print(f"[会话已保存] session_id={session_id}")
        except Exception as e:
            print(f"[保存失败] {e}", file=sys.stderr)

    def _render_exit(self) -> None:
        """渲染退出界面。"""
        self._interaction_gateway.render_exit(self._interaction_gateway.get_session_summary())


# ============================================================
# 入口
# ============================================================

def main() -> int:
    """MVP 主入口：创建 Application 实例并启动 REPL 循环。

    Returns:
        int: 0 表示正常退出，1 表示配置错误。
    """
    app = Application()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
