# Nexus v2 Architecture

## 1. 一句话目标

Nexus v2 是一个通用 Agent Runtime。它接收来自 CLI、Web API 或未来其他入口的用户请求，维护会话与上下文，调用模型，执行工具，管理记忆，并以统一事件流返回执行结果。

Nexus v2 不再被定义为 CLI coding agent。CLI 只是一个接口，Web API 也是一个接口。真正稳定的核心是 Agent Runtime。

## 2. 设计动机

当前 Nexus 已经具备模型调用、工具调用、上下文压缩、会话保存、RAG 等能力，但这些能力围绕 CLI 主循环逐步长大，导致项目一变复杂就不容易形成整体掌控感。

v2 的目标不是重写更多功能，而是重新划清边界：

- Interface 只负责输入输出。
- Runtime 只负责编排一次 Agent 执行。
- Model 只负责模型调用。
- Tools 只负责工具注册和执行。
- Memory 只负责会话、上下文和知识检索。

只要这五个边界清楚，项目可以继续长大，但不会每加一个能力都挤进同一个主流程。

## 3. 非目标

第一阶段不追求完整复刻 Claude、ChatGPT 或 Codex。

第一阶段暂不做：

- 多 Agent 协作。
- 复杂 Web UI。
- 插件市场。
- 长期人格记忆。
- 工作流编排器。
- 完整 RAGFlow 集成。
- 细粒度企业权限系统。

这些能力应该等 Runtime 的最小闭环稳定后，再作为可插拔模块加入。

## 4. 核心架构

```text
interfaces
  cli
  web

        |
        v

runtime
  AgentRuntime
  AgentRequest
  AgentEvent
  AgentResult

        |
        +------------+-------------+------------+
        v            v             v            v

      model        tools        memory      permissions
```

更完整的调用关系：

```text
User
  -> CLI / Web API
  -> AgentRequest
  -> AgentRuntime
  -> Memory builds context
  -> Model generates response or tool call
  -> Tools execute requested actions
  -> Runtime updates session
  -> AgentEvent stream
  -> CLI / Web renders result
```

## 5. 核心模块

### 5.1 interfaces

职责：

- 接收外部请求。
- 转换为统一的 `AgentRequest`。
- 渲染 `AgentEvent` 或 `AgentResult`。

接口类型：

- `interfaces/cli`: 终端交互、命令解析、流式输出。
- `interfaces/web`: HTTP API、SSE 或 WebSocket。

约束：

- 不直接调用模型。
- 不直接执行工具。
- 不直接读写 session 存储。
- 不包含 Agent 主循环。

### 5.2 runtime

职责：

- 执行一次 Agent 请求。
- 管理模型调用和工具调用循环。
- 产生统一事件流。
- 调用 Memory 读写会话和构造上下文。

核心契约：

```python
AgentRequest(
    session_id: str | None,
    input: str,
    interface: str,
    metadata: dict,
)

AgentEvent(
    type: str,
    payload: dict,
)

AgentResult(
    session_id: str,
    final_text: str,
    events: list[AgentEvent],
)
```

Runtime 是 Nexus v2 的心脏，但它不应该知道请求来自 CLI 还是 Web。

### 5.3 model

职责：

- 封装 OpenAI-compatible chat/completions 调用。
- 支持流式输出。
- 标准化模型响应和 tool call。

第一阶段只需要一个 OpenAI-compatible provider。Claude、Gemini、DeepSeek 等后续作为 provider 增加。

### 5.4 tools

职责：

- 注册工具。
- 暴露工具 schema。
- 执行工具调用。
- 返回结构化工具结果。

工具来源：

- 本地工具，例如文件、shell。
- MCP 工具。
- 未来的插件工具。

约束：

- Runtime 只依赖统一 `ToolRegistry` 和 `ToolExecutor`。
- Runtime 不关心工具来自本地、MCP 还是插件。

### 5.5 memory

职责：

- 管理 session history。
- 构造模型上下文。
- 执行上下文压缩。
- 接入知识检索。

Memory 不是单纯的 RAG。它包含三类能力：

- Short-term memory: 当前会话消息。
- Context management: token budget、compact、摘要。
- Knowledge retrieval: local RAG、RAGFlow、项目文档索引等。

RAGFlow 应作为 `memory` 的一个 backend，而不是 Runtime 的特殊分支。

### 5.6 permissions

职责：

- 判断工具调用是否允许。
- 管理文件、shell、网络等能力边界。
- 为 CLI 和 Web 提供一致的权限模型。

第一阶段可以很简单，只支持 allow/deny 和人工确认。

## 6. 最小闭环

v2 第一阶段只实现一个可解释、可运行、可测试的闭环：

```text
CLI input
  -> AgentRequest
  -> AgentRuntime
  -> Memory loads session
  -> Model returns assistant text
  -> Memory saves session
  -> CLI renders answer
```

然后增加第二条闭环：

```text
Web POST /sessions/{session_id}/messages
  -> AgentRequest
  -> same AgentRuntime
  -> AgentResult
  -> JSON response
```

再增加第三条闭环：

```text
Model requests tool call
  -> Runtime validates permission
  -> Tools execute
  -> Runtime sends tool result back to model
  -> final assistant answer
```

这三条闭环稳定之前，不引入 RAGFlow、多 Agent 或复杂 Web UI。

## 7. 建议目录结构

```text
src/
  nexus/
    interfaces/
      cli/
      web/
    runtime/
      __init__.py
      contracts.py
      agent_runtime.py
    model/
      __init__.py
      contracts.py
      openai_compatible.py
    tools/
      __init__.py
      contracts.py
      registry.py
      executor.py
    memory/
      __init__.py
      contracts.py
      session_store.py
      context_builder.py
      knowledge/
        local.py
        ragflow.py
    permissions/
      __init__.py
      policy.py
    config/
      settings.py
```

测试目录镜像源码目录：

```text
test/
  nexus/
    runtime/
    model/
    tools/
    memory/
    interfaces/
```

## 8. 关键原则

1. Runtime 不依赖 CLI。
2. Runtime 不依赖 Web。
3. Interface 不包含 Agent 主循环。
4. Tool 来源对 Runtime 透明。
5. RAG 是 Memory backend，不是主流程特例。
6. Session、Context、Knowledge 都属于 Memory。
7. 第一阶段先做少量闭环，不提前实现所有扩展点。

## 9. 从当前 Nexus 迁移的方向

当前模块可以这样映射到 v2：

```text
src/main.py              -> interfaces/cli + composition root
src/interaction          -> interfaces/cli
src/agent                -> runtime
src/client               -> model
src/tools                -> tools
src/session              -> memory/session_store
src/context              -> memory/context_builder
src/rag                  -> memory/knowledge/local
future RAGFlow adapter   -> memory/knowledge/ragflow
src/core_contracts       -> runtime/model/tools/memory contracts
```

迁移不需要一次完成。优先迁移最小闭环，再迁移 tool calling，最后迁移 context 和 RAG。

## 10. 第一阶段验收标准

第一阶段完成时，应该能做到：

- CLI 和 Web 都调用同一个 `AgentRuntime`。
- 一次普通对话请求可以完成模型调用并保存 session。
- Runtime 的核心流程可以用测试解释清楚。
- 新增一个接口类型时，不需要修改 Runtime。
- 新增一个模型 provider 时，不需要修改 interfaces。
- 新增一个工具来源时，不需要修改 Runtime 主循环。

如果这些标准满足，Nexus v2 就已经拥有了稳定骨架。
