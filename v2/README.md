# Nexus v2

Nexus v2 是一个从小处开始、逐步实现的通用 Agent Runtime。

第一目标是掌控感：每一步都应该容易解释、容易测试、容易替换，然后再允许项目慢慢长大。

## 实现原则

- **简单**：每个组件都应该尽可能简单，避免过度设计。
- **模块化**：组件之间应该高度解耦，允许独立开发和测试。
- **可测试**：每个组件都应该有明确的接口和行为，便于编写单元测试。

## 实现过程

1. 做计划
2. 拆分计划、小步实现
3. 每一步都要测试验证

## 当前能力

- 最小 `AgentRuntime`
- `FakeClient`
- `InMemorySessionStore`
- 单次 CLI 调用
- 最小 REPL

## 运行测试

```powershell
$env:PYTHONPATH="v2"
python -m pytest v2/test -v
```

## 运行 CLI

```powershell
$env:PYTHONPATH="v2"
python -m src.interfaces.cli hi
```

输出：

```text
Echo: hi
```

## 运行 REPL

```powershell
$env:PYTHONPATH="v2"
python -m src.interfaces.cli --repl
```

示例：

```text
> hi
Echo: hi
> /exit
```

当前 REPL 使用 `FakeClient` 和 `InMemorySessionStore`。历史消息只保存在当前 Python 进程内，程序退出后不会持久化。

## 模型 Provider

默认使用 `fake` provider：

```powershell
$env:PYTHONPATH="v2"
python -m src.interfaces.cli hi
```

使用 OpenAI-compatible provider：

```powershell
$env:PYTHONPATH="v2"
$env:NEXUS_MODEL_PROVIDER="openai"
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-4o-mini"
python -m src.interfaces.cli hi
```

如果使用自定义 OpenAI-compatible 服务，可以设置：

```powershell
$env:OPENAI_BASE_URL="https://example.test/v1"
```

当前 OpenAI-compatible provider 只支持普通非流式 chat completion，不支持 streaming 或 tool calls。
