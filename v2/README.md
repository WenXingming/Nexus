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

## 安装依赖

```powershell
python -m pip install -r v2/requirements.txt
```

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
session: afa463a8-5e7d-4689-88ac-367577246835
Echo: hi
> /exit
```

默认情况下，REPL 使用 `FakeClient` 和内存 session 存储。历史消息只保存在当前 Python 进程内，程序退出后不会持久化。

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
$env:OPENAI_MODEL="qwen3.6-plus"
python -m src.interfaces.cli hi
```

如果使用自定义 OpenAI-compatible 服务，可以设置：

```powershell
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

当前 OpenAI-compatible provider 只支持普通非流式 chat completion，不支持 streaming 或 tool calls。

## Session 存储

默认使用内存存储：

```powershell
$env:NEXUS_MEMORY_STORE="memory"
```

历史只存在当前 Python 进程内。

使用文件存储：

```powershell
$env:PYTHONPATH="v2"
$env:NEXUS_MEMORY_STORE="file"
$env:NEXUS_SESSION_ROOT=".nexus-v2/sessions"
python -m src.interfaces.cli --repl
```

session 文件会保存为：

```text
.nexus-v2/sessions/afa463a8-5e7d-4689-88ac-367577246835.json
```

session id 使用 UUID 自动生成。

当前限制：

- 只能在同一个进程内继续使用已有 session id。
- 暂无 resume/load 命令。
