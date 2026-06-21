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
- `FileSessionStore`
- 单次 CLI 调用
- 单次 CLI 流式输出
- 最小流式 REPL

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

指定已有 session：

```powershell
$env:PYTHONPATH="v2"
python -m src.interfaces.cli --session afa463a8-5e7d-4689-88ac-367577246835 hi
```

如果 session 不存在，会输出：

```text
Session not found: afa463a8-5e7d-4689-88ac-367577246835
```

流式输出：

```powershell
$env:PYTHONPATH="v2"
python -m src.interfaces.cli --stream hi
```

`--stream` 会边接收模型输出边打印。普通 CLI 调用会等模型完整返回后再一次性输出。

流式输出也可以继续已有 session：

```powershell
$env:PYTHONPATH="v2"
python -m src.interfaces.cli --stream --session afa463a8-5e7d-4689-88ac-367577246835 hi
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
session: afa463a8-5e7d-4689-88ac-367577246835
```

默认情况下，REPL 使用 `FakeClient` 和文件 session 存储，历史消息会保存到本地 session 文件。

REPL 会边接收模型输出边打印；每轮回复结束后自动换行。

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

当前 OpenAI-compatible provider 支持普通 chat completion 和 streaming，暂不支持 tool calls。

## Agent 配置

默认情况下，Agent 不设置 system prompt。

可以通过环境变量设置：

```powershell
$env:PYTHONPATH="v2"
$env:NEXUS_SYSTEM_PROMPT="You are Nexus."
python -m src.interfaces.cli hi
```

`NEXUS_SYSTEM_PROMPT` 会作为 system message 发送给模型，但不会保存到 session 历史。

## Session 存储

默认使用文件存储：

```powershell
$env:NEXUS_MEMORY_STORE="file"
```

session 文件默认保存到 `.nexus-v2/sessions`。

手动验证默认文件存储：

```powershell
$env:PYTHONPATH="v2"
$env:NEXUS_SESSION_ROOT=".nexus-v2/sessions"
python -m src.interfaces.cli hi
Get-ChildItem .nexus-v2/sessions
```

执行后，`.nexus-v2/sessions` 下会生成 `<session-id>.json`。

显式指定文件存储路径：

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

内存存储主要用于测试或临时会话：

```powershell
$env:NEXUS_MEMORY_STORE="memory"
```

使用内存存储时，历史只存在当前 Python 进程内。

使用文件存储时，可以通过 `--session <id>` 继续已有 session：

```powershell
$env:PYTHONPATH="v2"
$env:NEXUS_MEMORY_STORE="file"
$env:NEXUS_SESSION_ROOT=".nexus-v2/sessions"
python -m src.interfaces.cli --session afa463a8-5e7d-4689-88ac-367577246835 hi
```

也可以从已有 session 启动 REPL：

```powershell
$env:PYTHONPATH="v2"
$env:NEXUS_MEMORY_STORE="file"
$env:NEXUS_SESSION_ROOT=".nexus-v2/sessions"
python -m src.interfaces.cli --repl --session afa463a8-5e7d-4689-88ac-367577246835
```

如果 session 存在，REPL 会继续该 session 的历史，并在退出时输出 `session: <id>`。如果 session 不存在，会输出 `Session not found: <id>`。

当前限制：

- 暂无 session 列表命令。
