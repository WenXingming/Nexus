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
- `FakeModel`
- `InMemorySessionStore`
- 单次 CLI 调用

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
