# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
python -m pytest test/ -v

# Run a single test file
python -m pytest test/context/test_compactor.py -v

# Run a single test function
python -m pytest test/context/test_compactor.py::test_compact_reduces_message_count -v

# Run the application
python -m src.main
```

No linter or type checker is configured. No build step — this is a pure Python project.

## Architecture

Nexus is a coding agent CLI that follows a **Gateway/Facade + Contracts** architecture. Every domain module (`agent/`, `client/`, `context/`, `tools/`, `session/`, `interaction/`, `rag/`) exposes exactly one public class — a `*Gateway` dataclass — as its only external API. Module `__init__.py` files contain factory functions (`create_gateway`, `build_*`) that assemble dependencies via constructor injection.

All cross-module data structures live in `src/core_contracts/` as frozen dataclasses and Protocols. No module may import another module's internal implementation — only contracts and gateways.

### Module responsibilities

- **`agent/`** — Agent loop executor (pre-model → LLM → tool execution), ConversationOrchestrator (REPL input dispatch: slash commands vs. prompts), and slash command spec assembly (`src/agent/slash_commands.py`).
- **`client/`** — LLM client gateway. Validates `LlmRequest`, delegates to `OpenAIClient` which wraps the `openai` SDK. The gateway itself has zero dependency on the OpenAI SDK.
- **`context/`** — Token budget governance. `TokenEstimator` uses heuristic char-based estimation (4 chars ≈ 1 token). `Snipper` tombstone's old/long messages to stay under soft budget limits. `Compactor` calls the model to summarize conversation history into a system-reminder message. `ContextGateway` orchestrates: budget projection → snip → guard → auto-compact → guard, plus reactive compact on context-length errors.
- **`tools/`** — Tool registry (`MutableMapping`), executor (argument validation + handler dispatch), local tools (filesystem: list_dir/read_file/write_file/edit_file; shell: bash), and MCP tools (stdio + streamable-HTTP transports, async parallel loading via `ThreadPoolExecutor`).
- **`session/`** — Session persistence to `.nexus/sessions/*.json`. `SessionState` is the mutable runtime state; `SessionSnapshot` is the serializable frozen view.
- **`interaction/`** — CLI rendering (startup banner, exit summary, slash result panels), slash command parsing/dispatch (prefix matching, ambiguity detection), prompt_toolkit autocomplete for slash commands, and session tracking (tool call counts, wall-clock time).
- **`rag/`** — Document indexing (loader → chunker → embedding → vector store) and retrieval (query embedding → cosine similarity top-K search → prompt assembly). Supports PDF, DOCX, XLSX, PPTX via pypdf/python-docx/openpyxl/python-pptx.

### Key flows

1. **Startup**: `main.py` → `Application.run()` loads `ModelConfig.from_env()`, creates all gateways via factories, inits a session, enters `ConversationOrchestrator.run()` REPL.
2. **User prompt**: `ConversationOrchestrator._handle_prompt()` → append user message → `AgentGateway.run()` → `AgentLoopExecutor.execute()` (pre-model budget check → LLM call → tool execution loop, max 5 turns).
3. **Slash command**: `ConversationOrchestrator._handle_slash()` → parse → resolve (exact/prefix/ambiguous) → dispatch handler → apply result (exit, fork session, replace state).
4. **Context governance**: Before every LLM call, `ContextGateway.run_pre_model_cycle()` runs budget projection → snip if soft-over → guard check → auto-compact if over threshold → guard again. On context-length errors, `run_reactive_compact_cycle()` decides whether to compact and retry (max 2 attempts).
5. **Tool execution**: `ToolExecutor` checks required params, calls handler. MCP tools are loaded async in background via `AsyncMcpToolProvider`; `ToolsGateway.list_openai_tools()` merges them in when ready.

### Key patterns

- **Dataclass injection**: Gateways are `@dataclass` with injected dependencies. No DI framework — plain constructor injection in factory functions.
- **Frozen contracts**: All DTOs in `core_contracts/` are `@dataclass(frozen=True)`. `SessionState` in `session_contracts.py` is the main mutable exception.
- **Atomic steps**: Public methods delegate to private methods that are flat (no private-to-private calls). Complex orchestration methods (e.g., `run_pre_model_cycle`) compose private helpers in a flat DAG.
- **MCP config**: `.nexus/mcp.json` defines MCP servers. `McpToolConfig` defaults to this path.

## Environment variables

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes | — |
| `OPENAI_BASE_URL` | No | OpenAI default |
| `OPENAI_MODEL` | No | `gpt-4o` |
| `OPENAI_TEMPERATURE` | No | `0.7` |
| `OPENAI_MAX_TOKENS` | No | `1000000` |
| `RAG_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` |

## Tests

Tests mirror the `src/` structure under `test/`. `conftest.py` adds the project root to `sys.path`. Tests use `pytest` with `monkeypatch` for environment mocking and `unittest.mock.MagicMock` for client mocking. There is no test database or external service dependency — all external calls are mocked.
