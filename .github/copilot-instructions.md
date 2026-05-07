# Copilot Instructions for Nexus

## Build, test, and run commands

This repository is a pure Python project. There is no build step, linter, or type checker configured.

```bash
# Install runtime dependencies and pytest
python -m pip install -r requirements.txt pytest

# Run the CLI
python -m src.main

# Run the full test suite
python -m pytest test/ -v

# Run a single test file
python -m pytest test/context/test_compactor.py -v

# Run a single test
python -m pytest test/context/test_compactor.py::test_compact_reduces_message_count -v
```

## High-level architecture

Nexus is a coding-agent CLI organized around a **Gateway/Facade + Contracts** architecture.

- `src/main.py` is the composition root. `Application.run()` loads environment config, creates all gateways, initializes a session, and hands control to `ConversationOrchestrator`.
- Each domain exposes a single public gateway from its package entrypoint: `agent`, `client`, `context`, `interaction`, `rag`, `session`, and `tools`.
- Cross-module DTOs and Protocols live in `src/core_contracts/`. Modules are expected to depend on contracts and gateways, not on another module's internal classes.

The main runtime flow is:

1. `ConversationOrchestrator` reads REPL input and routes slash commands separately from normal prompts.
2. Normal prompts append a user message to `SessionState` and call `AgentGateway.run()`.
3. `AgentLoopExecutor` performs the main loop: pre-model context governance, LLM call, optional tool execution, then repeats for up to 5 turns.
4. `ContextGateway` manages token budgets with a staged flow: budget projection, snipping, guard checks, auto-compaction, and reactive compaction on context-length failures.
5. `ToolsGateway` exposes local tools immediately and can merge MCP tools loaded asynchronously from `.nexus/mcp.json`.
6. `SessionGateway` persists snapshots to `.nexus/sessions/*.json`; `SessionState` is the mutable in-memory state and `SessionSnapshot` is the frozen persisted view.
7. `RagGateway` is a separate indexing/retrieval pipeline used by slash commands such as `/rag-index` and `/rag-ask`.

## Key conventions

- Preserve package boundaries. External code should import from package `__init__.py` entrypoints and `src/core_contracts/*`, not internal implementation modules, unless a white-box test needs it.
- Keep the gateway pattern intact. Public APIs are gateway-oriented, with factory functions such as `create_gateway`, `create_interaction_gateway`, and `build_rag_gateway` assembling dependencies via constructor injection.
- Treat `src/core_contracts/` as the shared boundary. Contract dataclasses are generally frozen; `SessionState` is the main intentional mutable exception because it is the live runtime session object.
- Keep orchestration in the designated layers:
  - `src/main.py` owns composition-root wiring
  - `agent/conversation_orchestrator.py` owns REPL input routing and slash-result-to-state translation
  - `agent/agent_executor.py` owns the pre-model -> LLM -> tool-execution loop
- Keep slash command responsibilities split the current way: command specs and handlers are assembled in `src/agent/slash_commands.py`, while `interaction/` only parses, resolves, dispatches, renders, and autocompletes them.
- Session persistence rules matter. Session snapshots must remain JSON-serializable and continue to round-trip through `SessionSnapshot`/`SessionStore`; do not bypass `SessionGateway` for cross-module session work.
- MCP behavior is intentionally async by default in the composition root. Local tools should remain usable before MCP loading finishes, and MCP name collisions are resolved by prefixing duplicate remote tool names with `mcp_`.
- Tests mirror the source layout under `test/`. Existing tests rely on `pytest`, `monkeypatch`, and `unittest.mock.MagicMock`; keep tests isolated from real OpenAI, MCP, and other external services.
