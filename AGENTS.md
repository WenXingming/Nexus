# Repository Guidelines

## Project Structure & Module Organization

Nexus is a Python CLI coding-agent project. Runtime code lives in `src/`, with domains split into `agent/`, `client/`, `context/`, `interaction/`, `rag/`, `session/`, and `tools/`. Shared DTOs and Protocols live in `src/core_contracts/`; keep cross-module types there instead of importing another module's internals. Tests mirror the source layout under `test/` (for example, `src/context/compactor.py` is covered by `test/context/test_compactor.py`). Project notes are in `docs/`, local utility scripts are in `scripts/`, and runtime session/config data is stored under `.nexus/`.

## Build, Test, and Development Commands

Install dependencies with:

```bash
python -m pip install -r requirements.txt pytest
```

Run the application:

```bash
python -m src.main
```

Run the full test suite:

```bash
python -m pytest test/ -v
```

Run a focused test file or function:

```bash
python -m pytest test/context/test_compactor.py -v
python -m pytest test/context/test_compactor.py::test_compact_reduces_message_count -v
```

There is no configured build step, linter, or type checker.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes. Public domain APIs should remain gateway/facade oriented: each domain exposes a `*Gateway` dataclass and factory functions such as `create_gateway` or `build_*` from `__init__.py`. Prefer constructor injection over global state. Keep contract dataclasses frozen unless mutability is explicitly required, as with session state.

## Testing Guidelines

Use `pytest`. Place tests in the matching `test/<domain>/` package and name files `test_*.py`. Mock external services with `monkeypatch` or `unittest.mock.MagicMock`; tests should not require real OpenAI calls, MCP servers, or persistent external services. Add focused tests for gateway behavior, contract validation, and orchestration edge cases.

## Commit & Pull Request Guidelines

Recent commits use short subject lines with prefixes such as `Feat:`, `Optimization:`, and `refactor/` branch names, sometimes in Chinese. Keep commits imperative and scoped, for example `Feat: add compact slash command`. Pull requests should include a concise summary, linked issue or task when available, test evidence (`python -m pytest test/ -v`), and screenshots only for visible CLI/rendering changes.

## Security & Configuration Tips

Set `OPENAI_API_KEY` before running the CLI. Optional environment variables include `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`, and `RAG_EMBEDDING_MODEL`. Do not commit secrets, `.nexus/sessions/*.json`, or machine-specific MCP credentials.
