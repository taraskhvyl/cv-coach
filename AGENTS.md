# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, etc.) working in this repo.

## What this is
CV Coach: Gradio app. User uploads a CV, asks by text or voice, gets RAG-grounded coaching. LangChain orchestrates RAG (splitter, embeddings, vector store, chain); OpenAI for chat/embeddings/Whisper/TTS; Langfuse traces every call.

## Layout
```
src/cv_coach/
  config.py   model names and settings (single place to change models)
  llm.py      shared OpenAI SDK client (used by voice.py)
  rag.py      load, split, Index (LangChain InMemoryVectorStore)
  coach.py    LangChain chain: prompt | ChatOpenAI | parser
  voice.py    transcribe (STT) and speak (TTS)
  app.py      Gradio UI only; no business logic
tests/        pytest, no network (inject fake langchain Embeddings)
```
Dependency direction: `app -> coach, voice -> rag, llm -> config`. Never import upward.

## Commands
- Install: `python -m venv .venv && .venv/bin/pip install -e '.[dev]'`
- Test: `.venv/bin/pytest`
- Run locally: `.venv/bin/cv-coach` (needs `.env`, copy from `.env.example`)
- Docker: `docker compose up --build` -> http://localhost:7860

## Conventions
- Every function that calls an external model is decorated with `@observe(name=...)`. LangChain calls pass `CallbackHandler()` in `config={"callbacks": [...]}`; raw SDK calls use `from .llm import client`.
- Tests must not hit the network or need API keys.
- Keep it simple: no new dependency or abstraction without a concrete need. Mark deliberate shortcuts with a `ponytail:` comment naming the upgrade path.
- Secrets live in `.env` (gitignored). Never commit keys or log CV content outside Langfuse.

## Known scaling points
- `rag.Index` is an InMemoryVectorStore: replace with a vector store when supporting many docs/users.
- State is per-session in Gradio; add persistence/auth before multi-user deployment.
- No eval suite yet: next step is Langfuse datasets + scored runs.
