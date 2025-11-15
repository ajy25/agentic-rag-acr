# Agentic RAG ACR

Agentic Retrieval-Augmented Generation (RAG) system built with [Agno](https://docs.agno.com) that indexes local PDFs and answers questions with grounded citations.

## Architecture

- **Documents**: drop source material into `sources_pdf/` (any nested structure is fine).
- **Ingestion** (`python -m rag.ingest run`):
  - Parses each PDF with `pypdf`, batches contiguous pages into ~1.8k-character chunks, and enriches them with metadata (file name, relative path, page span, chunk id).
  - Embeds chunks using OpenAI's `text-embedding-3-small` via Agno's `Knowledge` abstraction and stores vectors in a local LanceDB table (`data/lancedb`).
  - Persists an ingestion manifest (`data/ingest_manifest.json`) to skip unchanged files unless `--force` is supplied.
- **Knowledge base**: `rag/knowledge.py` memoizes an Agno `Knowledge` object backed by LanceDB (hybrid search) and reuses it everywhere.
- **Agent runtime** (`python agent.py ask ...` / `chat`):
  - Single Agno `Agent` with `OpenAIChat` (model defaults to `gpt-4o-mini`), the shared knowledge base, chat history in SQLite (`data/agent_history.db`), and a `RagResponse` output schema that enforces `{answer, citations}`.
  - Optional context preview uses `knowledge.search` to show the top retrieved chunks before issuing the LLM call.

This keeps agent creation outside of loops, enables structured outputs, and separates ingestion from querying as suggested in `.cursorrules`.

## Setup

1. Ensure the `agentic-rag-acr` conda environment is active and install/update dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file (already present) with at least:

  ```env
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini           # optional override
  OPENAI_EMBEDDING_MODEL=text-embedding-3-small
  ```

3. Add PDFs to `sources_pdf/`. Subdirectories are supported.

## Ingest documents

```bash
python -m rag.ingest run            # full ingest
python -m rag.ingest run --force    # rebuild everything
python -m rag.ingest run --limit 2  # ingest the first two PDFs
python -m rag.ingest clear --yes    # drop LanceDB table + manifest
```

## Ask questions

```bash
python agent.py ask "What does the safety spec say about audits?"
python agent.py ask "Summarize sections 2-3" --show-context --context-k 5
python agent.py chat --context-k 2
```

Answers stream in Markdown, followed by a bullet list of citations populated by the structured output schema. Use `--show-context` to preview the retrieved snippets (without incurring an extra LLM call).

## Project layout

```
agent.py               # CLI for single-shot Q&A or chat
rag/
  __init__.py          # loads .env and exposes BASE_DIR
  config.py            # paths, models, and Settings dataclass
  knowledge.py         # LanceDB-backed Agno knowledge helper
  ingest.py            # PDF ingestion & management CLI
requirements.txt
sources_pdf/           # drop your documents here
```

## Next steps

- Configure `DATABASE_URL` + `PostgresDb` before moving to production (per `.cursorrules`).
- Extend `rag.ingest` with additional readers (Markdown, web) via Agno's reader registry.
- Add automated evaluations by scripting `agent.py ask` calls against expected answers.
