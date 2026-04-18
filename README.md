# Gemma 4 E2B RAG POC

Local-only POC targeting an **M2 Air (8 GB)**. No Ollama, no vector DB, no embeddings.

## What's here
- **General chat** with Gemma 4 E2B, streamed as markdown.
- **PDF RAG via PageIndex**: OpenDataLoader builds a section tree at ingest; the LLM navigates the document via JSON tool calls (`list_sections`, `read_section`, `read_pages`). No retrieval scoring, no vectors.
- **Long-term conversation**: SQLite-backed, with rolling auto-summaries.
- **Markdown-first UI** with per-message citations and a "New conversation" control.

## Requirements
- macOS Apple Silicon (Metal supported by llama.cpp)
- Python 3.10–3.12
- Node 18+
- **JDK 11+** (OpenDataLoader spawns a Java subprocess during ingest; verify with `java -version`)
- ~5 GB free disk for the Q4 GGUF download

## Backend
```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --app-dir .
```
The first run downloads the Gemma 4 E2B Q4_K_M GGUF from HuggingFace (~3 GB) into the standard `~/.cache/huggingface/` cache. The PyPI `llama-cpp-python` wheel for macOS arm64 already ships with Metal enabled.

## Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173.

## Configuration (env vars)
| Var | Default | Purpose |
|---|---|---|
| `GGUF_REPO` | `unsloth/gemma-4-E2B-it-GGUF` | HF repo for the GGUF |
| `GGUF_FILE` | `gemma-4-E2B-it-Q4_K_M.gguf` | file inside the repo |
| `N_CTX` | `8192` | context window (drop to 4096 if RAM is tight) |
| `N_GPU_LAYERS` | `-1` | all layers on Metal |
| `MAX_NEW_TOKENS` | `512` | per generation cap |
| `TEMPERATURE` | `0.7` | sampling |
| `TOP_P` | `0.95` | sampling |
| `TOP_K` | `64` | sampling |
| `RAG_MAX_HOPS` | `6` | max tool-call hops per RAG answer |
| `RAG_PAGE_READ_LIMIT` | `10` | max pages per `read_pages` call |
| `SUMMARY_TRIGGER` | `24` | start summarizing after N messages |
| `HISTORY_WINDOW` | `10` | last N messages kept in prompt context |

## 8 GB M2 Air notes
- Q4_K_M model weights ≈ 3.2 GB. With KV cache at `N_CTX=8192`, total ≈ 4–5 GB.
- Close Chrome/heavy apps **during PDF ingest** — the Java subprocess is transient but adds ~1 GB while running.
- If ingest or chat swaps to disk, drop `N_CTX` to `4096`, or set `GGUF_FILE=gemma-4-E2B-it-Q3_K_M.gguf` for ~2.4 GB weights.

## API
- `POST /api/documents` (multipart `file`) → `{doc_id, filename, page_count, section_count}`
- `POST /api/conversations` → `{conversation_id}`
- `GET /api/conversations/{id}` → messages + summary
- `POST /api/chat` (JSON; `mode: "chat" | "rag"`; streams SSE): emits `start`, `status`, `token`, `sources`, `done`, `error` events.

## Why PageIndex (no vectors)?
Vector embeddings require a pretrained embedding model, an index, and come with semantic failure modes that are hard to debug. PageIndex keeps retrieval fully **deterministic and inspectable**: the document is reduced at ingest to a section tree with titles + short leaders, the LLM is handed that outline, and it explicitly requests sections by id. Every answer has a visible navigation trace.
