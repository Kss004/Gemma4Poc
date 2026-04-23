# Gemma 4 E2B RAG POC

Local-only POC targeting an **M2 Air (8 GB)**. No Ollama, no vector DB, no embeddings. LLM inference is served by **LM Studio** (via the `lmstudio` Python SDK).

## What's here
- **General chat** with Gemma 4 E2B, streamed as markdown.
- **PDF RAG via PageIndex**: OpenDataLoader builds a section tree at ingest; the LLM is given the outline and navigates the document via JSON tool calls (`list_sections`, `read_section`, `read_pages`). No retrieval scoring, no vectors.
- **Long-term conversation**: SQLite-backed, with rolling auto-summaries.
- **Markdown-first UI** with per-message citations and a "New conversation" control.

## Requirements
- macOS Apple Silicon
- Python 3.10–3.12
- Node 18+
- **JDK 11+** (OpenDataLoader spawns a Java subprocess during ingest; verify with `java -version`)
- **LM Studio** (desktop app or the headless `llmster` daemon)

## One-time LM Studio setup

```bash
# 1. Install LM Studio (installs `lms` CLI + `llmster` daemon)
curl -fsSL https://lmstudio.ai/install.sh | bash

# 2. Start the daemon in the background
lms daemon up

# 3. Register Gemma 4 E2B. Either pull from the catalog:
lms get unsloth/gemma-4-E2B-it-GGUF

# ... or sideload an already-downloaded HuggingFace GGUF:
#     (useful if you've been running the older llama-cpp-python flow)
lms import ~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/blobs/<sha>

# 4. Confirm the registered model key, then export it
lms ls
export LMS_MODEL_KEY="unsloth/gemma-4-E2B-it-GGUF"   # whatever `lms ls` shows
```

## Backend
```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --app-dir .
```
The backend lazily connects to the LM Studio daemon on the first chat request. If the daemon is down or the model key isn't registered, `/api/chat` returns a clear remediation message.

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
| `LMS_MODEL_KEY` | `gemma-4-e2b-it` | LM Studio model key (from `lms ls`) |
| `N_CTX` | `8192` | context window at model load (drop to 4096 if RAM is tight) |
| `MAX_NEW_TOKENS` | `512` | per generation cap |
| `TEMPERATURE` | `0.7` | sampling |
| `TOP_P` | `0.95` | sampling (maps to `topPSampling`) |
| `TOP_K` | `64` | sampling (maps to `topKSampling`) |
| `RAG_MAX_HOPS` | `6` | max tool-call hops per RAG answer |
| `RAG_PAGE_READ_LIMIT` | `10` | max pages per `read_pages` call |
| `SUMMARY_TRIGGER` | `24` | start summarizing after N messages |
| `HISTORY_WINDOW` | `10` | last N messages kept in prompt context |

GPU layer / CPU thread tuning is owned by LM Studio; set it per-model in the LM Studio UI or via `lms load --gpu=max --context-length=8192`.

## 8 GB M2 Air notes
- Gemma 4 E2B Q4_K_M weights ≈ 3.2 GB, plus KV cache at `N_CTX=8192` ≈ 1–2 GB, plus `llmster` daemon overhead ≈ 200–400 MB.
- Close Chrome/heavy apps **during PDF ingest** — the Java subprocess is transient but adds ~1 GB while running.
- If swap kicks in, drop `N_CTX` to `4096`, or register a smaller quant (e.g. `gemma-4-E2B-it-Q3_K_M.gguf`) via `lms import`.
- Run only the `llmster` daemon, not the LM Studio GUI, to keep memory pressure low.

## API
- `POST /api/documents` (multipart `file`) → `{doc_id, filename, page_count, section_count}`
- `POST /api/conversations` → `{conversation_id}`
- `GET /api/conversations/{id}` → messages + summary
- `POST /api/chat` (JSON; `mode: "chat" | "rag"`; streams SSE): emits `start`, `status`, `token`, `sources`, `done`, `error` events.

## Why PageIndex (no vectors)?
Vector embeddings require a pretrained embedding model, an index, and come with semantic failure modes that are hard to debug. PageIndex keeps retrieval fully **deterministic and inspectable**: the document is reduced at ingest to a section tree with titles + short leaders, the LLM is handed that outline, and it explicitly requests sections by id. Every answer has a visible navigation trace.
