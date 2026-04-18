from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterator, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import (
    DATA_DIR,
    HISTORY_WINDOW,
    RAG_MAX_HOPS,
    SUMMARY_EVERY,
    SUMMARY_TRIGGER,
    UPLOADS_DIR,
)
from .model import ModelService
from .rag import (
    Document,
    DocumentStore,
    rag_system_prompt,
)
from .schemas import (
    ConversationResponse,
    ConversationSummary,
    DocumentResponse,
)
from .storage import (
    add_message,
    create_conversation,
    get_conversation,
    init_db,
    list_messages,
    touch_conversation,
    update_summary,
)


app = FastAPI(title="Gemma 4 E2B PageIndex RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
init_db()
model_service = ModelService()
document_store = DocumentStore()


def _load_history(conversation_id: str) -> List[Dict[str, str]]:
    return [
        {"role": role, "content": content}
        for role, content, _ in list_messages(conversation_id)
    ]


def _should_summarize(count: int) -> bool:
    if count < SUMMARY_TRIGGER:
        return False
    return (count - SUMMARY_TRIGGER) % SUMMARY_EVERY == 0


def _maybe_summarize(conversation_id: str, messages: List[Dict[str, str]]) -> str:
    if not _should_summarize(len(messages)):
        return ""
    body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    summary = model_service.generate(
        [
            {"role": "system", "content": "You are a concise summarizer."},
            {
                "role": "user",
                "content": (
                    "Summarize this conversation in up to 6 bullet points. "
                    "Focus on stable user preferences, goals, and unresolved questions.\n\n"
                    + body
                ),
            },
        ],
        max_tokens=256,
    )
    update_summary(conversation_id, summary)
    return summary


def _build_chat_messages(
    system_prompt: str,
    summary: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]
    if summary:
        msgs.append(
            {"role": "system", "content": f"Conversation summary so far:\n{summary}"}
        )
    msgs.extend(history[-HISTORY_WINDOW:])
    return msgs


def _sse(event: str, data) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _stream_tokens_and_persist(
    conversation_id: str,
    messages: List[Dict[str, str]],
    citations: Optional[List[Dict]] = None,
) -> Iterator[bytes]:
    pieces: List[str] = []
    try:
        for tok in model_service.stream(messages):
            pieces.append(tok)
            yield _sse("token", tok)
    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": str(exc)})
        return
    full = "".join(pieces).strip()
    if full:
        add_message(conversation_id, "assistant", full)
        touch_conversation(conversation_id)
    if citations:
        yield _sse("sources", citations)
    yield _sse("done", {"conversation_id": conversation_id})


def _run_rag(
    conversation_id: str,
    doc: Document,
    system_prompt: str,
    summary: str,
    history: List[Dict[str, str]],
    user_prompt: str,
) -> Iterator[bytes]:
    """Single-pass RAG: inject the full document text into the prompt."""
    full_system = rag_system_prompt(doc, system_prompt)
    if summary:
        full_system += f"\n\nConversation summary so far:\n{summary}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": full_system},
        *history[-HISTORY_WINDOW:],
        {"role": "user", "content": user_prompt},
    ]

    yield _sse("start", {"conversation_id": conversation_id})
    yield from _stream_tokens_and_persist(conversation_id, messages)


# ---------- Routes ----------

@app.post("/api/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as h:
        h.write(content)
    doc = document_store.ingest_pdf(file_path, file.filename, time.time())
    return DocumentResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        page_count=doc.page_count,
        section_count=len(doc.sections),
    )


@app.post("/api/conversations", response_model=ConversationSummary)
async def new_conversation() -> ConversationSummary:
    cid = create_conversation(title="Chat")
    return ConversationSummary(conversation_id=cid)


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_detail(conversation_id: str) -> ConversationResponse:
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = [
        {"role": role, "content": content, "created_at": created_at}
        for role, content, created_at in list_messages(conversation_id)
    ]
    return ConversationResponse(
        conversation_id=conversation_id,
        summary=conv[2] or "",
        messages=messages,
    )


@app.post("/api/chat")
async def chat(request: dict):
    system_prompt = str(request.get("system_prompt") or "You are a helpful assistant.")
    user_prompt = str(request.get("user_prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="user_prompt is required")
    mode = str(request.get("mode") or "chat")
    doc_id = request.get("doc_id")

    conversation_id = request.get("conversation_id")
    if conversation_id:
        if not get_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation_id = create_conversation(title="Chat")

    add_message(conversation_id, "user", user_prompt)
    touch_conversation(conversation_id)
    history = _load_history(conversation_id)
    conv = get_conversation(conversation_id)
    summary = conv[2] if conv else ""
    new_summary = _maybe_summarize(conversation_id, history)
    if new_summary:
        summary = new_summary

    if mode == "rag":
        if not doc_id:
            raise HTTPException(status_code=400, detail="doc_id is required for RAG mode")
        try:
            doc = document_store.get(doc_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown doc_id")
        return StreamingResponse(
            _run_rag(
                conversation_id=conversation_id,
                doc=doc,
                system_prompt=system_prompt,
                summary=summary,
                history=history[:-1],
                user_prompt=user_prompt,
            ),
            media_type="text/event-stream",
        )

    messages = _build_chat_messages(system_prompt, summary, history)

    def gen() -> Iterator[bytes]:
        yield _sse("start", {"conversation_id": conversation_id})
        yield from _stream_tokens_and_persist(conversation_id, messages)

    return StreamingResponse(gen(), media_type="text/event-stream")
