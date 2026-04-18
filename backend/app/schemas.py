from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(default=None)
    system_prompt: str
    user_prompt: str
    mode: str = Field(default="chat")
    doc_id: Optional[str] = None


class Source(BaseModel):
    section_id: Optional[str] = None
    title: str
    pages: List[int] = []


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    section_count: int


class ConversationSummary(BaseModel):
    conversation_id: str


class ConversationResponse(BaseModel):
    conversation_id: str
    summary: str
    messages: List[dict]
