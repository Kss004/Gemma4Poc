import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
import json

client = TestClient(app)

def test_new_conversation():
    response = client.post("/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert len(data["conversation_id"]) > 0

def test_get_conversation_detail():
    # First create one
    post_resp = client.post("/api/conversations")
    cid = post_resp.json()["conversation_id"]

    # Then get it
    get_resp = client.get(f"/api/conversations/{cid}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["conversation_id"] == cid
    assert "messages" in data
    assert isinstance(data["messages"], list)

def test_chat_creation():
    # Test chat validation
    response = client.post(
        "/api/chat",
        json={
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": ""
        }
    )
    assert response.status_code == 400
    assert "user_prompt is required" in response.text

def test_chat_streaming():
    # We will mock the model stream or just test the RAG missing doc error
    response = client.post(
        "/api/chat",
        json={
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Hello",
            "mode": "rag"
        }
    )
    assert response.status_code == 400
    assert "doc_id is required" in response.text

    response2 = client.post(
        "/api/chat",
        json={
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Hello",
            "mode": "rag",
            "doc_id": "invalid_doc_id"
        }
    )
    assert response2.status_code == 404
    assert "Unknown doc_id" in response2.text

from unittest.mock import patch, MagicMock

def test_upload_document(tmp_path):
    # Create a dummy PDF content mock
    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%EOF")
    
    with patch("app.main.document_store.ingest_pdf") as mock_ingest:
        # Create a mock Document object to return
        mock_doc = MagicMock()
        mock_doc.doc_id = "test_doc_123"
        mock_doc.filename = "dummy.pdf"
        mock_doc.page_count = 5
        mock_doc.sections = [1, 2, 3] # Used just for len(doc.sections)
        mock_ingest.return_value = mock_doc
        
        with open(fake_pdf, "rb") as f:
            response = client.post(
                "/api/documents", 
                files={"file": ("dummy.pdf", f, "application/pdf")}
            )
        
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "test_doc_123"
    assert data["filename"] == "dummy.pdf"
    assert data["page_count"] == 5
    assert data["section_count"] == 3

def test_upload_invalid_document(tmp_path):
    fake_txt = tmp_path / "dummy.txt"
    fake_txt.write_bytes(b"Hello World")
    
    with open(fake_txt, "rb") as f:
        response = client.post(
            "/api/documents", 
            files={"file": ("dummy.txt", f, "text/plain")}
        )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.text

def test_chat_success_stream():
    # Test valid chat session handling via generator
    response = client.post(
        "/api/chat",
        json={
            "system_prompt": "Reply 'hi' only.",
            "user_prompt": "Hello",
            "mode": "chat"
        }
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    
    # Check Server-Sent Events structure
    assert "event: start\n" in content
    assert "event: done\n" in content
    assert "data: {\"conversation_id\":" in content
