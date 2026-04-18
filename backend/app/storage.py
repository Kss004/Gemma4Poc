import sqlite3
import time
import uuid
from typing import List, Optional, Tuple

from .config import DB_PATH


def _now_ts() -> float:
    return time.time()


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_conversation(title: Optional[str] = None) -> str:
    conversation_id = str(uuid.uuid4())
    ts = _now_ts()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations (id, title, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, title, "", ts, ts),
    )
    conn.commit()
    conn.close()
    return conversation_id


def touch_conversation(conversation_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (_now_ts(), conversation_id),
    )
    conn.commit()
    conn.close()


def get_conversation(
    conversation_id: str,
) -> Optional[Tuple[str, str, str, float, float]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, summary, created_at, updated_at FROM conversations WHERE id = ?",
        (conversation_id,),
    )
    row = cur.fetchone()
    conn.close()
    return row


def add_message(conversation_id: str, role: str, content: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, _now_ts()),
    )
    conn.commit()
    conn.close()


def list_messages(
    conversation_id: str, limit: Optional[int] = None
) -> List[Tuple[str, str, float]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sql = "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC"
    params = [conversation_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def update_summary(conversation_id: str, summary: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversations SET summary = ?, updated_at = ? WHERE id = ?",
        (summary, _now_ts(), conversation_id),
    )
    conn.commit()
    conn.close()
