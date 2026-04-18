import os
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PKG_DIR.parent

DATA_DIR = os.getenv("DATA_DIR", str(_REPO_DIR / "data"))
DOCS_DIR = os.path.join(DATA_DIR, "docs")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "chat.db"))

GGUF_REPO = os.getenv("GGUF_REPO", "unsloth/gemma-4-E2B-it-GGUF")
GGUF_FILE = os.getenv("GGUF_FILE", "gemma-4-E2B-it-Q4_K_M.gguf")

N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "0")) or None
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "64"))

RAG_MAX_HOPS = int(os.getenv("RAG_MAX_HOPS", "6"))
RAG_PAGE_READ_LIMIT = int(os.getenv("RAG_PAGE_READ_LIMIT", "10"))
SUMMARY_TRIGGER = int(os.getenv("SUMMARY_TRIGGER", "24"))
SUMMARY_EVERY = int(os.getenv("SUMMARY_EVERY", "10"))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "10"))
