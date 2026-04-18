from __future__ import annotations

import threading
from typing import Dict, Iterator, List, Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .config import (
    GGUF_FILE,
    GGUF_REPO,
    MAX_NEW_TOKENS,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


class ModelService:
    def __init__(self) -> None:
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
        kwargs = dict(
            model_path=model_path,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )
        if N_THREADS:
            kwargs["n_threads"] = N_THREADS
        try:
            self._llm = Llama(**kwargs)
        except ValueError as e:
            if "Failed to load model from file" in str(e):
                import sys
                print("\n=======================================================\n", file=sys.stderr)
                print(f"CRITICAL ERROR: Failed to load GGUF file.", file=sys.stderr)
                print(f"The downloaded file might be corrupted.", file=sys.stderr)
                print(f"Please delete the cache at:", file=sys.stderr)
                print(f"  {model_path}", file=sys.stderr)
                print(f"and restart the app to download it again.", file=sys.stderr)
                print("\n=======================================================\n", file=sys.stderr)
            raise
        self._lock = threading.Lock()

    def _sampling(self, **overrides) -> Dict:
        params = dict(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_NEW_TOKENS,
        )
        params.update({k: v for k, v in overrides.items() if v is not None})
        return params

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> str:
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=messages,
                stream=False,
                **self._sampling(max_tokens=max_tokens),
            )
        return out["choices"][0]["message"]["content"].strip()

    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        with self._lock:
            for chunk in self._llm.create_chat_completion(
                messages=messages,
                stream=True,
                **self._sampling(max_tokens=max_tokens),
            ):
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
