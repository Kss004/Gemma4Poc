from __future__ import annotations

import os
import sys
import threading
from typing import Dict, Iterator, List, Optional

# On Windows the cu124 llama-cpp-python wheel's ggml-cuda.dll depends on the
# CUDA runtime DLLs (cudart64_12.dll, cublas64_12.dll, ...). Those ship in the
# nvidia-*-cu12 pip packages under site-packages/nvidia/<lib>/bin. llama_cpp
# loads its DLLs with LoadLibraryEx flags=0 (winmode=0), whose search order
# honors PATH (not os.add_dll_directory), so we prepend every nvidia bin dir
# to PATH BEFORE importing llama_cpp.
if sys.platform == "win32":
    import glob

    try:
        import site

        seen: list[str] = []
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            for bin_dir in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
                if bin_dir not in seen:
                    seen.append(bin_dir)
                    try:
                        os.add_dll_directory(bin_dir)
                    except (OSError, FileNotFoundError):
                        pass
        if seen:
            os.environ["PATH"] = os.pathsep.join(seen) + os.pathsep + os.environ["PATH"]
    except Exception:
        pass

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
