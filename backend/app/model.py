from __future__ import annotations

import threading
from typing import Dict, Iterator, List, Optional

import lmstudio as lms
from lmstudio import (
    LMStudioModelNotFoundError,
    LMStudioWebsocketError,
    LlmPredictionConfigDict,
)

from .config import (
    LMS_MODEL_KEY,
    MAX_NEW_TOKENS,
    N_CTX,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)


class ModelNotProvisionedError(RuntimeError):
    """Raised when the requested model key is not registered with LM Studio."""


class DaemonUnreachableError(RuntimeError):
    """Raised when the LM Studio daemon cannot be reached."""


def _build_chat(messages: List[Dict[str, str]]) -> lms.Chat:
    """Convert OpenAI-style role/content messages into an lmstudio Chat."""
    chat = lms.Chat()
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            chat.add_system_prompt(content)
        elif role == "user":
            chat.add_user_message(content)
        elif role == "assistant":
            chat.add_assistant_response(content)
        else:
            raise ValueError(f"Unsupported message role: {role!r}")
    return chat


class ModelService:
    """Thin wrapper over the lmstudio Python SDK.

    Uses the SDK's auto / convenience mode (``lms.llm(...)``), which delegates
    model lifecycle to the LM Studio daemon. The daemon must be running
    (``lms daemon up``) and the referenced model must already be registered
    (``lms import <gguf>`` or ``lms get <repo>``).

    The model handle is acquired lazily on the first call to
    :meth:`generate` or :meth:`stream`, so the backend process can start
    even when the daemon is temporarily unreachable.
    """

    def __init__(self) -> None:
        if not LMS_MODEL_KEY:
            raise RuntimeError(
                "LMS_MODEL_KEY is not set. Run `lms ls` to see available models "
                "and export LMS_MODEL_KEY=<key> before starting the backend."
            )
        self._lock = threading.Lock()
        self._llm: Optional[lms.LLM] = None

    def _get_llm(self) -> lms.LLM:
        if self._llm is not None:
            return self._llm
        load_config = {"contextLength": N_CTX}
        try:
            self._llm = lms.llm(LMS_MODEL_KEY, config=load_config)
        except LMStudioWebsocketError as exc:
            raise DaemonUnreachableError(
                "Cannot reach the LM Studio daemon. Start it with `lms daemon up` "
                "and try again."
            ) from exc
        except LMStudioModelNotFoundError as exc:
            try:
                available = [m.model_key for m in lms.list_downloaded_models()]
            except Exception:
                available = []
            raise ModelNotProvisionedError(
                f"Model {LMS_MODEL_KEY!r} is not registered with LM Studio. "
                f"Available: {available or 'unknown (run `lms ls`)'}. "
                "Register it with `lms import <path-to-gguf>` or `lms get <repo>`."
            ) from exc
        return self._llm

    def _prediction_config(
        self, max_tokens: Optional[int] = None
    ) -> LlmPredictionConfigDict:
        return {
            "temperature": TEMPERATURE,
            "topPSampling": TOP_P,
            "topKSampling": TOP_K,
            "maxTokens": max_tokens if max_tokens is not None else MAX_NEW_TOKENS,
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return a full assistant response for the given message history."""
        chat = _build_chat(messages)
        with self._lock:
            llm = self._get_llm()
            result = llm.respond(
                chat,
                config=self._prediction_config(max_tokens),
            )
        return result.content.strip()

    def stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield response fragments as they arrive from the model."""
        chat = _build_chat(messages)
        with self._lock:
            llm = self._get_llm()
            prediction_stream = llm.respond_stream(
                chat,
                config=self._prediction_config(max_tokens),
            )
            for fragment in prediction_stream:
                text = fragment.content
                if text:
                    yield text
