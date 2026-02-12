"""
OllamaChatModel — Lightweight Ollama LLM Wrapper
==================================================
Thin wrapper around Ollama's OpenAI-compatible endpoint.
No LangChain dependency — just plain OpenAI client calls.

Supports:
  - Text chat completion
  - Multimodal input (images via base64 data URIs)
  - <think>...</think> reasoning block stripping (Qwen3 etc.)
  - Step-based logging

Environment:
  OLLAMA_BASE_URL : Custom endpoint (default http://localhost:11434/v1/)

Usage::

    llm = OllamaChatModel(model_name="qwen2.5:7b")
    result = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Hello!"},
    ])
    print(result.content)
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


# ── Response dataclass ───────────────────────────────────────────────
@dataclass
class ChatResponse:
    """Simple response object returned by OllamaChatModel.invoke()."""
    content: str                            # clean text (thinking stripped)
    thinking: str = ""                      # extracted <think> block, if any
    raw: str = ""                           # original unprocessed content
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


# ── Message type alias ───────────────────────────────────────────────
# Messages are plain dicts following OpenAI chat format:
#   {"role": "system"|"user"|"assistant", "content": str | list[dict]}
Message = Dict[str, Any]


class OllamaChatModel:
    """
    Lightweight chat model backed by Ollama's OpenAI-compatible API.

    Parameters
    ----------
    model_name : str
        Ollama model tag, e.g. ``"qwen2.5:7b"`` or ``"kimi-k2.5:cloud"``.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        model_name: str = "qwen3-coder-next:cloud",
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self._step_name: str = ""

        self.client = OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
        )

    # ── Configuration helpers ────────────────────────────────────────
    def set_step_name(self, step_name: str):
        """Set current step label for logging (e.g. 'Story Expansion')."""
        self._step_name = step_name

    # ── Core API ─────────────────────────────────────────────────────
    def invoke(self, messages: List[Message], **kwargs: Any) -> ChatResponse:
        """
        Send a chat completion request and return a ``ChatResponse``.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list.  Content can be a plain string or
            a list of content parts (text + image_url) for multimodal input.
        **kwargs
            Extra params forwarded to the API (e.g. ``max_tokens``).

        Returns
        -------
        ChatResponse
        """
        # ── Log input ────────────────────────────────────────────────
        step = self._step_name or "LLM"
        logger.info(f"{'—'*50}")
        logger.info(f"[{step}]  model={self.model_name}")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            preview = self._preview_content(content)
            logger.debug(f"  [{role}]: {preview}")

        # ── Call API ─────────────────────────────────────────────────
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            **kwargs,
        )

        choice = response.choices[0]
        raw_content = choice.message.content or ""

        # Strip <think>...</think> blocks
        thinking, clean = self._parse_thinking(raw_content)

        # ── Log output ───────────────────────────────────────────────
        if thinking:
            logger.debug(f"  [thinking]: {thinking[:300]}")
        logger.info(f"  [output]: {clean[:200]}")
        logger.info(f"{'—'*50}")

        return ChatResponse(
            content=clean,
            thinking=thinking,
            raw=raw_content,
            model=self.model_name,
            usage={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            },
        )

    # ── Thinking block parser ────────────────────────────────────────
    @staticmethod
    def _parse_thinking(content: str) -> Tuple[str, str]:
        """
        Separate ``<think>…</think>`` reasoning traces from the response.

        Returns ``(thinking_text, clean_content)``.
        """
        if not content:
            return "", ""

        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        parts = pattern.findall(content)
        thinking = "\n".join(p.strip() for p in parts if p.strip())
        clean = pattern.sub("", content).strip()
        return thinking, clean

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _preview_content(content: Any, limit: int = 200) -> str:
        """Produce a short preview of message content for logging."""
        if isinstance(content, str):
            return content[:limit]
        if isinstance(content, list):
            # Multimodal: show text parts, mark images
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item["text"][:limit])
                elif item.get("type") == "image_url":
                    parts.append("[IMAGE]")
            return " | ".join(parts)
        return str(content)[:limit]
