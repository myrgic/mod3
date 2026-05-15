"""Inference providers for the Mod³ agent loop.

Abstracts LLM backends behind an InferenceProvider protocol.
Each provider returns structured ToolCall responses, not streaming text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger("mod3.providers")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool invocation from the LLM."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    """Structured response from an inference provider."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""  # fallback plain text (no tool call)
    raw: dict | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ProviderResponse: ...


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "output",
            "description": (
                "Unified output tool. Use for all agent responses. "
                "mode='both' (default for conversational replies): speaks the text aloud via TTS "
                "AND shows it as a chat bubble in the dashboard. Use this for everything the "
                "operator should be able to read. "
                "mode='audio': speaks aloud AND shows text bubble (identical to 'both' — "
                "reserved for cases where the text would be redundant on screen, e.g. filler). "
                "mode='text': shows in chat panel only, no audio. Use for code, lists, links, "
                "structured data that is better read than heard."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The content to output. Required.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["both", "audio", "text"],
                        "description": "Delivery mode. Default 'both' (voice + chat bubble).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "[DEPRECATED] Use output(text=..., mode='audio') instead. Speak text aloud via TTS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak aloud",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_text",
            "description": "[DEPRECATED] Use output(text=..., mode='text') instead. Send text to the chat panel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text to display in chat",
                    }
                },
                "required": ["content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# MLX provider (in-process, Apple Silicon)
# ---------------------------------------------------------------------------


def _format_tools_for_prompt(tools: list[dict]) -> str:
    """Format tool definitions into a text block for Gemma's system prompt."""
    lines = ["You have access to the following tools:\n"]
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- **{name}**: {desc}")
        props = params.get("properties", {})
        required = set(params.get("required", []))
        if props:
            lines.append("  Parameters:")
            for pname, pinfo in props.items():
                req_marker = " (required)" if pname in required else ""
                lines.append(
                    f"    - {pname} ({pinfo.get('type', 'string')}): {pinfo.get('description', '')}{req_marker}"
                )
    lines.append(
        "\nTo call a tool, output exactly:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {<args>}}\n'
        "</tool_call>\n"
        "\nYou may make multiple tool calls. Every tool call must be wrapped "
        "in its own <tool_call> block."
    )
    return "\n".join(lines)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract <tool_call> JSON blocks from model output."""
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"text": args}
            if name:
                calls.append(ToolCall(name=name, arguments=args))
        except json.JSONDecodeError:
            logger.warning("mlx: failed to parse tool_call JSON: %s", match.group(1))
    return calls


class MlxProvider:
    """In-process Gemma inference on Apple Silicon via mlx-lm.

    Lazy-loads the model on first call. The model stays resident in memory
    for the lifetime of the process — no cold-start on subsequent calls.
    """

    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or os.environ.get("MLX_MODEL", "mlx-community/gemma-3-4b-it-4bit")
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return f"mlx/{self._model_id}"

    def _ensure_loaded(self) -> None:
        """Load model + tokenizer on first use (synchronous, called from thread)."""
        if self._model is not None:
            return
        from mlx_lm import load

        logger.info("mlx: loading model %s (first call, this may take a moment)", self._model_id)
        self._model, self._tokenizer = load(self._model_id)
        logger.info("mlx: model loaded successfully")

    def _generate_sync(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        system: str,
    ) -> ProviderResponse:
        """Run generation synchronously (meant to be called via asyncio.to_thread)."""
        from mlx_lm import generate

        self._ensure_loaded()

        # Build messages list with system prompt
        msgs = list(messages)
        system_parts: list[str] = []
        if system:
            system_parts.append(system)
        if tools:
            system_parts.append(_format_tools_for_prompt(tools))
        if system_parts:
            msgs = [{"role": "system", "content": "\n\n".join(system_parts)}] + msgs

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        max_tokens = int(os.environ.get("MLX_MAX_TOKENS", "512"))
        raw_output = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        # Parse tool calls from output
        tool_calls = _parse_tool_calls(raw_output)

        # Strip tool_call blocks from the text to get any remaining content
        text = _TOOL_CALL_RE.sub("", raw_output).strip()

        return ProviderResponse(
            tool_calls=tool_calls,
            text=text,
            raw={"model": self._model_id, "output": raw_output},
        )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ProviderResponse:
        return await asyncio.to_thread(self._generate_sync, messages, tools, system)


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


class OllamaProvider:
    """Ollama inference with native tool calling (validated with Gemma 4 E4B)."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
    ):
        self._endpoint = endpoint or os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        self._model = model or os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ProviderResponse:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        body: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "stream": False,
            "think": False,
        }
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{self._endpoint}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        raw_tool_calls = msg.get("tool_calls", [])
        content = msg.get("content", "")

        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except Exception:
                    args = {"text": args}
            if name:
                tool_calls.append(ToolCall(name=name, arguments=args))

        return ProviderResponse(tool_calls=tool_calls, text=content, raw=data)


# ---------------------------------------------------------------------------
# CogOS provider (OpenAI-compatible SSE)
# ---------------------------------------------------------------------------


class CogOSProvider:
    """CogOS kernel — OpenAI-compatible chat/completions with tool support.

    Routes inference through the kernel at localhost:6931, which applies the
    Tier-1 provider ladder: Eclipse 26b A4B (on LAN) > Darkstar local runtimes
    (off-LAN) > Ollama E4B (safety net). The dashboard always gets the best
    available model, not the in-process Gemma 3 4B fallback.

    Model routing is controlled by the COGOS_MODEL env var (default:
    "lmstudio-eclipse" — the kernel provider alias for Eclipse 26b A4B). Set to
    "local" for the Ollama baseline, "google/gemma-4-26b-a4b" for an explicit
    model id, or any other kernel provider alias.
    """

    def __init__(self, endpoint: str | None = None, model: str | None = None):
        self._endpoint = endpoint or os.environ.get("COGOS_ENDPOINT", "http://localhost:6931")
        self._model = model or os.environ.get("COGOS_MODEL", "lmstudio-eclipse")

    @property
    def name(self) -> str:
        return f"cogos/{self._model}"

    @staticmethod
    def _make_traceparent() -> tuple[str, str]:
        """Generate a W3C traceparent header and return (header_value, trace_id).

        Format: ``00-<trace_id_32hex>-<parent_id_16hex>-01``

        Both IDs are derived from independent UUID4 values so they are
        cryptographically random and unique per request. The returned
        trace_id (32 hex chars) can be stored on the ProviderResponse.raw
        dict so downstream phase events (chat_flow_log.emit_phase) can
        include it as a correlation field without an OTel SDK dependency.
        """
        trace_id = uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 hex chars
        # uuid4().hex is 32 chars; take first 16 for parent span id
        parent_id = uuid.uuid4().hex[:16]
        traceparent = f"00-{trace_id}-{parent_id}-01"
        return traceparent, trace_id

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ProviderResponse:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        body: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        # Generate a W3C traceparent for this request so mod3 phase events
        # (chat_flow_log.emit_phase) share a trace_id with the kernel's
        # bus_traces sub-spans (kernel.chat.subspan.v1). The kernel propagates
        # the traceparent via the OTel SDK when a collector is configured;
        # even without a collector, both sides record the same trace_id so an
        # operator can join them manually.
        traceparent, trace_id = self._make_traceparent()

        headers = {
            "X-UCP-Identity": '{"name":"cog"}',
            "X-Session-ID": "mod3-dashboard",
            "X-Origin": "mod3-dashboard",
            "traceparent": traceparent,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._endpoint}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        raw_tool_calls = msg.get("tool_calls", [])

        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except Exception:
                    args = {"text": args}
            if name:
                tool_calls.append(ToolCall(name=name, arguments=args))

        # Attach trace_id to raw so callers (e.g. agent loop) can forward it
        # to chat_flow_log.emit_phase as a correlation field.
        if isinstance(data, dict):
            data["_mod3_trace_id"] = trace_id

        return ProviderResponse(tool_calls=tool_calls, text=content, raw=data)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


async def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return r.status_code < 500
    except Exception:
        return False


def _mlx_available() -> bool:
    """Check whether mlx-lm is importable (Apple Silicon with mlx installed)."""
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


async def auto_detect_provider_async() -> InferenceProvider:
    """Probe available backends: CogOS kernel > Ollama > MLX (in-process fallback).

    Priority rationale (2026-05-15):
      1. CogOS kernel (localhost:6931) — canonical Tier-1 router; applies the
         3-state ladder: Eclipse 26b A4B (on LAN) > Darkstar local runtimes >
         Ollama E4B safety net. This is the correct inference path for the
         dashboard — the kernel routes to the best available provider, not the
         in-process 4B model.
      2. Ollama — local daemon available when kernel is down.
      3. MLX in-process — last resort only; bypasses the kernel's routing
         entirely and blocks the event loop with CPU-bound generation.

    Override via MOD3_PROVIDER env var:
      MOD3_PROVIDER=cogos    — force kernel (default behaviour)
      MOD3_PROVIDER=ollama   — force Ollama (skip kernel check)
      MOD3_PROVIDER=mlx      — force in-process MLX (debug/offline)
    """
    forced = os.environ.get("MOD3_PROVIDER", "").lower()

    if forced == "mlx":
        logger.info("auto-detect: MOD3_PROVIDER=mlx — using MLX (in-process, forced)")
        return MlxProvider()
    if forced == "ollama":
        ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        logger.info("auto-detect: MOD3_PROVIDER=ollama — using Ollama at %s (forced)", ollama_endpoint)
        return OllamaProvider(endpoint=ollama_endpoint)

    # Default: probe CogOS kernel first
    cogos_endpoint = os.environ.get("COGOS_ENDPOINT", "http://localhost:6931")
    if await _probe(f"{cogos_endpoint}/health"):
        logger.info("auto-detect: using CogOS kernel at %s (Tier-1 router)", cogos_endpoint)
        return CogOSProvider(endpoint=cogos_endpoint)

    # Fallback: Ollama local daemon
    ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
    if await _probe(f"{ollama_endpoint}/api/tags"):
        logger.info("auto-detect: kernel unreachable — falling back to Ollama at %s", ollama_endpoint)
        return OllamaProvider(endpoint=ollama_endpoint)

    # Last resort: in-process MLX (only if importable)
    if _mlx_available():
        logger.warning(
            "auto-detect: kernel and Ollama unreachable — falling back to in-process MLX "
            "(bypasses Tier-1 routing; set COGOS_ENDPOINT if kernel is on a non-default port)"
        )
        return MlxProvider()

    logger.warning("auto-detect: no provider found, defaulting to Ollama (may fail if Ollama is not running)")
    return OllamaProvider()


def auto_detect_provider() -> InferenceProvider:
    """Synchronous wrapper for auto-detection.

    Checks MOD3_PROVIDER override first (no async probing needed for forced
    selection), then falls back to async probing via asyncio.run().
    """
    import asyncio

    forced = os.environ.get("MOD3_PROVIDER", "").lower()

    if forced == "mlx":
        logger.info("auto-detect: MOD3_PROVIDER=mlx — using MLX (in-process, forced)")
        return MlxProvider()
    if forced == "ollama":
        ollama_endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        logger.info("auto-detect: MOD3_PROVIDER=ollama — using Ollama at %s (forced)", ollama_endpoint)
        return OllamaProvider(endpoint=ollama_endpoint)
    if forced == "cogos" or forced == "":
        # CogOS is the default; no fast-path shortcut — need async probe to confirm
        # kernel is reachable. Fall through to asyncio.run path.
        pass

    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(auto_detect_provider_async())

    # Called from an async context (can't use asyncio.run).
    # Kernel probe not possible here without an await; default to CogOS since
    # the kernel is almost always up. Caller can use MOD3_PROVIDER=ollama to
    # override if the kernel is known-down.
    cogos_endpoint = os.environ.get("COGOS_ENDPOINT", "http://localhost:6931")
    logger.info(
        "auto-detect: async context — defaulting to CogOS at %s "
        "(set MOD3_PROVIDER=ollama or MOD3_PROVIDER=mlx to override)",
        cogos_endpoint,
    )
    return CogOSProvider(endpoint=cogos_endpoint)
