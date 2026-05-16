"""Agent loop — receives percepts, calls LLM with tools, dispatches actions.

The agent loop is the bridge between the ModalityBus (perception/action)
and the InferenceProvider (thinking). It maintains conversation history
and routes tool calls through the bus.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from typing import TYPE_CHECKING

import httpx

from bus import ModalityBus
from chat_flow_log import phase_timer
from draft_queue import DraftQueue
from modality import CognitiveEvent, CognitiveIntent, ModalityType
from pipeline_state import PipelineState
from providers import AGENT_TOOLS, InferenceProvider
from schemas.bargein import BargeinContext

if TYPE_CHECKING:
    from channels import BrowserChannel

logger = logging.getLogger("mod3.agent_loop")

# Base system prompt — kernel context is appended dynamically
_BASE_SYSTEM_PROMPT = (
    "You are Cog, a voice assistant running on Mod³ (Apple Silicon, fully local). "
    "You respond using tool calls. "
    "Default to output(text=..., mode='both') for conversational replies — this speaks the text "
    "aloud AND shows it as a chat bubble in the dashboard so the operator can read along. "
    "Use mode='audio' only when the text would be redundant on screen (e.g. purely filler like "
    "'mm-hmm', 'got it', 'one moment'). "
    "Use mode='text' only when the content is better read than heard (code, lists, links, "
    "structured data) and you are not speaking it. "
    "Keep audio responses concise, 1-3 sentences. No markdown in audio text. Speak naturally. "
    "If the user asks something you can't do, say so briefly. "
    "Legacy tools speak() and send_text() still work but are deprecated."
)

# CogOS kernel endpoint for context enrichment
_COGOS_ENDPOINT = os.environ.get("COGOS_ENDPOINT", "http://localhost:6931")

# Bus endpoint for logging exchanges (observation channel)
_COGOS_BUS_ENDPOINT = f"{_COGOS_ENDPOINT}/v1/bus"


def _fetch_kernel_context() -> str:
    """Pull active context from CogOS kernel to enrich the system prompt.

    Returns a context block string, or empty string if kernel unavailable.
    This is the afferent path: kernel → local model.
    """
    try:
        resp = httpx.get(f"{_COGOS_ENDPOINT}/health", timeout=2.0)
        if resp.status_code != 200:
            return ""
        health = resp.json()

        parts = []
        identity = health.get("identity", "cog")
        state = health.get("state", "unknown")
        parts.append(f"Kernel identity: {identity}, state: {state}")

        # Try to get active session context
        try:
            ctx_resp = httpx.get(f"{_COGOS_ENDPOINT}/v1/context", timeout=2.0)
            if ctx_resp.status_code == 200:
                ctx = ctx_resp.json()
                nucleus = ctx.get("nucleus", "")
                if nucleus:
                    parts.append(f"Active nucleus: {nucleus}")
                process_state = ctx.get("state", "")
                if process_state:
                    parts.append(f"Process state: {process_state}")
        except Exception:
            pass

        # Check for barge-in context (what was Claude saying when interrupted?)
        signal_file = os.environ.get("BARGEIN_SIGNAL", "/tmp/mod3-barge-in.json")
        try:
            if os.path.exists(signal_file):
                with open(signal_file) as f:
                    signal = _json.load(f)
                interrupted = signal.get("interrupted")
                if interrupted:
                    delivered = interrupted.get("delivered_text", "")
                    _full = interrupted.get("full_text", "")
                    pct = interrupted.get("spoken_pct", 0)
                    parts.append(
                        f"[barge-in] Claude's speech was interrupted at {pct * 100:.0f}%. "
                        f'Delivered: "{delivered}". '
                        f"The user interrupted to say something — acknowledge and respond to them."
                    )
        except Exception:
            pass

        if parts:
            return "\n\nKernel context:\n" + "\n".join(f"- {p}" for p in parts)
        return ""
    except Exception:
        return ""


def _log_exchange_to_bus(user_text: str, assistant_text: str, provider_name: str):
    """Log the local model exchange to the CogOS bus (observation channel).

    This is the efferent path: local model → kernel → Claude can observe.
    """
    try:
        payload = {
            "type": "modality.voice.exchange",
            "from": f"mod3-reflex:{provider_name}",
            "payload": {
                "user": user_text,
                "assistant": assistant_text,
                "provider": provider_name,
                "timestamp": time.time(),
            },
        }
        httpx.post(
            _COGOS_BUS_ENDPOINT,
            json=payload,
            timeout=2.0,
        )
    except Exception as e:
        logger.debug("Failed to log exchange to bus: %s", e)


MAX_HISTORY = 50


class AgentLoop:
    """Conversational agent that receives percepts and acts through the bus."""

    def __init__(
        self,
        bus: ModalityBus,
        provider: InferenceProvider,
        pipeline_state: PipelineState,
        channel_id: str = "",
    ):
        self.bus = bus
        self.provider = provider
        self.pipeline_state = pipeline_state
        self.channel_id = channel_id
        self.conversation: list[dict[str, str]] = []
        self._channel_ref: BrowserChannel | None = None
        self._processing = False
        self.draft_queue = DraftQueue()
        self._speculative_context: list[dict[str, str]] = []  # Context for speculative inference
        self._human_speaking = False  # Whether human is currently speaking
        # A2: typed barge-in context prepared before the next turn, consumed by A3
        # for prompt injection. Set by _prepare_bargein_context() on the WS path.
        self._pending_bargein: BargeinContext | None = None

    async def handle_event(self, event: CognitiveEvent) -> None:
        """Called when a CognitiveEvent arrives from the channel."""
        if not event.content.strip():
            return

        if self._processing:
            logger.warning("agent busy, dropping: %s", event.content[:50])
            return

        self._processing = True
        try:
            await self._process(event)
        except Exception as e:
            logger.error("agent_loop error: %s", e, exc_info=True)
            try:
                if self._channel_ref:
                    await self._channel_ref.send_response_text(f"[error: {e}]")
                    await self._channel_ref.send_response_complete()
            except Exception:
                pass  # channel may be dead, don't block finally
        finally:
            self._processing = False

    async def _process(self, event: CognitiveEvent) -> None:
        """Core: event → provider → tool dispatch."""
        # A2: build typed BargeinContext from pipeline_state.last_interrupt (if any)
        # and stash on self._pending_bargein. A3 will consume it for prompt injection.
        self._prepare_bargein_context(user_text=event.content)

        self.conversation.append({"role": "user", "content": event.content})
        self._trim_history()

        # --- Phase: agent_dispatch ------------------------------------------
        # Covers the time from event arrival (start of _process) to provider
        # call dispatch: history trim, kernel context fetch, prompt assembly.
        _session_id = self.channel_id
        _msg_id = event.metadata.get("message_id", "")
        _turn_t0 = time.perf_counter()

        # Pick up the utterance-level correlation ID generated by the STT path
        # in channels.py._process_utterance(). For voice turns this is set on
        # the CognitiveEvent before it reaches the agent loop. For text turns
        # it will be absent (None). When the CogOSProvider returns its own W3C
        # trace_id we use that instead; this utterance_trace_id is the fallback
        # that covers Ollama/local providers which don't generate traceparent headers.
        _utterance_trace_id: str | None = event.metadata.get("utterance_trace_id")

        t_start = time.perf_counter()

        # Assemble system prompt with kernel context (afferent path)
        kernel_ctx = _fetch_kernel_context()
        system_prompt = _BASE_SYSTEM_PROMPT + kernel_ctx
        system_prompt = self._inject_pending_bargein(system_prompt)

        _agent_dispatch_ms = int((time.perf_counter() - t_start) * 1000)

        # --- Phase: provider_call -------------------------------------------
        # Emit provider_call phase manually so we can attach the trace_id from
        # the provider response (CogOSProvider stores it on response.raw as
        # "_mod3_trace_id"). This correlates the mod3 chat.phase.provider_call
        # event with the kernel's bus_traces kernel.chat.subspan.v1 events that
        # share the same W3C trace_id injected via the traceparent header.
        # RTVI T4 — bot-llm-started before inference begins.
        try:
            from audio_subscribers import get_default_audio_subscribers as _get_rtvi_subs

            _get_rtvi_subs().emit_bot_llm_started(self.channel_id)
        except Exception:
            pass  # best-effort; never block inference

        t_provider_start = time.perf_counter()
        response = await self.provider.chat(
            messages=self.conversation,
            tools=AGENT_TOOLS,
            system=system_prompt,
        )
        _provider_call_ms = int((time.perf_counter() - t_provider_start) * 1000)

        # RTVI T4 — bot-llm-stopped after inference returns.
        try:
            from audio_subscribers import get_default_audio_subscribers as _get_rtvi_subs

            _get_rtvi_subs().emit_bot_llm_stopped(self.channel_id)
        except Exception:
            pass  # best-effort
        t_llm = float(_provider_call_ms)
        # Extract the trace_id set by CogOSProvider — propagate to ALL phase
        # events in this turn so the trace panel can correlate them with the
        # kernel's bus_traces sub-spans that share the same W3C trace_id.
        # Fall back to the utterance-level correlation ID from the STT path when
        # the provider is Ollama/local and doesn't set _mod3_trace_id.
        _provider_trace_id: str | None = None
        if isinstance(response.raw, dict):
            _provider_trace_id = response.raw.get("_mod3_trace_id")
        if not _provider_trace_id and _utterance_trace_id:
            _provider_trace_id = _utterance_trace_id

        try:
            from chat_flow_log import get_chat_flow_log

            # Emit agent_dispatch now (after provider call so we have trace_id).
            get_chat_flow_log().emit_phase(
                phase_name="agent_dispatch",
                session_id=_session_id,
                message_id=_msg_id,
                duration_ms=_agent_dispatch_ms,
                trace_id=_provider_trace_id,
            )
            get_chat_flow_log().emit_phase(
                "provider_call",
                _session_id,
                _msg_id,
                _provider_call_ms,
                trace_id=_provider_trace_id,
            )
        except Exception:  # noqa: BLE001
            pass

        # --- Phase: tool_execute (per tool call) ----------------------------
        # Dispatch tool calls
        assistant_parts: list[str] = []

        for tc in response.tool_calls:
            # Unified output() tool — mode dispatch: "audio", "text", "both"
            # Deprecated speak() and send_text() are handled below for backwards compat.
            if tc.name == "output":
                text = tc.arguments.get("text", "")
                mode = tc.arguments.get("mode", "audio")
                if text:
                    assistant_parts.append(text)
                    async with phase_timer("tool_execute", _session_id, _msg_id, trace_id=_provider_trace_id):
                        # Text path: send to dashboard chat panel.
                        # mode="audio" also emits the text bubble (Option A: baseline parity).
                        # This lets the operator read along while audio plays, and ensures
                        # the speculative-output block is visible after a barge-in.
                        # mode="audio" with purely conversational filler that the model judges
                        # should not clutter the chat panel is the ONLY exception — model
                        # should use mode="both" (the new default) for all other cases.
                        if mode in ("text", "both", "audio"):
                            if self._channel_ref:
                                await self._channel_ref.send_response_text(text)
                        # Audio path: route through bus → VoiceEncoder → TTS
                        if mode in ("audio", "both"):
                            intent = CognitiveIntent(
                                modality=ModalityType.VOICE,
                                content=text,
                                target_channel=self.channel_id,
                                metadata={
                                    "voice": tc.arguments.get("voice")
                                    or (
                                        self._channel_ref.config.get("voice", "bm_lewis")
                                        if self._channel_ref
                                        else "bm_lewis"
                                    ),
                                    "speed": tc.arguments.get("speed")
                                    or (self._channel_ref.config.get("speed", 1.25) if self._channel_ref else 1.25),
                                    "session_id": _session_id,
                                    "message_id": _msg_id,
                                    # Propagate trace_id so VoiceEncoder.encode() can attach
                                    # it to tts_synthesize and tts_playback_start phase events.
                                    "trace_id": _provider_trace_id,
                                },
                            )
                            self.bus.act(intent, channel=self.channel_id)

            elif tc.name == "speak":
                # DEPRECATED: use output(text=..., mode="audio") instead
                text = tc.arguments.get("text", "")
                if text:
                    assistant_parts.append(text)
                    async with phase_timer("tool_execute", _session_id, _msg_id, trace_id=_provider_trace_id):
                        # Show text in chat panel
                        if self._channel_ref:
                            await self._channel_ref.send_response_text(text)
                        # Route through bus → VoiceEncoder → TTS → channel.deliver
                        intent = CognitiveIntent(
                            modality=ModalityType.VOICE,
                            content=text,
                            target_channel=self.channel_id,
                            metadata={
                                "voice": self._channel_ref.config.get("voice", "bm_lewis")
                                if self._channel_ref
                                else "bm_lewis",
                                "speed": self._channel_ref.config.get("speed", 1.25) if self._channel_ref else 1.25,
                                # Pass through phase-timing identifiers so VoiceEncoder
                                # can attribute tts_synthesize events to this turn.
                                "session_id": _session_id,
                                "message_id": _msg_id,
                                "trace_id": _provider_trace_id,
                            },
                        )
                        # Fire-and-forget: bus.act(blocking=False) returns QueuedJob immediately,
                        # OutputQueue drain thread handles TTS encoding + delivery.
                        self.bus.act(intent, channel=self.channel_id)

            elif tc.name == "send_text":
                # DEPRECATED: use output(text=..., mode="text") instead
                text = tc.arguments.get("content", "") or tc.arguments.get("text", "")
                if text:
                    assistant_parts.append(text)
                    async with phase_timer("tool_execute", _session_id, _msg_id, trace_id=_provider_trace_id):
                        if self._channel_ref:
                            await self._channel_ref.send_response_text(text)

        # Fallback: if provider returned text but no tool calls, auto-speak
        if not response.tool_calls and response.text:
            text = response.text
            assistant_parts.append(text)
            async with phase_timer("tool_execute", _session_id, _msg_id, trace_id=_provider_trace_id):
                if self._channel_ref:
                    await self._channel_ref.send_response_text(text)
                intent = CognitiveIntent(
                    modality=ModalityType.VOICE,
                    content=text,
                    target_channel=self.channel_id,
                    metadata={
                        "voice": self._channel_ref.config.get("voice", "bm_lewis") if self._channel_ref else "bm_lewis",
                        "speed": self._channel_ref.config.get("speed", 1.25) if self._channel_ref else 1.25,
                        "session_id": _session_id,
                        "message_id": _msg_id,
                        "trace_id": _provider_trace_id,
                    },
                )
                self.bus.act(intent, channel=self.channel_id)

        # Update conversation history
        if assistant_parts:
            assistant_text = " ".join(assistant_parts)
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )

            # RTVI T4 — bot-transcription after response is complete.
            try:
                from audio_subscribers import get_default_audio_subscribers as _get_rtvi_subs

                _get_rtvi_subs().emit_bot_transcription(self.channel_id, assistant_text, is_final=True)
            except Exception:
                pass  # best-effort; ACP and channel delivery above are the primary paths

            # Log exchange to CogOS bus (observation channel — Claude can see this)
            _log_exchange_to_bus(event.content, assistant_text, self.provider.name)

        # --- Phase: turn_total ----------------------------------------------
        _turn_total_ms = int((time.perf_counter() - _turn_t0) * 1000)
        try:
            from chat_flow_log import get_chat_flow_log

            get_chat_flow_log().emit_phase(
                phase_name="turn_total",
                session_id=_session_id,
                message_id=_msg_id,
                duration_ms=_turn_total_ms,
                trace_id=_provider_trace_id,
            )
        except Exception:  # noqa: BLE001
            pass

        # Signal completion
        if self._channel_ref:
            await self._channel_ref.send_response_complete(
                metrics={"llm_ms": round(t_llm, 1), "provider": self.provider.name}
            )

    async def speculative_infer(self, committed_text: str) -> None:
        """D2: Speculative inference trigger.

        When T3 commits a sentence while the human is still speaking,
        launch background inference with context-so-far. Store result
        in the DraftQueue. Does NOT play — just buffers.
        """
        if not committed_text.strip():
            return

        logger.info("speculative_infer: '%s'", committed_text[:80])

        # Build speculative conversation with committed text so far
        spec_messages = list(self.conversation) + [
            {"role": "user", "content": committed_text},
        ]

        try:
            t_start = time.perf_counter()
            kernel_ctx = _fetch_kernel_context()
            system_prompt = _BASE_SYSTEM_PROMPT + kernel_ctx

            response = await self.provider.chat(
                messages=spec_messages,
                tools=AGENT_TOOLS,
                system=system_prompt,
            )

            t_ms = (time.perf_counter() - t_start) * 1000

            # Extract response text from tool calls (output, speak, send_text)
            response_text = ""
            for tc in response.tool_calls:
                if tc.name == "output":
                    response_text += tc.arguments.get("text", "") + " "
                elif tc.name == "speak":
                    response_text += tc.arguments.get("text", "") + " "
                elif tc.name == "send_text":
                    response_text += (tc.arguments.get("content") or tc.arguments.get("text", "")) + " "
            if not response_text and response.text:
                response_text = response.text

            response_text = response_text.strip()
            if not response_text:
                return

            # Add to draft queue
            import hashlib

            ctx_hash = hashlib.md5(committed_text.encode()).hexdigest()[:8]
            block = self.draft_queue.add_block(
                text=response_text,
                context_hash=ctx_hash,
                generation_ms=t_ms,
            )

            logger.info(
                "speculative block %s: '%s' (%.0fms)",
                block.id,
                response_text[:60],
                t_ms,
            )

            # F2: Speculative TTS pre-synthesis
            # Generate audio immediately but don't play
            await self._presynthesise_block(block)

            # Notify dashboard of draft queue state
            if self._channel_ref:
                await self._channel_ref.ws.send_json(
                    {
                        "type": "draft_queue",
                        "blocks": [b.to_dict() for b in self.draft_queue.get_pending()],
                    }
                )

        except Exception as e:
            logger.debug("speculative_infer failed: %s", e)

    async def self_barge_snip(self, block_id: str) -> bool:
        """E1: Remove a queued block that's no longer relevant."""
        result = self.draft_queue.snip(block_id)
        if result:
            logger.info("self-barge: snipped block %s", block_id)
            await self._push_draft_queue_state()
        return result

    async def self_barge_inject(self, position: int, text: str) -> None:
        """E1: Insert a new block at position."""
        block = self.draft_queue.inject(position, text)
        logger.info("self-barge: injected block %s at pos %d", block.id, position)
        # Pre-synthesize the new block
        await self._presynthesise_block(block)
        await self._push_draft_queue_state()

    async def self_barge_revise(self, block_id: str, new_text: str) -> bool:
        """E1: Replace a block's content and re-synthesize TTS."""
        result = self.draft_queue.revise(block_id, new_text)
        if result:
            logger.info("self-barge: revised block %s -> '%s'", block_id, new_text[:60])
            # Find the block and re-synthesize
            for block in self.draft_queue.all_blocks:
                if block.id == block_id:
                    await self._presynthesise_block(block)
                    break
            await self._push_draft_queue_state()
        return result

    async def _push_draft_queue_state(self) -> None:
        """Push current draft queue state to the dashboard."""
        if self._channel_ref:
            try:
                await self._channel_ref.ws.send_json(
                    {
                        "type": "draft_queue",
                        "blocks": [b.to_dict() for b in self.draft_queue.all_blocks],
                    }
                )
            except Exception:
                pass

    async def invalidate_stale_drafts(self, new_context: str) -> int:
        """D3: Draft block invalidation.

        When a new T3 sentence arrives, check if existing draft blocks
        are still valid given the updated context. Mark stale ones.

        Uses context hash comparison: if a block was generated with
        different context than what we have now, it's potentially stale.

        Returns count of invalidated blocks.
        """
        import hashlib

        new_hash = hashlib.md5(new_context.encode()).hexdigest()[:8]
        invalidated = 0

        for block in self.draft_queue.get_pending():
            if block.context_hash and block.context_hash != new_hash:
                self.draft_queue.invalidate(block.id)
                invalidated += 1
                logger.info("invalidated stale draft block %s (context changed)", block.id)

        if invalidated > 0 and self._channel_ref:
            try:
                await self._channel_ref.ws.send_json(
                    {
                        "type": "draft_queue",
                        "blocks": [b.to_dict() for b in self.draft_queue.all_blocks],
                    }
                )
            except Exception:
                pass

        return invalidated

    async def _presynthesise_block(self, block) -> None:
        """F2: Pre-synthesize TTS audio for a draft block.

        Generates audio immediately and attaches it to the block.
        Ready for instant playback when the human stops speaking.
        """
        from modules.voice import _encode_wav

        try:
            voice = "bm_lewis"
            speed = 1.25
            if self._channel_ref:
                voice = self._channel_ref.config.get("voice", "bm_lewis")
                speed = self._channel_ref.config.get("speed", 1.25)

            def _synth():
                from engine import synthesize

                samples, sample_rate = synthesize(
                    block.text,
                    voice=voice,
                    speed=speed,
                )
                wav_bytes = _encode_wav(samples, sample_rate)
                duration = len(samples) / sample_rate
                return wav_bytes, duration

            wav_bytes, duration = await asyncio.to_thread(_synth)
            block.tts_audio = wav_bytes
            block.tts_duration_sec = duration
            logger.info("pre-synthesized block %s: %.1fs audio", block.id, duration)

        except Exception as e:
            logger.debug("pre-synthesis failed for block %s: %s", block.id, e)

    async def background_validate_drafts(self, latest_user_text: str) -> None:
        """E2: Background validation loop.

        After each new human sentence, re-evaluate all queued draft blocks.
        Snips/revises if context has invalidated them. This runs between
        TTS synthesis and playback — the revision window.
        """
        pending = self.draft_queue.get_pending()
        if not pending:
            return

        logger.info("background_validate: checking %d pending blocks", len(pending))

        # First, invalidate any blocks whose context is clearly stale
        await self.invalidate_stale_drafts(latest_user_text)

        # Then re-evaluate remaining valid blocks
        still_pending = self.draft_queue.get_pending()
        if not still_pending:
            return

        # Build context with latest human input
        _check_messages = list(self.conversation) + [
            {"role": "user", "content": latest_user_text},
        ]

        for block in still_pending:
            try:
                # Quick relevance check: ask the model if this block is still appropriate
                check_prompt = (
                    f'Given the user just said: "{latest_user_text}"\n'
                    f"Is this planned response still appropriate? "
                    f'Response: "{block.text}"\n'
                    f"Answer KEEP or REVISE in one word."
                )

                response = await self.provider.chat(
                    messages=[{"role": "user", "content": check_prompt}],
                    tools=[],
                    system="You are evaluating whether a planned response is still valid. Answer KEEP or REVISE.",
                )

                answer = (response.text or "").strip().upper()
                if "REVISE" in answer:
                    logger.info("background_validate: block %s needs revision", block.id)
                    self.draft_queue.invalidate(block.id)
                else:
                    logger.debug("background_validate: block %s still valid", block.id)

            except Exception as e:
                logger.debug("background_validate error for block %s: %s", block.id, e)

        await self._push_draft_queue_state()

    def _prepare_bargein_context(self, user_text: str | None) -> None:
        """Read pipeline_state.last_interrupt and stash a typed BargeinContext.

        Called at the top of each WS turn. If the previous assistant reply was
        interrupted (and the interrupt is still fresh, < 30s), build a
        BargeinContext via the A1 schema and store it on ``self._pending_bargein``
        for A3 to pick up during prompt construction. Clears last_interrupt so
        the next turn does not re-consume a stale record.
        """
        info = self.pipeline_state.last_interrupt
        if info is None:
            self._pending_bargein = None
            return

        # Only use recent interrupts (within last 30 seconds)
        if time.time() - info.timestamp > 30:
            # Stale — clear and skip.
            with self.pipeline_state._lock:
                self.pipeline_state._last_interrupt = None
            self._pending_bargein = None
            return

        # Consume the interrupt so we don't re-inject it on subsequent turns.
        # pipeline_state has no public consume helper yet; clear the private
        # slot under its lock (matches the pre-existing pattern on this path).
        with self.pipeline_state._lock:
            self.pipeline_state._last_interrupt = None

        self._pending_bargein = BargeinContext.from_interrupt_info(
            info,
            source="browser_vad",
            user_said=user_text or None,
        )

    def _inject_pending_bargein(self, system_prompt: str) -> str:
        """Append the pending BargeinContext (if any) to the system prompt.

        Consumes ``self._pending_bargein`` so it does not leak into subsequent
        turns. Returns the prompt unchanged if no barge-in is pending.
        """
        pending = self._pending_bargein
        if pending is None:
            return system_prompt
        self._pending_bargein = None
        return system_prompt + "\n\n" + pending.format_for_prompt()

    def _trim_history(self) -> None:
        """Keep conversation within MAX_HISTORY messages."""
        if len(self.conversation) > MAX_HISTORY:
            self.conversation = self.conversation[-MAX_HISTORY:]
