"""schemas.acp.notifications — ACP server-to-client notification models.

ACP uses JSON-RPC 2.0 notifications (no ``id``) to stream agent responses
back to the client during a ``session/prompt`` call.

The primary notification is ``session/update``, which wraps the update
payload inside an ``update: {}`` envelope per the ACP spec
([[reference/acp-protocol-spec]] — schema.json ``SessionNotification``).

Wire shape per spec::

    {
      "jsonrpc": "2.0",
      "method": "session/update",
      "params": {
        "sessionId": "mod3-<uuid>",
        "update": {
          "sessionUpdate": "agent_message_chunk",
          "content": {"type": "text", "text": "Hello "}
        }
      }
    }

Prior to this fix (flagged in acp-spec-research.md, 2026-05-19), mod3 emitted
a FLAT params shape (``sessionUpdate`` and ``content`` at the top level of
params, not nested under ``update``). The dashboard's acp-transport.js
agreed with the old flat shape, so both sides were internally consistent but
deviated from the spec. Any compliant external ACP client (e.g. Zed pointed
at mod3 as a custom agent) would misparse the notifications.

Both the server emission (this file) and the client parser
(dashboard/acp-transport.js ``_handleNotification``) were updated together to
use the spec-compliant nested form. See: [[reference/acp-protocol-spec]],
[[semantic/research/acp-redesign-decision-2026-05-19]].

Reference: https://github.com/agentclientprotocol/agent-client-protocol
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .content import ContentBlock, TextContent


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------------------------------------------------------------------------
# session/update
# ---------------------------------------------------------------------------

# The ``sessionUpdate`` discriminator values from the ACP spec.
SessionUpdateKind = Literal[
    "agent_message_chunk",
    "user_message_chunk",
    "thought_chunk",
    "plan",
    "tool_use",
    "tool_result",
    "error",
]


class SessionUpdatePayload(_Base):
    """The inner ``update`` object inside a ``session/update`` notification params.

    Per [[reference/acp-protocol-spec]] (schema.json ``SessionUpdate``):
    ``sessionUpdate`` is the discriminator; ``content`` carries the chunk.
    For ``agent_message_chunk``, ``content`` is a ``TextContent`` block
    with the incremental text.

    This is the nested ``update`` object, not the full params envelope.
    The full params envelope is ``SessionUpdateParams``.

    Wire shape (the ``update`` field value)::

        {
          "sessionUpdate": "agent_message_chunk",
          "content": {"type": "text", "text": "Hello "}
        }
    """

    sessionUpdate: SessionUpdateKind = "agent_message_chunk"
    # content is a single ContentBlock for message chunks; may be None for
    # other update kinds (e.g. plan / thought).
    content: ContentBlock | None = None
    # Forward-compat — extra fields like stopReason are passed through.


class SessionUpdateParams(_Base):
    """The full ``params`` envelope for a ``session/update`` notification.

    Per [[reference/acp-protocol-spec]] (schema.json ``SessionNotification``):
    params must be ``{sessionId, update: SessionUpdate}``. The discriminator
    and content are nested under ``update``, not flat in params.

    Wire shape (the ``params`` field value)::

        {
          "sessionId": "mod3-<uuid>",
          "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Hello "}
          }
        }
    """

    sessionId: str = ""
    update: SessionUpdatePayload = Field(default_factory=SessionUpdatePayload)


class SessionUpdateNotification(_Base):
    """A ``session/update`` notification sent by the server.

    This is the JSON-RPC notification envelope; params is a
    ``SessionUpdateParams`` with the spec-compliant nested shape.

    Wire shape per ACP spec ([[reference/acp-protocol-spec]])::

        {
          "jsonrpc": "2.0",
          "method": "session/update",
          "params": {
            "sessionId": "mod3-<uuid>",
            "update": {
              "sessionUpdate": "agent_message_chunk",
              "content": {"type": "text", "text": "Hello "}
            }
          }
        }

    NOTE: mod3 previously emitted a flat params shape where ``sessionUpdate``
    and ``content`` appeared directly in params (not nested under ``update``).
    This was a wire-shape divergence from the ACP spec, flagged in
    acp-spec-research.md (2026-05-19). The dashboard's acp-transport.js parser
    was updated in the same commit to consume the spec-compliant nested shape.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["session/update"] = "session/update"
    params: SessionUpdateParams = Field(default_factory=SessionUpdateParams)

    @classmethod
    def text_chunk(cls, *, session_id: str, text: str) -> "SessionUpdateNotification":
        """Convenience constructor for an agent_message_chunk with text content."""
        return cls(
            params=SessionUpdateParams(
                sessionId=session_id,
                update=SessionUpdatePayload(
                    sessionUpdate="agent_message_chunk",
                    content=TextContent(text=text),
                ),
            )
        )


# Backward-compat alias — code that imported SessionUpdatePayload expecting the
# old flat-params model should be updated to SessionUpdateParams (the envelope)
# or SessionUpdatePayload (the inner update object). This alias preserves
# importability while the rename propagates.
_SessionUpdatePayloadLegacy = SessionUpdatePayload


__all__ = [
    "SessionUpdateKind",
    "SessionUpdateNotification",
    "SessionUpdateParams",
    "SessionUpdatePayload",
]
