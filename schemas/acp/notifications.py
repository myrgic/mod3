"""schemas.acp.notifications — ACP server-to-client notification models.

ACP uses JSON-RPC 2.0 notifications (no ``id``) to stream agent responses
back to the client during a ``session/prompt`` call.

The primary notification is ``session/update``, which carries a discriminated
``sessionUpdate`` field indicating the update kind, plus a ``content`` block
carrying the actual payload.

Reference: https://github.com/zed-industries/agent-client-protocol
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
    """The ``params`` carried inside a ``session/update`` notification.

    ``sessionUpdate`` is the discriminator; ``content`` carries the chunk.
    For ``agent_message_chunk``, ``content`` is a ``TextContent`` block
    with the incremental text.

    Wire shape (streaming text chunk)::

        {
          "sessionId": "mod3-<uuid>",
          "sessionUpdate": "agent_message_chunk",
          "content": {"type": "text", "text": "Hello "}
        }
    """

    sessionId: str = ""
    sessionUpdate: SessionUpdateKind = "agent_message_chunk"
    # content is a single ContentBlock for message chunks; may be None for
    # other update kinds (e.g. plan / thought).
    content: ContentBlock | None = None
    # Forward-compat — extra fields like stopReason are passed through.


class SessionUpdateNotification(_Base):
    """A ``session/update`` notification sent by the server.

    This is the JSON-RPC notification envelope; params is a
    ``SessionUpdatePayload``.

    Wire shape::

        {
          "jsonrpc": "2.0",
          "method": "session/update",
          "params": {
            "sessionId": "mod3-<uuid>",
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Hello "}
          }
        }
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["session/update"] = "session/update"
    params: SessionUpdatePayload = Field(default_factory=SessionUpdatePayload)

    @classmethod
    def text_chunk(cls, *, session_id: str, text: str) -> "SessionUpdateNotification":
        """Convenience constructor for an agent_message_chunk with text content."""
        return cls(
            params=SessionUpdatePayload(
                sessionId=session_id,
                sessionUpdate="agent_message_chunk",
                content=TextContent(text=text),
            )
        )


__all__ = [
    "SessionUpdateKind",
    "SessionUpdateNotification",
    "SessionUpdatePayload",
]
