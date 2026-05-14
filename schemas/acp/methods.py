"""schemas.acp.methods — ACP request/response parameter models.

Each ACP method has a Params and Result type. These are carried inside the
JSON-RPC envelope (``JsonRpcRequest.params`` / ``JsonRpcResponse.result``).

Methods implemented:
  initialize        — capability negotiation
  session/new       — create a new session
  session/prompt    — submit a user prompt and stream the response
  session/cancel    — cancel an in-flight prompt (notification, no response)

Reference: https://github.com/zed-industries/agent-client-protocol
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .content import ContentBlock


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class ClientCapabilities(_Base):
    """Capabilities the client declares to the agent."""

    fs: dict[str, Any] = Field(default_factory=dict)
    terminal: bool = False


class ClientInfo(_Base):
    """Identifying information about the connecting client."""

    name: str = "mod3-dashboard"
    version: str = "1.0"


class InitializeParams(_Base):
    """Parameters for the ``initialize`` request.

    Wire shape::

        {
          "protocolVersion": 1,
          "clientCapabilities": {"fs": {}, "terminal": false},
          "clientInfo": {"name": "mod3-dashboard", "version": "1.0"}
        }
    """

    protocolVersion: int = 1
    clientCapabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)
    clientInfo: ClientInfo = Field(default_factory=ClientInfo)


class PromptCapabilities(_Base):
    """Agent's declared prompt capabilities."""

    audio: bool = False
    image: bool = False
    embeddedContext: bool = False


class AgentCapabilities(_Base):
    """Capabilities the agent declares to the client."""

    promptCapabilities: PromptCapabilities = Field(default_factory=PromptCapabilities)
    sessionCapabilities: dict[str, Any] = Field(default_factory=dict)


class AgentInfo(_Base):
    """Identifying information about the agent (mod3)."""

    name: str = "mod3"
    title: str = "Mod3 — Voice Modality Provider"
    version: str = "0.4.0"


class InitializeResult(_Base):
    """Result returned by the ``initialize`` method.

    Wire shape (per ACP spec)::

        {
          "protocolVersion": 1,
          "agentCapabilities": {
            "promptCapabilities": {"audio": false, "image": false, "embeddedContext": false},
            "sessionCapabilities": {}
          },
          "agentInfo": {"name": "mod3", "title": "...", "version": "..."},
          "authMethods": []
        }

    The earlier shape omitted ``protocolVersion`` and ``agentInfo`` which the
    ACP spec lists as required client-side validation fields; some clients
    refuse to proceed without them. Restored 2026-05-13.
    """

    protocolVersion: int = 1
    agentCapabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    agentInfo: AgentInfo = Field(default_factory=AgentInfo)
    authMethods: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# session/new
# ---------------------------------------------------------------------------


class McpServer(_Base):
    """An MCP server declaration passed by the client (may be empty list)."""

    name: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class SessionNewParams(_Base):
    """Parameters for the ``session/new`` request.

    Wire shape::

        {"cwd": "/", "mcpServers": []}
    """

    cwd: str = "/"
    mcpServers: list[McpServer] = Field(default_factory=list)


class SessionNewResult(_Base):
    """Result returned by the ``session/new`` method.

    Wire shape::

        {"sessionId": "mod3-<uuid>"}
    """

    sessionId: str


# ---------------------------------------------------------------------------
# session/prompt
# ---------------------------------------------------------------------------


class SessionPromptParams(_Base):
    """Parameters for the ``session/prompt`` request.

    ``prompt`` is a list of content blocks. For text-only clients, a single
    ``TextContent`` block is sufficient.

    Wire shape::

        {
          "sessionId": "mod3-<uuid>",
          "prompt": [{"type": "text", "text": "Hello!"}]
        }
    """

    sessionId: str
    prompt: list[ContentBlock] = Field(default_factory=list)


class SessionPromptResult(_Base):
    """Final result returned after a ``session/prompt`` completes.

    Streaming chunks are delivered via ``session/update`` notifications
    while the prompt is in-flight; this result signals completion.

    Wire shape::

        {"stopReason": "end_turn"}
    """

    stopReason: str = "end_turn"


# ---------------------------------------------------------------------------
# session/cancel  (notification — no result)
# ---------------------------------------------------------------------------


class SessionCancelParams(_Base):
    """Parameters for the ``session/cancel`` notification.

    Sent by the client as a JSON-RPC notification (no ``id``, no response).

    Wire shape::

        {"sessionId": "mod3-<uuid>"}
    """

    sessionId: str


__all__ = [
    "AgentCapabilities",
    "ClientCapabilities",
    "ClientInfo",
    "InitializeParams",
    "InitializeResult",
    "McpServer",
    "PromptCapabilities",
    "SessionCancelParams",
    "SessionNewParams",
    "SessionNewResult",
    "SessionPromptParams",
    "SessionPromptResult",
]
