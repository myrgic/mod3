"""schemas.acp.methods — ACP request/response parameter models.

Each ACP method has a Params and Result type. These are carried inside the
JSON-RPC envelope (``JsonRpcRequest.params`` / ``JsonRpcResponse.result``).

Methods implemented:
  initialize        — capability negotiation
  session/new       — create a new session
  session/prompt    — submit a user prompt and stream the response
  session/cancel    — cancel an in-flight prompt (notification, no response)
  session/list      — list available sessions (optional; requires sessionCapabilities.list)
  session/load      — retrieve state of a specific session (optional; requires loadSession)
  session/resume    — reconnect to an existing session (optional; requires sessionCapabilities.resume)
  authenticate      — auth handshake; no-op when authMethods is empty

Reference: https://github.com/agentclientprotocol/agent-client-protocol
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


class SessionCapabilities(_Base):
    """Session-level capabilities the agent declares.

    Gating flags for optional session methods:
      list   — agent supports ``session/list``
      resume — agent supports ``session/resume``
    """

    list: bool = False
    resume: bool = False


class AgentCapabilities(_Base):
    """Capabilities the agent declares to the client."""

    promptCapabilities: PromptCapabilities = Field(default_factory=PromptCapabilities)
    sessionCapabilities: SessionCapabilities = Field(default_factory=SessionCapabilities)
    loadSession: bool = False


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


# ---------------------------------------------------------------------------
# session/list  (optional — requires sessionCapabilities.list: true)
# ---------------------------------------------------------------------------


class SessionListParams(_Base):
    """Parameters for the ``session/list`` request.

    No required fields; the spec does not define filter parameters.

    Wire shape::

        {}
    """


class SessionListItem(_Base):
    """Metadata for a single session in the ``session/list`` response.

    Wire shape::

        {
          "sessionId": "default",
          "state": "idle",
          "participantId": "chaz",
          "participantType": "human"
        }
    """

    sessionId: str
    state: str = "idle"
    participantId: str = ""
    participantType: str = ""


class SessionListResult(_Base):
    """Result returned by the ``session/list`` method.

    Wire shape::

        {"sessions": [{...}, ...]}
    """

    sessions: list[SessionListItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# session/load  (optional — requires loadSession: true)
# ---------------------------------------------------------------------------


class SessionLoadParams(_Base):
    """Parameters for the ``session/load`` request.

    Wire shape::

        {"sessionId": "default"}
    """

    sessionId: str


class SessionLoadResult(_Base):
    """Result returned by the ``session/load`` method.

    Wire shape::

        {
          "sessionId": "default",
          "state": {
            "state": "idle",
            "participantId": "chaz",
            ...
          }
        }
    """

    sessionId: str
    state: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# session/resume  (optional — requires sessionCapabilities.resume: true)
# ---------------------------------------------------------------------------


class SessionResumeParams(_Base):
    """Parameters for the ``session/resume`` request.

    Wire shape::

        {"sessionId": "default"}
    """

    sessionId: str


class SessionResumeResult(_Base):
    """Result returned by the ``session/resume`` method.

    Wire shape::

        {"sessionId": "default"}

    On success the ACP session is bound to the named mod3 session_id so
    subsequent ``session/prompt`` calls fan to seats in that session.
    """

    sessionId: str


# ---------------------------------------------------------------------------
# authenticate  (called by client when authMethods is non-empty)
# ---------------------------------------------------------------------------


class AuthenticateParams(_Base):
    """Parameters for the ``authenticate`` request.

    ``methodId`` names the auth method the client selected from the
    ``authMethods`` list returned by ``initialize``.  When ``authMethods``
    is ``[]`` mod3 returns an immediate success and never calls this method.

    Wire shape::

        {"methodId": ""}
    """

    methodId: str = ""


class AuthenticateResult(_Base):
    """Result returned by the ``authenticate`` method.

    Wire shape::

        {"success": true}
    """

    success: bool = True


__all__ = [
    "AgentCapabilities",
    "AuthenticateParams",
    "AuthenticateResult",
    "ClientCapabilities",
    "ClientInfo",
    "InitializeParams",
    "InitializeResult",
    "McpServer",
    "PromptCapabilities",
    "SessionCapabilities",
    "SessionCancelParams",
    "SessionListItem",
    "SessionListParams",
    "SessionListResult",
    "SessionLoadParams",
    "SessionLoadResult",
    "SessionNewParams",
    "SessionNewResult",
    "SessionPromptParams",
    "SessionPromptResult",
    "SessionResumeParams",
    "SessionResumeResult",
]
