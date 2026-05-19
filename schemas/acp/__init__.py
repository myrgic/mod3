"""schemas.acp — Pydantic models for the Agent Client Protocol (ACP).

ACP is the Agent Client Protocol — JSON-RPC 2.0 over WebSocket.
Spec: https://github.com/agentclientprotocol/agent-client-protocol

This package exposes the minimum viable subset needed to implement
the /ws/acp endpoint in mod3: initialization, session lifecycle,
prompt submission, streaming responses, and error handling.

Public surface
--------------
From ``envelope``:
    JsonRpcRequest, JsonRpcResponse, JsonRpcNotification, JsonRpcError

From ``methods``:
    InitializeParams, InitializeResult,
    SessionNewParams, SessionNewResult,
    SessionPromptParams, SessionPromptResult,
    SessionCancelParams

From ``notifications``:
    SessionUpdateNotification, SessionUpdateParams, SessionUpdatePayload

    NOTE on wire shape: the spec-compliant ``session/update`` params are
    ``{sessionId, update: {sessionUpdate, content}}``. ``SessionUpdateParams``
    is the full params envelope; ``SessionUpdatePayload`` is the inner
    ``update`` object. Prior to 2026-05-19 mod3 used a flat params shape
    (``sessionUpdate`` and ``content`` at the top level of params); that was
    a divergence from the ACP spec. See notifications.py module docstring.

From ``content``:
    TextContent, ImageContent, AudioContent,
    ResourceLink, EmbeddedResource, ContentBlock
"""

from .content import (
    AudioContent,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from .envelope import (
    JsonRpcError,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)
from .methods import (
    AgentCapabilities,
    InitializeParams,
    InitializeResult,
    PromptCapabilities,
    SessionCancelParams,
    SessionNewParams,
    SessionNewResult,
    SessionPromptParams,
    SessionPromptResult,
)
from .notifications import SessionUpdateNotification, SessionUpdateParams, SessionUpdatePayload

__all__ = [
    # envelope
    "JsonRpcError",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
    # methods
    "AgentCapabilities",
    "InitializeParams",
    "InitializeResult",
    "PromptCapabilities",
    "SessionCancelParams",
    "SessionNewParams",
    "SessionNewResult",
    "SessionPromptParams",
    "SessionPromptResult",
    # notifications
    "SessionUpdateNotification",
    "SessionUpdateParams",
    "SessionUpdatePayload",
    # content
    "AudioContent",
    "ContentBlock",
    "EmbeddedResource",
    "ImageContent",
    "ResourceLink",
    "TextContent",
]
