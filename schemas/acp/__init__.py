"""schemas.acp — Pydantic models for the Agent Client Protocol (ACP).

ACP is Zed's Agent Client Protocol — JSON-RPC 2.0 over WebSocket.
Spec: https://github.com/zed-industries/agent-client-protocol

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
    SessionUpdateNotification, SessionUpdatePayload

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
from .notifications import SessionUpdateNotification, SessionUpdatePayload

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
    "SessionUpdatePayload",
    # content
    "AudioContent",
    "ContentBlock",
    "EmbeddedResource",
    "ImageContent",
    "ResourceLink",
    "TextContent",
]
