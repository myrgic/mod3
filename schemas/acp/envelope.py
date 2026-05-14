"""schemas.acp.envelope — JSON-RPC 2.0 envelope models.

These are the base message shapes for ACP's JSON-RPC 2.0 transport.
Field names match the JSON-RPC 2.0 spec byte-identically.

Reference: https://www.jsonrpc.org/specification
ACP wire: https://github.com/zed-industries/agent-client-protocol
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    jsonrpc: Literal["2.0"] = "2.0"


class JsonRpcRequest(_Base):
    """A JSON-RPC 2.0 request that expects a response.

    Wire shape::

        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {...}}

    ``id`` may be a string or integer. ``params`` is method-specific.
    """

    id: int | str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcNotification(_Base):
    """A JSON-RPC 2.0 notification — no ``id``, no response expected.

    Wire shape::

        {"jsonrpc": "2.0", "method": "session/cancel", "params": {...}}
    """

    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object embedded in ``JsonRpcResponse.error``.

    Standard codes:
      -32700  Parse error
      -32600  Invalid request
      -32601  Method not found
      -32602  Invalid params
      -32603  Internal error
      -32000..-32099  Server-defined errors
    """

    model_config = ConfigDict(populate_by_name=True)

    code: int
    message: str
    data: Any = None


class JsonRpcResponse(_Base):
    """A JSON-RPC 2.0 response.

    Exactly one of ``result`` or ``error`` is present (spec rule).

    Wire shape (success)::

        {"jsonrpc": "2.0", "id": 1, "result": {...}}

    Wire shape (error)::

        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "..."}}
    """

    id: int | str | None = None
    result: Any = None
    error: JsonRpcError | None = None

    @classmethod
    def ok(cls, *, request_id: int | str, result: Any) -> "JsonRpcResponse":
        """Convenience constructor for a successful response."""
        return cls(id=request_id, result=result)

    @classmethod
    def err(
        cls,
        *,
        request_id: int | str | None,
        code: int,
        message: str,
        data: Any = None,
    ) -> "JsonRpcResponse":
        """Convenience constructor for an error response."""
        return cls(id=request_id, error=JsonRpcError(code=code, message=message, data=data))


__all__ = [
    "JsonRpcError",
    "JsonRpcNotification",
    "JsonRpcRequest",
    "JsonRpcResponse",
]
