"""Channel descriptor — Python mirror of ``cogos/pkg/modality/channel.go``.

A channel is a transport-bound identity that declares which modalities
it can receive and deliver. The kernel's ``ChannelRegistry`` routes
output to every channel that supports a requested modality; mod3's
dashboard registers itself as a channel with both Voice and Text I/O.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .modality import ModalityType


class ChannelDescriptor(BaseModel):
    """A channel's identity, transport, and modality capabilities."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description='unique channel ID, e.g. "mod3-dashboard-0xab12"')
    transport: str = Field(
        ...,
        description='transport identifier, e.g. "websocket", "mcp", "http", "stdio"',
    )
    input: list[ModalityType] = Field(
        default_factory=list,
        description="modalities this channel can receive (mic, keyboard, ...)",
    )
    output: list[ModalityType] = Field(
        default_factory=list,
        description="modalities this channel can deliver (speakers, screen, ...)",
    )
    session_key: str = Field(
        default="",
        description='binding pattern for session routing, e.g. "mod3:{session_id}"',
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def supports_output(self, modality: ModalityType) -> bool:
        """Whether this channel can deliver the given modality."""
        return modality in self.output

    def supports_input(self, modality: ModalityType) -> bool:
        """Whether this channel can receive the given modality."""
        return modality in self.input
