"""schemas.acp.content — ACP content block union types.

ACP prompts and responses carry structured content blocks rather than raw
strings. The minimum viable type for text chat is ``TextContent``; the
others are present in the schema but not yet implemented server-side.

Reference: https://github.com/zed-industries/agent-client-protocol
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _ContentBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class TextContent(_ContentBase):
    """Plain text content block.

    Wire shape::

        {"type": "text", "text": "Hello, world!"}
    """

    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(_ContentBase):
    """Image content block (not yet implemented server-side).

    Wire shape::

        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    """

    type: Literal["image"] = "image"
    source: dict[str, Any] = Field(default_factory=dict)


class AudioContent(_ContentBase):
    """Audio content block (not yet implemented server-side).

    Wire shape::

        {"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": "..."}}
    """

    type: Literal["audio"] = "audio"
    source: dict[str, Any] = Field(default_factory=dict)


class ResourceLink(_ContentBase):
    """A link to an external resource (file, URL).

    Wire shape::

        {"type": "resource_link", "uri": "file:///path/to/file", "name": "myfile.txt"}
    """

    type: Literal["resource_link"] = "resource_link"
    uri: str = ""
    name: str = ""
    description: str = ""
    mime_type: str = ""


class EmbeddedResource(_ContentBase):
    """An embedded resource with inline content.

    Wire shape::

        {"type": "embedded_resource", "resource": {"uri": "...", "text": "..."}}
    """

    type: Literal["embedded_resource"] = "embedded_resource"
    resource: dict[str, Any] = Field(default_factory=dict)


# Discriminated union of all content block types.
ContentBlock = Annotated[
    Union[TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource],
    Field(discriminator="type"),
]

__all__ = [
    "AudioContent",
    "ContentBlock",
    "EmbeddedResource",
    "ImageContent",
    "ResourceLink",
    "TextContent",
]
