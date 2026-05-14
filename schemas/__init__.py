"""Mod3 shared schemas.

Three layers live here:

* :mod:`.wire` and :mod:`.operations` — the D2 wire protocol shared with
  the CogOS kernel (Go). These are the canonical contract; field names
  must stay byte-identical to the matching Go structs in
  ``cogos/pkg/modality``.
* :mod:`.modality` and :mod:`.channel` — Python mirrors of the kernel's
  canonical modality types (``CognitiveEvent``, ``ChannelDescriptor`` …).
* :mod:`.primitives` — Python-side dual of what the inference engines
  emit (``AudioChunk``, ``TranscriptResult`` …). These are not on the
  wire by themselves; they round-trip through the operation schemas.

The legacy in-process dataclasses in :mod:`mod3.modality` are retained
for existing call sites and will be progressively migrated.
"""

from .bargein import BargeinContext
from .channel import ChannelDescriptor
from .modality import (
    CognitiveEvent,
    CognitiveIntent,
    EncodedOutput,
    GateResult,
    ModalityType,
    ModuleState,
    ModuleStatus,
)
from .operations import (
    STTStreamingRequest,
    STTStreamingResponse,
    STTTranscribeRequest,
    STTTranscribeResponse,
    TTSChunkEvent,
    TTSStreamRequest,
    TTSSynthesizeRequest,
    TTSSynthesizeResponse,
    VADDetectRequest,
    VADDetectResponse,
)
from .primitives import (
    AudioChunk,
    PartialTranscript,
    TranscriptResult,
    TranscriptSegment,
    VADResult,
)
from .wire import MAX_WIRE_LINE_SIZE, WireMessage, WireType

__all__ = [
    "MAX_WIRE_LINE_SIZE",
    # primitives
    "AudioChunk",
    # bargein (existing)
    "BargeinContext",
    # channel
    "ChannelDescriptor",
    # modality
    "CognitiveEvent",
    "CognitiveIntent",
    "EncodedOutput",
    "GateResult",
    "ModalityType",
    "ModuleState",
    "ModuleStatus",
    "PartialTranscript",
    # operations
    "STTStreamingRequest",
    "STTStreamingResponse",
    "STTTranscribeRequest",
    "STTTranscribeResponse",
    "TTSChunkEvent",
    "TTSStreamRequest",
    "TTSSynthesizeRequest",
    "TTSSynthesizeResponse",
    "TranscriptResult",
    "TranscriptSegment",
    "VADDetectRequest",
    "VADDetectResponse",
    "VADResult",
    # wire
    "WireMessage",
    "WireType",
]
