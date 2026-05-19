from __future__ import annotations

import hashlib
import pathlib
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Registry-level VoiceProfile (Primitive 3 generative head, Wave 6c)
# ---------------------------------------------------------------------------
# This is the on-disk metadata record for a cloned voice stored in the mod3
# voice registry at ~/.mod3/voices/<name>.{safetensors,json}.
# Paired with VoiceGenerativeHead / VoiceDiscriminativeHead (below) which are
# the schema representations of the identity-level voice config that arrives
# via the CogOS identity projection.
# ---------------------------------------------------------------------------


@dataclass
class VoiceProfile:
    name: str
    engine: str
    source_audio_path: str
    source_sha256: str
    ref_text: str | None
    exaggeration: float
    model_id: str
    created_at: str
    # --- curation metadata (added 2026-05-15) ---
    # Missing fields in pre-existing JSON files are filled with defaults on load.
    favorite: bool = False
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    last_used_at: Optional[str] = None
    rating: Optional[int] = None
    # --- discriminative head (Primitive 3, Wave 6c) ---
    # embedding_ref is a cog://voices/<name>/ecapa-embedding URI that resolves
    # to the ECAPA-TDNN 192-dim speaker embedding vector. Schema only in this
    # wave; no live ECAPA integration yet — field is None until enrollment runs.
    embedding_ref: Optional[str] = None

    @classmethod
    def from_json(cls, data: dict) -> "VoiceProfile":
        return cls(
            name=data["name"],
            engine=data["engine"],
            source_audio_path=data["source_audio_path"],
            source_sha256=data["source_sha256"],
            ref_text=data.get("ref_text"),
            exaggeration=data["exaggeration"],
            model_id=data["model_id"],
            created_at=data["created_at"],
            # curation fields — backward-compat defaults when absent
            favorite=data.get("favorite", False),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            last_used_at=data.get("last_used_at", None),
            rating=data.get("rating", None),
            # discriminative head — absent in pre-Primitive-3 files
            embedding_ref=data.get("embedding_ref", None),
        )

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Identity-level voice schema (Primitive 3 — mirrors cogos IdentityExpression)
# ---------------------------------------------------------------------------
# These dataclasses mirror the VoiceGenerativeHead / VoiceDiscriminativeHead /
# VoiceProfile structs from cogos identity_crd.go and are used when mod3
# receives voice config from the CogOS kernel via identity projection events.
# They are distinct from the registry-level VoiceProfile above.
# ---------------------------------------------------------------------------


@dataclass
class IdentityVoiceGenerativeHead:
    """TTS conditioning parameters for one engine.

    conditionals_ref is a cog://voices/<name> URI. The mod3 URI resolver
    maps this to the local registry path ~/.mod3/voices/<name>.safetensors.
    """

    engine: str  # e.g. "chatterbox-turbo"
    conditionals_ref: str  # cog://voices/<name>
    enrolled_at: Optional[str] = None  # RFC3339

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityVoiceGenerativeHead":
        return cls(
            engine=data["engine"],
            conditionals_ref=data["conditionals_ref"],
            enrolled_at=data.get("enrolled_at"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IdentityVoiceDiscriminativeHead:
    """Speaker-recognition head. Schema only — no live ECAPA integration yet.

    embedding_ref is a cog://voices/<name>/ecapa-embedding URI that would
    resolve to the 192-dim ECAPA-TDNN speaker embedding once enrollment
    is implemented (Wave 6d+).
    """

    model: str  # e.g. "speechbrain/spkrec-ecapa-voxceleb"
    embedding_ref: str  # cog://voices/<name>/ecapa-embedding
    enrolled_at: Optional[str] = None  # RFC3339

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityVoiceDiscriminativeHead":
        return cls(
            model=data["model"],
            embedding_ref=data["embedding_ref"],
            enrolled_at=data.get("enrolled_at"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IdentityVoiceProfile:
    """Structured voice config for an identity, mirroring cogos VoiceProfile.

    Received from the CogOS kernel as part of an identity projection event.
    Generative head enables TTS via mod3. Discriminative head enables speaker
    recognition (schema only in this wave).
    """

    generative: Optional[IdentityVoiceGenerativeHead] = None
    discriminative: Optional[IdentityVoiceDiscriminativeHead] = None

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityVoiceProfile":
        gen_data = data.get("generative")
        disc_data = data.get("discriminative")
        return cls(
            generative=IdentityVoiceGenerativeHead.from_dict(gen_data) if gen_data else None,
            discriminative=IdentityVoiceDiscriminativeHead.from_dict(disc_data) if disc_data else None,
        )

    def to_dict(self) -> dict:
        return {
            "generative": self.generative.to_dict() if self.generative else None,
            "discriminative": self.discriminative.to_dict() if self.discriminative else None,
        }


# ---------------------------------------------------------------------------
# cog://voices/* URI resolver
# ---------------------------------------------------------------------------
# Resolves cog://voices/<name> URIs to the local mod3 voice registry path.
# Resolution semantics:
#   cog://voices/<name>                  → ~/.mod3/voices/<name> (registry entry)
#   cog://voices/<name>/ecapa-embedding  → ~/.mod3/voices/<name>.ecapa.npy (future)
#
# This is a local resolver — mod3 owns the voices namespace for its own library.
# The kernel may later expose a cross-node resolver that delegates to this one.
# ---------------------------------------------------------------------------

_DEFAULT_VOICE_REGISTRY_ROOT = pathlib.Path.home() / ".mod3" / "voices"


def resolve_voices_uri(uri: str, registry_root: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve a cog://voices/* URI to a local filesystem path.

    Args:
        uri: A cog://voices/* URI. Both bare (cog:voices/<name>) and authority
             (cog://voices/<name>) forms are accepted.
        registry_root: Override the registry root. Defaults to ~/.mod3/voices/.

    Returns:
        pathlib.Path for the resolved resource (file may not exist yet).

    Raises:
        ValueError: URI does not start with a voices namespace prefix.
        ValueError: Name segment is empty.

    Resolution table:
        cog://voices/<name>                  → <root>/<name>  (directory entry stub)
        cog://voices/<name>/ecapa-embedding  → <root>/<name>.ecapa.npy
    """
    root = registry_root if registry_root is not None else _DEFAULT_VOICE_REGISTRY_ROOT

    # Normalise both cog: and cog:// forms.
    for prefix in ("cog://voices/", "cog:voices/"):
        if uri.startswith(prefix):
            remainder = uri[len(prefix) :]
            break
    else:
        raise ValueError(f"resolve_voices_uri: not a voices URI: {uri!r}")

    if not remainder:
        raise ValueError(f"resolve_voices_uri: missing name segment in URI: {uri!r}")

    parts = remainder.split("/", 1)
    name = parts[0]
    if not name:
        raise ValueError(f"resolve_voices_uri: empty name segment in URI: {uri!r}")

    if len(parts) == 1:
        # cog://voices/<name> — resolve to the safetensors conditionals file
        return root / f"{name}.safetensors"
    else:
        sub = parts[1]
        if sub == "ecapa-embedding":
            return root / f"{name}.ecapa.npy"
        # Unknown sub-path — return path as-is for forward compatibility
        return root / name / sub


def compute_source_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
