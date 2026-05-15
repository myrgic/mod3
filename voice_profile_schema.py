from __future__ import annotations

import hashlib
import pathlib
from dataclasses import asdict, dataclass, field
from typing import List, Optional


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
        )

    def to_json(self) -> dict:
        return asdict(self)


def compute_source_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
