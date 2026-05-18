"""On-disk registry for voice-lab compositions.

A Composition is a saved draft: an ordered list of reference-segment paths
plus engine/exaggeration/gap settings and free-form notes. Compositions are
the iteration unit in the voice lab — you can have many drafts in flight
(`kronk_solo`, `kronk_top3`, `proposal_a`) and register any of them into a
voice profile via POST /v1/voices/profiles/compose.

Compositions are JSON files at:

  <root>/<name>.json

`<root>` defaults to ``~/.mod3/voices/compositions/`` and is created on first
write.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import threading
from dataclasses import asdict, dataclass, field

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_ROOT = pathlib.Path.home() / ".mod3" / "voices" / "compositions"


@dataclass
class Segment:
    path: str
    label: str = ""
    duration_sec: float | None = None


@dataclass
class Composition:
    name: str
    segments: list[Segment] = field(default_factory=list)
    engine: str = "chatterbox-turbo"
    exaggeration: float = 0.5
    gap_sec: float = 0.15
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, data: dict) -> "Composition":
        segs = [Segment(**s) for s in data.get("segments", [])]
        return cls(
            name=data["name"],
            segments=segs,
            engine=data.get("engine", "chatterbox-turbo"),
            exaggeration=data.get("exaggeration", 0.5),
            gap_sec=data.get("gap_sec", 0.15),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class CompositionRegistry:
    """Disk-backed registry for compositions. Thread-safe per-file."""

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self._root = pathlib.Path(root) if root is not None else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Path / lock helpers
    # ------------------------------------------------------------------

    def _path(self, name: str) -> pathlib.Path:
        if not _NAME_RE.match(name):
            raise ValueError(f"composition name must match [A-Za-z0-9_-]+; got {name!r}")
        return self._root / f"{name}.json"

    def _lock(self, name: str) -> threading.Lock:
        with self._locks_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Composition | None:
        path = self._path(name)
        if not path.exists():
            return None
        with self._lock(name):
            data = json.loads(path.read_text())
        return Composition.from_json(data)

    def list(self) -> list[Composition]:
        out: list[Composition] = []
        for p in sorted(self._root.glob("*.json")):
            try:
                out.append(Composition.from_json(json.loads(p.read_text())))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create(self, comp: Composition) -> Composition:
        path = self._path(comp.name)
        if path.exists():
            raise ValueError(f"composition {comp.name!r} already exists")
        now = self._now()
        comp.created_at = comp.created_at or now
        comp.updated_at = now
        with self._lock(comp.name):
            path.write_text(json.dumps(comp.to_json(), indent=2))
        return comp

    def update(self, name: str, patch: dict) -> Composition:
        """Apply a partial update; preserves created_at, bumps updated_at."""
        existing = self.get(name)
        if existing is None:
            raise KeyError(name)
        merged = existing.to_json()
        for k, v in patch.items():
            if k in {"name", "created_at"}:
                continue
            if k == "segments" and v is not None:
                merged["segments"] = [asdict(Segment(**s)) if not isinstance(s, Segment) else asdict(s) for s in v]
            elif v is not None:
                merged[k] = v
        merged["updated_at"] = self._now()
        comp = Composition.from_json(merged)
        with self._lock(name):
            self._path(name).write_text(json.dumps(comp.to_json(), indent=2))
        return comp

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.exists():
            return False
        with self._lock(name):
            path.unlink()
        return True
