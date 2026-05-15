"""On-disk registry for mod3 voice profiles.

Profiles are stored as paired files under the registry root:
  <root>/<name>.safetensors  — Conditionals tensors
  <root>/<name>.json         — VoiceProfile metadata sidecar

Both files must be present for a profile to be considered registered.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import threading
from typing import Any

from voice_profile_io import load_conditionals, save_conditionals
from voice_profile_schema import VoiceProfile, compute_source_sha256

# Per-file locks for atomic JSON updates
_file_locks: dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()


def _get_file_lock(path: pathlib.Path) -> threading.Lock:
    key = str(path)
    with _file_locks_lock:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_DEFAULT_ROOT = pathlib.Path.home() / ".mod3" / "voices"


class VoiceProfileRegistry:
    """Wraps disk persistence for voice profiles."""

    def __init__(self, root: pathlib.Path | None = None) -> None:
        """Initialise the registry.

        Args:
            root: Directory where profiles are stored. Defaults to
                  ~/.mod3/voices/. Created if absent.
        """
        self._root = pathlib.Path(root) if root is not None else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        engine: str,
        ref_audio_path: str,
        conds,
        ref_text: str | None = None,
        exaggeration: float = 0.5,
        model_id: str = "",
    ) -> VoiceProfile:
        """Persist a Conditionals blob + sidecar JSON. Returns the VoiceProfile.

        Raises:
            ValueError: name is empty or contains invalid characters.
            ValueError: engine does not support cloning.
            ValueError: a profile with this name already exists.
            FileNotFoundError: ref_audio_path does not exist on disk.
        """
        # --- validate name ---
        if not name:
            raise ValueError("profile name must be non-empty")
        if not _NAME_RE.match(name):
            raise ValueError(f"profile name {name!r} is invalid; only A-Z a-z 0-9 _ - are allowed")

        # --- validate engine (lazy import to avoid circular dependency) ---
        from engine import MODELS  # noqa: PLC0415

        engine_info = MODELS.get(engine)
        if engine_info is None or not engine_info.get("supports_cloning"):
            raise ValueError(
                f"engine {engine!r} does not support voice cloning; "
                f"choose one of: {[k for k, v in MODELS.items() if v.get('supports_cloning')]}"
            )

        # --- validate source audio ---
        audio_path = pathlib.Path(ref_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"ref_audio_path does not exist: {audio_path}")

        # --- refuse to overwrite ---
        sidecar = self._root / f"{name}.json"
        if sidecar.exists():
            raise ValueError(f"profile {name!r} already exists")

        # --- compute sha256 of source audio ---
        source_sha256 = compute_source_sha256(audio_path)

        # --- build metadata ---
        profile = VoiceProfile(
            name=name,
            engine=engine,
            source_audio_path=str(audio_path.resolve()),
            source_sha256=source_sha256,
            ref_text=ref_text,
            exaggeration=exaggeration,
            model_id=model_id,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # --- persist tensors ---
        safetensors_path = self._root / f"{name}.safetensors"
        save_conditionals(conds, safetensors_path)

        # --- persist sidecar ---
        sidecar.write_text(json.dumps(profile.to_json(), indent=2))

        return profile

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def list(self) -> list[VoiceProfile]:
        """Return all registered profile metadata, sorted by name."""
        profiles: list[VoiceProfile] = []
        for json_path in sorted(self._root.glob("*.json")):
            safetensors_path = json_path.with_suffix(".safetensors")
            if not safetensors_path.exists():
                continue
            try:
                data = json.loads(json_path.read_text())
                profiles.append(VoiceProfile.from_json(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return profiles

    def get(self, name: str) -> VoiceProfile | None:
        """Return profile metadata or None if not registered."""
        json_path = self._root / f"{name}.json"
        safetensors_path = self._root / f"{name}.safetensors"
        if not json_path.exists() or not safetensors_path.exists():
            return None
        data = json.loads(json_path.read_text())
        return VoiceProfile.from_json(data)

    def get_conditionals(self, name: str):
        """Return the Conditionals object loaded from <name>.safetensors.

        Returns None if the profile is not registered. Lets safetensors
        errors propagate to the caller.
        """
        safetensors_path = self._root / f"{name}.safetensors"
        json_path = self._root / f"{name}.json"
        if not safetensors_path.exists() or not json_path.exists():
            return None
        return load_conditionals(safetensors_path)

    def names(self) -> list[str]:
        """Cheap path-only scan; does not load any tensors. Returns sorted names."""
        stems = sorted(
            p.stem for p in self._root.glob("*.json") if (self._root / p.stem).with_suffix(".safetensors").exists()
        )
        return stems

    # ------------------------------------------------------------------
    # Curation metadata patch
    # ------------------------------------------------------------------

    _CURATION_FIELDS = frozenset({"favorite", "notes", "tags", "last_used_at", "rating"})

    def patch_metadata(self, name: str, updates: dict[str, Any]) -> VoiceProfile | None:
        """Atomically update curation metadata fields on an existing profile.

        Only the fields in ``_CURATION_FIELDS`` may be patched.  Unrecognised
        keys are silently ignored.  Returns the updated VoiceProfile, or None
        if the profile does not exist.

        Raises:
            ValueError: rating out of range (must be None or 1-5).
        """
        json_path = self._root / f"{name}.json"
        safetensors_path = json_path.with_suffix(".safetensors")
        if not json_path.exists() or not safetensors_path.exists():
            return None

        if "rating" in updates and updates["rating"] is not None:
            r = updates["rating"]
            if not isinstance(r, int) or r < 1 or r > 5:
                raise ValueError(f"rating must be an integer 1-5 or null, got {r!r}")

        lock = _get_file_lock(json_path)
        with lock:
            data = json.loads(json_path.read_text())
            for field in self._CURATION_FIELDS:
                if field in updates:
                    data[field] = updates[field]
            json_path.write_text(json.dumps(data, indent=2))

        return VoiceProfile.from_json(data)

    def update_last_used_at(self, name: str) -> None:
        """Record the current UTC timestamp as last_used_at for the named profile.

        No-op if the profile does not exist.  Writes atomically under a per-file lock.
        """
        json_path = self._root / f"{name}.json"
        safetensors_path = json_path.with_suffix(".safetensors")
        if not json_path.exists() or not safetensors_path.exists():
            return

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lock = _get_file_lock(json_path)
        with lock:
            try:
                data = json.loads(json_path.read_text())
                data["last_used_at"] = now
                json_path.write_text(json.dumps(data, indent=2))
            except Exception:
                pass  # never break synthesis over a metadata write

    # ------------------------------------------------------------------
    # Delete path
    # ------------------------------------------------------------------

    def delete(self, name: str) -> bool:
        """Remove both <name>.safetensors and <name>.json.

        Returns True if at least one file was removed, False if the
        profile did not exist.
        """
        json_path = self._root / f"{name}.json"
        safetensors_path = self._root / f"{name}.safetensors"

        removed = False
        if safetensors_path.exists():
            safetensors_path.unlink()
            removed = True
        if json_path.exists():
            json_path.unlink()
            removed = True
        return removed
