#!/usr/bin/env python3
"""Fetch Smart Turn v3.1 ONNX weight from HuggingFace and update MANIFEST.toml.

Usage:
    python scripts/fetch_smart_turn_weights.py [--dest vendor/smart_turn/data]

Requires: huggingface_hub (pip install huggingface_hub)

After fetch, updates vendor/MANIFEST.toml with the revision SHA and SHA-256
checksum of the downloaded file.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tomllib
from pathlib import Path

log = logging.getLogger("fetch_smart_turn_weights")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weight(dest_dir: Path, force: bool = False) -> Path:
    """Download smart-turn-v3.1.onnx to dest_dir; return the local path."""
    try:
        from huggingface_hub import hf_hub_download, model_info
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "smart-turn-v3.1.onnx"

    if dest_path.exists() and not force:
        log.info("Weight already present at %s (use --force to re-download)", dest_path)
        return dest_path

    log.info("Fetching smart-turn-v3.1.onnx from pipecat-ai/smart-turn ...")
    local = hf_hub_download(
        repo_id="pipecat-ai/smart-turn",
        filename="smart-turn-v3.1.onnx",
        local_dir=str(dest_dir),
    )
    log.info("Downloaded to %s", local)
    return Path(local)


def update_manifest(manifest_path: Path, revision: str, sha256: str) -> None:
    """Write revision + sha256 into MANIFEST.toml [smart_turn.hf_weight] section."""
    content = manifest_path.read_text()

    # Simple line-by-line update — MANIFEST.toml is human-maintained so we use
    # targeted string replacement rather than toml round-trip (toml serializers
    # may reformat the file).
    def replace_field(text: str, field: str, value: str) -> str:
        import re

        pattern = rf'(^{re.escape(field)}\s*=\s*")[^"]*(")'
        replacement = rf"\g<1>{value}\g<2>"
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)

    content = replace_field(content, "revision", revision)
    content = replace_field(content, "sha256", sha256)
    manifest_path.write_text(content)
    log.info("Updated MANIFEST.toml: revision=%s sha256=%s...", revision, sha256[:16])


def get_hf_revision() -> str:
    """Return the current HEAD commit SHA for pipecat-ai/smart-turn on HF."""
    try:
        from huggingface_hub import model_info

        info = model_info("pipecat-ai/smart-turn")
        return info.sha or ""
    except Exception as exc:
        log.warning("Could not fetch HF model info: %s", exc)
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).parent.parent / "vendor" / "smart_turn" / "data",
        help="Destination directory for the ONNX weight file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent.parent / "vendor" / "MANIFEST.toml",
        help="Path to MANIFEST.toml to update",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    weight_path = fetch_weight(args.dest, force=args.force)
    sha256 = sha256_file(weight_path)
    revision = get_hf_revision()

    log.info("SHA-256: %s", sha256)
    log.info("HF revision: %s", revision or "(unknown)")

    if args.manifest.exists():
        update_manifest(args.manifest, revision, sha256)
        log.info("MANIFEST.toml updated.")
    else:
        log.warning("MANIFEST.toml not found at %s — skipping update", args.manifest)

    log.info("Done. Weight at: %s", weight_path)


if __name__ == "__main__":
    main()
