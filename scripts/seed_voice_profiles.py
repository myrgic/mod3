#!/usr/bin/env python3
"""Seed the mod3 voice profile registry from a local JSON config.

The config is a flat array of {"name": "...", "path": "..."} entries:

    [
      {"name": "alex_v1", "path": "/abs/path/to/reference.wav"},
      {"name": "narrator_2", "path": "/abs/path/to/another.wav"}
    ]

Default location: ~/.mod3/seeds.json (deliberately outside this repo so personal
voice references stay personal). Override with --seed-list. See
scripts/seed_voice_profiles.example.json for the canonical format.
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

DEFAULT_SEED_LIST = pathlib.Path.home() / ".mod3" / "seeds.json"


def load_seeds(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        sys.exit(
            f"Seed list not found at {path}.\n"
            f"Create one (see scripts/seed_voice_profiles.example.json) "
            f"or pass --seed-list <path>."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"Seed list at {path} is not valid JSON: {exc}")
    if not isinstance(data, list):
        sys.exit(f"Seed list at {path} must be a JSON array of {{name,path}} entries.")
    for entry in data:
        if not (isinstance(entry, dict) and "name" in entry and "path" in entry):
            sys.exit(f"Each entry in {path} must be an object with 'name' and 'path' keys.")
    return data


def post_profile(base_url: str, name: str, path: str, engine: str) -> str:
    url = f"{base_url}/v1/voices/profiles"
    payload = json.dumps({"name": name, "ref_audio_path": path, "engine": engine}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status == 200:
                return "ok"
            return f"unexpected status {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            return "exists"
        body = ""
        try:
            body = exc.read().decode(errors="replace")[:200]
        except Exception:
            pass
        return f"http {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return f"url error: {exc.reason}"


def patch_metadata(base_url: str, name: str, metadata: dict) -> str:
    """PATCH curation metadata fields onto an existing profile.

    Silently skips keys that are not curation fields; the server validates.
    Returns "ok" on success, a short error string otherwise.
    """
    url = f"{base_url}/v1/voices/profiles/{name}"
    payload = json.dumps(metadata).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status == 200:
                return "ok"
            return f"unexpected status {resp.status}"
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")[:200]
        except Exception:
            pass
        return f"http {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return f"url error: {exc.reason}"


def main():
    parser = argparse.ArgumentParser(
        description="Seed the mod3 voice profile registry from a JSON config.",
    )
    parser.add_argument(
        "--seed-list",
        default=str(DEFAULT_SEED_LIST),
        metavar="PATH",
        help=f"JSON file with seed entries (default: {DEFAULT_SEED_LIST}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be registered without making any network calls.",
    )
    parser.add_argument(
        "--engine",
        default="chatterbox-turbo",
        help="Target TTS engine (default: chatterbox-turbo).",
    )
    parser.add_argument(
        "--mod3",
        default="http://localhost:7860",
        metavar="URL",
        help="mod3 base URL (default: http://localhost:7860).",
    )
    args = parser.parse_args()

    seeds = load_seeds(pathlib.Path(args.seed_list).expanduser())

    if args.dry_run:
        print(f"Dry run — would register {len(seeds)} profile(s) against {args.mod3}  [engine: {args.engine}]")
        print()
        for s in seeds:
            exists = os.path.isfile(s["path"])
            marker = "  " if exists else "  [missing] "
            print(f"{marker}{s['name']:30s}  {s['path']}")
        return

    # Curation metadata fields that may be present in seed entries.
    _CURATION_FIELDS = {"favorite", "notes", "tags", "rating"}

    print(f"Seeding {len(seeds)} profile(s) → {args.mod3}  [engine: {args.engine}]")
    print()
    for s in seeds:
        name = s["name"]
        path = s["path"]

        if not os.path.isfile(path):
            print(f"  ? {name}: skipped (file not found: {path})")
            continue

        result = post_profile(args.mod3, name, path, args.engine)
        if result == "ok":
            print(f"  + {name}: registered")
        elif result == "exists":
            print(f"  - {name}: already registered (skipping re-registration)")
        else:
            print(f"  ! {name}: {result}")
            continue

        # Merge any curation metadata present in the seed entry onto the profile.
        # This is idempotent: running the seed script again will apply the same
        # values without overwriting unrelated fields the operator may have set.
        curation = {k: s[k] for k in _CURATION_FIELDS if k in s}
        if curation:
            meta_result = patch_metadata(args.mod3, name, curation)
            if meta_result == "ok":
                print(f"    metadata merged: {list(curation)}")
            else:
                print(f"    metadata merge failed: {meta_result}")


if __name__ == "__main__":
    main()
