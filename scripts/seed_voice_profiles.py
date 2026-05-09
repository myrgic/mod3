#!/usr/bin/env python3
"""Seed the mod3 voice profile registry from known reference WAV files."""

import argparse
import glob
import json
import os
import urllib.error
import urllib.request

RESEMBLE_DIR = "/tmp/voice_lab/resemble_demos/24k"

SEEDS = [
    {"name": "chaz_demo", "path": "/tmp/ref_chaz_demo.wav"},
    {"name": "sagan", "path": "/tmp/voice_lab/celeb_refs/sagan_commencement.wav"},
    {"name": "einstein", "path": "/tmp/voice_lab/celeb_refs/einstein_speech.wav"},
    {"name": "hawking_dectalk", "path": "/tmp/voice_lab/celeb_refs/hawking_dectalk_24k.wav"},
]


def discover_resemble_seeds():
    pattern = os.path.join(RESEMBLE_DIR, "*_prompt.wav")
    paths = sorted(glob.glob(pattern))
    seeds = []
    for p in paths:
        basename = os.path.basename(p)  # e.g. jerry_seinfeld_prompt.wav
        name = basename[: -len("_prompt.wav")]  # strip suffix
        seeds.append({"name": name, "path": p})
    return seeds


def build_seed_list():
    return SEEDS + discover_resemble_seeds()


def post_profile(base_url, name, path, engine):
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


def main():
    parser = argparse.ArgumentParser(description="Seed mod3 voice profile registry from reference WAV files.")
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

    seeds = build_seed_list()

    if args.dry_run:
        print(f"Dry run — would register {len(seeds)} profile(s) against {args.mod3}  [engine: {args.engine}]")
        print()
        for s in seeds:
            exists = os.path.isfile(s["path"])
            marker = "  " if exists else "  [missing] "
            print(f"{marker}{s['name']:30s}  {s['path']}")
        return

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
            print(f"  - {name}: already registered")
        else:
            print(f"  ! {name}: {result}")


if __name__ == "__main__":
    main()
