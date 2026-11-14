#!/usr/bin/env bash
# mod3-tts.sh — Hermes command-type TTS provider backed by the local Mod³ server.
#
# Usage (as invoked by Hermes tts.providers.mod3.command):
#   mod3-tts.sh <input_text_path> <output_audio_path> [voice] [model] [speed]
#
# Mod³'s /v1/audio/speech always returns WAV regardless of response_format,
# so we transcode with ffmpeg to whatever extension Hermes asked for.
set -euo pipefail

IN="$1"; OUT="$2"
VOICE="${3:-eng_uk_m_davids}"
MODEL="${4:-chatterbox-turbo}"
SPEED="${5:-1.0}"
[ -z "$SPEED" ] && SPEED="1.0"
MOD3_URL="${MOD3_URL:-http://127.0.0.1:7860}"

TMP_WAV="$(mktemp -t mod3tts).wav"
trap 'rm -f "$TMP_WAV"' EXIT

# JSON-encode the input text safely (python3 stdlib only).
PAYLOAD="$(python3 - "$IN" "$VOICE" "$MODEL" "$SPEED" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
print(json.dumps({"input": text, "voice": sys.argv[2], "model": sys.argv[3],
                  "speed": float(sys.argv[4]), "response_format": "wav"}))
EOF
)"

HTTP_CODE="$(curl -s -m 180 -o "$TMP_WAV" -w '%{http_code}' \
  -X POST "$MOD3_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD")"

if [ "$HTTP_CODE" != "200" ]; then
  echo "mod3 returned HTTP $HTTP_CODE: $(head -c 300 "$TMP_WAV")" >&2
  exit 1
fi

EXT="${OUT##*.}"
case "$EXT" in
  wav) cp "$TMP_WAV" "$OUT" ;;
  mp3) ffmpeg -y -loglevel error -i "$TMP_WAV" -c:a libmp3lame -b:a 128k "$OUT" ;;
  ogg|opus) ffmpeg -y -loglevel error -i "$TMP_WAV" -c:a libopus -b:a 64k "$OUT" ;;
  *) ffmpeg -y -loglevel error -i "$TMP_WAV" "$OUT" ;;
esac
