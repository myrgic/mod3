"""Entry point for ``python -m mod3.worker {tts,vad,stt}``.

The CogOS kernel spawns this module as a subprocess for each modality worker:
    python -m mod3.worker tts
    python -m mod3.worker vad
    python -m mod3.worker stt

Each subcommand starts the D2 wire-protocol loop for its module.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python -m mod3.worker {tts,vad,stt}",
            file=sys.stderr,
        )
        sys.exit(1)

    subcommand = sys.argv[1].lower()

    if subcommand == "tts":
        from mod3.worker.tts import main as run

        run()
    elif subcommand == "vad":
        from mod3.worker.vad import main as run

        run()
    elif subcommand == "stt":
        from mod3.worker.stt import main as run

        run()
    else:
        print(
            f"unknown subcommand: {subcommand!r}. choose from: tts, vad, stt",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
