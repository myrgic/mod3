"""mod3.worker — kernel-facing subprocess CLI for mod3 inference workers.

The CogOS kernel spawns ``python -m mod3.worker {tts,vad,stt}`` as long-running
subprocesses and communicates with each over the D2 wire protocol (JSON-lines
over stdin/stdout, as defined in ``schemas/wire.py``).

Each subcommand:
1. Emits a ``ready`` event on startup.
2. Reads ``WireMessage`` records from stdin in a loop.
3. Dispatches operations to the matching mod3 inference code.
4. Writes ``WireMessage`` responses/events to stdout.
5. Handles ``health`` and ``shutdown`` lifecycle commands.
"""
