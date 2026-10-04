"""The deployed server must be able to accept WebSocket upgrades.

uvicorn without a WebSocket library (websockets or wsproto) answers every /ws/* upgrade with a plain 404 and only
logs a warning, so /ws/audio, /ws/chat, /ws/acp and /ws/body were all silently dead in production while every HTTP
test passed. This checks the dependency at the level that failed: a real upgrade through a real uvicorn server.
"""

import asyncio
import socket
import threading
import time

import pytest


def test_websocket_library_installed():
    try:
        import websockets  # noqa: F401
    except ImportError:  # pragma: no cover - the failure this test exists for
        pytest.fail("websockets is not installed: uvicorn will 404 every /ws/* upgrade (add it to requirements.txt)")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_real_websocket_upgrade_through_uvicorn():
    import uvicorn
    import websockets
    from fastapi import FastAPI, WebSocket

    app = FastAPI()

    @app.websocket("/ws/probe")
    async def probe(ws: WebSocket):
        await ws.accept()
        await ws.send_text("ok")
        await ws.close()

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)

        async def go():
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/probe") as c:
                return await c.recv()

        assert asyncio.run(go()) == "ok"
    finally:
        server.should_exit = True
        t.join(timeout=5)
