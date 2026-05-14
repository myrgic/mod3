/**
 * WebSocket transport for voice chat.
 * Binary frames = audio, text frames = JSON control messages.
 *
 * Supported handler keys:
 *   onAudio, onTranscript, onPartialTranscript, onResponseText,
 *   onResponseComplete, onInterrupted, onDraftQueue, onError,
 *   onOpen, onClose
 *
 * Note: tts_progress and standalone metrics handlers were removed (no
 * server-side producer). The error handler now receives structured
 * { type: "error", error: { code, message, data? } } frames from the
 * server rather than raw WebSocket error events.
 */
class VoiceTransport {
  constructor(url, handlers) {
    this.url = url;
    this.handlers = handlers; // { onAudio, onTranscript, onPartialTranscript, onResponseText, onResponseComplete, onInterrupted, onDraftQueue, onError, onOpen, onClose }
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnects = 3;
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      if (this.handlers.onOpen) this.handlers.onOpen();
    };

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        console.log(`[WS] Binary frame: ${event.data.byteLength} bytes`);
        if (this.handlers.onAudio) this.handlers.onAudio(event.data);
      } else if (event.data instanceof Blob) {
        console.log(`[WS] Blob frame: ${event.data.size} bytes (converting)`);
        event.data.arrayBuffer().then(buf => {
          if (this.handlers.onAudio) this.handlers.onAudio(buf);
        });
      } else {
        try {
          const msg = JSON.parse(event.data);
          console.log(`[WS] JSON: ${msg.type}`, msg.type === 'response_text' ? msg.text?.substring(0, 80) : '');
          this._dispatch(msg);
        } catch (e) {
          console.error("Failed to parse WS message:", e, "data:", typeof event.data, event.data?.substring?.(0, 100));
        }
      }
    };

    this.ws.onclose = (event) => {
      if (this.handlers.onClose) this.handlers.onClose(event);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      if (this.handlers.onError) this.handlers.onError(error);
    };
  }

  _dispatch(msg) {
    // (trace_event side-channel dispatch removed 2026-05-13 — the Cycle Trace
    // drawer was a side-channel UX artifact, not part of the chat data plane.)

    // Handle base64 audio message — decode and route to onAudio
    if (msg.type === "audio" && msg.data) {
      const binary = atob(msg.data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      if (this.handlers.onAudio) this.handlers.onAudio(bytes.buffer);
      return;
    }

    // Structured error frame from the server — { type: "error", error: { code, message, data? } }
    if (msg.type === "error") {
      console.error("[WS] Server error frame:", msg.error);
      if (this.handlers.onError) this.handlers.onError(msg);
      return;
    }

    // tts_progress and standalone metrics removed (no server-side producer)
    const handlerMap = {
      transcript: "onTranscript",
      partial_transcript: "onPartialTranscript",
      response_text: "onResponseText",
      response_complete: "onResponseComplete",
      interrupted: "onInterrupted",
      draft_queue: "onDraftQueue",
    };
    const handler = handlerMap[msg.type];
    if (handler && this.handlers[handler]) {
      this.handlers[handler](msg);
    }
    // Push to substrate visibility panel (if open).
    if (window.__mod3OpsEvent) {
      const preview = msg.type === 'response_text' ? ` "${msg.text?.substring(0, 40)}"` :
                      msg.type === 'transcript' ? ` "${msg.text?.substring(0, 40)}"` : '';
      window.__mod3OpsEvent(`${msg.type}${preview}`);
    }
  }

  sendAudio(pcmBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(pcmBuffer);
    }
  }

  sendControl(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  endSpeech() {
    this.sendControl({ type: "end_of_speech" });
  }

  interrupt() {
    this.sendControl({ type: "interrupt" });
  }

  setConfig(config) {
    this.sendControl({ type: "config", ...config });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get connected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}
