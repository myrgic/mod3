/**
 * ACP (Agent Client Protocol) transport for mod3 dashboard.
 *
 * Implements a minimal ACP client over WebSocket using JSON-RPC 2.0.
 * Wire shapes are byte-identical to the Zed ACP spec:
 * https://github.com/zed-industries/agent-client-protocol
 *
 * Supported methods:
 *   initialize      - capability negotiation
 *   session/new     - create a session
 *   session/prompt  - submit a prompt (streaming via session/update)
 *   session/cancel  - cancel in-flight prompt (notification)
 *
 * Usage:
 *   const transport = new AcpTransport('/ws/acp');
 *   transport.onAgentChunk = (text) => { ... };
 *   transport.onResponseComplete = () => { ... };
 *   transport.onError = (msg) => { ... };
 *   await transport.connect();
 *   await transport.initialize();
 *   await transport.sessionNew();
 *   await transport.sessionPrompt('Hello!');
 *
 * Manual smoke test procedure:
 *   1. Open the dashboard at http://localhost:7860/dashboard?acp=1
 *   2. Open DevTools Network tab, filter by WS
 *   3. Type a message in the chat input and press Enter
 *   4. Verify the WS connection to /ws/acp shows:
 *      - SEND: {"jsonrpc":"2.0","id":0,"method":"initialize",...}
 *      - RECV: {"jsonrpc":"2.0","id":0,"result":{"agentCapabilities":...}}
 *      - SEND: {"jsonrpc":"2.0","id":1,"method":"session/new",...}
 *      - RECV: {"jsonrpc":"2.0","id":1,"result":{"sessionId":"mod3-..."}}
 *      - SEND: {"jsonrpc":"2.0","id":2,"method":"session/prompt",...}
 *      - RECV: {"jsonrpc":"2.0","method":"session/update",...} (one or more)
 *      - RECV: {"jsonrpc":"2.0","id":2,"result":{"stopReason":"end_turn"}}
 *   5. Verify the response text appears in the chat panel
 *   6. Confirm /ws/chat is still available as a fallback (?legacy=1)
 */
class AcpTransport {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this._nextId = 0;
    this._pending = {}; // id -> { resolve, reject }
    this._sessionId = null;
    this._initialized = false;

    // Callbacks
    this.onAgentChunk = null;        // (text: string) => void -- incremental text
    this.onResponseComplete = null;  // () => void
    this.onError = null;             // (message: string) => void
    this.onOpen = null;              // () => void
    this.onClose = null;             // () => void
  }

  // ---------------------------------------------------------------------------
  // Connection management
  // ---------------------------------------------------------------------------

  connect(url) {
    const wsUrl = url || this.url;
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(wsUrl);
      } catch (e) {
        reject(e);
        return;
      }

      this.ws.onopen = () => {
        console.log('[ACP] Connected to', wsUrl);
        if (this.onOpen) this.onOpen();
        resolve();
      };

      this.ws.onmessage = (ev) => {
        this._handleMessage(ev.data);
      };

      this.ws.onclose = (ev) => {
        console.log('[ACP] Closed', ev.code, ev.reason);
        this._rejectAllPending('WebSocket closed');
        if (this.onClose) this.onClose(ev);
      };

      this.ws.onerror = (err) => {
        console.error('[ACP] WebSocket error', err);
        if (this.onError) this.onError('WebSocket error');
        reject(err);
      };
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this._initialized = false;
    this._sessionId = null;
  }

  get connected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  // ---------------------------------------------------------------------------
  // ACP methods
  // ---------------------------------------------------------------------------

  /**
   * Send ACP initialize request.
   * Returns the agentCapabilities from the server response.
   */
  async initialize() {
    const result = await this._request('initialize', {
      protocolVersion: 1,
      clientCapabilities: { fs: {}, terminal: false },
      clientInfo: { name: 'mod3-dashboard', version: '1.0' },
    });
    this._initialized = true;
    console.log('[ACP] Initialized, capabilities:', result.agentCapabilities);
    return result;
  }

  /**
   * Create a new ACP session.
   * Returns the sessionId string.
   */
  async sessionNew() {
    const result = await this._request('session/new', {
      cwd: '/',
      mcpServers: [],
    });
    this._sessionId = result.sessionId;
    console.log('[ACP] Session created:', this._sessionId);
    return this._sessionId;
  }

  /**
   * Submit a text prompt to the current session.
   * Streaming chunks are delivered via onAgentChunk(text) callback.
   * Returns the final SessionPromptResult when complete.
   */
  async sessionPrompt(text) {
    if (!this._sessionId) {
      throw new Error('AcpTransport: no session -- call sessionNew() first');
    }
    const result = await this._request('session/prompt', {
      sessionId: this._sessionId,
      prompt: [{ type: 'text', text }],
    });
    if (this.onResponseComplete) this.onResponseComplete();
    return result;
  }

  /**
   * Cancel an in-flight prompt. This is a JSON-RPC notification (no response).
   */
  sessionCancel() {
    if (!this._sessionId) return;
    this._notify('session/cancel', { sessionId: this._sessionId });
  }

  // ---------------------------------------------------------------------------
  // Internal: JSON-RPC plumbing
  // ---------------------------------------------------------------------------

  _nextRequestId() {
    return this._nextId++;
  }

  _request(method, params) {
    return new Promise((resolve, reject) => {
      if (!this.connected) {
        reject(new Error('AcpTransport: not connected'));
        return;
      }
      const id = this._nextRequestId();
      this._pending[id] = { resolve, reject };
      this._send({ jsonrpc: '2.0', id, method, params });
    });
  }

  _notify(method, params) {
    if (!this.connected) return;
    this._send({ jsonrpc: '2.0', method, params });
  }

  _send(obj) {
    try {
      this.ws.send(JSON.stringify(obj));
    } catch (e) {
      console.error('[ACP] send failed:', e);
    }
  }

  _handleMessage(raw) {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      console.error('[ACP] Failed to parse message:', raw);
      return;
    }

    // Notification: method present, no id.
    if (msg.method && !('id' in msg)) {
      this._handleNotification(msg);
      return;
    }

    // Response: has id, either result or error.
    const id = msg.id;
    const pending = this._pending[id];
    if (!pending) {
      console.warn('[ACP] Received response for unknown id:', id, msg);
      return;
    }
    delete this._pending[id];

    if (msg.error) {
      const err = new Error(`ACP error ${msg.error.code}: ${msg.error.message}`);
      if (this.onError) this.onError(err.message);
      pending.reject(err);
    } else {
      pending.resolve(msg.result);
    }
  }

  _handleNotification(msg) {
    if (msg.method === 'session/update') {
      const params = msg.params || {};
      const updateKind = params.sessionUpdate;
      if (updateKind === 'agent_message_chunk') {
        const content = params.content;
        if (content && content.type === 'text' && this.onAgentChunk) {
          this.onAgentChunk(content.text || '');
        }
      }
      // Other update kinds (thought_chunk, plan, etc.) are no-ops for now.
      return;
    }
    console.log('[ACP] Unhandled notification:', msg.method, msg.params);
  }

  _rejectAllPending(reason) {
    for (const id of Object.keys(this._pending)) {
      const p = this._pending[id];
      delete this._pending[id];
      p.reject(new Error(reason));
    }
  }
}

// ---------------------------------------------------------------------------
// Node.js smoke test (run with: node dashboard/acp-transport.js)
//
// This validates the JSON shapes constructed by AcpTransport without a live
// server. It monkey-patches WebSocket to capture outbound messages and
// manually triggers inbound messages.
// ---------------------------------------------------------------------------

if (typeof module !== 'undefined' && typeof require !== 'undefined' && require.main === module) {
  const assert = require('assert');

  // Minimal WebSocket stub.
  // connect() expects new WebSocket(url), then sets .onopen/.onmessage/.onclose.
  // We fire onopen via microtask so the transport's handler assignment runs first.
  let _latestMockWs = null;
  class MockWebSocket {
    constructor(url) {
      this.url = url;
      this.sent = [];
      this.readyState = 1; // OPEN
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      _latestMockWs = this;
    }
    triggerOpen() { if (this.onopen) this.onopen(); }
    send(data) { this.sent.push(data); }
    close() { this.readyState = 3; if (this.onclose) this.onclose({}); }
    inject(data) { if (this.onmessage) this.onmessage({ data: typeof data === 'string' ? data : JSON.stringify(data) }); }
  }
  MockWebSocket.OPEN = 1;
  global.WebSocket = MockWebSocket;

  async function runSmokeTest() {
    const transport = new AcpTransport('/ws/acp');

    // MockWebSocket is constructed synchronously inside connect(); trigger onopen
    // before awaiting so the Promise resolves.
    const connectPromise = transport.connect('/ws/acp');
    _latestMockWs.triggerOpen();
    await connectPromise;
    const mockWs = _latestMockWs;

    assert.ok(transport.connected, 'should be connected');

    // Test 1: initialize request shape.
    const initPromise = transport.initialize();
    assert.ok(mockWs.sent.length === 1, 'one message sent');
    const initMsg = JSON.parse(mockWs.sent[0]);
    assert.strictEqual(initMsg.jsonrpc, '2.0');
    assert.strictEqual(initMsg.method, 'initialize');
    assert.ok(typeof initMsg.id === 'number');
    assert.strictEqual(initMsg.params.protocolVersion, 1);
    assert.strictEqual(initMsg.params.clientInfo.name, 'mod3-dashboard');
    // Inject response.
    mockWs.inject({
      jsonrpc: '2.0', id: initMsg.id,
      result: { agentCapabilities: { promptCapabilities: { audio: false, image: false, embeddedContext: false }, sessionCapabilities: {} } },
    });
    const initResult = await initPromise;
    assert.ok(initResult.agentCapabilities, 'result has agentCapabilities');
    console.log('  PASS: initialize request/response shape');

    // Test 2: session/new request shape.
    const newPromise = transport.sessionNew();
    const newMsg = JSON.parse(mockWs.sent[1]);
    assert.strictEqual(newMsg.method, 'session/new');
    assert.strictEqual(newMsg.params.cwd, '/');
    mockWs.inject({ jsonrpc: '2.0', id: newMsg.id, result: { sessionId: 'mod3-test123' } });
    const sessionId = await newPromise;
    assert.strictEqual(sessionId, 'mod3-test123');
    console.log('  PASS: session/new request/response shape');

    // Test 3: session/prompt + streaming session/update notification.
    const chunks = [];
    transport.onAgentChunk = (t) => chunks.push(t);
    transport.onResponseComplete = () => {};
    const promptPromise = transport.sessionPrompt('Hello?');
    const promptMsg = JSON.parse(mockWs.sent[2]);
    assert.strictEqual(promptMsg.method, 'session/prompt');
    assert.strictEqual(promptMsg.params.sessionId, 'mod3-test123');
    assert.deepStrictEqual(promptMsg.params.prompt, [{ type: 'text', text: 'Hello?' }]);
    // Inject streaming chunk.
    mockWs.inject({
      jsonrpc: '2.0', method: 'session/update',
      params: { sessionId: 'mod3-test123', sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Hi there ' } },
    });
    // Inject final response.
    mockWs.inject({ jsonrpc: '2.0', id: promptMsg.id, result: { stopReason: 'end_turn' } });
    const promptResult = await promptPromise;
    assert.strictEqual(promptResult.stopReason, 'end_turn');
    assert.deepStrictEqual(chunks, ['Hi there ']);
    console.log('  PASS: session/prompt + session/update streaming');

    // Test 4: session/cancel notification shape (no id).
    transport.sessionCancel();
    const cancelMsg = JSON.parse(mockWs.sent[3]);
    assert.strictEqual(cancelMsg.method, 'session/cancel');
    assert.ok(!('id' in cancelMsg), 'cancel is a notification -- no id');
    assert.strictEqual(cancelMsg.params.sessionId, 'mod3-test123');
    console.log('  PASS: session/cancel notification shape');

    console.log('\nAll smoke tests passed.');
  }

  runSmokeTest().catch((e) => { console.error('FAIL:', e.message); process.exit(1); });
}
