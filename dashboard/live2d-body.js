/**
 * live2d-body.js — a Live2D model as a Mod³ body.
 *
 * Adapted from Vroku's Live2D Cubism 4 controller suite (Storm/Luna canvas,
 * 2025): same Live2DManipulator idea (clamped writes into the Cubism core) and
 * the same seven named states, rebuilt as a body an agent can drive:
 *
 *   - declares a manifest (params with ranges + states) on /ws/body/{id}
 *   - applies `act` commands and answers every one with a receipt of the
 *     values actually reached (read back from the core, not echoed)
 *   - agent-held params sit on top of the procedural layer until released
 *   - optional lip sync from the speaking session's /ws/audio/{session} feed
 *   - seeded RNG, so a given seed replays the same idle motion
 *
 * Fixes vs the original: lerpFactorBase/lerpFactorEye were undefined (body,
 * smile and eye-smile never moved); smile/tilt timers were scheduled but never
 * read; body sway/bounce/follow-gaze sliders were unwired; bodyTargetZ was
 * computed but never applied; writes now happen inside the model's update
 * (after motions/physics inputs) instead of in a separate rAF loop.
 */

// ------------------------------------------------------------------ helpers
const clamp = (v, lo, hi) => Math.min(Math.max(Number.isFinite(v) ? v : 0, lo), hi);
const lerp = (a, b, f) => a + (b - a) * clamp(f, 0, 1);

/** mulberry32: small, fast, seedable. */
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const STATES = [
  "IDLE", "DANCING", "VIBING", "EXCITED",
  "LEANING_FORWARD", "PRETENDING_NOT_TO_NOTICE", "REFUSING_TO_LOOK",
];

const DEFAULT_TUNING = {
  focusLerp: 0.075, baseLerp: 0.06, eyeLerp: 0.14,
  lookIntervalMin: 2500, lookIntervalMax: 5000, lookDurationMin: 400, lookDurationMax: 1000,
  blinkIntervalMin: 1000, blinkIntervalMax: 8000,
  angleXRange: 30, angleYRange: 20, angleZTiltMax: 15,
  bodyAngleXMax: 10, bodyAngleZMax: 10, breathAmp: 0.08,
  saccadeIntervalMin: 150, saccadeIntervalMax: 750, saccadeMagnitude: 0.2,
  smileIntervalMin: 5000, smileIntervalMax: 15000, smileDuration: 2000, smileVal: 0.7, eyeSmileVal: 0.8,
  tiltIntervalMin: 6000, tiltIntervalMax: 12000, tiltDuration: 3000,
  danceEnergy: 1.0, leanAmount: 10, bodySwayAmount: 1.0, bodyBounceAmount: 1.0, followGazeIntensity: 0.1,
  mouthGain: 4.0,
};

// Procedural channels → standard Cubism parameter ids.
const CHANNEL_PARAM = {
  angleX: "ParamAngleX", angleY: "ParamAngleY", angleZ: "ParamAngleZ",
  eyeBallX: "ParamEyeBallX", eyeBallY: "ParamEyeBallY",
  eyeLOpen: "ParamEyeLOpen", eyeROpen: "ParamEyeROpen",
  eyeLSmile: "ParamEyeLSmile", eyeRSmile: "ParamEyeRSmile",
  bodyAngleX: "ParamBodyAngleX", bodyAngleY: "ParamBodyAngleY", bodyAngleZ: "ParamBodyAngleZ",
  breath: "ParamBreath", mouthForm: "ParamMouthForm", mouthOpenY: "ParamMouthOpenY",
};

// ------------------------------------------------------------ core access
export class CubismParams {
  constructor(model) {
    const core = model?.internalModel?.coreModel;
    if (!core?._parameterIds) throw new Error("not a Cubism 4 model (no coreModel._parameterIds)");
    this.core = core;
    this.ids = Array.from(core._parameterIds, String);
    this.min = Array.from(core._parameterMinimumValues, Number);
    this.max = Array.from(core._parameterMaximumValues, Number);
    this.def = Array.from(core._parameterDefaultValues ?? core._parameterValues, Number);
    this.index = new Map(this.ids.map((id, i) => [id, i]));
  }
  has(id) { return this.index.has(id); }
  get(id) { const i = this.index.get(id); return i === undefined ? undefined : this.core._parameterValues[i]; }
  set(id, v) {
    const i = this.index.get(id);
    if (i === undefined) return;
    this.core._parameterValues[i] = clamp(v, this.min[i], this.max[i]);
  }
  manifestParams() {
    return this.ids.map((id, i) => ({ id, min: this.min[i], max: this.max[i], default: this.def[i] }))
      .filter((p) => Math.abs(p.max - p.min) > 1e-4);
  }
}

// ------------------------------------------------------------ the body
export class Live2DBody {
  constructor(model, { seed = 1, tuning = {} } = {}) {
    this.model = model;
    this.params = new CubismParams(model);
    this.tuning = { ...DEFAULT_TUNING, ...tuning };
    this.rng = makeRng(seed);
    this.state = "IDLE";
    this.held = new Map();      // paramId → value; agent layer, wins over procedural
    this.holdTimers = new Map();
    this.mouthLevel = 0;        // 0..1 from lip sync
    this.idleRunning = true;
    this.s = {};                // procedural state
    this._resetProcedural(performance.now());

    // pixi-live2d-display's Cubism4InternalModel.update() runs, per frame:
    //   motion → emit("afterMotionUpdate") → saveParameters() → expression,
    //   eyeBlink, focus, natural movements, physics, pose →
    //   emit("beforeModelUpdate") → coreModel.update() → loadParameters().
    // Writing before update() (the old approach) let motions overwrite us, so a
    // held param only stuck if nothing else touched it. We write at two points:
    //   afterMotionUpdate  – physics sees our head/body angles, and the saved
    //                        snapshot (restored by loadParameters) carries them;
    //   beforeModelUpdate  – last word over eyeBlink/focus/physics before render.
    // The receipt reads `rendered`, captured right before coreModel.update().
    const internal = model.internalModel;
    let last = performance.now();
    this.rendered = new Map();
    internal.on("afterMotionUpdate", () => {
      const t = performance.now();
      const delta = Math.min(t - last, 100); last = t;
      if (this.idleRunning) this._tick(delta, t);
      this._apply();
    });
    internal.on("beforeModelUpdate", () => {
      this._apply();
      for (const id of this.params.ids) this.rendered.set(id, this.params.get(id));
    });
  }

  /** Value as last rendered (falls back to the live core value). */
  renderedValue(id) { return this.rendered.has(id) ? this.rendered.get(id) : this.params.get(id); }

  manifest() {
    return {
      kind: "live2d",
      runtime: "cubism4 / pixi-live2d-display",
      states: STATES,
      params: this.params.manifestParams(),
      channels: Object.fromEntries(Object.entries(CHANNEL_PARAM).filter(([, p]) => this.params.has(p))),
      lip_sync: this.params.has("ParamMouthOpenY"),
      adapted_from: "Vroku's Live2D controller suite (Storm/Luna canvas)",
    };
  }

  /** Apply an `act` command; returns the receipt (values read back next frame). */
  async act(cmd) {
    if (cmd.state && STATES.includes(cmd.state)) this.setState(cmd.state);
    for (const [id, v] of Object.entries(cmd.params || {})) {
      if (!this.params.has(id)) continue;
      this.held.set(id, Number(v));
      clearTimeout(this.holdTimers.get(id));
      if (cmd.hold_ms) this.holdTimers.set(id, setTimeout(() => this.held.delete(id), cmd.hold_ms));
    }
    for (const id of cmd.release || []) { this.held.delete(id); clearTimeout(this.holdTimers.get(id)); }
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    return this.receipt(cmd);
  }

  receipt(cmd = {}) {
    const touched = new Set([...Object.keys(cmd.params || {}), ...(cmd.release || [])]);
    const params = {};
    for (const id of touched) if (this.params.has(id)) params[id] = round(this.renderedValue(id));
    const channels = {};
    for (const [ch, id] of Object.entries(CHANNEL_PARAM)) if (this.params.has(id)) channels[ch] = round(this.renderedValue(id));
    return { state: this.state, params, held: Object.fromEntries(this.held), channels, t: Date.now() };
  }

  setState(state) {
    this.state = state;
    this.s.lookState = "WAITING";
    this.s.nextLook = performance.now();
  }

  // ---------------------------------------------------------- procedural layer
  _rand(min, max) { return min + (max - min) * this.rng(); }
  _schedule(key, min, max, now) { this.s[key] = now + this._rand(min, max); }

  _resetProcedural(now) {
    const T = this.tuning;
    const s = (this.s = {});
    for (const ch of Object.keys(CHANNEL_PARAM)) s[ch] = this.params.get(CHANNEL_PARAM[ch]) ?? 0;
    s.eyeLOpen = s.eyeROpen = 1;
    Object.assign(s, { lookX: 0, lookY: 0, lookState: "WAITING", sacX: 0, sacY: 0, blink: "open", smiling: false, smileEnd: 0, tilting: false, tiltEnd: 0, tiltZ: 0 });
    this._schedule("nextLook", 500, T.lookIntervalMin, now);
    this._schedule("nextBlink", T.blinkIntervalMin, T.blinkIntervalMax, now);
    this._schedule("nextSaccade", 100, T.saccadeIntervalMin, now);
    this._schedule("nextSmile", T.smileIntervalMin, T.smileIntervalMax, now);
    this._schedule("nextTilt", T.tiltIntervalMin, T.tiltIntervalMax, now);
  }

  _tick(delta, now) {
    const T = this.tuning, s = this.s;
    const ff = clamp(delta / (1000 / 60), 0.1, 3);
    const k = (f) => 1 - Math.pow(1 - f, ff);
    const focus = k(T.focusLerp), base = k(T.baseLerp), eye = k(T.eyeLerp);

    let hx = 0, hy = 0, hz = 0, ex = 0, ey = 0, smile = 0, eyeSmile = 0, bx = 0, by = 0, bz = 0;
    s.breath = ((Math.sin((now / 4500) * 2 * Math.PI) + 1) / 2) * T.breathAmp;
    // Always-on sway (was an unwired slider in the original).
    bz = Math.sin(now / 3700) * T.bodyAngleZMax * 0.3 * T.bodySwayAmount;

    switch (this.state) {
      case "IDLE":
      case "PRETENDING_NOT_TO_NOTICE": {
        const pretend = this.state !== "IDLE";
        if (now >= s.nextLook) {
          if (s.lookState === "WAITING") {
            s.lookState = "AWAY";
            s.lookX = pretend ? (this.rng() > 0.5 ? 1 : -1) * T.angleXRange : this._rand(-T.angleXRange, T.angleXRange);
            s.lookY = pretend ? 0 : this._rand(-T.angleYRange * 0.7, T.angleYRange * 0.7);
            pretend ? this._schedule("nextLook", 150, 400, now) : this._schedule("nextLook", T.lookDurationMin, T.lookDurationMax, now);
          } else {
            s.lookState = "WAITING"; s.lookX = 0; s.lookY = 0;
            pretend ? this._schedule("nextLook", 2000, 5000, now) : this._schedule("nextLook", T.lookIntervalMin, T.lookIntervalMax, now);
          }
        }
        hx = s.lookX; hy = s.lookY;
        break;
      }
      case "DANCING": {
        const p = now / (2000 / T.danceEnergy);
        hx = Math.sin(p) * T.angleXRange * 0.5;
        hy = (Math.cos(p * 2) * 0.5 + 0.5) * T.angleYRange * 0.4;
        hz = Math.cos(p) * T.angleZTiltMax;
        bx = Math.sin(p + Math.PI) * T.bodyAngleXMax * 0.8 * T.bodySwayAmount;
        by = (Math.cos(p * 4) * 0.5 + 0.5) * T.bodyBounceAmount * 10;
        eyeSmile = 1;
        break;
      }
      case "VIBING": {
        const p = now / (5000 / T.danceEnergy);
        hy = (Math.cos(p) * 0.5 + 0.5) * T.angleYRange * 0.2;
        hz = Math.sin(p) * T.angleZTiltMax * 0.7;
        bx = Math.sin(p) * T.bodyAngleXMax * 0.5 * T.bodySwayAmount;
        eyeSmile = 0.5;
        break;
      }
      case "EXCITED":
        if (now >= s.nextLook) {
          s.lookX = this._rand(-T.angleXRange, T.angleXRange) * 0.5;
          s.lookY = this._rand(-T.angleYRange, T.angleYRange) * 0.5;
          this._schedule("nextLook", 100, 300, now);
        }
        hx = s.lookX; hy = s.lookY;
        by = Math.abs(Math.sin(now / 200)) * 5 * T.bodyBounceAmount;
        smile = 1; eyeSmile = 1;
        break;
      case "LEANING_FORWARD":
        hy = T.leanAmount; by = T.leanAmount;
        break;
      case "REFUSING_TO_LOOK":
        hx = -T.angleXRange; hy = -T.angleYRange * 0.5; bx = -T.bodyAngleXMax;
        break;
    }

    // Idle-only flourishes (scheduled in the original but never applied).
    if (this.state === "IDLE") {
      if (!s.smiling && now >= s.nextSmile) { s.smiling = true; s.smileEnd = now + T.smileDuration; }
      if (s.smiling && now >= s.smileEnd) { s.smiling = false; this._schedule("nextSmile", T.smileIntervalMin, T.smileIntervalMax, now); }
      if (s.smiling) { smile = T.smileVal; eyeSmile = T.eyeSmileVal; }
      if (!s.tilting && now >= s.nextTilt) { s.tilting = true; s.tiltEnd = now + T.tiltDuration; s.tiltZ = this._rand(-T.angleZTiltMax, T.angleZTiltMax); }
      if (s.tilting && now >= s.tiltEnd) { s.tilting = false; this._schedule("nextTilt", T.tiltIntervalMin, T.tiltIntervalMax, now); }
      if (s.tilting) hz += s.tiltZ;
    }

    if (this.state === "IDLE" || this.state === "EXCITED") {
      if (now >= s.nextSaccade) {
        s.sacX = s.angleX * 0.1 * (1 + T.followGazeIntensity * 10) / 30 + this._rand(-T.saccadeMagnitude, T.saccadeMagnitude);
        s.sacY = s.angleY * 0.1 * (1 + T.followGazeIntensity * 10) / 30 + this._rand(-T.saccadeMagnitude, T.saccadeMagnitude);
        this._schedule("nextSaccade", T.saccadeIntervalMin, T.saccadeIntervalMax, now);
      }
      ex = s.sacX; ey = s.sacY;
    } else {
      // Eyes follow the head a little in every other state.
      ex = (hx / Math.max(T.angleXRange, 1)) * T.followGazeIntensity * 5;
      ey = (hy / Math.max(T.angleYRange, 1)) * T.followGazeIntensity * 5;
    }

    const excited = this.state === "EXCITED" ? 3 : 1;
    if (now >= s.nextBlink && s.blink === "open") {
      s.blink = "closing";
      this._schedule("nextBlink", T.blinkIntervalMin / excited, T.blinkIntervalMax / excited, now);
    }
    if (s.blink === "closing") {
      s.eyeLOpen = s.eyeROpen = Math.max(0, s.eyeLOpen - 0.15 * ff);
      if (s.eyeLOpen <= 0) s.blink = "opening";
    } else if (s.blink === "opening") {
      s.eyeLOpen = s.eyeROpen = Math.min(1, s.eyeLOpen + 0.2 * ff);
      if (s.eyeLOpen >= 1) s.blink = "open";
    }

    s.angleX = lerp(s.angleX, hx, focus);
    s.angleY = lerp(s.angleY, hy, focus);
    s.angleZ = lerp(s.angleZ, hz + s.angleX * 0.3, focus);
    s.bodyAngleX = lerp(s.bodyAngleX, bx + s.angleX * 0.1, base);
    s.bodyAngleY = lerp(s.bodyAngleY, by, base);
    s.bodyAngleZ = lerp(s.bodyAngleZ, bz, base);
    s.eyeBallX = lerp(s.eyeBallX, ex, eye);
    s.eyeBallY = lerp(s.eyeBallY, ey, eye);
    s.mouthForm = lerp(s.mouthForm, smile, base);
    s.eyeLSmile = lerp(s.eyeLSmile, eyeSmile, base);
    s.eyeRSmile = lerp(s.eyeRSmile, eyeSmile, base);
    s.mouthOpenY = lerp(s.mouthOpenY, this.mouthLevel, 0.5);
  }

  _apply() {
    for (const [ch, id] of Object.entries(CHANNEL_PARAM)) {
      if (this.held.has(id)) continue;
      if (this.s[ch] !== undefined) this.params.set(id, this.s[ch]);
    }
    for (const [id, v] of this.held) this.params.set(id, v);
  }
}

function round(v) { return typeof v === "number" ? Math.round(v * 1000) / 1000 : v; }

// ------------------------------------------------------------ wiring to Mod³
/** Connect a Live2DBody to Mod³'s body channel. Reconnects on drop. */
export function connectBody(body, { bodyId, origin = location.origin, onStatus = () => {} }) {
  const url = origin.replace(/^http/, "ws") + `/ws/body/${encodeURIComponent(bodyId)}`;
  let ws, closed = false;
  const open = () => {
    ws = new WebSocket(url);
    ws.onopen = () => { ws.send(JSON.stringify({ type: "hello", manifest: body.manifest() })); };
    ws.onmessage = async (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "welcome") onStatus("connected");
      if (msg.type === "act") {
        const receipt = await body.act(msg);
        ws.send(JSON.stringify({ type: "receipt", id: msg.id, ...receipt }));
        onStatus(`act → ${receipt.state}`);
      }
    };
    ws.onclose = () => { onStatus("disconnected"); if (!closed) setTimeout(open, 1500); };
  };
  open();
  return () => { closed = true; ws?.close(); };
}

/**
 * Lip sync: subscribe to a Mod³ speaking session's audio feed and drive the
 * mouth from playback loudness. Audio is played here too (the feed replaces
 * local playback for that session), so voice and mouth stay in sync.
 */
export function connectLipSync(body, { sessionId, origin = location.origin }) {
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.connect(ctx.destination);
  const buf = new Float32Array(analyser.fftSize);
  let playAt = 0;
  const ws = new WebSocket(origin.replace(/^http/, "ws") + `/ws/audio/${encodeURIComponent(sessionId)}`);
  ws.onopen = () => ws.send(JSON.stringify({ label: "rtvi-ai", type: "client-ready", id: crypto.randomUUID(), data: { version: "1.3.0", about: { library: "live2d-body" } } }));
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return;
    const msg = JSON.parse(ev.data);
    if (msg.type !== "bot-tts-audio") return;
    const bytes = Uint8Array.from(atob(msg.data.audio), (c) => c.charCodeAt(0));
    const pcm = new Int16Array(bytes.buffer, 0, bytes.byteLength >> 1);
    const ab = ctx.createBuffer(1, pcm.length, msg.data.sample_rate || 24000);
    const ch = ab.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = ab; src.connect(analyser);
    playAt = Math.max(playAt, ctx.currentTime);
    src.start(playAt); playAt += ab.duration;
  };
  const loop = () => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0; for (const v of buf) sum += v * v;
    body.mouthLevel = clamp(Math.sqrt(sum / buf.length) * body.tuning.mouthGain, 0, 1);
    requestAnimationFrame(loop);
  };
  loop();
  return { resume: () => ctx.resume(), close: () => { ws.close(); ctx.close(); } };
}
