/**
 * avatar3d-body.js — a "three" (three.js) Mod3 body, phase 1 scaffold.
 *
 * Phase 1 scope (3D avatar rig spec, section B): declares the real Mod3 body
 * protocol (manifest/act/play/stop, same wire contract as live2d-body.js's
 * connectBody()) and proves the env-driven mouth-movement path end to end --
 * NO rig.json, NO GLTFLoader, NO three.js scene yet, because there is no
 * avatar asset to render (that's phase 2, the Digi-el pack). This body
 * renders a minimal placeholder (a plane whose scale.y tracks mouth-open)
 * so the avatar3d Stage view has something real and inspectable to show,
 * driven purely by a real say() call's env envelope.
 *
 * Phonetic mouth shapes (section A's `cues`) are accepted by .speak() but
 * NOT YET applied to any morph -- there is no morph target until phase 2's
 * rig.json exists. A body with no cues (every say() call today, since no
 * cue source is wired up) falls back to env exactly like Live2D does; this
 * body's speak() falls back the same way, by design, whether or not cues
 * are ever supplied, until phase 2 lands real viseme morphs.
 */

export const STATES = ["IDLE", "DANCING", "VIBING", "EXCITED",
  "LEANING_FORWARD", "PRETENDING_NOT_TO_NOTICE", "REFUSING_TO_LOOK"];

// Same semantic channel names clip.py already defines (head.x/y/z, body.x/y/z,
// eyes.x/y, eyes.open, mouth.open, mouth.smile, brows.y/angle, cheek, breath) --
// so a clip authored today is already shape-compatible with this body kind,
// even though phase 1 has nothing real behind most of them yet (mouth.open is
// the one channel this phase actually drives, from env).
const SEMANTIC_CHANNELS = {
  "head.x": "bi", "head.y": "bi", "head.z": "bi",
  "body.x": "bi", "body.y": "bi", "body.z": "bi",
  "eyes.x": "bi", "eyes.y": "bi", "eyes.open": "uni",
  "mouth.open": "uni", "mouth.smile": "bi",
  "brows.y": "bi", "brows.angle": "bi",
  "cheek": "uni", "breath": "uni",
};

const round = (v) => Math.round(v * 1000) / 1000;

export class Avatar3DBody {
  constructor({ seed = 1 } = {}) {
    this.state = "IDLE";
    this.held = new Map();          // channel -> value; agent layer, same semantics as live2d-body.js
    this.holdTimers = new Map();
    this.mouthLevel = 0;            // 0..1 from env playback (pad.js sets this directly, same as Live2D)
    this.channels = {};
    for (const [ch, polarity] of Object.entries(SEMANTIC_CHANNELS)) {
      this.channels[ch] = { polarity, value: 0 };
    }
    // The one real, demoable render for phase 1: a placeholder mesh whose
    // mouth-open channel tracks env/mouthLevel every frame. No three.js
    // Scene/Camera/Renderer is constructed here -- the Stage view (avatar3d.html,
    // task 3) owns the three.js scene and calls renderFrame() on this body each
    // animation frame, passing it the elapsed time; this class stays
    // render-agnostic so it is testable without a WebGL context.
    this._cuesActive = null;        // the current speak() cue timeline, or null (env-only fallback)
    this._speakT0 = null;
  }

  /** Mod3 body protocol: declared on /ws/body/{id} hello (same wire shape as live2d-body.js). */
  manifest() {
    return {
      kind: "three",
      runtime: "three.js (phase 1 scaffold: placeholder geometry, no rig)",
      states: STATES,
      // body.py's validate_command() only forwards act params whose id is listed here (with min/max).
      // The param ids ARE the semantic channel names: this body's act() is keyed by channel, so the
      // namespace stays one thing (channel name) end to end. Ranges come from polarity: bi -1..1, uni 0..1.
      params: Object.entries(SEMANTIC_CHANNELS).map(([id, polarity]) => ({
        id, min: polarity === "bi" ? -1 : 0, max: 1, default: 0,
      })),
      semantic_channels: this.channels,
      clip_generators: ["const", "keys", "wave", "jitter", "follow"],
      lip_sync: true,                 // env-driven mouth.open; cue-driven visemes await phase 2's rig.json
      rig: null,                      // phase 2: the loaded rig.json's bone/morph map, once an avatar pack exists
    };
  }

  /** window.body.speak({cues, env, hop_ms, t0}) -- called once per say message (pad.js), on the SAME
   * clock (t0) env/mouthLevel already use. Phase 1: records the cue timeline for later use but does not
   * yet apply it to any morph (no rig = no morph targets) -- env-driven mouth.open via mouthLevel is the
   * only real behavior today, exactly the same whether cues is empty or populated. This method must never
   * throw: pad.js calls it inside a try/catch, but a body that breaks its own call contract is still a bug. */
  speak({ cues = [], env = [], hop_ms = 20, t0 = null } = {}) {
    this._cuesActive = Array.isArray(cues) && cues.length ? cues : null;
    this._speakT0 = t0;
    return { ok: true, cues_received: this._cuesActive ? this._cuesActive.length : 0, fallback: "env" };
  }

  /** Writes the procedural + held layers into every channel's displayed value, held winning (same
   * precedence as live2d-body.js's agent-held params over the procedural layer). Call before reading a
   * receipt or a render frame so both see the same, current values -- never read channels[ch].value
   * directly without calling this first. */
  _apply() {
    // Reset every channel to rest (0) FIRST -- otherwise a released or hold_ms-expired param leaves a
    // stale value behind forever (real bug, caught by review: act({params:{"head.x":.5}}) then
    // act({release:["head.x"]}) left semantic["head.x"] at 0.5 with nothing in `held` to explain it).
    for (const ch of Object.keys(this.channels)) this.channels[ch].value = 0;
    this.channels["mouth.open"].value = Math.max(0, Math.min(1, this.mouthLevel));
    for (const [ch, v] of this.held) {
      if (ch in this.channels) this.channels[ch].value = v;
    }
  }

  /** One frame of the placeholder render. Not wired to a real three.js scene yet (task 3 owns that) --
   * exposed so a browser probe (or a future WebGL harness) can drive and assert on it directly. */
  renderFrame() {
    this._apply();
    return { "mouth.open": this.channels["mouth.open"].value };
  }

  /** Clamp a requested value to the channel's declared range (same as live2d-body.js's clamp()).
   * Returns null for a non-finite value, so NaN/Infinity never reach `held` or the receipt. */
  _clamp(ch, v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    const { min, max } = this.channels[ch].polarity === "bi" ? { min: -1, max: 1 } : { min: 0, max: 1 };
    return Math.min(max, Math.max(min, n));
  }

  /** Apply an `act` command; returns the receipt -- same shape/semantics as live2d-body.js's act(). */
  async act(cmd) {
    if (cmd.state && STATES.includes(cmd.state)) this.setState(cmd.state);
    for (const [ch, raw] of Object.entries(cmd.params || {})) {
      if (!(ch in this.channels)) continue;
      const v = this._clamp(ch, raw);
      if (v === null) continue;
      this.held.set(ch, v);
      clearTimeout(this.holdTimers.get(ch));
      if (cmd.hold_ms != null) {
        this.holdTimers.set(ch, setTimeout(() => this.held.delete(ch), Math.max(0, Number(cmd.hold_ms) || 0)));
      }
    }
    for (const ch of cmd.release || []) { this.held.delete(ch); clearTimeout(this.holdTimers.get(ch)); }
    this._apply();
    return this.receipt(cmd);
  }

  receipt(cmd = {}) {
    // receipt() is the protocol's source of truth (play()/stop() call it directly, not through
    // act()/renderFrame()) -- it must apply the current state itself rather than trust a caller to
    // have called _apply() first (real bug, caught by review: play()/stop() returned a stale
    // mouth.open because nothing upstream of them called _apply()).
    this._apply();
    const all_semantic = {};
    for (const ch of Object.keys(this.channels)) all_semantic[ch] = round(this.channels[ch].value);
    return { state: this.state, held: Object.fromEntries(this.held), semantic: all_semantic, clips: [], t: Date.now() };
  }

  /** Clips: phase 1 has no ClipPlayer wired up (no rig to drive beyond mouth.open) -- stub that answers
   * the protocol honestly rather than silently no-op, so a caller can tell nothing real happened. */
  async play(msg) {
    return { ...this.receipt(), playing: null, note: "phase 1 scaffold: no clip player yet" };
  }

  async stop(msg) {
    return { ...this.receipt(), stopped: [], note: "phase 1 scaffold: no clip player yet" };
  }

  setState(state) {
    this.state = state;
  }
}
