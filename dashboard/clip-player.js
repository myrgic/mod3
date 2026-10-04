/**
 * clip-player.js — runs compiled clips (see clip.py) on a body.
 *
 * A clip is data: per-channel tracks built from five generators that are
 * summed each frame — const, keys, wave, jitter, follow — then optionally
 * smoothed. The server has already checked the clip and resolved every
 * expression to a number; nothing here evaluates agent-supplied code.
 *
 * Channel values are normalised: "bi" channels are -1..1 around the
 * parameter's default, "uni" channels are 0..1 across its range. The body
 * supplies the mapping (semantic channel → model parameter ids).
 */

const clamp = (v, lo, hi) => Math.min(Math.max(Number.isFinite(v) ? v : 0, lo), hi);

const EASE = {
  linear: (x) => x,
  in: (x) => x * x,
  out: (x) => 1 - (1 - x) * (1 - x),
  inOut: (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2),
  step: () => 0,
};

const SHAPE = {
  sin: (x) => Math.sin(2 * Math.PI * x),
  abs: (x) => Math.abs(Math.sin(2 * Math.PI * x)),
  tri: (x) => { const f = x - Math.floor(x); return f < 0.25 ? 4 * f : f < 0.75 ? 2 - 4 * f : 4 * f - 4; },
  square: (x) => (x - Math.floor(x) < 0.5 ? 1 : -1),
};

function sampleKeys(keys, t, ease) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [t1, v1] = keys[i];
    if (t <= t1) {
      const [t0, v0] = keys[i - 1];
      const x = t1 === t0 ? 1 : (t - t0) / (t1 - t0);
      return v0 + (v1 - v0) * EASE[ease](x);
    }
  }
  return keys[keys.length - 1][1];
}

export class ClipPlayer {
  /**
   * @param {object} opts
   * @param {object} opts.channels  semantic channel → {params: [ids], polarity}
   * @param {(id: string) => number} opts.get  current param value
   * @param {(id: string) => {min:number,max:number,def:number}} opts.range
   * @param {() => number} opts.rng  seeded random in [0, 1)
   * @param {(state: string|null) => string|null} opts.setState  for clip.base
   */
  constructor({ channels, get, range, rng, setState }) {
    Object.assign(this, { channels, get, range, rng, setState });
    this.playing = new Map(); // name → instance
    this.out = new Map();     // paramId → [target, weight]
    this.norm = {};           // channel → normalised value written this frame
  }

  play(clip, now = performance.now()) {
    const prev = this.playing.get(clip.name);
    const inst = {
      clip, t0: now, stopAt: null,
      prevState: prev ? prev.prevState : null,
      smooth: {}, jit: {},
    };
    if (clip.base) inst.prevState = this.setState(clip.base);
    this.playing.delete(clip.name); // re-insert so the newest clip wins ties
    this.playing.set(clip.name, inst);
    return inst;
  }

  stop(name, now = performance.now()) {
    const names = name ? [name] : [...this.playing.keys()];
    for (const n of names) {
      const inst = this.playing.get(n);
      if (inst && inst.stopAt === null) inst.stopAt = now;
    }
    return names.filter((n) => this.playing.has(n));
  }

  list() {
    const now = performance.now();
    return [...this.playing.values()].map((i) => ({
      name: i.clip.name, t_ms: Math.round(now - i.t0), loop: i.clip.loop,
      duration_ms: i.clip.duration_ms, stopping: i.stopAt !== null,
    }));
  }

  // channel value → param units
  toParam(ch, id, v) {
    const r = this.range(id);
    if (!r) return undefined;
    if (this.channels[ch]?.polarity === "uni") return r.min + clamp(v, 0, 1) * (r.max - r.min);
    v = clamp(v, -1, 1);
    return r.def + (v >= 0 ? v * (r.max - r.def) : v * (r.def - r.min));
  }

  // param units → channel value (for follow of a channel no clip drives)
  fromParam(ch, id) {
    const r = this.range(id), p = this.get(id);
    if (!r || p === undefined) return 0;
    if (this.channels[ch]?.polarity === "uni") return r.max === r.min ? 0 : (p - r.min) / (r.max - r.min);
    if (p >= r.def) return r.max === r.def ? 0 : (p - r.def) / (r.max - r.def);
    return r.def === r.min ? 0 : (p - r.def) / (r.def - r.min);
  }

  channelNow(ch) {
    if (ch in this.norm) return this.norm[ch];
    const ids = this.channels[ch]?.params;
    return ids?.length ? this.fromParam(ch, ids[0]) : 0;
  }

  _jitter(inst, key, g, t) {
    let st = inst.jit[key];
    const pick = () => (g.pick ? g.pick[Math.floor(this.rng() * g.pick.length)] : g.min + (g.max - g.min) * this.rng());
    const between = ([a, b]) => a + (b - a) * this.rng();
    if (!st) st = inst.jit[key] = { v: g.hold_ms ? g.rest : pick(), next: t + between(g.every_ms), until: null };
    if (st.until !== null && t >= st.until) { st.v = g.rest; st.until = null; }
    if (t >= st.next) {
      st.v = pick();
      st.next = t + between(g.every_ms);
      st.until = g.hold_ms ? t + between(g.hold_ms) : null;
    }
    return st.v;
  }

  _value(inst, ch, track, t, span) {
    let v = 0;
    for (const c of track.const || []) v += c;
    for (const keys of track.keys || []) v += sampleKeys(keys, t % span, track.ease);
    for (const w of track.wave || []) v += w.offset + w.amp * SHAPE[w.shape](t / w.period_ms + w.phase);
    (track.jitter || []).forEach((g, i) => { v += this._jitter(inst, `${ch}:${i}`, g, t); });
    for (const f of track.follow || []) v += f.offset + f.gain * this.channelNow(f.of);
    return v;
  }

  /** Advance one frame. Fills this.out (paramId → [target, weight]). */
  tick(now, delta) {
    this.out.clear();
    this.norm = {};
    const ff = clamp(delta / (1000 / 60), 0.1, 3);
    for (const [name, inst] of this.playing) {
      const c = inst.clip, t = now - inst.t0;
      const span = c.duration_ms || Math.max(1, ...Object.values(c.tracks).flatMap((tr) => (tr.keys || []).map((k) => k[k.length - 1][0])));
      // weight: fade in, fade out at the end or after stop()
      let w = c.fade_ms ? clamp(t / c.fade_ms, 0, 1) : 1;
      let done = false;
      if (!c.loop && c.duration_ms) {
        if (c.fade_ms) w = Math.min(w, clamp((c.duration_ms - t) / c.fade_ms, 0, 1));
        if (t >= c.duration_ms) done = true;
      }
      if (inst.stopAt !== null) {
        const ts = now - inst.stopAt;
        w = Math.min(w, c.fade_ms ? clamp(1 - ts / c.fade_ms, 0, 1) : 0);
        if (w <= 0) done = true;
      }
      if (done) {
        this.playing.delete(name);
        if (c.base && inst.prevState) this.setState(inst.prevState);
        continue;
      }
      // follow tracks read other channels, so do them last
      const order = Object.entries(c.tracks).sort(([, a], [, b]) => (a.follow ? 1 : 0) - (b.follow ? 1 : 0));
      for (const [ch, track] of order) {
        let v = this._value(inst, ch, track, t, span);
        if (track.smooth < 1) {
          const k = 1 - Math.pow(1 - track.smooth, ff);
          const prev = inst.smooth[ch] ?? this.channelNow(ch);
          v = prev + (v - prev) * k;
          inst.smooth[ch] = v;
        }
        if (ch.startsWith("param:")) {
          const id = ch.slice(6), r = this.range(id);
          if (r) this.out.set(id, [clamp(v, r.min, r.max), w]);
          continue;
        }
        this.norm[ch] = v;
        for (const id of this.channels[ch]?.params || []) {
          const p = this.toParam(ch, id, v);
          if (p !== undefined) this.out.set(id, [p, w]);
        }
      }
    }
  }

  /** Blend this frame's clip output over whatever is in the model now. */
  apply(set) {
    for (const [id, [target, w]] of this.out) {
      const cur = this.get(id);
      set(id, cur === undefined ? target : cur + (target - cur) * w);
    }
  }
}
