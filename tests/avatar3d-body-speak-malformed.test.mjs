// speak() must never throw on a malformed cue element (review findings on mod3#163 and stage#27).
import assert from "node:assert/strict";
import { Avatar3DBody } from "../dashboard/avatar3d-body.js";

const b = new Avatar3DBody();
assert.doesNotThrow(() => b.speak({ cues: [null, 5, "x", [0, "X"], { t: 0.1, s: "B" }], env: [], t0: 0 }));
assert.equal(b.cueSource, "server");
assert.equal(b.cueAt(35), "X", "the array cue at 0 ms survives the malformed neighbours");
assert.equal(b.cueAt(135), "B", "the object cue at 100 ms survives too");

const allBad = new Avatar3DBody();
const r = allBad.speak({ cues: [null, undefined], env: [0.5], hop_ms: 20, t0: 0 });
assert.equal(r.ok, true);
assert.equal(allBad.cueSource, "env", "no usable cue falls back to env, not an empty server timeline");
console.log("ok   speak() drops malformed cue elements and falls back when none are usable");

// Wrong-shape elements with no finite time must not leave a 'server' source with an empty timeline.
const shaped = new Avatar3DBody();
const rs = shaped.speak({ cues: [[], {}], env: [0.5], hop_ms: 20, t0: 0 });
assert.equal(rs.fallback, "env", "wrong-shape cues with no finite time fall back to env");
assert.equal(rs.cues_received, 0);
assert.equal(shaped.cueSource, "env");
console.log("ok   wrong-shape cues ([] / {}) fall back to env instead of an empty server timeline");

// A Symbol field cannot throw out of speak(); a null argument is accepted.
const sym = new Avatar3DBody();
assert.doesNotThrow(() => sym.speak({ cues: [{ t: Symbol("x") }, [Symbol("y"), 1]], env: [0.5], t0: 0 }));
assert.equal(sym.cueSource, "env");
assert.doesNotThrow(() => new Avatar3DBody().speak(null));
console.log("ok   Symbol fields and a null argument do not throw");

// The analyser branch gets the same treatment: an analyser timeline with no usable element is no timeline.
const fakeAudio = { getChannelData: () => new Float32Array(8), sampleRate: 16000, duration: 0.5 };
const an = new Avatar3DBody();
an.lipsync = () => [{ t: NaN, s: "A" }, {}];   // no null: a null would throw inside the old map and mask the bug
const ra = an.speak({ cues: [], env: [], t0: 0, audio: fakeAudio });
assert.equal(an.cueSource, "env", "an analyser timeline with no finite time falls back to env, not 'analyser'");
assert.equal(ra.source, "env");
assert.equal(ra.fallback, "env");
assert.equal(ra.cues_received, 0);
// ...and a good analyser timeline (lipsync.js shape: {t: seconds, s}) is used, in ms, with bad neighbours dropped.
const ag = new Avatar3DBody();
ag.lipsync = () => [{ t: 0.1, s: "C" }, { t: NaN, s: "Q" }, { t: 0, s: "B" }];
const rg = ag.speak({ cues: [], env: [], t0: 0, audio: fakeAudio });
assert.equal(rg.source, "analyser");
assert.equal(rg.cues_received, 2);
assert.equal(rg.fallback, null);
assert.equal(ag.cueAt(0, 0), "B");
assert.equal(ag.cueAt(100, 0), "C", "analyser seconds become ms on the same clock as server cues");
console.log("ok   analyser cues: none usable -> env; usable ones kept (s -> ms), bad ones dropped");

// Review of 15cf7b0: wrong-TYPED times are not coerced (null/""/true/false would become t=0/1), and no opts field
// can throw out of speak() (Symbol/BigInt arithmetic, a throwing getter).
const typed = new Avatar3DBody();
const rt = typed.speak({ cues: [[null, "A"], ["", "B"], [true, "C"], [false, "D"], ["5", "E"], { t: null, s: "F" }, { t: "", s: "G" }, { t: true, s: "H" }], env: [0.5], t0: 0 });
assert.equal(rt.cues_received, 0, "no wrong-typed time survives as a cue");
assert.equal(rt.source, "env");
assert.equal(rt.fallback, "env");
const mixed = new Avatar3DBody();
const rm = mixed.speak({ cues: [[null, "A"], [50, "B"], { t: "", s: "C" }, { t: 0.2, s: "D" }], env: [0.5], hop_ms: 20, t0: 0 });
assert.equal(rm.cues_received, 2, "only the two number-typed times survive");
assert.equal(mixed.cueAt(50, 0), "B");
assert.equal(mixed.cueAt(200, 0), "D");
const thrower = { get duration() { throw new Error("boom"); }, getChannelData: () => new Float32Array(4), sampleRate: 16000 };
for (const opts of [
  { cues: [], env: [0.5], hop_ms: Symbol("h"), t0: 0 },
  { cues: [], env: [0.5], hop_ms: 10n, t0: 0 },
  { cues: [], env: [], t0: 0, audio: { duration: 2n, getChannelData: () => new Float32Array(4), sampleRate: 16000 } },
  { cues: [], env: [], t0: 0, audio: thrower },
  { cues: [], env: [], t0: Symbol("t") },
  Object.defineProperty({}, "cues", { get() { throw new Error("getter"); } }),
]) {
  const b2 = new Avatar3DBody();
  assert.doesNotThrow(() => b2.speak(opts), "speak() must never throw");
  assert.ok(Number.isFinite(b2._speakDur), "duration stays a finite number");
  assert.doesNotThrow(() => b2.cueAt(10));
}
const hb = new Avatar3DBody();
hb.speak({ cues: [], env: [0.5, 0.5], hop_ms: 10n, t0: 0 });
assert.equal(hb._speakDur, 40, "a non-number hop_ms falls back to 20 ms per env frame");
assert.doesNotThrow(() => new Avatar3DBody().cueAt(Symbol("now")));
console.log("ok   wrong-typed times are dropped and no opts field can throw out of speak()/cueAt()");

const m = new Avatar3DBody();
assert.equal(m.setMood("sad"), "sad");
assert.equal(m.setMood("bogus"), "sad", "an unknown mood is ignored");
assert.equal(m.gesture("nod").clip, "nod");
assert.equal(m.receipt().mood, "sad");
console.log("ok   setMood() and gesture() update the body and the receipt");
