// speak() must never throw on a malformed cue element (review finding on mod3#163).
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

const m = new Avatar3DBody();
assert.equal(m.setMood("sad"), "sad");
assert.equal(m.setMood("bogus"), "sad", "an unknown mood is ignored");
assert.equal(m.gesture("nod").clip, "nod");
assert.equal(m.receipt().mood, "sad");
console.log("ok   setMood() and gesture() update the body and the receipt");
