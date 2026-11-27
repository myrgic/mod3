// avatar3d-body-speak.test.mjs: speak() stores the cue timeline and cueAt() reads it on the same clock as env.
// Pure Node, no DOM/WebGL: the body is render-agnostic by design.
import assert from "node:assert/strict";
import { Avatar3DBody } from "../dashboard/avatar3d-body.js";

const t0 = 5000;
const b = new Avatar3DBody();
b.speak({ cues: [[0, "X"], [100, "B"], [300, "E"]], env: [], hop_ms: 20, t0 });
assert.equal(b.cueSource, "server");
// ahead defaults to 35 ms, so cueAt(t0 + 70) looks at t = 105 -> the 100 ms cue "B"
assert.equal(b.cueAt(t0 + 70), "B");
assert.equal(b.cueAt(t0 + 0), "X");          // before the 100 ms cue, the 0 ms cue still holds
assert.equal(b.cueAt(t0 + 400), "X");        // past the line (duration 0 -> rest)
assert.equal(new Avatar3DBody().cueAt(t0), "X", "no timeline is rest, not a throw");
console.log("ok   speak() stores cues; cueAt() reads them on the speak clock");
