// avatar3d-body.test.mjs — phase 1 smoke test for the "three" Mod3 body scaffold.
// No test runner/package.json exists in this repo for dashboard JS (confirmed before writing this);
// run directly with plain Node (ESM, no DOM/WebGL dependency by design): `node avatar3d-body.test.mjs`
import assert from "node:assert/strict";
import { Avatar3DBody } from "./avatar3d-body.js";

function test(name, fn) {
  try {
    fn();
    console.log(`ok   ${name}`);
  } catch (e) {
    console.error(`FAIL ${name}: ${e.message}`);
    process.exitCode = 1;
  }
}

async function asyncTest(name, fn) {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (e) {
    console.error(`FAIL ${name}: ${e.message}`);
    process.exitCode = 1;
  }
}

test("manifest declares kind=three and the exact clip.py semantic channels", () => {
  const b = new Avatar3DBody();
  const m = b.manifest();
  assert.equal(m.kind, "three");
  assert.equal(m.semantic_channels["mouth.open"].polarity, "uni");
  const expected = ["head.x", "head.y", "head.z", "body.x", "body.y", "body.z", "eyes.x", "eyes.y",
    "eyes.open", "mouth.open", "mouth.smile", "brows.y", "brows.angle", "cheek", "breath"].sort();
  assert.deepEqual(Object.keys(m.semantic_channels).sort(), expected);
});

test("speak() with no cues falls back to env (the only real behavior today)", () => {
  const b = new Avatar3DBody();
  const r = b.speak({ cues: [], env: [0.1, 0.5, 0.9], hop_ms: 20, t0: 1000 });
  assert.equal(r.ok, true);
  assert.equal(r.fallback, "env");
  assert.equal(r.cues_received, 0);
});

test("speak() with real cues records them, still reports the env fallback (no rig to drive visemes yet)", () => {
  const b = new Avatar3DBody();
  const r = b.speak({ cues: [[0, "X"], [100, "B"]], env: [], hop_ms: 20, t0: 2000 });
  assert.equal(r.cues_received, 2);
  assert.equal(r.fallback, "env");
});

test("speak() never throws on a missing/malformed argument", () => {
  const b = new Avatar3DBody();
  assert.doesNotThrow(() => b.speak());
  assert.doesNotThrow(() => b.speak({}));
  assert.doesNotThrow(() => b.speak({ cues: null, env: null }));
});

test("renderFrame tracks mouthLevel exactly and clamps to [0,1]", () => {
  const b = new Avatar3DBody();
  b.mouthLevel = 0.75;
  assert.equal(b.renderFrame()["mouth.open"], 0.75);
  b.mouthLevel = 1.5;
  assert.equal(b.renderFrame()["mouth.open"], 1);
  b.mouthLevel = -0.5;
  assert.equal(b.renderFrame()["mouth.open"], 0);
});

await asyncTest("act() sets state and holds a param, which appears in both held and semantic", async () => {
  const b = new Avatar3DBody();
  await b.act({ state: "VIBING", params: { "mouth.smile": 0.5 } });
  assert.equal(b.state, "VIBING");
  const r = b.receipt();
  assert.equal(r.held["mouth.smile"], 0.5);
  assert.equal(r.semantic["mouth.smile"], 0.5);
});

await asyncTest("act() release clears a held param from both held AND semantic (not just held)", async () => {
  // Real bug caught by review: the original fix applied held values but never reset a released
  // channel back to rest, so `held` was correctly empty while `semantic` kept the stale value.
  const b = new Avatar3DBody();
  await b.act({ params: { "mouth.smile": 0.5 } });
  await b.act({ release: ["mouth.smile"] });
  const r = b.receipt();
  assert.ok(!("mouth.smile" in r.held));
  assert.equal(r.semantic["mouth.smile"], 0, "a released channel must return to rest, not stay stale");
});

await asyncTest("a hold_ms-expired param returns to rest, not a stale value", async () => {
  const b = new Avatar3DBody();
  await b.act({ params: { "head.y": 0.3 }, hold_ms: 10 });
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(b.receipt().semantic["head.y"], 0, "an expired hold_ms timer must not leave a stale value");
});

await asyncTest("play()/stop() reflect live mouthLevel, not a stale value from before _apply ran", async () => {
  // Real bug caught by review: receipt() didn't call _apply() itself, so play()/stop() (which call
  // receipt() directly, not through act()/renderFrame()) returned a stale mouth.open.
  const b = new Avatar3DBody();
  b.mouthLevel = 0.7;
  const p = await b.play({ clip: { name: "x" } });
  assert.equal(p.semantic["mouth.open"], 0.7);
  const s = await b.stop({ clip: { name: "x" } });
  assert.equal(s.semantic["mouth.open"], 0.7);
});

await asyncTest("act() ignores an unknown channel name rather than throwing", async () => {
  const b = new Avatar3DBody();
  await assert.doesNotReject(b.act({ params: { "not.a.real.channel": 1 } }));
});

await asyncTest("play()/stop() honestly report no clip player exists yet (phase 1)", async () => {
  const b = new Avatar3DBody();
  const p = await b.play({ clip: { name: "test" } });
  assert.equal(p.playing, null);
  const s = await b.stop({ clip: { name: "test" } });
  assert.deepEqual(s.stopped, []);
});

if (process.exitCode) {
  console.error("SOME TESTS FAILED");
} else {
  console.log("ALL TESTS PASSED");
}
