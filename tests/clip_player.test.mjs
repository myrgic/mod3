// Unit tests for dashboard/clip-player.js (node:test, no dependencies).
// Run: node --test tests/clip_player.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";

const { ClipPlayer } = await import("../dashboard/clip-player.js");

function rig() {
  const values = { ParamAngleX: 0 };
  let state = "IDLE";
  const player = new ClipPlayer({
    channels: { "head.x": { polarity: "bi", params: ["ParamAngleX"] } },
    get: (id) => values[id],
    range: (id) => (id in values ? { min: -30, max: 30, def: 0 } : null),
    rng: () => 0.5,
    setState: (s) => { const prev = state; state = s; return prev; },
  });
  return { player, values, state: () => state };
}

const clip = (name, extra = {}) => ({
  name, loop: false, duration_ms: 1000, fade_ms: 0,
  tracks: { "head.x": { const: [0.5], ease: "linear", smooth: 1 } }, ...extra,
});

test("overlapping base clips: the first to end does not clobber the second", () => {
  const { player, state } = rig();
  player.play(clip("a", { base: "VIBING", duration_ms: 2000 }), 0);
  assert.equal(state(), "VIBING");
  player.play(clip("b", { base: "EXCITED", loop: true, duration_ms: null }), 500);
  assert.equal(state(), "EXCITED");
  player.tick(2001, 16); // a ends while b is still looping
  assert.equal(state(), "EXCITED", "b still owns the base state");
  player.stop("b", 2100);
  player.tick(2101, 16);
  assert.equal(state(), "IDLE", "the pre-base state comes back after the last base clip");
});

test("inner base clip ending hands the state back to the outer one", () => {
  const { player, state } = rig();
  player.play(clip("outer", { base: "VIBING", loop: true, duration_ms: null }), 0);
  player.play(clip("inner", { base: "EXCITED", duration_ms: 500 }), 100);
  player.tick(601, 16);
  assert.equal(state(), "VIBING");
  player.stop("outer", 700);
  player.tick(701, 16);
  assert.equal(state(), "IDLE");
});

test("a clip writes its channel in model units and ends on time", () => {
  const { player } = rig();
  player.play(clip("x"), 0);
  player.tick(10, 16);
  assert.deepEqual(player.out.get("ParamAngleX"), [15, 1]); // 0.5 of +30
  player.tick(1001, 16);
  assert.equal(player.playing.size, 0);
});
