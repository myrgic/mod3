// Contract tests for Live2DBody.manifest() / receipt() shape (node:test).
// #154 shipped `channels` as {proceduralName: value-or-paramId}; this PR must
// only ADD fields, never rename or move existing ones.
// Run: node --test tests/live2d_body_contract.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";

globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 1);

const P = [
  ["ParamAngleX", -30, 30, 0], ["ParamAngleY", -30, 30, 0], ["ParamMouthForm", -1, 1, 0],
  ["ParamEyeLOpen", 0, 1, 1], ["ParamEyeROpen", 0, 1, 1], ["ParamBreath", 0, 1, 0],
];
function fakeModel() {
  const internal = new EventEmitter();
  internal.coreModel = {
    _parameterIds: P.map((p) => p[0]), _parameterMinimumValues: P.map((p) => p[1]),
    _parameterMaximumValues: P.map((p) => p[2]), _parameterDefaultValues: P.map((p) => p[3]),
    _parameterValues: Float32Array.from(P.map((p) => p[3])),
  };
  return { internalModel: internal };
}

const { Live2DBody } = await import("../dashboard/live2d-body.js");

test("manifest keeps #154's channels shape; clip channels are a new field", () => {
  const body = new Live2DBody(fakeModel());
  const m = body.manifest();
  assert.equal(m.channels.angleX, "ParamAngleX");
  assert.equal(m.channels.mouthForm, "ParamMouthForm");
  assert.equal(m.channels.breath, "ParamBreath");
  assert.ok(!("head.x" in m.channels), "semantic names must not leak into channels");
  assert.deepEqual(m.semantic_channels["head.x"], { polarity: "bi", params: ["ParamAngleX"] });
  assert.deepEqual(m.semantic_channels["eyes.open"].params, ["ParamEyeLOpen", "ParamEyeROpen"]);
});

test("receipt keeps #154's channels keys; semantic and clips are additions", async () => {
  const body = new Live2DBody(fakeModel());
  const r = await body.act({ params: { ParamAngleX: 12 } });
  for (const k of ["angleX", "angleY", "mouthForm", "eyeLOpen", "breath"]) {
    assert.equal(typeof r.channels[k], "number", `channels.${k}`);
  }
  assert.ok(!("raw" in r), "no renamed field");
  assert.equal(typeof r.semantic["head.x"], "number");
  assert.ok(Array.isArray(r.clips));
  assert.deepEqual(Object.keys(r).filter((k) => ["state", "params", "held", "channels", "t"].includes(k)).sort(),
    ["channels", "held", "params", "state", "t"]);
});
