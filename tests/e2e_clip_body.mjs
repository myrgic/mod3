// e2e: real live2d-body.js + clip-player.js in Node against a real mod3 server.
// A fake Cubism core stands in for the model; the frame loop calls the same
// afterMotionUpdate/beforeModelUpdate hooks pixi-live2d-display does.
// Usage: node tests/e2e_clip_body.mjs http://127.0.0.1:7871 <bodyId>
import { EventEmitter } from "node:events";

globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 16);
globalThis.location = { origin: process.argv[2] };

const P = [
  ["ParamAngleX", -30, 30, 0], ["ParamAngleY", -30, 30, 0], ["ParamAngleZ", -30, 30, 0],
  ["ParamEyeLOpen", 0, 1, 1], ["ParamEyeROpen", 0, 1, 1], ["ParamEyeLSmile", 0, 1, 0], ["ParamEyeRSmile", 0, 1, 0],
  ["ParamEyeBallX", -1, 1, 0], ["ParamEyeBallY", -1, 1, 0], ["ParamBrowLY", -1, 1, 0], ["ParamBrowRY", -1, 1, 0],
  ["ParamMouthForm", -1, 1, 0], ["ParamMouthOpenY", 0, 1, 0],
  ["ParamBodyAngleX", -10, 10, 0], ["ParamBodyAngleY", -10, 10, 0], ["ParamBodyAngleZ", -10, 10, 0], ["ParamBreath", 0, 1, 0],
];
const internal = new EventEmitter();
internal.coreModel = {
  _parameterIds: P.map((p) => p[0]), _parameterMinimumValues: P.map((p) => p[1]),
  _parameterMaximumValues: P.map((p) => p[2]), _parameterDefaultValues: P.map((p) => p[3]),
  _parameterValues: Float32Array.from(P.map((p) => p[3])),
};
const model = { internalModel: internal };
setInterval(() => { internal.emit("afterMotionUpdate"); internal.emit("beforeModelUpdate"); }, 16);

const { Live2DBody, connectBody } = await import("../dashboard/live2d-body.js");
const body = new Live2DBody(model, { seed: 7 });
body.idleRunning = false; // isolate clip motion from the procedural idle
connectBody(body, { bodyId: process.argv[3] || "e2e", origin: process.argv[2], onStatus: (s) => console.error("[body]", s) });
setTimeout(() => process.exit(0), Number(process.argv[4] || 20000));
