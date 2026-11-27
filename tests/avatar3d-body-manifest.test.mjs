// avatar3d-body manifest + act contract. Run by CI: node --test tests/*.test.mjs
// Pins tests/fixtures/avatar3d_manifest.json (read by tests/test_avatar3d_body_params.py through body.py)
// to what manifest() actually emits, and checks that act() clamps and drops non-finite values.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { Avatar3DBody } from "../dashboard/avatar3d-body.js";

const fixture = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/avatar3d_manifest.json", import.meta.url)), "utf8"));

test("manifest params list matches the python-side fixture exactly", () => {
  assert.deepEqual(new Avatar3DBody().manifest().params, fixture.params);
});

test("every declared param id is a semantic channel (one namespace for act and manifest)", () => {
  const b = new Avatar3DBody();
  for (const p of b.manifest().params) assert.ok(p.id in b.channels, p.id);
  assert.equal(b.manifest().params.length, Object.keys(b.channels).length);
});

test("act() clamps to polarity range and drops non-finite values", async () => {
  const b = new Avatar3DBody();
  await b.act({ params: { "head.x": 5, "mouth.open": -3, "mouth.smile": -9, "eyes.x": NaN, "cheek": Infinity } });
  const r = b.receipt();
  assert.equal(r.held["head.x"], 1);
  assert.equal(r.held["mouth.open"], 0);
  assert.equal(r.held["mouth.smile"], -1);
  assert.ok(!("eyes.x" in r.held), "NaN must not be held");
  assert.ok(!("cheek" in r.held), "Infinity must not be held");
  for (const v of Object.values(r.semantic)) assert.ok(Number.isFinite(v));
});
