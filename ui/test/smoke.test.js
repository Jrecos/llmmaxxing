import { test } from "node:test";
import assert from "node:assert/strict";
import { APP_VERSION } from "../src/app.js";

test("app module loads", () => {
  assert.equal(typeof APP_VERSION, "string");
  assert.match(APP_VERSION, /^\d+\.\d+\.\d+$/);
});
