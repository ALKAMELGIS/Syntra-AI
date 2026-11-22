// node --test src/lib  (npm test): the label editing geometry
import test from "node:test";
import assert from "node:assert/strict";
import { splitRing, mergeRings, tracePath, snapPoint, smoothClosed, ringArea } from "./geomEdit.js";

const sq = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }];
const areas = (rings) => rings.map(ringArea).sort((a, b) => a - b);

test("a straight line across splits a square in two halves", () => {
  assert.deepEqual(areas(splitRing(sq, [{ x: 5, y: -2 }, { x: 5, y: 12 }])), [50, 50]);
});
test("a bent line splits along both of its segments", () => {
  assert.deepEqual(areas(splitRing(sq, [{ x: -1, y: 5 }, { x: 5, y: 5 }, { x: 5, y: 11 }])), [25, 75]);
});
test("a line that misses the polygon splits nothing", () => {
  assert.equal(splitRing(sq, [{ x: 20, y: 0 }, { x: 20, y: 10 }]).length, 0);
});
test("overlapping squares merge into one ring; apart they stay two", () => {
  const right = sq.map((p) => ({ x: p.x + 8, y: p.y }));
  assert.deepEqual(areas(mergeRings([sq, right])), [180]);
  assert.equal(mergeRings([sq, sq.map((p) => ({ x: p.x + 30, y: p.y }))]).length, 2);
});
test("trace follows the shorter way round the edge", () => {
  assert.deepEqual(tracePath(sq, { i: 0, t: 0.5 }, { i: 1, t: 0.5 }), [{ x: 10, y: 0 }]);
  assert.deepEqual(tracePath(sq, { i: 0, t: 0.5 }, { i: 3, t: 0.5 }), [{ x: 0, y: 0 }]);
  assert.deepEqual(tracePath(sq, { i: 0, t: 0.2 }, { i: 0, t: 0.8 }), []);
});
test("snapping prefers a corner, then an edge, within the tolerance", () => {
  const rings = [{ id: 1, ring: sq }];
  assert.equal(snapPoint({ x: 9.5, y: 0.4 }, rings, 2).kind, "vertex");
  const e = snapPoint({ x: 5, y: 0.4 }, rings, 2);
  assert.equal(e.kind, "edge"); assert.deepEqual(e.q, { x: 5, y: 0 });
  assert.equal(snapPoint({ x: 5, y: 5 }, rings, 2), null);
});
test("a curve passes through its points", () => {
  const c = smoothClosed(sq, 4);
  assert.equal(c.length, 16);
  assert.deepEqual(c[0], sq[0]); assert.deepEqual(c[4], sq[1]);
});
