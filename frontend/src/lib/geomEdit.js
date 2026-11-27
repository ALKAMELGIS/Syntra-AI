// Planar geometry for the label editing tools (split, merge, trace, snap, curve).
// Points are {x, y} in one plane: the map's container pixels while drawing.
// Rings are open (the first point is not repeated at the end).
import pc from "polygon-clipping";

const toPc = (ring) => [[...ring.map((p) => [p.x, p.y]), [ring[0].x, ring[0].y]]];
const fromPc = (multi) => multi.map((poly) => poly[0].slice(0, -1).map(([x, y]) => ({ x, y })));

export function ringArea(ring) {
  let a = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) a += (ring[j].x + ring[i].x) * (ring[j].y - ring[i].y);
  return Math.abs(a / 2);
}

// The nearest point on segment a-b to p, with t along it (0..1).
export function nearestOnSegment(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2)) : 0;
  const q = { x: a.x + t * dx, y: a.y + t * dy };
  return { q, t, d: Math.hypot(p.x - q.x, p.y - q.y) };
}

// The nearest point on a closed ring: {q, i (edge from vertex i), t, d}.
export function nearestOnRing(p, ring) {
  let best = null;
  for (let i = 0; i < ring.length; i++) {
    const r = nearestOnSegment(p, ring[i], ring[(i + 1) % ring.length]);
    if (!best || r.d < best.d) best = { ...r, i };
  }
  return best;
}

// Snap p to the nearest vertex within tol, else the nearest edge within tol.
// rings: [{id, ring}]. Returns {q, kind: "vertex"|"edge", id, i, t} or null.
export function snapPoint(p, rings, tol = 10) {
  let v = null, e = null;
  for (const { id, ring } of rings) {
    for (let i = 0; i < ring.length; i++) {
      const d = Math.hypot(p.x - ring[i].x, p.y - ring[i].y);
      if (d <= tol && (!v || d < v.d)) v = { q: ring[i], d, kind: "vertex", id, i, t: 0 };
    }
    const n = nearestOnRing(p, ring);
    if (n && n.d <= tol && (!e || n.d < e.d)) e = { q: n.q, d: n.d, kind: "edge", id, i: n.i, t: n.t };
  }
  return v || e;
}

// The ring's boundary from position a to position b ({i, t} each), the shorter
// way round, without the end points themselves.
export function tracePath(ring, a, b) {
  const n = ring.length;
  const pos = (s) => s.i + s.t;
  const fwd = [], back = [];
  // forward: vertices a.i+1 .. b.i
  let k = (a.i + 1) % n, guard = 0;
  if (!(a.i === b.i && a.t <= b.t)) {
    while (guard++ <= n) { fwd.push(ring[k]); if (k === b.i) break; k = (k + 1) % n; }
  }
  // backward: vertices a.i .. b.i+1
  k = a.i; guard = 0;
  if (!(a.i === b.i && a.t >= b.t)) {
    while (guard++ <= n) { back.push(ring[k]); if (k === (b.i + 1) % n) break; k = (k - 1 + n) % n; }
  }
  const len = (pts, from, to) => {
    const all = [from, ...pts, to];
    let s = 0;
    for (let i = 1; i < all.length; i++) s += Math.hypot(all[i].x - all[i - 1].x, all[i].y - all[i - 1].y);
    return s;
  };
  const at = (s) => { const p = ring[s.i], q = ring[(s.i + 1) % n]; return { x: p.x + (q.x - p.x) * s.t, y: p.y + (q.y - p.y) * s.t }; };
  if (pos(a) === pos(b)) return [];
  return len(fwd, at(a), at(b)) <= len(back, at(a), at(b)) ? fwd : back;
}

// Split a polygon with a polyline that crosses it. The line is stretched past
// both ends, and the side to its left is cut out of the polygon and kept as the
// other part. Returns the parts (two or more), or [] when the line misses.
export function splitRing(ring, line) {
  if (line.length < 2 || ring.length < 3) return [];
  const xs = ring.map((p) => p.x), ys = ring.map((p) => p.y);
  const far = 4 * (Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys) + 1);
  const dir = (a, b) => { const l = Math.hypot(b.x - a.x, b.y - a.y) || 1; return { x: (b.x - a.x) / l, y: (b.y - a.y) / l }; };
  const d0 = dir(line[1], line[0]), d1 = dir(line[line.length - 2], line[line.length - 1]);
  const s = { x: line[0].x + d0.x * far, y: line[0].y + d0.y * far };
  const e = { x: line.at(-1).x + d1.x * far, y: line.at(-1).y + d1.y * far };
  const path = [s, ...line, e];
  const left = (d) => ({ x: d.y, y: -d.x });        // screen y points down: this is the left-hand side
  const ln = left(dir(s, e));
  const side = [...path, { x: e.x + ln.x * far, y: e.y + ln.y * far }, { x: s.x + ln.x * far, y: s.y + ln.y * far }];
  let a, b;
  try {
    a = fromPc(pc.intersection(toPc(ring), toPc(side)));
    b = fromPc(pc.difference(toPc(ring), toPc(side)));
  } catch { return []; }
  const min = ringArea(ring) * 1e-4;
  const parts = [...a, ...b].filter((r) => r.length >= 3 && ringArea(r) > min);
  return a.length && b.length && parts.length >= 2 ? parts : [];
}

// Take a drawn area out of a polygon. Returns the parts left (none when the
// area covers it all), or null when the polygon would be left with a hole: a
// training label can't hold one (YOLO polygons have a single outline).
export function cutRing(ring, area) {
  let out;
  try { out = pc.difference(toPc(ring), toPc(area)); } catch { return [ring]; }
  if (out.some((poly) => poly.length > 1)) return null;
  const min = ringArea(ring) * 1e-4;
  return fromPc(out).filter((r) => r.length >= 3 && ringArea(r) > min);
}

function segHit(p, q, a, b) {
  const r = { x: q.x - p.x, y: q.y - p.y }, s = { x: b.x - a.x, y: b.y - a.y };
  const den = r.x * s.y - r.y * s.x;
  if (Math.abs(den) < 1e-12) return null;
  const t = ((a.x - p.x) * s.y - (a.y - p.y) * s.x) / den;
  const u = ((a.x - p.x) * r.y - (a.y - p.y) * r.x) / den;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? { t, u } : null;
}

// The ring's vertices walking from position a to position b ({i, t}), forwards
// (dir 1) or backwards (dir -1), without the end points.
function walk(ring, a, b, dir) {
  const n = ring.length, out = [];
  if (dir > 0) {
    if (a.i === b.i && a.t <= b.t) return out;
    for (let k = (a.i + 1) % n, g = 0; g <= n; g++, k = (k + 1) % n) { out.push(ring[k]); if (k === b.i) break; }
  } else {
    if (a.i === b.i && a.t >= b.t) return out;
    for (let k = a.i, g = 0; g <= n; g++, k = (k - 1 + n) % n) { out.push(ring[k]); if (k === (b.i + 1) % n) break; }
  }
  return out;
}

// Reshape (as in ArcGIS): a line that crosses the outline twice replaces the
// outline between those crossings. Drawn outside it adds area, inside it takes
// some away; of the two outlines this could give, the larger is kept. The line's
// ends may stop just short of the outline (tol): they are put onto it.
export function reshapeRing(ring, line, tol = 0) {
  if (ring.length < 3 || line.length < 2) return null;
  const pts = line.slice();
  for (const k of [0, pts.length - 1]) {
    const n = nearestOnRing(pts[k], ring);
    if (n.d <= tol) pts[k] = n.q;
  }
  const hits = [];
  for (let j = 0; j < pts.length - 1; j++) {
    for (let i = 0; i < ring.length; i++) {
      const h = segHit(pts[j], pts[j + 1], ring[i], ring[(i + 1) % ring.length]);
      if (h) hits.push({ j, along: j + h.t, i, t: Math.min(h.u, 1 - 1e-9) });
    }
  }
  if (hits.length < 2) return null;
  hits.sort((a, b) => a.along - b.along);
  const first = hits[0], last = hits[hits.length - 1];
  const at = (h) => ({ x: pts[h.j].x + (pts[h.j + 1].x - pts[h.j].x) * (h.along - h.j), y: pts[h.j].y + (pts[h.j + 1].y - pts[h.j].y) * (h.along - h.j) });
  const mid = [at(first)];
  for (let j = first.j + 1; j <= last.j; j++) mid.push(pts[j]);
  mid.push(at(last));
  const a = [...mid, ...walk(ring, last, first, 1)];
  const b = [...mid, ...walk(ring, last, first, -1)];
  const best = ringArea(a) >= ringArea(b) ? a : b;
  return best.length >= 3 ? best : null;
}

// Union of rings: one ring when they touch or overlap, else each part.
export function mergeRings(rings) {
  if (rings.length < 2) return rings;
  try {
    return fromPc(pc.union(...rings.map(toPc))).filter((r) => r.length >= 3);
  } catch { return []; }
}

// A smooth closed curve through the points (centripetal Catmull-Rom).
export function smoothClosed(pts, perSeg = 12) {
  const n = pts.length;
  if (n < 3) return pts.slice();
  const out = [];
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
    for (let k = 0; k < perSeg; k++) {
      const t = k / perSeg, t2 = t * t, t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push({ x: f(p0.x, p1.x, p2.x, p3.x), y: f(p0.y, p1.y, p2.y, p3.y) });
    }
  }
  return out;
}
