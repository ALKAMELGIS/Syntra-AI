import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

// Training labels on the map. Lives inside <MapContainer>; it has no UI of
// its own: the Train panel picks the class and the tool and owns the saved
// list. This layer draws the saved shapes and turns mouse gestures into a
// shape, which it hands back through onDrawn(latlngs, {tool, shape}).
//
// Tools
//   polygon   click corners; double-click or Enter closes; Backspace undoes a corner
//   rect      drag corner to corner (Shift: square), or click three times for a
//             rotated rectangle: two clicks set one edge, the third its width
//   square    drag from a corner (Alt: from the centre)
//   circle    drag from the centre; saved as a 64-point polygon
//   freehand  press and draw
//   click     AI: click inside an object (Alt or right-click: remove), Enter
//             saves the outline, Esc discards, Backspace removes the last click
//   box       AI: drag a box around the object; the outline inside it is
//             previewed the same way (Alt or right-click removes a part)
//   boxsel    drag a box: selects the labels whose middle is inside (Shift adds)
//   lasso     draw around labels to select them (Shift adds)
// With an exact size set in the panel, one click places the shape there.
// Lengths shown while drawing are geodesic metres (map.distance).

const PALETTE = ["#ffcf5c", "#5b8cff", "#5fd98a", "#ff7a7a", "#c792ea", "#4fd1c5", "#f5a623", "#ec6ea0"];

export const TRAIN_AREA = "__train_area__";
export const CHECK_AREA = "__check_area__";
export const isArea = (name) => name === TRAIN_AREA || name === CHECK_AREA;
export const AREA_LABEL = { [TRAIN_AREA]: "Training area", [CHECK_AREA]: "Accuracy area" };

export function classColor(className) {
  if (className === TRAIN_AREA) return "#c8ccd4";
  if (className === CHECK_AREA) return "#f0b54a";
  let h = 0;
  for (let i = 0; i < className.length; i++) h = (h * 31 + className.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

const CIRCLE_POINTS = 64;
const EDIT = { color: "#ffffff", weight: 2, fillColor: "#ffffff", fillOpacity: 0.14 };
const HALO = { color: "#0d0f12", weight: 5, fill: false, interactive: false };

export const fmtM = (m) => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : m >= 100 ? `${Math.round(m)} m` : `${m.toFixed(1)} m`);

// metres <-> degrees around a latitude (exact enough for one object)
function offset(ll, dxM, dyM) {
  const lat = ll.lat + dyM / 111320;
  const lng = ll.lng + dxM / (111320 * Math.cos((ll.lat * Math.PI) / 180));
  return L.latLng(lat, lng);
}

function ringArea(map, latlngs) {
  // planar area in m² from local metres (fine at object scale)
  const o = latlngs[0];
  const k = 111320 * Math.cos((o.lat * Math.PI) / 180);
  const pts = latlngs.map((p) => [(p.lng - o.lng) * k, (p.lat - o.lat) * 111320]);
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

export { fmtAreaM2 as fmtArea } from "../../lib/prefs.js";

// Vertex, midpoint and rotate handles for editing a saved label
const VX = L.divIcon({ className: "edit-vx", iconSize: [10, 10] });
const MID = L.divIcon({ className: "edit-mid", iconSize: [8, 8] });
const ROT = L.divIcon({ className: "edit-rot", iconSize: [14, 14] });

export default function AnnotationLayer({
  visible, mode, className, annotations = [], onDrawn, onShapeClick, onSegment, exact, onPreview,
  editable = false, editId = null, onEdit, onUpdate, onDeleteSelected,
  selectedIds = [], onSelect,
}) {
  const map = useMap();
  const st = useRef({});
  const group = useRef(null);
  const live = useRef({ onDrawn, onSegment, onPreview, exact });
  live.current = { onDrawn, onSegment, onPreview, exact, onEdit, onUpdate, onDeleteSelected, onSelect, annotations };
  const selKey = selectedIds.join(",");

  // saved labels
  useEffect(() => {
    if (group.current) { map.removeLayer(group.current); group.current = null; }
    if (!visible || !annotations.length) return;
    const g = L.layerGroup();
    const sel = new Set(selectedIds);
    for (const a of annotations) {
      if (a.id === editId && editable && !mode) continue;      // drawn by the editor below
      const color = classColor(a.class_name);
      const area = isArea(a.class_name);
      if (sel.has(a.id)) {                                     // selected: a white edge over a stronger fill
        g.addLayer(L.polygon(a.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon)), { ...HALO, interactive: false }));
      }
      // while a tool is on, saved shapes let clicks through: you often label next to (or inside) one
      const poly = L.polygon(a.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon)),
        { ...(area ? { color, weight: 2, dashArray: "8 6", fill: false } : { color, weight: 2, fillOpacity: 0.18 }),
          ...(sel.has(a.id) ? { color: "#ffffff", weight: 2.5, fillColor: color, fillOpacity: 0.4 } : {}), interactive: !mode });
      if (!mode) {
        poly.bindTooltip(AREA_LABEL[a.class_name] || a.class_name, { direction: "center" });
        poly.on("click", (e) => {
          L.DomEvent.stopPropagation(e);
          const ev = e.originalEvent;
          if (editable && (ev.shiftKey || ev.ctrlKey || ev.metaKey)) {       // add to or take from the selection
            live.current.onSelect?.((cur) => (cur.includes(a.id) ? cur.filter((x) => x !== a.id) : [...cur, a.id]));
            onEdit?.(null);
            return;
          }
          if (editable) live.current.onSelect?.([]);
          (editable ? onEdit : onShapeClick)?.(a.id);
        });
      }
      g.addLayer(poly);
    }
    g.addTo(map);
    group.current = g;
    return () => { if (group.current) map.removeLayer(group.current); };
  }, [visible, annotations, map, mode, onShapeClick, editable, editId, onEdit, selKey]);

  // with a selection and no tool: Delete removes it, Esc lets go of it
  useEffect(() => {
    if (!visible || !editable || mode || editId || !selectedIds.length) return;
    function key(e) {
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "Escape") { e.preventDefault(); live.current.onSelect?.([]); }
      else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); live.current.onDeleteSelected?.(selectedIds); }
    }
    const clear = () => live.current.onSelect?.([]);
    window.addEventListener("keydown", key);
    map.on("click", clear);
    return () => { window.removeEventListener("keydown", key); map.off("click", clear); };
  }, [visible, editable, mode, editId, selKey, map]);

  // editing the selected label: drag vertices, add one from a midpoint, Alt+click
  // or right-click a vertex to remove it, drag the shape to move it, the round
  // handle above it to rotate. Every finished change is saved at once.
  useEffect(() => {
    if (!visible || mode || !editable || !editId) return;
    const a = annotations.find((x) => x.id === editId);
    if (!a) return;
    let pts = a.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon));
    if (pts.length > 3 && pts[0].equals(pts[pts.length - 1])) pts = pts.slice(0, -1);
    const color = classColor(a.class_name);
    const g = L.layerGroup().addTo(map);
    const halo = L.polygon(pts, { ...HALO }).addTo(g);
    const body = L.polygon(pts, { color: "#ffffff", weight: 2, fillColor: color, fillOpacity: 0.22, bubblingMouseEvents: false, className: "edit-body" }).addTo(g);
    const tip = L.tooltip({ permanent: true, direction: "top", offset: [0, -12], className: "draw-readout" });
    const px = (q) => map.latLngToContainerPoint(q);
    const ll = (q) => map.containerPointToLatLng(q);
    let vx = [], mids = [], rot = null;
    const commit = () => live.current.onUpdate?.(editId, pts);
    const centre = () => { const p = pts.map(px); return L.point(p.reduce((s2, q) => s2 + q.x, 0) / p.length, p.reduce((s2, q) => s2 + q.y, 0) / p.length); };
    const rotAt = () => { const c = centre(); const r = Math.max(...pts.map((q) => px(q).distanceTo(c))); return ll(L.point(c.x, c.y - r - 24)); };
    function redraw() {
      halo.setLatLngs(pts); body.setLatLngs(pts);
      vx.forEach((m, i) => pts[i] && m.setLatLng(pts[i]));
      if (mids.length === pts.length) mids.forEach((m, i) => m.setLatLng(L.latLngBounds(pts[i], pts[(i + 1) % pts.length]).getCenter()));
      if (rot) rot.setLatLng(rotAt());
    }
    function removeAt(i) {
      if (pts.length <= 3) return;
      pts.splice(i, 1); build(); commit();
    }
    function build() {
      [...vx, ...mids].forEach((m) => g.removeLayer(m));
      if (rot) g.removeLayer(rot);
      redraw();
      const many = pts.length > 40;                       // a traced outline: corners only every few points
      vx = pts.map((q, i) => {
        const m = L.marker(q, { icon: VX, draggable: true, keyboard: false, opacity: many && i % 4 ? 0 : 1 }).addTo(g);
        m.on("drag", (e) => { pts[i] = e.target.getLatLng(); redraw(); });
        m.on("dragend", commit);
        m.on("click", (e) => { if (e.originalEvent.altKey) removeAt(i); });
        m.on("contextmenu", (e) => { L.DomEvent.preventDefault(e.originalEvent); removeAt(i); });
        return m;
      });
      mids = many ? [] : pts.map((q, i) => {
        const m = L.marker(L.latLngBounds(q, pts[(i + 1) % pts.length]).getCenter(), { icon: MID, draggable: true, keyboard: false }).addTo(g);
        let at = -1;
        m.on("dragstart", () => { at = i + 1; pts.splice(at, 0, m.getLatLng()); });
        m.on("drag", (e) => { pts[at] = e.target.getLatLng(); halo.setLatLngs(pts); body.setLatLngs(pts); });
        m.on("dragend", () => { build(); commit(); });
        return m;
      });
      rot = L.marker(rotAt(), { icon: ROT, draggable: true, keyboard: false, title: "Rotate" }).addTo(g);
      let c0, a0, orig;
      rot.on("dragstart", () => { c0 = centre(); a0 = null; orig = pts.map(px); });
      rot.on("drag", (e) => {
        const p = px(e.target.getLatLng());
        const ang = Math.atan2(p.y - c0.y, p.x - c0.x);
        if (a0 == null) a0 = ang;
        let d = ang - a0;
        if (e.originalEvent?.shiftKey) d = Math.round(d / (Math.PI / 12)) * (Math.PI / 12);   // 15° steps
        const cs = Math.cos(d), sn = Math.sin(d);
        pts = orig.map((q) => ll(L.point(c0.x + (q.x - c0.x) * cs - (q.y - c0.y) * sn, c0.y + (q.x - c0.x) * sn + (q.y - c0.y) * cs)));
        halo.setLatLngs(pts); body.setLatLngs(pts);
        vx.forEach((m, i) => m.setLatLng(pts[i]));
        tip.setLatLng(e.target.getLatLng()).setContent(`${Math.round((d * 180) / Math.PI)}°`);
        if (!map.hasLayer(tip)) tip.addTo(map);
      });
      rot.on("dragend", () => { map.removeLayer(tip); build(); commit(); });
    }
    build();

    // drag the whole shape
    let start = null, orig = null, moved = false;
    function bodyDown(e) {
      if (e.originalEvent.button !== 0) return;
      L.DomEvent.stop(e);
      start = px(e.latlng); orig = pts.map(px); moved = false;
      map.dragging.disable();
      map.on("mousemove", bodyMove);
      map.once("mouseup", bodyUp);
    }
    function bodyMove(e) {
      const d = px(e.latlng).subtract(start);
      if (!moved && Math.hypot(d.x, d.y) < 3) return;
      moved = true;
      pts = orig.map((q) => ll(q.add(d)));
      redraw();
    }
    function bodyUp() {
      map.off("mousemove", bodyMove);
      map.dragging.enable();
      if (moved) commit();
      start = null;
    }
    body.on("mousedown", bodyDown);
    const deselect = () => live.current.onEdit?.(null);
    map.on("click", deselect);
    function key(e) {
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "Escape") { e.preventDefault(); deselect(); }
      else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); live.current.onDeleteSelected?.(editId); }
    }
    window.addEventListener("keydown", key);
    return () => {
      body.off("mousedown", bodyDown);
      map.off("mousemove", bodyMove);
      map.off("click", deselect);
      window.removeEventListener("keydown", key);
      map.dragging.enable();
      map.removeLayer(tip);
      map.removeLayer(g);
    };
  }, [visible, mode, editable, editId, annotations, map]);

  // drawing
  useEffect(() => {
    const s = { layers: [], points: [], start: null, drag: false, clicks: [], preview: null, busy: false, seq: 0 };
    st.current = s;
    if (!visible || !mode) return;
    const color = classColor(className || "object");
    const container = map.getContainer();
    map.dragging.disable();
    map.doubleClickZoom.disable();
    container.style.cursor = "crosshair";
    const tip = L.tooltip({ permanent: true, direction: "right", offset: [14, 0], className: "draw-readout" });

    const clear = () => { s.layers.forEach((l) => map.removeLayer(l)); s.layers = []; };
    const add = (l) => { l.addTo(map); s.layers.push(l); return l; };
    const say = (ll, text) => {
      if (!text) { map.removeLayer(tip); return; }
      tip.setLatLng(ll).setContent(text);
      if (!map.hasLayer(tip)) tip.addTo(map);
    };
    const finish = (latlngs, meta) => {
      clear(); say(null); s.points = []; s.start = null; s.drag = false;
      if (latlngs && latlngs.length >= 3) live.current.onDrawn?.(latlngs, meta);
    };
    const shape = (latlngs, fillOnly) => {
      add(L.polygon(latlngs, { ...HALO, interactive: false }));
      return add(L.polygon(latlngs, { ...EDIT, color: fillOnly ? color : "#ffffff", interactive: false }));
    };
    const px = (ll) => map.latLngToContainerPoint(ll);
    const ll = (p) => map.containerPointToLatLng(p);

    function rectFrom(a, b, square, fromCentre) {
      let dx = b.x - a.x, dy = b.y - a.y;
      if (square) { const m = Math.max(Math.abs(dx), Math.abs(dy)); dx = Math.sign(dx || 1) * m; dy = Math.sign(dy || 1) * m; }
      const p0 = fromCentre ? L.point(a.x - dx, a.y - dy) : a;
      const p2 = L.point(a.x + dx, a.y + dy);
      return [p0, L.point(p2.x, p0.y), p2, L.point(p0.x, p2.y)].map(ll);
    }
    function rotatedRect(p1, p2, p3) {
      const ex = p2.x - p1.x, ey = p2.y - p1.y;
      const len = Math.hypot(ex, ey) || 1;
      const nx = -ey / len, ny = ex / len;
      const h = (p3.x - p1.x) * nx + (p3.y - p1.y) * ny;
      return [p1, p2, L.point(p2.x + nx * h, p2.y + ny * h), L.point(p1.x + nx * h, p1.y + ny * h)].map(ll);
    }
    function circleAround(c, edge) {
      const r = c.distanceTo(edge);
      const out = [];
      for (let i = 0; i < CIRCLE_POINTS; i++) {
        const t = (2 * Math.PI * i) / CIRCLE_POINTS;
        out.push(ll(L.point(c.x + r * Math.cos(t), c.y + r * Math.sin(t))));
      }
      return out;
    }
    const sizeText = (pts) => `${fmtM(map.distance(pts[0], pts[1]))} × ${fmtM(map.distance(pts[1], pts[2]))}`;

    function placeExact(at) {
      const ex = live.current.exact;
      if (!ex) return false;
      if (mode === "circle" && ex.r > 0) {
        const pts = [];
        for (let i = 0; i < CIRCLE_POINTS; i++) {
          const t = (2 * Math.PI * i) / CIRCLE_POINTS;
          pts.push(offset(at, ex.r * Math.cos(t), ex.r * Math.sin(t)));
        }
        finish(pts, { tool: "circle", shape: "circle" });
        return true;
      }
      const w = mode === "square" ? ex.s : ex.w, h = mode === "square" ? ex.s : ex.h;
      if ((mode === "rect" || mode === "square") && w > 0 && h > 0) {
        finish([offset(at, -w / 2, h / 2), offset(at, w / 2, h / 2), offset(at, w / 2, -h / 2), offset(at, -w / 2, -h / 2)],
          { tool: mode, shape: mode === "square" ? "square" : "rectangle" });
        return true;
      }
      return false;
    }

    // ----- click to segment, or a box around the object (SAM box prompt: corners labelled 2 and 3) -----
    function marks() {
      const tl = s.clicks.find((c) => c.label === 2), br = s.clicks.find((c) => c.label === 3);
      if (tl && br) add(L.rectangle(L.latLngBounds(tl.ll, br.ll), { color: "#ffffff", weight: 1.5, dashArray: "5 4", fill: false, interactive: false }));
      for (const c of s.clicks) {
        if (c.label > 1) continue;
        add(L.circleMarker(c.ll, { radius: 6, color: "#0d0f12", weight: 2, fillColor: c.label ? "#4cc38a" : "#f06a6a", fillOpacity: 1, interactive: false }));
      }
    }
    async function segment() {
      const seq = ++s.seq;
      clear();
      marks();
      const boxed = s.clicks.some((c) => c.label === 2) && s.clicks.some((c) => c.label === 3);
      if (!boxed && !s.clicks.some((c) => c.label === 1)) { s.preview = null; live.current.onPreview?.(null); return; }
      s.busy = true;
      live.current.onPreview?.({ busy: true });
      container.style.cursor = "progress";
      try {
        const b = map.getBounds();
        const res = await live.current.onSegment?.(s.clicks.map((c) => [c.ll.lng, c.ll.lat, c.label]),
          [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
        if (seq !== s.seq || st.current !== s) return;              // a newer click or tool change won
        clear();
        const latlngs = res.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon));
        s.layers.push(L.polygon(latlngs, HALO).addTo(map));
        s.layers.push(L.polygon(latlngs, { ...EDIT, interactive: false }).addTo(map));
        marks();
        s.preview = { latlngs, res };
        live.current.onPreview?.({ res });
      } catch (err) {
        if (seq !== s.seq || st.current !== s) return;
        s.preview = null;
        live.current.onPreview?.({ error: err.detail?.message || err.detail || err.message });
      } finally {
        if (seq === s.seq) { s.busy = false; container.style.cursor = "crosshair"; }
      }
    }
    function saveClick() {
      if (!s.preview) return;
      const { latlngs, res } = s.preview;
      s.clicks = []; s.preview = null; clear();
      live.current.onPreview?.(null);
      live.current.onDrawn?.(latlngs, { tool: mode, shape: res.shape });
    }

    // ----- mouse -----
    function down(e) {
      if (e.originalEvent.button === 2) return;
      const p = px(e.latlng);
      if (mode === "rect" || mode === "square" || mode === "circle") {
        if (s.points.length) return;                        // 3-click rectangle in progress: clicks only
        s.start = p; s.drag = false;
      } else if (mode === "freehand") {
        s.points = [e.latlng]; s.drag = true;
      } else if (mode === "box" && !e.originalEvent.altKey) {
        s.start = p; s.drag = false;
      } else if (mode === "boxsel") {
        s.start = p; s.drag = false;
      } else if (mode === "lasso") {
        s.points = [e.latlng]; s.drag = true;
      }
    }
    // select the labels whose middle falls inside a ring drawn on screen (container pixels)
    function selectInside(ring, additive) {
      const inside = (pt) => {
        let c = false;
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
          const a = ring[i], b = ring[j];
          if ((a.y > pt.y) !== (b.y > pt.y) && pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x) c = !c;
        }
        return c;
      };
      const ids = (live.current.annotations || []).filter((a) => {
        const q = a.points_lonlat.map(([lon, lat]) => px(L.latLng(lat, lon)));
        const mid = L.point(q.reduce((t, v) => t + v.x, 0) / q.length, q.reduce((t, v) => t + v.y, 0) / q.length);
        return inside(mid);
      }).map((a) => a.id);
      live.current.onSelect?.((cur) => (additive ? [...new Set([...cur, ...ids])] : ids));
    }
    function move(e) {
      const p = px(e.latlng);
      const ev = e.originalEvent;
      if ((mode === "rect" || mode === "square" || mode === "circle") && s.start) {
        if (!s.drag && s.start.distanceTo(p) < 4) return;
        s.drag = true;
        clear();
        if (mode === "circle") {
          shape(circleAround(s.start, p));
          add(L.polyline([ll(s.start), e.latlng], { color: "#fff", weight: 1, dashArray: "3 3", interactive: false }));
          say(e.latlng, `r ${fmtM(map.distance(ll(s.start), e.latlng))}`);
        } else {
          const pts = rectFrom(s.start, p, mode === "square" || ev.shiftKey, mode === "square" && ev.altKey);
          shape(pts);
          say(e.latlng, mode === "square" ? fmtM(map.distance(pts[0], pts[1])) : sizeText(pts));
        }
      } else if (mode === "rect" && s.points.length) {
        clear();
        if (s.points.length === 1) {
          add(L.polyline([s.points[0], e.latlng], { color: "#fff", weight: 2, interactive: false }));
          say(e.latlng, fmtM(map.distance(s.points[0], e.latlng)));
        } else {
          const pts = rotatedRect(px(s.points[0]), px(s.points[1]), p);
          shape(pts);
          say(e.latlng, sizeText(pts));
        }
      } else if (mode === "polygon" && s.points.length) {
        clear();
        const pts = [...s.points, e.latlng];
        if (pts.length > 2) add(L.polygon(pts, { ...EDIT, weight: 1, fillOpacity: 0.08, interactive: false }));
        add(L.polyline(pts, { color: "#fff", weight: 2, interactive: false }));
        for (const q of s.points) add(L.circleMarker(q, { radius: 4, color: "#fff", weight: 1.5, fillColor: "#0d0f12", fillOpacity: 1, interactive: false }));
        say(e.latlng, fmtM(map.distance(s.points[s.points.length - 1], e.latlng)));
      } else if (mode === "boxsel" && s.start) {
        if (!s.drag && s.start.distanceTo(p) < 4) return;
        s.drag = true;
        clear();
        add(L.rectangle(L.latLngBounds(ll(s.start), e.latlng), { color: "#ffffff", weight: 1, dashArray: "4 3", fillColor: "#ffffff", fillOpacity: 0.06, interactive: false }));
      } else if (mode === "lasso" && s.drag) {
        if (px(s.points[s.points.length - 1]).distanceTo(p) < 3) return;
        s.points.push(e.latlng);
        clear();
        add(L.polygon(s.points, { color: "#ffffff", weight: 1, dashArray: "4 3", fillColor: "#ffffff", fillOpacity: 0.06, interactive: false }));
      } else if (mode === "box" && s.start) {
        if (!s.drag && s.start.distanceTo(p) < 4) return;
        s.drag = true;
        clear();
        add(L.rectangle(L.latLngBounds(ll(s.start), e.latlng), { color: "#ffffff", weight: 1.5, dashArray: "5 4", fill: false, interactive: false }));
      } else if (mode === "freehand" && s.drag) {
        const last = s.points[s.points.length - 1];
        if (px(last).distanceTo(p) < 3) return;
        s.points.push(e.latlng);
        clear();
        add(L.polyline(s.points, { color: "#fff", weight: 2, interactive: false }));
      }
    }
    function up(e) {
      const p = px(e.latlng);
      const ev = e.originalEvent;
      if ((mode === "rect" || mode === "square" || mode === "circle") && s.start) {
        const start = s.start;
        s.start = null;
        if (s.drag) {
          s.drag = false;
          if (mode === "circle") {
            if (start.distanceTo(p) < 3) { clear(); say(null); return; }
            finish(circleAround(start, p), { tool: "circle", shape: "circle" });
          } else {
            const pts = rectFrom(start, p, mode === "square" || ev.shiftKey, mode === "square" && ev.altKey);
            finish(pts, { tool: mode, shape: mode === "square" || ev.shiftKey ? "square" : "rectangle" });
          }
          return;
        }
        // a click, not a drag
        if (placeExact(e.latlng)) return;
        if (mode === "rect") { s.points = [e.latlng]; return; }
      } else if (mode === "boxsel" && s.start) {
        const a = s.start;
        s.start = null; clear();
        if (!s.drag) { if (!ev.shiftKey) live.current.onSelect?.([]); return; }   // a click on nothing lets go
        s.drag = false;
        selectInside([a, L.point(p.x, a.y), p, L.point(a.x, p.y)], ev.shiftKey);
      } else if (mode === "lasso" && s.drag) {
        s.drag = false; clear();
        if (s.points.length >= 3) selectInside(s.points.map(px), ev.shiftKey);
        else if (!ev.shiftKey) live.current.onSelect?.([]);
        s.points = [];
      } else if (mode === "box" && s.start) {
        const start = s.start;
        s.start = null;
        if (!s.drag) return;
        s.drag = false;
        const a = ll(start), b = e.latlng;
        const tl = L.latLng(Math.max(a.lat, b.lat), Math.min(a.lng, b.lng)), br = L.latLng(Math.min(a.lat, b.lat), Math.max(a.lng, b.lng));
        s.clicks = [...s.clicks.filter((c) => c.label < 2), { ll: tl, label: 2 }, { ll: br, label: 3 }];
        segment();
      } else if (mode === "freehand" && s.drag) {
        s.drag = false;
        const pts = s.points;
        finish(pts.length >= 3 ? pts : null, { tool: "freehand", shape: "polygon" });
      }
    }
    function click(e) {
      const ev = e.originalEvent;
      if (mode === "polygon") {
        s.points.push(e.latlng);
        move(e);
      } else if (mode === "rect" && s.points.length === 1 && px(s.points[0]).distanceTo(px(e.latlng)) > 3) {
        s.points.push(e.latlng);
      } else if (mode === "rect" && s.points.length === 2) {
        const pts = rotatedRect(px(s.points[0]), px(s.points[1]), px(e.latlng));
        finish(pts, { tool: "rect", shape: "rectangle" });
      } else if (mode === "click") {
        s.clicks.push({ ll: e.latlng, label: ev.altKey ? 0 : 1 });
        segment();
      } else if (mode === "box" && ev.altKey && s.clicks.length) {
        s.clicks.push({ ll: e.latlng, label: 0 });
        segment();
      }
    }
    function context(e) {
      if (mode !== "click" && !(mode === "box" && s.clicks.length)) return;
      L.DomEvent.preventDefault(e.originalEvent);
      s.clicks.push({ ll: e.latlng, label: 0 });
      segment();
    }
    function dbl() {
      if (mode === "polygon" && s.points.length >= 3) {
        // the double-click's two clicks added one extra point at the same spot
        const pts = s.points.slice(0, -1);
        finish(pts.length >= 3 ? pts : s.points, { tool: "polygon", shape: "polygon" });
      }
    }
    function key(e) {
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "Enter") {
        if (mode === "polygon" && s.points.length >= 3) { e.preventDefault(); finish(s.points, { tool: "polygon", shape: "polygon" }); }
        if (mode === "click" || mode === "box") { e.preventDefault(); saveClick(); }
      } else if (e.key === "Escape") {
        if (s.points.length || s.clicks.length || s.start) {
          e.preventDefault(); e.stopPropagation();
          s.clicks = []; s.preview = null; s.points = []; s.start = null; s.drag = false;
          clear(); say(null); live.current.onPreview?.(null);
        }
      } else if (e.key === "Backspace") {
        if (mode === "polygon" && s.points.length) { e.preventDefault(); s.points.pop(); clear(); say(null); }
        if ((mode === "click" || mode === "box") && s.clicks.length) {
          e.preventDefault(); s.clicks.pop();
          if (s.clicks.at(-1)?.label === 2) s.clicks.pop();          // a box goes as one
          segment();
        }
        if (mode === "rect" && s.points.length) { e.preventDefault(); s.points.pop(); clear(); say(null); }
      }
    }
    map.on("mousedown", down); map.on("mousemove", move); map.on("mouseup", up);
    map.on("click", click); map.on("dblclick", dbl); map.on("contextmenu", context);
    window.addEventListener("keydown", key, true);
    return () => {
      map.off("mousedown", down); map.off("mousemove", move); map.off("mouseup", up);
      map.off("click", click); map.off("dblclick", dbl); map.off("contextmenu", context);
      window.removeEventListener("keydown", key, true);
      s.seq++;
      clear(); map.removeLayer(tip);
      live.current.onPreview?.(null);
      map.dragging.enable();
      map.doubleClickZoom.enable();
      container.style.cursor = "";
    };
  }, [visible, mode, className, map]);

  return null;
}

export { ringArea };
