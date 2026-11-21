import React, { useEffect, useRef, useState } from "react";
import dayUrl from "../../assets/earth/earth-day.jpg";
import nightUrl from "../../assets/earth/earth-night.jpg";

// The sign-in globe: NASA's Blue Marble drawn on a real orthographic sphere
// (not a scrolling flat image), lit from the upper left, with city lights on
// the night side. A satellite crosses on its orbit and the imaged places under
// it light up; the readout names the place nearest the viewer, with its real
// coordinates. Textures and their licence: assets/earth/SOURCE.md.
// Decorative: aria-hidden, still when the user asks for reduced motion.

const PLACES = [
  ["Dubai", 25.2, 55.27], ["Riyadh", 24.71, 46.68], ["Doha", 25.29, 51.53], ["Cairo", 30.04, 31.24],
  ["Mumbai", 19.08, 72.88], ["Singapore", 1.35, 103.82], ["London", 51.51, -0.13], ["Rotterdam", 51.92, 4.48],
  ["Tokyo", 35.68, 139.69], ["Sydney", -33.87, 151.21], ["New York", 40.71, -74.0], ["Santos", -23.96, -46.33],
];
const TILT = (22 * Math.PI) / 180;            // the Gulf faces the viewer
const START = (-55 * Math.PI) / 180;
const SUN = (() => { const v = [-0.55, 0.35, 0.76]; const n = Math.hypot(...v); return v.map((a) => a / n); })();
const ORBIT_TILT = (-20 * Math.PI) / 180;
const SVGNS = "http://www.w3.org/2000/svg";

function loadPixels(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      const x = c.getContext("2d");
      x.drawImage(img, 0, 0);
      resolve({ w: img.width, h: img.height, d: x.getImageData(0, 0, img.width, img.height).data });
    };
    img.onerror = reject;
    img.src = src;
  });
}

function project(lat, lon, rot) {
  const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180 + rot, c = Math.cos(TILT), s = Math.sin(TILT);
  const x = Math.cos(la) * Math.sin(lo), y0 = Math.sin(la), z0 = Math.cos(la) * Math.cos(lo);
  return [x, y0 * c - z0 * s, y0 * s + z0 * c];
}

const fmt = (lat, lon) =>
  `${Math.abs(lat).toFixed(2)} ${lat >= 0 ? "N" : "S"} · ${Math.abs(lon).toFixed(2)} ${lon >= 0 ? "E" : "W"}`;

export default function EarthGlobe() {
  const host = useRef(null), canvas = useRef(null), overlay = useRef(null);
  const [place, setPlace] = useState(PLACES[0]);

  useEffect(() => {
    let alive = true, raf = 0, geo = null, tex = null, lastPlace = "";
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const ctx = canvas.current.getContext("2d");

    function layout() {
      const box = host.current.getBoundingClientRect();
      const css = Math.max(160, Math.min(box.width * 0.78, box.height * 0.7, 520));
      const dpr = Math.min(window.devicePixelRatio || 1, 2), n = Math.round(css * dpr);
      canvas.current.width = canvas.current.height = n;
      canvas.current.style.width = canvas.current.style.height = `${css}px`;
      const lat = new Float32Array(n * n), lon = new Float32Array(n * n), sh = new Float32Array(n * n), mask = new Uint8Array(n * n);
      const c = Math.cos(TILT), s = Math.sin(TILT);
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const x = ((i + 0.5) / n) * 2 - 1, y = 1 - ((j + 0.5) / n) * 2, r = x * x + y * y, k = j * n + i;
        if (r > 1) continue;
        const z = Math.sqrt(1 - r);
        mask[k] = r > 0.985 ? 2 : 1;                     // soft rim
        lat[k] = Math.asin(y * c + z * s);
        lon[k] = Math.atan2(x, z * c - y * s);
        sh[k] = x * SUN[0] + y * SUN[1] + z * SUN[2];
      }
      const w = css + 120;
      overlay.current.setAttribute("width", w); overlay.current.setAttribute("height", w);
      overlay.current.setAttribute("viewBox", `0 0 ${w} ${w}`);
      geo = { n, css, w, lat, lon, sh, mask, img: ctx.createImageData(n, n) };
    }

    function draw(t) {
      if (!geo || !tex) return;
      const sec = t / 1000, rot = reduce ? START : START + sec * 0.06;
      const { n, css, w, lat, lon, sh, mask, img } = geo, [day, night] = tex, d = img.data, TAU = Math.PI * 2;
      for (let k = 0, m = n * n; k < m; k++) {
        const p = k * 4;
        if (!mask[k]) { d[p + 3] = 0; continue; }
        const lo = (((lon[k] - rot + Math.PI) % TAU) + TAU) % TAU;
        const u = lo / TAU, v = 0.5 - lat[k] / Math.PI;
        const di = (((v * day.h) | 0) * day.w + ((u * day.w) | 0)) * 4;
        const ni = (((v * night.h) | 0) * night.w + ((u * night.w) | 0)) * 4;
        const s = sh[k], lit = 0.08 + 0.92 * Math.min(1, Math.max(0, s * 1.6 + 0.12));
        const dark = 1 - Math.min(1, Math.max(0, s * 3 + 0.35));
        d[p] = day.d[di] * lit + night.d[ni] * dark * 0.9;
        d[p + 1] = day.d[di + 1] * lit + night.d[ni + 1] * dark * 0.85;
        d[p + 2] = day.d[di + 2] * lit + night.d[ni + 2] * dark * 0.7;
        d[p + 3] = mask[k] === 2 ? 150 : 255;
      }
      ctx.putImageData(img, 0, 0);

      // overlay: rim, crosshair, orbit, satellite, imaged places
      const svg = overlay.current, cx = w / 2, R = css / 2;
      svg.replaceChildren();
      const el = (tag, attrs) => { const e = document.createElementNS(SVGNS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); svg.append(e); return e; };
      el("circle", { cx, cy: cx, r: R + 0.5, class: "globe-rim" });
      [[cx, cx - R - 40, cx, cx - R - 16], [cx, cx + R + 16, cx, cx + R + 40], [cx - R - 40, cx, cx - R - 16, cx], [cx + R + 16, cx, cx + R + 40, cx]]
        .forEach(([x1, y1, x2, y2]) => el("line", { x1, y1, x2, y2, class: "globe-hair" }));
      const A = R + 44, B = R * 0.32, phase = reduce ? 2.2 : sec * 0.5;
      const at = (a) => {
        const ex = Math.cos(a) * A, ey = Math.sin(a) * B;
        return [cx + ex * Math.cos(ORBIT_TILT) - ey * Math.sin(ORBIT_TILT), cx + ex * Math.sin(ORBIT_TILT) + ey * Math.cos(ORBIT_TILT), Math.sin(a)];
      };
      let front = "";
      for (let a = 0; a <= TAU + 0.01; a += 0.05) {
        const [x, y, z] = at(a);
        if (z >= 0) front += `${front ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      el("path", { d: front, class: "globe-orbit" });
      const [sx, sy, sz] = at(phase);
      let best = null;
      for (const pl of PLACES) {
        const [x, y, z] = project(pl[1], pl[2], rot);
        if (z < 0.15) continue;
        const px = cx + x * R, py = cx - y * R, seen = sz > 0 && Math.abs(px - sx) < R * 0.35;
        if (!best || z > best[1]) best = [pl, z];
        el("rect", { x: px - 5, y: py - 4, width: 10, height: 8, class: seen ? "globe-spot seen" : "globe-spot" });
      }
      if (sz > 0) el("line", { x1: sx, y1: sy, x2: sx, y2: sy + R * 0.18, class: "globe-beam" });
      el("circle", { cx: sx, cy: sy, r: 4, class: sz > 0 ? "globe-sat" : "globe-sat behind" });
      if (best && best[0][0] !== lastPlace) { lastPlace = best[0][0]; setPlace(best[0]); }
    }

    let last = 0;
    function frame(t) {
      if (!alive) return;
      raf = requestAnimationFrame(frame);
      if (t - last < 33) return;                       // ~30 fps is plenty for a slow turn
      last = t;
      draw(t);
    }

    layout();
    const ro = new ResizeObserver(() => { layout(); draw(performance.now()); });
    ro.observe(host.current);
    Promise.all([loadPixels(dayUrl), loadPixels(nightUrl)]).then((t) => {
      if (!alive) return;
      tex = t;
      if (reduce) draw(0); else raf = requestAnimationFrame(frame);
    }).catch(() => {});                                // no globe is fine; the form still works
    return () => { alive = false; cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return (
    <div className="auth-globe" ref={host} aria-hidden="true">
      <canvas ref={canvas} />
      <svg ref={overlay} />
      <div className="auth-globe-read mono">
        <span>over <b>{place[0]}</b></span><span>{fmt(place[1], place[2])}</span>
      </div>
    </div>
  );
}
