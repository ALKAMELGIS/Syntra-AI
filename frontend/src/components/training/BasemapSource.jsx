// Label > Learn from > From basemap: save a basemap area as a GeoTIFF in the
// library, then label it like any image. The area is the one drawn with
// Edit > Rectangle area (the AOI). Only sources whose terms allow keeping their
// tiles are offered; Google and Esri imagery are listed with the reason.
import React, { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client.js";

const BLOCKED = [
  ["google", "Google satellite", "Google's terms don't allow saving its tiles or training on them"],
  ["esri", "Esri World Imagery", "Esri's terms don't allow keeping its tiles without a licence"],
];
const R = 6378137;

function bboxOf(aoi) {
  const pts = [];
  const walk = (c) => (typeof c[0] === "number" ? pts.push(c) : c.forEach(walk));
  walk(aoi?.geometry?.coordinates || aoi?.features?.[0]?.geometry?.coordinates || []);
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function estimate(b, z) {
  const t = (lon, lat) => {
    const n = 2 ** z;
    return [((lon + 180) / 360) * n, ((1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2) * n];
  };
  const [x0, y0] = t(b[0], b[3]), [x1, y1] = t(b[2], b[1]);
  const tiles = (Math.ceil(x1) - Math.floor(x0)) * (Math.ceil(y1) - Math.floor(y0));
  const gsd = (2 * Math.PI * R * Math.cos((((b[1] + b[3]) / 2) * Math.PI) / 180)) / (256 * 2 ** z);
  return { tiles, gsd, w: Math.round((x1 - x0) * 256), h: Math.round((y1 - y0) * 256) };
}

export default function BasemapSource({ aoi, onImage }) {
  const [srcs, setSrcs] = useState(null);
  const [source, setSource] = useState(() => { try { return localStorage.getItem("syntra_basemap_url") ? "custom" : "usgs"; } catch { return "usgs"; } });
  // the last service used is kept in this browser
  const [url, setUrlState] = useState(() => { try { return localStorage.getItem("syntra_basemap_url") || ""; } catch { return ""; } });
  const setUrl = (v) => { setUrlState(v); try { localStorage.setItem("syntra_basemap_url", v); } catch { /* private mode */ } };
  const [zoom, setZoom] = useState(16);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  useEffect(() => { api.basemapSources().then(setSrcs).catch((e) => setErr(e.message)); }, []);

  const box = useMemo(() => bboxOf(aoi), [aoi]);
  const src = srcs?.sources.find((s) => s.id === source);
  const maxZoom = src?.max_zoom || 21;
  const z = Math.min(zoom, maxZoom);
  const est = box ? estimate(box, z) : null;
  const tooMany = est && srcs && est.tiles > srcs.max_tiles;

  async function make() {
    setBusy(true); setErr(null);
    try {
      const img = await api.fromBasemap({ source, url: source === "custom" ? url : null, bbox: box, zoom: z });
      onImage(img);
    } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
    setBusy(false);
  }

  return (
    <div className="basemap-src">
      <label className="train-field">Imagery
        <select className="train-select" value={source} onChange={(e) => setSource(e.target.value)}>
          {(srcs?.sources || []).map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          <option value="custom">Your service (XYZ or ArcGIS MapServer)</option>
          {BLOCKED.map(([id, l]) => <option key={id} value={id} disabled>{l}: not allowed</option>)}
        </select>
      </label>
      {source === "custom" && (
        <input className="train-input mono" value={url} onChange={(e) => setUrl(e.target.value)} spellCheck={false}
               placeholder="XYZ tile URL or ArcGIS …/MapServer" aria-label="Tile URL" />
      )}
      {src && source !== "custom" && <p className="train-dim">{src.licence}</p>}
      <label className="train-field">Zoom
        <select className="train-select" value={z} onChange={(e) => setZoom(Number(e.target.value))}>
          {Array.from({ length: maxZoom - 13 }, (_, i) => 14 + i).map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      </label>
      {!box ? <p className="train-dim">No area · Edit &gt; Rectangle area</p> : (
        <p className="train-dim mono">{est.w} × {est.h} px · {est.gsd < 1 ? `${Math.round(est.gsd * 100)} cm` : `${est.gsd.toFixed(1)} m`} a pixel · {est.tiles} tiles</p>
      )}
      {tooMany && <p className="train-error">Too large: zoom out or draw a smaller area (up to {srcs.max_tiles} tiles)</p>}
      <button type="button" className="quiet-button" onClick={make}
              disabled={!box || busy || tooMany || (source === "custom" && !url.trim())}>
        {busy ? "Saving the area…" : "Save area as GeoTIFF"}
      </button>
      {err && <p className="train-error">{err}</p>}
    </div>
  );
}
