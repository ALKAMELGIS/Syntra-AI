import React, { useRef, useState } from "react";
import { api } from "../api/client.js";
import { IconClose, IconDownload, IconUpload } from "./Icons.jsx";

// Label step: labels drawn elsewhere (Shapefile zip, GeoPackage, GeoJSON,
// KML) added to this image. The server reads the file, says which fields
// look like class and role, and nothing is added until "Add" is pressed.
// The standard the file follows is in backend/app/annotations/importer.py.

const clean = (v) => String(v ?? "").replace(/\s+/g, " ").trim().toLowerCase().slice(0, 32);
const SKIPS = {
  outside: "outside the image", not_polygon: "not polygons", no_class: "without a class", duplicates: "already labelled",
  too_small: "under 2 px", left_out: "left out", unknown_role: "unknown role", broken: "broken geometry",
};

// A starting file for ArcGIS or QGIS: the three kinds of feature, placed on this image
function template(image) {
  const b = image.bounds_wgs84;
  const w = b.east - b.west, h = b.north - b.south;
  const rect = (x0, y0, x1, y1) => [[[b.west + w * x0, b.south + h * y0], [b.west + w * x1, b.south + h * y0],
    [b.west + w * x1, b.south + h * y1], [b.west + w * x0, b.south + h * y1], [b.west + w * x0, b.south + h * y0]]];
  const f = (props, c) => ({ type: "Feature", properties: props, geometry: { type: "Polygon", coordinates: c } });
  return { type: "FeatureCollection", name: "syntra_labels", crs: { type: "name", properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    features: [
      f({ class: "ship", role: "object" }, rect(0.45, 0.45, 0.5, 0.48)),
      f({ class: "", role: "complete" }, rect(0.3, 0.3, 0.7, 0.7)),
      f({ class: "", role: "accuracy" }, rect(0.75, 0.1, 0.95, 0.3)),
    ] };
}

export default function LabelImport({ image, onImported }) {
  const input = useRef(null);
  const [file, setFile] = useState(null);
  const [pre, setPre] = useState(null);
  const [field, setField] = useState("");            // "" = one class for the file
  const [fixed, setFixed] = useState("");
  const [roleField, setRoleField] = useState("");
  const [map, setMap] = useState({});                // file value -> class, null = leave out
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);
  const [showStd, setShowStd] = useState(false);

  const reset = () => { setFile(null); setPre(null); setMap({}); setError(null); };
  async function choose(f) {
    if (!f) return;
    reset(); setReport(null); setBusy(true); setFile(f);
    try {
      const p = await api.previewLabelImport(image.image_id, f);
      setPre(p); setField(p.class_field || ""); setRoleField(p.role_field || "");
    } catch (err) { setError(err.detail?.message || err.detail || err.message); setFile(null); }
    finally { setBusy(false); if (input.current) input.current.value = ""; }
  }
  async function add() {
    setBusy(true); setError(null);
    try {
      const fields = { class_field: field, fixed_class: field ? "" : fixed, role_field: roleField };
      if (field) fields.class_map = JSON.stringify(map);
      const r = await api.importLabels(image.image_id, file, fields);
      onImported(r.annotations || []);
      setReport(r); reset();
    } catch (err) { setError(err.detail?.message || err.detail || err.message); }
    finally { setBusy(false); }
  }
  function downloadTemplate() {
    const blob = new Blob([JSON.stringify(template(image), null, 1)], { type: "application/geo+json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "syntra_labels.geojson" });
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  const values = pre?.fields.find((f) => f.name === field)?.values || [];
  const ready = pre && (field || clean(fixed));
  return (
    <div className="limp">
      <div className="limp-row">
        <button type="button" className="quiet-button limp-pick" disabled={!image?.image_id || busy} onClick={() => input.current?.click()}>
          <IconUpload /> {busy && !pre ? "Reading…" : "From a file"}
        </button>
        <button type="button" className="limp-link" aria-expanded={showStd} onClick={() => setShowStd((v) => !v)}>Standard</button>
        <input ref={input} type="file" hidden accept=".zip,.gpkg,.geojson,.json,.kml,.fgb" onChange={(e) => choose(e.target.files?.[0])} />
      </div>

      {showStd && (
        <div className="limp-std">
          <dl>
            <dt className="mono">geometry</dt><dd>Polygon or MultiPolygon, one object each</dd>
            <dt className="mono">class</dt><dd>Text, required. ship, building, tank…</dd>
            <dt className="mono">role</dt><dd>Optional. object, complete (training area) or accuracy</dd>
            <dt className="mono">crs</dt><dd>Any, with its .prj. Shapefile as a .zip</dd>
          </dl>
          {image?.bounds_wgs84 && (
            <button type="button" className="limp-link" onClick={downloadTemplate}><IconDownload /> Template for this image</button>
          )}
        </div>
      )}

      {pre && (
        <div className="limp-card">
          <div className="limp-head">
            <span className="limp-name" title={file?.name}>{pre.name}</span>
            <button type="button" className="lp-icon" aria-label="Cancel" onClick={reset}><IconClose /></button>
          </div>
          <p className="mono train-dim">{pre.polygons} polygons · {pre.on_image} on this image{pre.other_geometry ? ` · ${pre.other_geometry} not polygons` : ""}</p>
          {pre.note && <p className="train-dim">{pre.note}</p>}

          <label className="limp-field">
            <span>Class from</span>
            <select className="train-input" value={field} onChange={(e) => { setField(e.target.value); setMap({}); }}>
              <option value="">One class for all</option>
              {pre.fields.map((f) => <option key={f.name} value={f.name}>{f.name} · {f.distinct}</option>)}
            </select>
          </label>
          {!field && (
            <input className="train-input" placeholder="Class, e.g. ship" value={fixed} onChange={(e) => setFixed(e.target.value)} aria-label="Class for every polygon" />
          )}
          {field && values.length > 0 && (
            <ul className="limp-map">
              {values.map(([v, n]) => {
                const out = map[v] === null;
                return (
                  <li key={v} className={out ? "off" : ""}>
                    <input type="checkbox" checked={!out} aria-label={`Include ${v}`}
                           onChange={() => setMap((m) => ({ ...m, [v]: out ? clean(v) : null }))} />
                    <span className="mono limp-val" title={v}>{v}</span>
                    <span className="mono train-dim">{n}</span>
                    <input className="train-input" disabled={out} value={out ? "" : map[v] ?? clean(v)} aria-label={`Class for ${v}`}
                           onChange={(e) => setMap((m) => ({ ...m, [v]: e.target.value }))} />
                  </li>
                );
              })}
            </ul>
          )}
          <label className="limp-field">
            <span>Role from</span>
            <select className="train-input" value={roleField} onChange={(e) => setRoleField(e.target.value)}>
              <option value="">None · all are objects</option>
              {pre.fields.filter((f) => f.name !== field).map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
            </select>
          </label>
          <button type="button" className="run-button" disabled={!ready || busy || !pre.on_image} onClick={add}>
            {busy ? "Adding…" : pre.on_image ? `Add ${pre.on_image} label${pre.on_image === 1 ? "" : "s"}` : "Nothing on this image"}
          </button>
        </div>
      )}

      {report && (
        <p className="limp-report train-dim">
          <span className="mono">{report.added}</span> added
          {Object.entries(report.by_class).filter(([c]) => !c.startsWith("__")).map(([c, n]) => ` · ${c} ${n}`).join("")}
          {Object.entries(report.skipped || {}).filter(([k]) => SKIPS[k]).map(([k, n]) => ` · ${n} ${SKIPS[k]}`).join("")}
        </p>
      )}
      {error && <p className="train-error">{error}</p>}
    </div>
  );
}
