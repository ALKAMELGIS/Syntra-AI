import React, { useCallback, useEffect, useRef, useState } from "react";
import { fmtAreaKm2 } from "../../lib/prefs.js";
import {
  IconPlus, IconArrowDown, IconArrowUp, IconClose, IconDownload, IconEye, IconEyeOff, IconInfo, IconMore,
  IconTable, IconTrash, IconZoomTo, IconEdit, IconChevronLeft, IconCopy, IconClickAI, IconCheck, IconPolygon, IconSliders,
} from "../ui/Icons.jsx";
import { LAYER_COLOURS, downloadGeojson, featuresBounds, fieldType, fieldsOf, unitOf } from "./layerUtils.js";
import { BASEMAPS, BASEMAP_ORDER, thumbUrl } from "./basemaps.js";
import { api } from "../../api/client.js";
import SymbologyPanel, { Legend } from "./SymbologyPanel.jsx";
import { classesOf } from "./symbology.js";

// Layers, beside the map toolbar (replaces Legend and the show/hide-all eye).
// Results: one row per detection run, each with its own eye, colour, count and
// options. Drawn: the area of interest. Imagery: the GeoTIFF with opacity and
// display contrast / brightness. Basemap at the foot. Everything shown here is
// read from the layers themselves; nothing is a placeholder.

function Toggle({ on, onChange, label }) {
  return (
    <button type="button" className={`lp-toggle ${on ? "on" : ""}`} role="switch" aria-checked={on} onClick={() => onChange(!on)}>
      <span className="lp-toggle-label">{label}</span><span className="lp-switch" aria-hidden="true" />
    </button>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }) {
  return (
    <label className="lp-slider">
      <span>{label}</span>
      <input type="range" className="gauge" min={min} max={max} step={step} value={value}
             style={{ "--fill": `${((value - min) / (max - min)) * 100}%` }} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="mono lp-val">{fmt ? fmt(value) : value}</span>
    </label>
  );
}

function Swatch({ layer }) {
  if (layer.symbology && layer.symbology.type !== "single" && layer.symbology.field) {
    const cs = classesOf(layer.symbology, layer.shown, layer.color).slice(0, 3);
    return <span className="lp-swatch multi" aria-hidden="true">{cs.map((c) => <i key={c.label} style={{ background: c.color }} />)}</span>;
  }
  const style = layer.isArea
    ? { background: layer.color, opacity: 0.6 }
    : { borderColor: layer.color };
  return <span className={`lp-swatch ${layer.isArea ? "fill" : ""}`} style={style} aria-hidden="true" />;
}

// The ⋯ list of a layer: a small floating menu like ArcGIS Online's. Actions
// run at once; settings live on the layer's Properties page.
function LayerMenu({ top, items, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    const down = (e) => { if (ref.current && !ref.current.contains(e.target) && !e.target.closest?.("[data-menu-anchor]")) onClose(); };
    const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    document.addEventListener("mousedown", down);
    window.addEventListener("keydown", key, true);
    ref.current?.querySelector("button:not(:disabled)")?.focus();
    return () => { document.removeEventListener("mousedown", down); window.removeEventListener("keydown", key, true); };
  }, [onClose]);
  return (
    <div className="lp-menu" role="menu" ref={ref} style={{ top }}>
      {items.map((it, i) => it === "-" ? <div key={i} className="lp-menu-sep" role="separator" /> : it.slider ? (
        <label key={it.label} className="lp-menu-slider">
          <span>{it.label}</span>
          <input type="range" className="gauge" min={0} max={it.max ?? 100} step={5} value={it.value}
                 style={{ "--fill": `${(it.value / (it.max ?? 100)) * 100}%` }} onChange={(e) => it.onChange(Number(e.target.value))} />
          <span className="mono">{it.value} %</span>
        </label>
      ) : (
        <button key={it.label} type="button" role="menuitem" disabled={it.disabled} className={it.danger ? "danger" : ""}
                onClick={() => { onClose(); it.run(); }}>
          <it.Icon />{it.label}{it.hint != null && <span className="mono">{it.hint}</span>}
        </button>
      ))}
    </div>
  );
}

function PropsHeader({ title, sub, onBack, onClose }) {
  return (
    <header className="lp-head">
      <button type="button" className="lp-icon" onClick={onBack} aria-label="Back to layers"><IconChevronLeft /></button>
      <span className="lp-title" title={title}>{title}</span><span className="lp-sub">{sub}</span>
      <button type="button" className="lp-icon" onClick={onClose} aria-label="Close"><IconClose /></button>
    </header>
  );
}

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)} %`);

// Properties of a results or added layer: appearance, filter, then its source and fields.
function VectorProps({ layer, onPatch, zoom, onBack, onClose }) {
  const b = featuresBounds(layer.features);
  const fields = fieldsOf(layer.features);
  const geomTypes = [...new Set(layer.features.map((f) => f.geometry?.type).filter(Boolean))].join(", ") || "—";
  const hasScore = layer.features.some((f) => typeof f.properties?.score === "number");
  return (
    <>
      <PropsHeader title={layer.name} sub="Properties" onBack={onBack} onClose={onClose} />
      <div className="lp-body">
        <div className="lp-props">
          <div className="lp-sec">Appearance</div>
          <div className="lp-colours" role="radiogroup" aria-label="Colour">
            {LAYER_COLOURS.map((c) => (
              <button key={c} type="button" role="radio" aria-checked={layer.color === c} aria-label={c}
                      className={layer.color === c ? "on" : ""} style={{ background: c }} onClick={() => onPatch({ color: c })} />
            ))}
          </div>
          <Slider label="Transparency" value={Math.round((1 - layer.opacity) * 100)} min={0} max={90} step={5}
                  onChange={(v) => onPatch({ opacity: 1 - v / 100 })} fmt={(v) => `${v} %`} />
          <div className="lp-range">
            <span>Visible from</span>
            <select value={layer.minZoom ?? ""} onChange={(e) => onPatch({ minZoom: e.target.value === "" ? null : Number(e.target.value) })} aria-label="Visible from zoom">
              <option value="">any</option>{ZOOMS.map((z) => <option key={z} value={z}>z {z}</option>)}
            </select>
            <span>to</span>
            <select value={layer.maxZoom ?? ""} onChange={(e) => onPatch({ maxZoom: e.target.value === "" ? null : Number(e.target.value) })} aria-label="Visible to zoom">
              <option value="">any</option>{ZOOMS.map((z) => <option key={z} value={z}>z {z}</option>)}
            </select>
            {zoom != null && <span className="mono lp-muted" title="The map's zoom now">now {zoom}</span>}
          </div>
          <Toggle label="Labels" on={layer.labels} onChange={(v) => onPatch({ labels: v })} />
          <Toggle label="Pop-ups" on={layer.popups} onChange={(v) => onPatch({ popups: v })} />
          {hasScore && (<>
            <div className="lp-sec">Filter</div>
            <Slider label="Score at least" value={layer.minScore} min={0} max={0.95} step={0.05}
                    onChange={(v) => onPatch({ minScore: v })} fmt={(v) => v.toFixed(2)} />
          </>)}
        </div>
        {(layer.jobId && (layer.accuracy || layer.task || layer.taskError)) && (<>
          <div className="lp-group-label">Accuracy</div>
          {layer.accuracy ? (
            <dl className="lp-dl">
              <dt>F1</dt><dd className="mono lp-key">{pct(layer.accuracy.f1)}</dd>
              <dt>Precision</dt><dd className="mono">{pct(layer.accuracy.precision)}</dd>
              <dt>Recall</dt><dd className="mono">{pct(layer.accuracy.recall)}</dd>
              <dt>Outline IoU</dt><dd className="mono">{pct(layer.accuracy.mean_iou)}</dd>
              <dt>Matched</dt><dd className="mono">{layer.accuracy.matched} of {layer.accuracy.labels} labels · {layer.accuracy.results} found</dd>
              <dt>Checked</dt><dd className="mono">{layer.accuracy.areas} area{layer.accuracy.areas === 1 ? "" : "s"} · IoU ≥ {layer.accuracy.iou_min} · {layer.accuracy.at}</dd>
            </dl>
          ) : <p className="lp-note" style={{ padding: "0 16px 8px" }}>{layer.task || layer.taskError}</p>}
          {layer.accuracy && layer.taskError && <p className="lp-note" style={{ padding: "0 16px 8px" }}>{layer.taskError}</p>}
        </>)}
        <div className="lp-group-label">Source</div>
        <dl className="lp-dl">
          {layer.modelId && (<><dt>Model</dt><dd className="mono">{layer.modelId}{layer.modelVersion ? ` · ${layer.modelVersion}` : ""}</dd></>)}
          {layer.ranAt && (<><dt>Run</dt><dd className="mono">{layer.ranAt}</dd></>)}
          {layer.imageName && (<><dt>{layer.modelId ? "Image" : "File"}</dt><dd className="mono" title={layer.imageName}>{layer.imageName}</dd></>)}
          <dt>Objects</dt><dd className="mono">{layer.features.length}</dd>
          <dt>Geometry</dt><dd className="mono">{geomTypes}</dd>
          <dt>CRS</dt><dd className="mono">EPSG:4326 · WGS84{layer.sourceCrs && !/4326/.test(layer.sourceCrs) ? ` (from ${layer.sourceCrs})` : ""}</dd>
          {b && (<><dt>Extent</dt><dd className="mono">{b[0][1].toFixed(4)} – {b[1][1].toFixed(4)} E<br />{b[0][0].toFixed(4)} – {b[1][0].toFixed(4)} N</dd></>)}
        </dl>
        {layer.note && <p className="lp-note" style={{ padding: "0 16px 8px" }}>{layer.note}</p>}
        <div className="lp-group-label">Fields</div>
        <ul className="lp-fields">
          {fields.map((k) => (
            <li key={k}><span className="mono">{k}</span><span className="mono lp-muted">{fieldType(layer.features, k)}{unitOf(k) ? ` · ${unitOf(k)}` : ""}</span></li>
          ))}
          {fields.length === 0 && <li className="lp-muted">No attributes</li>}
        </ul>
      </div>
    </>
  );
}

// Properties of the image: display settings (the map only), then what the file is.
function RasterProps({ image, imageLayer, onImageLayer, onBack, onClose }) {
  const nBands = image?.bands || 0;
  const display = imageLayer.displayFor === image?.image_id ? imageLayer : {};
  const setDisplay = (p) => onImageLayer({ ...imageLayer, bands: display.bands || null, stretch: display.stretch || null, ...p, displayFor: image.image_id });
  const gsd = image?.health?.gsd_m || image?.gsd_x_m;
  const b = image?.bounds_wgs84;
  return (
    <>
      <PropsHeader title={image.filename} sub="Properties" onBack={onBack} onClose={onClose} />
      <div className="lp-body">
        <div className="lp-props">
          <div className="lp-sec">Display</div>
          <Slider label="Opacity" value={Math.round(imageLayer.opacity * 100)} min={0} max={100} step={5}
                  onChange={(v) => onImageLayer({ ...imageLayer, opacity: v / 100 })} fmt={(v) => `${v} %`} />
          <Slider label="Contrast" value={imageLayer.contrast ?? 0} min={-50} max={100} step={5}
                  onChange={(v) => onImageLayer({ ...imageLayer, contrast: v })} />
          <Slider label="Brightness" value={imageLayer.brightness ?? 0} min={-50} max={100} step={5}
                  onChange={(v) => onImageLayer({ ...imageLayer, brightness: v })} />
          {nBands > 1 && (
            <div className="lp-bands">
              <span>Bands</span>
              <select value={display.bands ? "custom" : ""} aria-label="Band choice"
                      onChange={(e) => setDisplay({ bands: e.target.value ? (display.bands || [Math.min(3, nBands), Math.min(2, nBands), 1]) : null })}>
                <option value="">Auto</option><option value="custom">Choose R G B</option>
              </select>
              {display.bands && ["R", "G", "B"].map((c, i) => (
                <select key={c} value={display.bands[i]} aria-label={`${c} band`}
                        onChange={(e) => { const bb = [...display.bands]; bb[i] = Number(e.target.value); setDisplay({ bands: bb }); }}>
                  {Array.from({ length: nBands }, (_, k) => <option key={k} value={k + 1}>{c} {k + 1}</option>)}
                </select>
              ))}
            </div>
          )}
          <div className="lp-bands">
            <span>Stretch</span>
            <select value={display.stretch || ""} onChange={(e) => setDisplay({ stretch: e.target.value || null })} aria-label="Stretch">
              {STRETCHES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
          <p className="lp-note">Display only: detection reads the original pixels with its own bands and stretch</p>
          <div className="lp-actions">
            <button type="button" disabled={!imageLayer.contrast && !imageLayer.brightness && !display.bands && !display.stretch}
                    onClick={() => onImageLayer({ ...imageLayer, contrast: 0, brightness: 0, bands: null, stretch: null })}>Reset display</button>
          </div>
        </div>
        <div className="lp-group-label">Source</div>
        <dl className="lp-dl">
          <dt>File</dt><dd className="mono">{image.filename}</dd>
          {image.width && (<><dt>Size</dt><dd className="mono">{image.width.toLocaleString("en-US")} × {image.height.toLocaleString("en-US")} px</dd></>)}
          {gsd && (<><dt>Resolution</dt><dd className="mono">{gsd < 1 ? `${Math.round(gsd * 100)} cm` : `${+gsd.toFixed(2)} m`}</dd></>)}
          {nBands > 0 && (<><dt>Bands</dt><dd className="mono">{nBands}{image.dtype ? ` · ${image.dtype}` : ""}</dd></>)}
          {image.crs && (<><dt>CRS</dt><dd className="mono">{image.crs}</dd></>)}
          {image.area_km2 && (<><dt>Area</dt><dd className="mono">{fmtAreaKm2(image.area_km2, 0)}</dd></>)}
          {b && (<><dt>Extent</dt><dd className="mono">{b.west.toFixed(4)} – {b.east.toFixed(4)} E<br />{b.south.toFixed(4)} – {b.north.toFixed(4)} N</dd></>)}
        </dl>
      </div>
    </>
  );
}

function ServiceProps({ sv, onPatch, onBack, onClose }) {
  return (
    <>
      <PropsHeader title={sv.name} sub="Properties" onBack={onBack} onClose={onClose} />
      <div className="lp-body"><div className="lp-props">
        <Slider label="Opacity" value={Math.round(sv.opacity * 100)} min={0} max={100} step={5}
                onChange={(v) => onPatch({ opacity: v / 100 })} fmt={(v) => `${v} %`} />
        <dl className="lp-dl" style={{ padding: 0 }}><dt>Type</dt><dd className="mono">{sv.kind.toUpperCase()}</dd><dt>Address</dt><dd className="mono">{sv.url}</dd>
          {sv.layers && (<><dt>Layers</dt><dd className="mono">{sv.layers}</dd></>)}</dl>
      </div></div>
    </>
  );
}

// A row's name becomes a text field while renaming (Enter saves, Esc cancels).
function RenameField({ value, onSave, onCancel, label = "Layer name", placeholder, blurSaves = true }) {
  const [v, setV] = useState(value);
  return (
    <input className="lp-rename-inline" autoFocus value={v} maxLength={60} aria-label={label} placeholder={placeholder}
           onChange={(e) => setV(e.target.value)} onBlur={() => (blurSaves ? onSave(v) : onCancel())}
           onKeyDown={(e) => { if (e.key === "Enter") onSave(v); if (e.key === "Escape") { e.stopPropagation(); onCancel(); } }} />
  );
}

const ZOOMS = Array.from({ length: 22 }, (_, i) => i + 1);
// "" is the map's default, a percent clip like ArcGIS (raster_prep.DISPLAY_STRETCH)
const STRETCHES = [["", "Percent clip 0.5 %"], ["2-98", "Percent clip 2 %"], ["5-95", "Percent clip 5 %"], ["0-100", "Min – max"], ["none", "None (as stored)"]];

// Add > From a file (read by the server, reprojected to WGS84) or a map service (XYZ / WMS).
function AddLayer({ onFile, onService, onDone }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [kind, setKind] = useState("xyz");
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [wmsLayers, setWmsLayers] = useState("");
  const fileRef = useRef(null);
  async function file(e) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    setBusy(true); setError(null);
    try { await onFile(f); onDone(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  function service(e) {
    e.preventDefault();
    const u = url.trim();
    if (!/^https?:\/\//i.test(u)) { setError("The address must start with https:// or http://"); return; }
    if (kind === "xyz" && !/\{z\}.*\{x\}.*\{y\}|\{z\}.*\{y\}.*\{x\}/.test(u)) { setError("An XYZ address needs {z}, {x} and {y}"); return; }
    if (kind === "wms" && !wmsLayers.trim()) { setError("Name at least one WMS layer"); return; }
    onService({ kind, url: u, name: name.trim() || (kind === "wms" ? wmsLayers.trim() : new URL(u.replace(/[{}]/g, "")).hostname), layers: wmsLayers.trim() });
    onDone();
  }
  return (
    <div className="lp-add">
      <div className="lp-sec">From a file</div>
      <input ref={fileRef} type="file" accept=".zip,.gpkg,.kml,.geojson,.json,.fgb" hidden onChange={file} />
      <button type="button" className="lp-add-file" disabled={busy} onClick={() => fileRef.current?.click()}>
        {busy ? "Reading…" : "Choose a file"}<span className="mono">SHP (.zip) · GPKG · KML · GeoJSON</span>
      </button>
      <div className="lp-sec">From a map service</div>
      <form onSubmit={service} className="lp-add-form">
        <div className="lp-seg" role="radiogroup" aria-label="Service type">
          {[["xyz", "XYZ tiles"], ["wms", "WMS"]].map(([k, l]) => (
            <button key={k} type="button" role="radio" aria-checked={kind === k} className={kind === k ? "on" : ""} onClick={() => setKind(k)}>{l}</button>
          ))}
        </div>
        <input value={url} onChange={(e) => setUrl(e.target.value)} aria-label="Service address"
               placeholder={kind === "xyz" ? "https://…/{z}/{x}/{y}.png" : "https://…/wms"} />
        {kind === "wms" && <input value={wmsLayers} onChange={(e) => setWmsLayers(e.target.value)} placeholder="Layer names, comma separated" aria-label="WMS layers" />}
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name (optional)" aria-label="Layer name" maxLength={60} />
        <button type="submit" className="lp-add-go">Add service</button>
      </form>
      <p className="lp-note">Use services you are licensed to use; they are shown for viewing only</p>
      {error && <p className="lp-error">{error}</p>}
    </div>
  );
}

export function BasemapGallery({ basemap, onBasemap }) {
  const [google, setGoogle] = useState(null);           // {configured}
  const [keyOpen, setKeyOpen] = useState(false);
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  useEffect(() => { api.googleBasemap().then(setGoogle).catch(() => setGoogle({ configured: false })); }, []);
  function pick(k) {
    if (k === "google" && !google?.configured) { setKeyOpen(true); return; }
    onBasemap(k);
  }
  async function save(e) {
    e.preventDefault(); setBusy(true); setErr(null);
    try {
      const r = await api.googleBasemapSave(key);
      setGoogle(r); setKey(""); setKeyOpen(false);
      if (r.configured) onBasemap("google");
    } catch (ex) { setErr(ex.detail || ex.message); } finally { setBusy(false); }
  }
  return (
    <>
      <div className="bm-grid" role="radiogroup" aria-label="Basemap">
        {BASEMAP_ORDER.map((k) => {
          const b = BASEMAPS[k];
          const src = k === "google" ? (google?.configured ? thumbUrl(k) : null) : thumbUrl(k);
          return (
            <button key={k} type="button" role="radio" aria-checked={basemap === k}
                    className={`bm-item ${basemap === k ? "on" : ""}`} onClick={() => pick(k)}
                    title={k === "google" && !google?.configured ? "Google satellite · add a key" : b?.label || "None"}>
              <span className="bm-thumb">{src ? <img src={src} alt="" loading="lazy" /> : <span className="bm-none" />}</span>
              <span className="bm-label">{b?.label || "None"}</span>
            </button>
          );
        })}
      </div>
      {keyOpen && (
        <form className="bm-key" onSubmit={save}>
          <input className="train-input mono" type="password" placeholder="Google Maps key" value={key} autoFocus
                 onChange={(e) => setKey(e.target.value)} aria-label="Google Maps key (Map Tiles API)" />
          <button type="submit" className="quiet-button" disabled={!key.trim() || busy}>{busy ? "…" : "Save"}</button>
          <button type="button" className="lp-icon" aria-label="Cancel" onClick={() => { setKeyOpen(false); setErr(null); }}><IconClose /></button>
        </form>
      )}
      {err && <p className="lp-note bm-err">{err}</p>}
    </>
  );
}

export default function LayersPanel({
  open, onClose, layers, onPatch, onMove, onRemove, onZoom, onTable, onDuplicate, onRefine, onScore, onToLabels, allVisible, onToggleAll,
  aoi, aoiVisible, onAoiVisible, aoiLabel, image, imageLayer, onImageLayer, onZoomImage, onRemoveImage, basemap, onBasemap,
  zoom, onAddFile, onAddService, services = [], onServicePatch, onServiceRemove,
}) {
  const [adding, setAdding] = useState(false);
  const [menu, setMenu] = useState(null);           // {key, top}: the ⋯ list that is open
  const [page, setPage] = useState(null);           // {kind, key}: a layer's Properties
  const [renaming, setRenaming] = useState(null);   // key whose name is being edited
  const [labelling, setLabelling] = useState(null); // key being sent to labels: its row asks for the class
  const ref = useRef(null);
  const closeMenu = useCallback(() => setMenu(null), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape" && !menu && !renaming) { if (page) setPage(null); else onClose(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, page, menu, renaming, onClose]);

  if (!open) return null;
  const total = layers.length + services.length + (aoi ? 1 : 0) + (image?.image_id ? 1 : 0);
  const gsd = image?.health?.gsd_m || image?.gsd_x_m;
  const res = gsd ? (gsd < 1 ? `${Math.round(gsd * 100)} cm` : `${+gsd.toFixed(2)} m`) : null;

  function toggleMenu(e, key) {
    if (menu?.key === key) { setMenu(null); return; }
    // right under the ⋯ button; above it when the window has no room below
    const pr = ref.current.getBoundingClientRect(), br = e.currentTarget.getBoundingClientRect();
    const h = key === "raster" ? 170 : services.some((x) => x.key === key) ? 140 : 350;
    const fitsBelow = br.bottom + 4 + h <= window.innerHeight - 8;
    setMenu({ key, top: fitsBelow ? br.bottom - pr.top + 4 : Math.max(8 - pr.top, br.top - pr.top - h - 4) });
  }
  const moreBtn = (key, name) => (
    <button type="button" className={`lp-icon ${menu?.key === key ? "on" : ""}`} data-menu-anchor aria-haspopup="menu" aria-expanded={menu?.key === key}
            aria-label={`Options for ${name}`} onClick={(e) => toggleMenu(e, key)}><IconMore /></button>
  );

  const back = () => setPage(null);
  if (page?.kind === "vector") {
    const l = layers.find((x) => x.key === page.key);
    if (l) return <section className="layers-panel" aria-label="Layers" ref={ref}><VectorProps layer={l} onPatch={(p) => onPatch(l.key, p)} zoom={zoom} onBack={back} onClose={onClose} /></section>;
  }
  if (page?.kind === "symbology") {
    const l = layers.find((x) => x.key === page.key);
    if (l) return <section className="layers-panel" aria-label="Layers" ref={ref}><SymbologyPanel layer={l} onPatch={(p) => onPatch(l.key, p)} onBack={back} onClose={onClose} /></section>;
  }
  if (page?.kind === "raster" && image?.image_id) {
    return <section className="layers-panel" aria-label="Layers" ref={ref}><RasterProps image={image} imageLayer={imageLayer} onImageLayer={onImageLayer} onBack={back} onClose={onClose} /></section>;
  }
  if (page?.kind === "service") {
    const sv = services.find((x) => x.key === page.key);
    if (sv) return <section className="layers-panel" aria-label="Layers" ref={ref}><ServiceProps sv={sv} onPatch={(p) => onServicePatch(sv.key, p)} onBack={back} onClose={onClose} /></section>;
  }

  let items = null;
  if (menu) {
    const i = layers.findIndex((x) => x.key === menu.key);
    const l = layers[i];
    const sv = services.find((x) => x.key === menu.key);
    if (l) items = [
      { label: "Zoom to layer", Icon: IconZoomTo, run: () => onZoom(l.shown) },
      { label: "Show properties", Icon: IconInfo, run: () => setPage({ kind: "vector", key: l.key }) },
      { label: "Symbology", Icon: IconSliders, run: () => setPage({ kind: "symbology", key: l.key }) },
      { label: "Show table", Icon: IconTable, hint: l.shown.length, run: () => onTable(l.key) },
      "-",
      { slider: true, label: "Transparency", max: 90, value: Math.round((1 - l.opacity) * 100), onChange: (v) => onPatch(l.key, { opacity: 1 - v / 100 }) },
      "-",
      { label: "Rename", Icon: IconEdit, run: () => setRenaming(l.key) },
      { label: "Duplicate", Icon: IconCopy, run: () => onDuplicate(l.key) },
      ...(onToLabels ? [{ label: "Use as labels", Icon: IconPolygon, disabled: !!l.task, run: () => setLabelling(l.key) }] : []),
      ...(l.jobId ? [
        ...(String(l.resultId).endsWith("__refined") ? [] : [{ label: "Refine outlines", Icon: IconClickAI, hint: "SAM", disabled: !!l.task, run: () => onRefine(l.key) }]),
        { label: "Measure accuracy", Icon: IconCheck, disabled: !!l.task, run: () => { onScore(l.key); setPage({ kind: "vector", key: l.key }); } },
      ] : []),
      { label: "Export GeoJSON", Icon: IconDownload, hint: l.shown.length, run: () => downloadGeojson(l.name, l.shown) },
      { label: "Remove", Icon: IconTrash, danger: true, run: () => onRemove(l.key) },
      "-",
      { label: "Move up", Icon: IconArrowUp, disabled: i === 0, run: () => onMove(l.key, -1) },
      { label: "Move down", Icon: IconArrowDown, disabled: i === layers.length - 1, run: () => onMove(l.key, 1) },
    ];
    else if (sv) items = [
      { label: "Show properties", Icon: IconInfo, run: () => setPage({ kind: "service", key: sv.key }) },
      "-",
      { slider: true, label: "Transparency", value: Math.round((1 - sv.opacity) * 100), onChange: (v) => onServicePatch(sv.key, { opacity: 1 - v / 100 }) },
      "-",
      { label: "Remove", Icon: IconTrash, danger: true, run: () => onServiceRemove(sv.key) },
    ];
    else if (menu.key === "raster" && image?.image_id) items = [
      { label: "Zoom to layer", Icon: IconZoomTo, run: onZoomImage },
      { label: "Show properties", Icon: IconInfo, run: () => setPage({ kind: "raster" }) },
      "-",
      { slider: true, label: "Transparency", value: Math.round((1 - imageLayer.opacity) * 100), onChange: (v) => onImageLayer({ ...imageLayer, opacity: 1 - v / 100 }) },
      "-",
      { label: "Remove", Icon: IconTrash, danger: true, run: onRemoveImage },
    ];
  }

  const name = (key, value, base, cls) => (labelling === key
    ? <RenameField value="" label="Class for these labels" placeholder="Class, e.g. date palm" blurSaves={false}
                   onSave={(v) => { if (v.trim()) onToLabels(key, v.trim()); setLabelling(null); }} onCancel={() => setLabelling(null)} />
    : renaming === key
    ? <RenameField value={value} onSave={(v) => { onPatch(key, { name: v.trim() || base }); setRenaming(null); }} onCancel={() => setRenaming(null)} />
    : <span className={cls} title={value}>{value}</span>);

  return (
    <section className="layers-panel" aria-label="Layers" ref={ref}>
      <header className="lp-head">
        <span className="lp-title">Layers</span><span className="lp-sub mono">{total || ""}</span>
        <button type="button" className={`lp-add-btn ${adding ? "on" : ""}`} aria-expanded={adding} onClick={() => setAdding((a) => !a)}>
          <IconPlus />Add</button>
        <button type="button" className="lp-icon" onClick={onClose} aria-label="Close layers"><IconClose /></button>
      </header>
      <div className="lp-body" onScroll={closeMenu}>
        {adding && <AddLayer onFile={onAddFile} onService={onAddService} onDone={() => setAdding(false)} />}
        {layers.length > 0 && (
          <div className="lp-group">
            <div className="lp-group-label">
              <button type="button" className="lp-eye" onClick={onToggleAll} aria-pressed={allVisible}
                      aria-label={allVisible ? "Hide all results" : "Show all results"} title={allVisible ? "Hide all results" : "Show all results"}>
                {allVisible ? <IconEye /> : <IconEyeOff />}
              </button>
              Results<span className="mono">{layers.reduce((n, l) => n + l.shown.length, 0)}</span>
            </div>
            {layers.map((l) => (
              <React.Fragment key={l.key}>
              <div className={`lp-row ${menu?.key === l.key ? "on" : ""}`}>
                <button type="button" className="lp-eye" onClick={() => onPatch(l.key, { visible: !l.visible })}
                        aria-pressed={l.visible} aria-label={l.visible ? `Hide ${l.name}` : `Show ${l.name}`}>
                  {l.visible && allVisible ? <IconEye /> : <IconEyeOff />}
                </button>
                <Swatch layer={l} />
                {name(l.key, l.name, l.baseName, `lp-name ${l.visible && allVisible ? "" : "dim"}`)}
                <span className="mono lp-val" title={l.taskError || undefined}>{l.task || (l.taskError ? "failed" : l.value)}</span>
                {moreBtn(l.key, l.name)}
              </div>
              {l.visible && l.symbology && l.symbology.type !== "single" && l.symbology.field && (
                <div className="lp-legend"><span className="lp-legend-field mono">{l.symbology.field}</span>
                  <Legend compact classes={classesOf(l.symbology, l.shown, l.color)} /></div>
              )}
              </React.Fragment>
            ))}
          </div>
        )}

        {services.length > 0 && (
          <div className="lp-group">
            <div className="lp-group-label">Map services</div>
            {services.map((sv) => (
              <div key={sv.key} className={`lp-row ${menu?.key === sv.key ? "on" : ""}`}>
                <button type="button" className="lp-eye" onClick={() => onServicePatch(sv.key, { visible: !sv.visible })}
                        aria-pressed={sv.visible} aria-label={sv.visible ? `Hide ${sv.name}` : `Show ${sv.name}`}>
                  {sv.visible ? <IconEye /> : <IconEyeOff />}
                </button>
                <span className="lp-swatch raster" aria-hidden="true" />
                <span className={`lp-name ${sv.visible ? "" : "dim"}`} title={sv.url}>{sv.name}</span>
                <span className="mono lp-val">{sv.kind.toUpperCase()}</span>
                {moreBtn(sv.key, sv.name)}
              </div>
            ))}
          </div>
        )}

        {aoi && (
          <div className="lp-group">
            <div className="lp-group-label">Drawn</div>
            <div className="lp-row">
              <button type="button" className="lp-eye" onClick={() => onAoiVisible(!aoiVisible)} aria-pressed={aoiVisible}
                      aria-label={aoiVisible ? "Hide area of interest" : "Show area of interest"}>
                {aoiVisible ? <IconEye /> : <IconEyeOff />}
              </button>
              <span className="lp-swatch aoi" aria-hidden="true" />
              <span className={`lp-name ${aoiVisible ? "" : "dim"}`}>Area of interest</span>
              {aoiLabel && <span className="mono lp-val">{aoiLabel}</span>}
              <button type="button" className="lp-icon" aria-label="Zoom to area of interest" title="Zoom to layer" onClick={() => onZoom(aoi.features || [aoi])}><IconZoomTo /></button>
            </div>
          </div>
        )}

        {image?.image_id && (
          <div className="lp-group">
            <div className="lp-group-label">Imagery</div>
            <div className={`lp-row ${menu?.key === "raster" ? "on" : ""}`}>
              <button type="button" className="lp-eye" onClick={() => onImageLayer({ ...imageLayer, visible: !imageLayer.visible })}
                      aria-pressed={imageLayer.visible} aria-label={imageLayer.visible ? "Hide image" : "Show image"}>
                {imageLayer.visible ? <IconEye /> : <IconEyeOff />}
              </button>
              <span className="lp-swatch raster" aria-hidden="true" />
              <span className={`lp-name ${imageLayer.visible ? "" : "dim"}`} title={image.filename}>{image.filename}</span>
              {imageLayer.opacity < 1 && <span className="mono lp-val lp-op" title="Transparency">{Math.round((1 - imageLayer.opacity) * 100)} %</span>}
              {res && <span className="mono lp-val" title="Ground size of one pixel">{res}</span>}
              {moreBtn("raster", image.filename)}
            </div>
          </div>
        )}

        {total === 0 && <p className="lp-empty">No layers yet: open an image and run a model</p>}
      </div>
      <footer className="lp-foot">
        <div className="lp-group-label">Basemap</div>
        <BasemapGallery basemap={basemap} onBasemap={onBasemap} />
      </footer>
      {items && <LayerMenu top={menu.top} items={items} onClose={closeMenu} />}
    </section>
  );
}
