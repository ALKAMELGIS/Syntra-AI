import React, { useEffect, useRef, useState } from "react";
import {
  IconPlus, IconArrowDown, IconArrowUp, IconClose, IconDownload, IconEye, IconEyeOff, IconInfo, IconMore,
  IconTable, IconTrash, IconZoomTo, IconEdit, IconChevronLeft, IconChevronDown,
} from "./Icons.jsx";
import { LAYER_COLOURS, downloadGeojson, featuresBounds, fieldType, fieldsOf, unitOf } from "./layerUtils.js";

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
  const style = layer.isArea
    ? { background: layer.color, opacity: 0.6 }
    : { borderColor: layer.color };
  return <span className={`lp-swatch ${layer.isArea ? "fill" : ""}`} style={style} aria-hidden="true" />;
}

function VectorOptions({ layer, onPatch, onZoom, onTable, onInfo, onMove, onRemove, first, last, zoom }) {
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(layer.name);
  const hasScore = layer.features.some((f) => typeof f.properties?.score === "number");
  return (
    <div className="lp-options">
      <div className="lp-actions">
        <button type="button" onClick={onZoom}><IconZoomTo />Zoom to</button>
        <button type="button" onClick={onTable}><IconTable />Show table<span className="mono">{layer.count}</span></button>
        <button type="button" onClick={onInfo}><IconInfo />Information</button>
      </div>
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
      <div className="lp-sec">Layer</div>
      {renaming ? (
        <form className="lp-rename" onSubmit={(e) => { e.preventDefault(); onPatch({ name: name.trim() || layer.baseName }); setRenaming(false); }}>
          <input autoFocus value={name} onChange={(e) => setName(e.target.value)} maxLength={60} aria-label="Layer name" />
          <button type="submit">Save</button>
        </form>
      ) : null}
      <div className="lp-actions">
        <button type="button" onClick={() => { setName(layer.name); setRenaming(true); }}><IconEdit />Rename</button>
        <button type="button" disabled={first} onClick={() => onMove(-1)}><IconArrowUp />Move up</button>
        <button type="button" disabled={last} onClick={() => onMove(1)}><IconArrowDown />Move down</button>
        <button type="button" onClick={() => downloadGeojson(layer.name, layer.shown)}><IconDownload />Export GeoJSON<span className="mono">{layer.shown.length}</span></button>
        <button type="button" className="danger" onClick={onRemove}><IconTrash />Remove from map</button>
      </div>
    </div>
  );
}

function LayerInfo({ layer, onBack, onClose }) {
  const b = featuresBounds(layer.features);
  const fields = fieldsOf(layer.features);
  const geomTypes = [...new Set(layer.features.map((f) => f.geometry?.type).filter(Boolean))].join(", ") || "—";
  return (
    <>
      <header className="lp-head">
        <button type="button" className="lp-icon" onClick={onBack} aria-label="Back to layers"><IconChevronLeft /></button>
        <span className="lp-title">{layer.name}</span><span className="lp-sub">Information</span>
        <button type="button" className="lp-icon" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>
      <div className="lp-body">
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

const ZOOMS = Array.from({ length: 22 }, (_, i) => i + 1);
const STRETCHES = [["", "Auto (2–98 %)"], ["0.5-99.5", "0.5–99.5 %"], ["5-95", "5–95 %"], ["0-100", "Min – max"]];

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

const BASEMAP_CHOICES = [["satellite", "Satellite"], ["streets", "Streets"], ["none", "None"]];

export default function LayersPanel({
  open, onClose, layers, onPatch, onMove, onRemove, onZoom, onTable, allVisible, onToggleAll,
  aoi, aoiVisible, onAoiVisible, aoiLabel, image, imageLayer, onImageLayer, onZoomImage, basemap, onBasemap,
  zoom, onAddFile, onAddService, services = [], onServicePatch, onServiceRemove,
}) {
  const [adding, setAdding] = useState(false);
  const [menu, setMenu] = useState(null);           // key whose options are open
  const [info, setInfo] = useState(null);           // key whose Information is shown
  const [rasterOpen, setRasterOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape") { if (info) setInfo(null); else onClose(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, info, onClose]);

  if (!open) return null;
  const infoLayer = info && layers.find((l) => l.key === info);
  const total = layers.length + services.length + (aoi ? 1 : 0) + (image?.image_id ? 1 : 0);
  const nBands = image?.bands || 0;
  const display = imageLayer.displayFor === image?.image_id ? imageLayer : {};
  const setDisplay = (p) => onImageLayer({ ...imageLayer, bands: display.bands || null, stretch: display.stretch || null, ...p, displayFor: image.image_id });
  const gsd = image?.health?.gsd_m || image?.gsd_x_m;
  const res = gsd ? (gsd < 1 ? `${Math.round(gsd * 100)} cm` : `${+gsd.toFixed(2)} m`) : null;

  return (
    <section className="layers-panel" aria-label="Layers" ref={ref}>
      {infoLayer ? <LayerInfo layer={infoLayer} onBack={() => setInfo(null)} onClose={onClose} /> : (<>
        <header className="lp-head">
          <span className="lp-title">Layers</span><span className="lp-sub mono">{total || ""}</span>
          <button type="button" className={`lp-add-btn ${adding ? "on" : ""}`} aria-expanded={adding} onClick={() => setAdding((a) => !a)}>
            <IconPlus />Add</button>
          <button type="button" className="lp-icon" onClick={onClose} aria-label="Close layers"><IconClose /></button>
        </header>
        <div className="lp-body">
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
              {layers.map((l, i) => (
                <div key={l.key} className={`lp-item ${menu === l.key ? "open" : ""}`}>
                  <div className="lp-row">
                    <button type="button" className="lp-eye" onClick={() => onPatch(l.key, { visible: !l.visible })}
                            aria-pressed={l.visible} aria-label={l.visible ? `Hide ${l.name}` : `Show ${l.name}`}>
                      {l.visible && allVisible ? <IconEye /> : <IconEyeOff />}
                    </button>
                    <Swatch layer={l} />
                    <span className={`lp-name ${l.visible && allVisible ? "" : "dim"}`} title={l.name}>{l.name}</span>
                    <span className="mono lp-val">{l.value}</span>
                    <button type="button" className="lp-icon" aria-expanded={menu === l.key} aria-label={`Options for ${l.name}`}
                            onClick={() => setMenu((m) => (m === l.key ? null : l.key))}><IconMore /></button>
                  </div>
                  {menu === l.key && (
                    <VectorOptions layer={l} first={i === 0} last={i === layers.length - 1}
                                   onPatch={(p) => onPatch(l.key, p)} onZoom={() => onZoom(l.shown)} onTable={() => onTable(l.key)}
                                   onInfo={() => { setInfo(l.key); setMenu(null); }} onMove={(d) => onMove(l.key, d)}
                                   onRemove={() => { setMenu(null); onRemove(l.key); }} zoom={zoom} />
                  )}
                </div>
              ))}
            </div>
          )}

          {services.length > 0 && (
            <div className="lp-group">
              <div className="lp-group-label">Map services</div>
              {services.map((sv) => (
                <div key={sv.key} className={`lp-item ${menu === sv.key ? "open" : ""}`}>
                  <div className="lp-row">
                    <button type="button" className="lp-eye" onClick={() => onServicePatch(sv.key, { visible: !sv.visible })}
                            aria-pressed={sv.visible} aria-label={sv.visible ? `Hide ${sv.name}` : `Show ${sv.name}`}>
                      {sv.visible ? <IconEye /> : <IconEyeOff />}
                    </button>
                    <span className="lp-swatch raster" aria-hidden="true" />
                    <span className={`lp-name ${sv.visible ? "" : "dim"}`} title={sv.url}>{sv.name}</span>
                    <span className="mono lp-val">{sv.kind.toUpperCase()}</span>
                    <button type="button" className="lp-icon" aria-expanded={menu === sv.key} aria-label={`Options for ${sv.name}`}
                            onClick={() => setMenu((m) => (m === sv.key ? null : sv.key))}><IconMore /></button>
                  </div>
                  {menu === sv.key && (
                    <div className="lp-options">
                      <Slider label="Opacity" value={Math.round(sv.opacity * 100)} min={0} max={100} step={5}
                              onChange={(v) => onServicePatch(sv.key, { opacity: v / 100 })} fmt={(v) => `${v} %`} />
                      <p className="lp-note mono" style={{ overflowWrap: "anywhere" }}>{sv.url}</p>
                      <div className="lp-actions">
                        <button type="button" className="danger" onClick={() => { setMenu(null); onServiceRemove(sv.key); }}><IconTrash />Remove</button>
                      </div>
                    </div>
                  )}
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
                <button type="button" className="lp-icon" aria-label="Zoom to area of interest" onClick={() => onZoom(aoi.features || [aoi])}><IconZoomTo /></button>
              </div>
            </div>
          )}

          {image?.image_id && (
            <div className="lp-group">
              <div className="lp-group-label">Imagery</div>
              <div className={`lp-item ${rasterOpen ? "open" : ""}`}>
                <div className="lp-row">
                  <button type="button" className="lp-eye" onClick={() => onImageLayer({ ...imageLayer, visible: !imageLayer.visible })}
                          aria-pressed={imageLayer.visible} aria-label={imageLayer.visible ? "Hide image" : "Show image"}>
                    {imageLayer.visible ? <IconEye /> : <IconEyeOff />}
                  </button>
                  <span className="lp-swatch raster" aria-hidden="true" />
                  <span className={`lp-name ${imageLayer.visible ? "" : "dim"}`} title={image.filename}>{image.filename}</span>
                  {res && <span className="mono lp-val" title="Ground size of one pixel">{res}</span>}
                  <button type="button" className="lp-icon" aria-expanded={rasterOpen} aria-label="Image display settings"
                          onClick={() => setRasterOpen((o) => !o)}><IconChevronDown /></button>
                </div>
                {rasterOpen && (
                  <div className="lp-options">
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
                                  onChange={(e) => { const b = [...display.bands]; b[i] = Number(e.target.value); setDisplay({ bands: b }); }}>
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
                    <p className="lp-note">Display only: detection always reads the original pixels with its own bands and stretch</p>
                    <div className="lp-actions">
                      <button type="button" onClick={onZoomImage}><IconZoomTo />Zoom to</button>
                      <button type="button" disabled={!imageLayer.contrast && !imageLayer.brightness && !display.bands && !display.stretch}
                              onClick={() => onImageLayer({ ...imageLayer, contrast: 0, brightness: 0, bands: null, stretch: null })}>Reset display</button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {total === 0 && <p className="lp-empty">No layers yet: open an image and run a model</p>}
        </div>
        <footer className="lp-foot">
          <div className="lp-group-label">Basemap</div>
          <div className="lp-seg" role="radiogroup" aria-label="Basemap">
            {BASEMAP_CHOICES.map(([k, label]) => (
              <button key={k} type="button" role="radio" aria-checked={basemap === k} className={basemap === k ? "on" : ""}
                      onClick={() => onBasemap(k)}>{label}</button>
            ))}
          </div>
        </footer>
      </>)}
    </section>
  );
}
