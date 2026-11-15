import React, { useEffect, useRef, useState } from "react";
import {
  IconArrowDown, IconArrowUp, IconClose, IconDownload, IconEye, IconEyeOff, IconInfo, IconMore,
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
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
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

function VectorOptions({ layer, onPatch, onZoom, onTable, onInfo, onMove, onRemove, first, last }) {
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
          {layer.imageName && (<><dt>Image</dt><dd className="mono" title={layer.imageName}>{layer.imageName}</dd></>)}
          <dt>Objects</dt><dd className="mono">{layer.features.length}</dd>
          <dt>Geometry</dt><dd className="mono">{geomTypes}</dd>
          <dt>CRS</dt><dd className="mono">EPSG:4326 · WGS84</dd>
          {b && (<><dt>Extent</dt><dd className="mono">{b[0][1].toFixed(4)} – {b[1][1].toFixed(4)} E<br />{b[0][0].toFixed(4)} – {b[1][0].toFixed(4)} N</dd></>)}
        </dl>
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

const BASEMAP_CHOICES = [["satellite", "Satellite"], ["streets", "Streets"], ["none", "None"]];

export default function LayersPanel({
  open, onClose, layers, onPatch, onMove, onRemove, onZoom, onTable, allVisible, onToggleAll,
  aoi, aoiVisible, onAoiVisible, aoiLabel, image, imageLayer, onImageLayer, onZoomImage, basemap, onBasemap,
}) {
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
  const total = layers.length + (aoi ? 1 : 0) + (image?.image_id ? 1 : 0);
  const gsd = image?.health?.gsd_m || image?.gsd_x_m;
  const res = gsd ? (gsd < 1 ? `${Math.round(gsd * 100)} cm` : `${+gsd.toFixed(2)} m`) : null;

  return (
    <section className="layers-panel" aria-label="Layers" ref={ref}>
      {infoLayer ? <LayerInfo layer={infoLayer} onBack={() => setInfo(null)} onClose={onClose} /> : (<>
        <header className="lp-head">
          <span className="lp-title">Layers</span><span className="lp-sub mono">{total || ""}</span>
          <button type="button" className="lp-icon" onClick={onClose} aria-label="Close layers"><IconClose /></button>
        </header>
        <div className="lp-body">
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
                                   onRemove={() => { setMenu(null); onRemove(l.key); }} />
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
                    <p className="lp-note">Display only: detection always reads the original pixels</p>
                    <div className="lp-actions">
                      <button type="button" onClick={onZoomImage}><IconZoomTo />Zoom to</button>
                      <button type="button" disabled={!imageLayer.contrast && !imageLayer.brightness}
                              onClick={() => onImageLayer({ ...imageLayer, contrast: 0, brightness: 0 })}>Reset display</button>
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
