import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client.js";
import { IconClose, IconDownload, IconZoomTo } from "./Icons.jsx";

// Remote sensing: a Sentinel-2 index over the AOI, live. Every change of
// index, scene or cloud limit redraws the layer (debounced, answers cached
// by the engine). The legend shows the AOI's own numbers; "Add to layer"
// writes per-feature values into a result layer's attributes.
const day = (d) => d.toISOString().slice(0, 10);
const ago = (n) => { const d = new Date(); d.setDate(d.getDate() - n); return day(d); };
const fmt = (v, n = 2) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(n));

function Ramp({ index }) {
  const lo = index.min, hi = index.max;
  const stops = index.ramp.map(([v, c]) => `${c} ${(((v - lo) / (hi - lo)) * 100).toFixed(1)}%`).join(", ");
  return (
    <div className="rs-ramp">
      <span className="rs-ramp-bar" style={{ background: `linear-gradient(90deg, ${stops})` }} />
      <span className="rs-ramp-ticks mono">
        {index.ramp.filter((_, i, a) => i === 0 || i === a.length - 1 || i === Math.floor(a.length / 2)).map(([v]) => <span key={v}>{v}</span>)}
      </span>
    </div>
  );
}

function Timeline({ data, index }) {
  const pts = (data?.series || []).filter((p) => p.mean != null);
  const [hover, setHover] = useState(null);
  if (!pts.length) return <p className="rs-dim">{data?.stopped || "No clear scene in the range"}</p>;
  const W = 256, H = 88, P = 4;
  const t0 = new Date(pts[0].date).getTime(), t1 = new Date(pts[pts.length - 1].date).getTime() || t0 + 1;
  const vs = pts.map((p) => p.mean);
  const lo = Math.min(...vs), hi = Math.max(...vs), span = hi - lo || 0.1;
  const x = (p) => P + ((new Date(p.date).getTime() - t0) / Math.max(1, t1 - t0)) * (W - 2 * P);
  const y = (v) => H - P - ((v - lo) / span) * (H - 2 * P);
  const h = hover != null ? pts[hover] : pts[pts.length - 1];
  return (
    <div className="rs-tl">
      <div className="rs-tl-read mono"><span>{h.date}</span><span>{index.name} {fmt(h.mean, 3)}</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} className="rs-tl-svg" onMouseLeave={() => setHover(null)}>
        <line x1={P} x2={W - P} y1={H - P} y2={H - P} className="rs-tl-axis" />
        <polyline points={pts.map((p) => `${x(p)},${y(p.mean)}`).join(" ")} className="rs-tl-line" />
        {pts.map((p, i) => (
          <circle key={p.date} cx={x(p)} cy={y(p.mean)} r={hover === i ? 4 : 2.5} className="rs-tl-dot"
                  onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      <div className="rs-tl-range mono"><span>{pts[0].date}</span><span>{fmt(lo, 2)} – {fmt(hi, 2)}</span><span>{pts[pts.length - 1].date}</span></div>
      {data.stopped && <p className="rs-dim">{data.stopped}</p>}
    </div>
  );
}

// Names shown before the engine answers (the ramps come with its answer)
const INDEX_NAMES = [["ndvi", "NDVI", "Vegetation health"], ["ndre", "NDRE", "Early crop stress"], ["savi", "SAVI", "Sparse vegetation on soil"],
  ["msavi", "MSAVI", "Sparse vegetation, no soil factor"], ["evi", "EVI", "Dense vegetation"], ["ndmi", "NDMI", "Vegetation moisture"],
  ["ndwi", "NDWI", "Open water"], ["ndsi", "NDSI", "Snow and ice"], ["ndbi", "NDBI", "Built-up areas"], ["bsi", "BSI", "Bare soil"],
  ["nbr", "NBR", "Burnt areas"]].map(([id, name, use]) => ({ id, name, use }));

export default function RemoteSensingPanel({ open, onClose, aoi, layers, overlay, onOverlay, onLayerValues, onZoom, onUseView }) {
  const [st, setSt] = useState(null);
  const [index, setIndex] = useState("ndvi");
  const [cloud, setCloud] = useState(20);
  const [start, setStart] = useState(ago(90));
  const [end, setEnd] = useState(day(new Date()));
  const [scene, setScene] = useState("composite");       // a date, or the least cloudy pixel in the range
  const [scenes, setScenes] = useState(null);
  const [mask, setMask] = useState(true);
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState(null);
  const [tl, setTl] = useState(null);
  const [target, setTarget] = useState("");
  const [added, setAdded] = useState(null);
  const ticket = useRef(0);

  const [down, setDown] = useState(false);               // the running engine has no Remote sensing yet
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!open || st) return;
    api.rsStatus().then((r) => { setSt(r); setDown(false); })
      .catch((e) => (/not found/i.test(e.message) || /404/.test(e.message) ? setDown(true) : setErr(e.message)));
  }, [open, st, nonce]);
  const idx = useMemo(() => st?.indices?.find((i) => i.id === index), [st, index]);
  const span = scene === "composite" ? { start, end } : { date: scene };
  const body = { index, max_cloud: cloud, mask_clouds: mask, ...span };
  const aoiKey = aoi ? JSON.stringify(aoi).length + ":" + JSON.stringify(aoi).slice(0, 80) : "";
  const targets = (layers || []).filter((l) => l.jobId && l.resultId && !l.isArea);

  // scenes in the range
  useEffect(() => {
    if (!open || !st?.configured || !aoi) { setScenes(null); return; }
    let live = true;
    const t = setTimeout(() => api.rsScenes({ aoi, start, end, max_cloud: cloud })
      .then((r) => live && setScenes(r.scenes)).catch((e) => live && setErr(e.message)), 400);
    return () => { live = false; clearTimeout(t); };
  }, [open, st?.configured, aoiKey, start, end, cloud]);              // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (scene !== "composite" && scenes && !scenes.some((s) => s.date === scene)) setScene("composite"); }, [scenes]); // eslint-disable-line

  // the live layer
  useEffect(() => {
    if (!open || !st?.configured || !aoi) return undefined;
    const my = ++ticket.current;
    setBusy("Loading"); setErr(null);
    const t = setTimeout(() => api.rsRender({ ...body, aoi })
      .then((r) => { if (my === ticket.current) { onOverlay({ ...r, opacity: overlay?.opacity ?? 0.85 }); setBusy(null); } })
      .catch((e) => { if (my === ticket.current) { setErr(e.message); setBusy(null); } }), 600);
    return () => clearTimeout(t);
  }, [open, st?.configured, aoiKey, index, cloud, start, end, scene, mask, nonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { setTl(null); }, [aoiKey, index, cloud, start, end, mask]);
  useEffect(() => { if (!target && targets.length) setTarget(targets[0].key); }, [targets.length]); // eslint-disable-line

  async function timeline() {
    setBusy("Timeline"); setErr(null);
    try { setTl(await api.rsTimeline({ index, aoi, start, end, max_cloud: cloud, mask_clouds: mask })); }
    catch (e) { setErr(e.message); }
    setBusy(null);
  }
  async function addToLayer() {
    const l = targets.find((x) => x.key === target);
    if (!l) return;
    setBusy("Values"); setErr(null); setAdded(null);
    try {
      const r = await api.rsAttributes({ ...body, job_id: l.jobId, model_id: l.resultId });
      setAdded(r);
      onLayerValues?.(l);
    } catch (e) { setErr(e.message); }
    setBusy(null);
  }
  async function exportTif() {
    setBusy("Export"); setErr(null);
    try {
      const blob = await api.rsExport({ ...body, aoi });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${index}_${(scene === "composite" ? end : scene).replaceAll("-", "")}.tif`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    } catch (e) { setErr(e.message); }
    setBusy(null);
  }

  if (!open) return null;
  const s = overlay?.stats;
  return (
    <section className="analysis-flyout rs-flyout" aria-label="Remote sensing">
      <header className="analysis-head">
        <span className="analysis-title">Remote sensing</span>
        <span className="rs-src mono">{st?.collection || "Sentinel-2 L2A"}</span>
        {busy && <span className="job-spinner" aria-label={busy} />}
        <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>
      <div className="rs-body">
        {down && (
          <div className="rs-note rs-note-act">
            <span>Restart the engine to load this tool</span>
            <button type="button" className="quiet-button" onClick={() => setNonce((n) => n + 1)}>Retry</button>
          </div>
        )}
        {st && !st.configured && <p className="rs-note">Add <span className="mono">SENTINELHUB_INSTANCE_ID</span> to .env, then restart</p>}
        <label className="rs-field"><span>Index</span>
          <select className="train-select" value={index} onChange={(e) => setIndex(e.target.value)}>
            {(st?.indices || INDEX_NAMES).map((i) => <option key={i.id} value={i.id}>{i.name} · {i.use}</option>)}
          </select>
        </label>
        <label className="rs-field">
          <span>Max cloud <b className="mono">{cloud}%</b></span>
          <input type="range" min="0" max="100" step="5" value={cloud} onChange={(e) => setCloud(Number(e.target.value))} />
        </label>
        <div className="rs-grid">
          <label className="rs-field"><span>From</span><input type="date" className="train-input mono" value={start} max={end} onChange={(e) => setStart(e.target.value)} /></label>
          <label className="rs-field"><span>To</span><input type="date" className="train-input mono" value={end} min={start} onChange={(e) => setEnd(e.target.value)} /></label>
        </div>
        <label className="rs-field"><span>Imagery</span>
          <select className="train-select mono" value={scene} onChange={(e) => setScene(e.target.value)}>
            <option value="composite">Least cloudy in range</option>
            {(scenes || []).map((x) => <option key={x.date} value={x.date}>{x.date} · {x.cloud == null ? "?" : `${x.cloud}%`}</option>)}
          </select>
        </label>
        <div className="rs-row">
          <span>Mask clouds</span>
          <div className="set-seg" role="radiogroup" aria-label="Mask clouds">
            {[[true, "On"], [false, "Off"]].map(([v, t]) => (
              <button key={t} type="button" role="radio" aria-checked={mask === v} className={mask === v ? "on" : ""} onClick={() => setMask(v)}>{t}</button>
            ))}
          </div>
        </div>

        {!aoi && (
          <div className="rs-aoi">
            <span className="rs-aoi-label">Area</span>
            <div className="rs-aoi-btns">
              <button type="button" className="quiet-button" onClick={() => window.dispatchEvent(new CustomEvent("syntra:draw", { detail: "rect" }))}>Rectangle</button>
              <button type="button" className="quiet-button" onClick={() => window.dispatchEvent(new CustomEvent("syntra:draw", { detail: "pencil" }))}>Polygon</button>
              <button type="button" className="quiet-button" onClick={onUseView} disabled={!onUseView}>Map view</button>
            </div>
          </div>
        )}
        {aoi && idx && (
          <div className="rs-legend">
            <div className="rs-legend-head"><b>{idx.name}</b><span className="mono">{overlay?.time || ""}</span>
              {overlay && onZoom && <button type="button" className="lp-icon" aria-label="Zoom to area" title="Zoom to area" onClick={onZoom}><IconZoomTo /></button>}
              {overlay && <button type="button" className="lp-icon" aria-label="Remove layer" title="Remove layer" onClick={() => onOverlay(null)}><IconClose /></button>}
            </div>
            <Ramp index={idx} />
            {s && (s.valid ? (
              <dl className="rs-stats mono">
                <dt>Mean</dt><dd className="rs-key">{fmt(s.mean, 3)}</dd>
                <dt>P10 – P90</dt><dd>{fmt(s.p10)} – {fmt(s.p90)}</dd>
                <dt>Clear</dt><dd>{Math.round((s.valid_fraction || 0) * 100)}%</dd>
              </dl>
            ) : <p className="rs-dim">No clear pixel</p>)}
            {overlay && (
              <label className="rs-field">
                <span>Opacity <b className="mono">{Math.round((overlay.opacity ?? 0.85) * 100)}%</b></span>
                <input type="range" min="0" max="1" step="0.05" value={overlay.opacity ?? 0.85}
                       onChange={(e) => onOverlay({ ...overlay, opacity: Number(e.target.value) })} />
              </label>
            )}
          </div>
        )}

        {aoi && st?.configured && (
          <>
            <div className="rs-actions">
              <button type="button" className="rs-primary" onClick={() => setNonce((n) => n + 1)} disabled={!!busy}>
                {busy === "Loading" ? "Loading…" : "Show index"}
              </button>
              <button type="button" className="quiet-button" onClick={timeline} disabled={!!busy}>{busy === "Timeline" ? "…" : "Timeline"}</button>
            </div>
            {tl && idx && <Timeline data={tl} index={idx} />}
            <div className="rs-attr">
              <select className="train-select" value={target} onChange={(e) => setTarget(e.target.value)} disabled={!targets.length} aria-label="Result layer">
                {targets.length ? targets.map((l) => <option key={l.key} value={l.key}>{l.name}</option>) : <option>No result layer</option>}
              </select>
              <button type="button" className="quiet-button" onClick={addToLayer} disabled={!!busy || !targets.length}>Add to layer</button>
            </div>
            {added && (
              <p className="rs-dim mono">{added.with_values}/{added.features} with values{added.too_small ? ` · ${added.too_small} too small` : ""}{added.no_data ? ` · ${added.no_data} cloudy` : ""}</p>
            )}
            <button type="button" className="quiet-button rs-export" onClick={exportTif} disabled={!!busy}><IconDownload />Export GeoTIFF</button>
          </>
        )}
        {err && <p className="lp-error">{err}</p>}
      </div>
    </section>
  );
}
