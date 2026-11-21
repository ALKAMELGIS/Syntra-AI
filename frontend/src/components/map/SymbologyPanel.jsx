// A layer's Symbology page (Layers > ⋯ > Symbology): pick a style, a field and
// colours; the map, the legend and this page show the same classes.
import React, { useMemo } from "react";
import { IconChevronLeft, IconClose } from "../ui/Icons.jsx";
import { fieldType, fieldsOf } from "./layerUtils.js";
import { CATEGORICAL, METHODS, RAMPS, classesOf, rampColors } from "./symbology.js";

const STYLES = [
  ["single", "Location", "One symbol for every feature"],
  ["unique", "Types", "A colour per value of a field"],
  ["graduated", "Counts and amounts", "Colour classes of a number field"],
];

function Seg({ value, options, onChange, label }) {
  return (
    <div className="set-seg sym-seg" role="radiogroup" aria-label={label}>
      {options.map(([v, t]) => (
        <button key={v} type="button" role="radio" aria-checked={value === v} className={value === v ? "on" : ""} onClick={() => onChange(v)}>{t}</button>
      ))}
    </div>
  );
}

export function Legend({ classes, compact = false }) {
  return (
    <ul className={`sym-legend ${compact ? "compact" : ""}`}>
      {classes.map((c) => (
        <li key={c.label}><span className="sym-chip" style={{ background: c.color }} aria-hidden="true" />
          <span className="sym-label" title={c.label}>{c.label}</span><span className="mono">{c.count}</span></li>
      ))}
    </ul>
  );
}

export default function SymbologyPanel({ layer, onPatch, onBack, onClose }) {
  const sym = layer.symbology || { type: "single" };
  const set = (p) => onPatch({ symbology: { ...sym, ...p } });
  const fields = useMemo(() => fieldsOf(layer.features), [layer.features]);
  const numFields = fields.filter((k) => fieldType(layer.features, k) === "number");
  const textFields = fields.filter((k) => fieldType(layer.features, k) !== "number" && fieldType(layer.features, k) !== "empty");
  const classes = useMemo(() => classesOf(sym, layer.shown, layer.color), [sym, layer.shown, layer.color]);

  function pickStyle(type) {
    if (type === "single") return onPatch({ symbology: { ...sym, type, field: null } });
    const pool = type === "unique" ? textFields.concat(numFields) : numFields;
    const field = pool.includes(sym.field) ? sym.field : (type === "unique" ? (pool.includes("class") ? "class" : pool[0]) : (pool.includes("score") ? "score" : pool[0]));
    onPatch({ symbology: { method: "jenks", classes: 5, ramp: "viridis", ...sym, type, field: field || null, colors: type === "unique" ? sym.colors || {} : sym.colors } });
  }

  const pool = sym.type === "unique" ? textFields.concat(numFields) : numFields;
  return (
    <>
      <header className="lp-head">
        <button type="button" className="lp-icon" onClick={onBack} aria-label="Back to layers"><IconChevronLeft /></button>
        <span className="lp-title" title={layer.name}>{layer.name}</span><span className="lp-sub">Symbology</span>
        <button type="button" className="lp-icon" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>
      <div className="lp-body">
        <div className="lp-props sym">
          <div className="lp-sec">Style</div>
          <div className="sym-styles" role="radiogroup" aria-label="Style">
            {STYLES.map(([k, t, d]) => {
              const off = (k === "graduated" && !numFields.length) || (k === "unique" && !pool.length && !textFields.length && !numFields.length);
              return (
                <button key={k} type="button" role="radio" aria-checked={sym.type === k} disabled={off}
                        className={`sym-style ${sym.type === k ? "on" : ""}`} onClick={() => pickStyle(k)} title={off ? "No suitable field" : d}>
                  <span className={`sym-ico ${k}`} aria-hidden="true">
                    {k === "single" && <i style={{ background: sym.color || layer.color }} />}
                    {k === "unique" && CATEGORICAL.slice(0, 3).map((c) => <i key={c} style={{ background: c }} />)}
                    {k === "graduated" && rampColors(sym.ramp || "viridis", 3).map((c) => <i key={c} style={{ background: c }} />)}
                  </span>
                  <span>{t}</span>
                </button>
              );
            })}
          </div>

          {sym.type !== "single" && (
            <label className="sym-row"><span>Field</span>
              <select className="train-select" value={sym.field || ""} onChange={(e) => set({ field: e.target.value })} aria-label="Field">
                {pool.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </label>
          )}

          {sym.type === "graduated" && (<>
            <label className="sym-row"><span>Method</span>
              <select className="train-select" value={sym.method || "jenks"} onChange={(e) => set({ method: e.target.value })} aria-label="Classification method">
                {Object.entries(METHODS).map(([k, t]) => <option key={k} value={k}>{t}</option>)}
              </select>
            </label>
            <div className="sym-row"><span>Classes</span>
              <Seg label="Classes" value={sym.classes || 5} onChange={(v) => set({ classes: v })} options={[3, 4, 5, 6, 7].map((n) => [n, String(n)])} />
            </div>
            <div className="sym-row"><span>Ramp</span>
              <div className="sym-ramps" role="radiogroup" aria-label="Colour ramp">
                {Object.entries(RAMPS).map(([k, r]) => (
                  <button key={k} type="button" role="radio" aria-checked={(sym.ramp || "viridis") === k} title={r.label} aria-label={r.label}
                          className={`sym-ramp ${(sym.ramp || "viridis") === k ? "on" : ""}`}
                          style={{ background: `linear-gradient(90deg, ${(sym.reverse ? [...r.colors].reverse() : r.colors).join(", ")})` }}
                          onClick={() => set({ ramp: k })} />
                ))}
              </div>
            </div>
            <div className="sym-row"><span>Order</span>
              <Seg label="Ramp order" value={sym.reverse ? "rev" : "fwd"} onChange={(v) => set({ reverse: v === "rev" })} options={[["fwd", "Low → high"], ["rev", "High → low"]]} />
            </div>
          </>)}

          {sym.type === "single" && (
            <div className="sym-row"><span>Colour</span>
              <input type="color" className="sym-color" value={sym.color || layer.color} onChange={(e) => onPatch({ color: e.target.value, symbology: { ...sym, color: e.target.value } })} aria-label="Fill colour" />
            </div>
          )}

          <div className="lp-sec">Symbol</div>
          <label className="lp-slider"><span>Fill</span>
            <input type="range" className="gauge" min={0} max={1} step={0.05} value={sym.fillOpacity ?? 0.3}
                   style={{ "--fill": `${(sym.fillOpacity ?? 0.3) * 100}%` }} onChange={(e) => set({ fillOpacity: Number(e.target.value) })} />
            <span className="mono lp-val">{Math.round((sym.fillOpacity ?? 0.3) * 100)} %</span>
          </label>
          <label className="lp-slider"><span>Outline</span>
            <input type="range" className="gauge" min={0} max={5} step={0.5} value={sym.width ?? 1.5}
                   style={{ "--fill": `${((sym.width ?? 1.5) / 5) * 100}%` }} onChange={(e) => set({ width: Number(e.target.value) })} />
            <span className="mono lp-val">{(sym.width ?? 1.5).toFixed(1)} px</span>
          </label>
          <div className="sym-row"><span>Outline colour</span>
            <Seg label="Outline colour" value={sym.outline ? "custom" : "fill"} onChange={(v) => set({ outline: v === "fill" ? null : sym.outline || "#ffffff" })}
                 options={[["fill", "As fill"], ["custom", "Custom"]]} />
            {sym.outline && <input type="color" className="sym-color" value={sym.outline} onChange={(e) => set({ outline: e.target.value })} aria-label="Outline colour" />}
          </div>
          <label className="lp-slider"><span>Point size</span>
            <input type="range" className="gauge" min={2} max={14} step={1} value={sym.size ?? 5}
                   style={{ "--fill": `${(((sym.size ?? 5) - 2) / 12) * 100}%` }} onChange={(e) => set({ size: Number(e.target.value) })} />
            <span className="mono lp-val">{sym.size ?? 5} px</span>
          </label>

          <div className="lp-sec">Legend</div>
          {sym.type === "unique" ? (
            <ul className="sym-legend">
              {classes.map((c) => (
                <li key={c.label}>
                  {c.label.startsWith("Other (") ? <span className="sym-chip" style={{ background: c.color }} aria-hidden="true" /> : (
                    <input type="color" className="sym-chip-pick" value={c.color} aria-label={`Colour of ${c.label}`}
                           onChange={(e) => set({ colors: { ...(sym.colors || {}), [c.label]: e.target.value } })} />
                  )}
                  <span className="sym-label" title={c.label}>{c.label}</span><span className="mono">{c.count}</span>
                </li>
              ))}
            </ul>
          ) : <Legend classes={classes} />}
          {sym.type !== "single" && (
            <button type="button" className="link-button sym-reset" onClick={() => onPatch({ symbology: { type: "single" } })}>Reset to one symbol</button>
          )}
        </div>
      </div>
    </>
  );
}
