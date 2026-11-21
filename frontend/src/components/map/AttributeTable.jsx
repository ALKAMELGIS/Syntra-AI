import React, { useMemo, useRef, useState } from "react";
import { IconClose, IconDownload, IconZoomTo } from "../ui/Icons.jsx";
import { downloadGeojson, fieldType, fieldsOf, fmtValue, inView } from "./layerUtils.js";

// The attribute table of a results layer, docked under the map. One tab per
// layer; click a header to sort, a row to select it (Ctrl / Shift add to the
// selection), double-click to zoom. Selected rows are outlined on the map and
// the map's own clicks select rows here. Rows are windowed, so a layer with
// thousands of objects scrolls as smoothly as one with ten.

const ROW = 30;

export default function AttributeTable({ open, layers, activeKey, onActive, selection, onSelect, view, onZoom, onClose, height, onHeight }) {
  const [sort, setSort] = useState({ key: "score", dir: -1 });
  const [query, setQuery] = useState("");
  const [extentOnly, setExtentOnly] = useState(false);
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  const anchor = useRef(null);
  const layer = layers.find((l) => l.key === activeKey) || layers[0];
  const sel = (layer && selection[layer.key]) || new Set();

  const fields = useMemo(() => (layer ? fieldsOf(layer.shown) : []), [layer]);
  // numbers line up on the right, text on the left
  const textCols = useMemo(() => new Set(fields.filter((k) => fieldType(layer?.shown || [], k) !== "number")), [fields, layer]);
  const rows = useMemo(() => {
    if (!layer) return [];
    const q = query.trim().toLowerCase();
    let r = layer.shown.filter((f) =>
      (!extentOnly || inView(f, view)) && (!selectedOnly || sel.has(f.id)) &&
      (!q || Object.values(f.properties || {}).some((v) => String(v).toLowerCase().includes(q))));
    const k = fields.includes(sort.key) ? sort.key : null;
    r = [...r].sort((a, b) => {
      if (!k) return (a.id - b.id) * (sort.key === "__oid" ? sort.dir : 1);
      const x = a.properties?.[k], y = b.properties?.[k];
      if (x == null) return 1; if (y == null) return -1;
      return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sort.dir;
    });
    return r;
  }, [layer, query, extentOnly, selectedOnly, sel, view, sort, fields]);

  if (!open || !layer) return null;

  const statKey = fields.includes(sort.key) && typeof layer.shown[0]?.properties?.[sort.key] === "number" && sort.key !== "lat" && sort.key !== "lon"
    ? sort.key : fields.find((k) => k.startsWith("area")) || null;
  const statRows = sel.size ? layer.shown.filter((f) => sel.has(f.id)) : rows;
  const nums = statKey ? statRows.map((f) => f.properties?.[statKey]).filter((v) => typeof v === "number") : [];
  const sum = nums.reduce((a, b) => a + b, 0);
  const inViewCount = view ? layer.shown.filter((f) => inView(f, view)).length : null;

  function pick(e, f, i) {
    const next = new Set(e.ctrlKey || e.metaKey || e.shiftKey ? sel : []);
    if (e.shiftKey && anchor.current != null) {
      const a = Math.min(anchor.current, i), b = Math.max(anchor.current, i);
      for (let j = a; j <= b; j++) next.add(rows[j].id);
    } else if ((e.ctrlKey || e.metaKey) && sel.has(f.id)) next.delete(f.id);
    else next.add(f.id);
    if (!e.shiftKey) anchor.current = i;
    onSelect(layer.key, next);
  }

  const bodyH = height - 44 - 32 - 32;
  const first = Math.max(0, Math.floor(scrollTop / ROW) - 6);
  const last = Math.min(rows.length, first + Math.ceil(bodyH / ROW) + 12);

  function startResize(e) {
    e.preventDefault();
    const y0 = e.clientY, h0 = height;
    const move = (ev) => onHeight(Math.max(160, Math.min(window.innerHeight * 0.7, h0 - (ev.clientY - y0))));
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  return (
    <section className="attr-table" style={{ height }} aria-label="Attribute table">
      <div className="attr-grip" onPointerDown={startResize} role="separator" aria-orientation="horizontal" aria-label="Resize table" />
      <header className="attr-head">
        <div className="attr-tabs" role="tablist">
          {layers.map((l) => (
            <button key={l.key} type="button" role="tab" aria-selected={l.key === layer.key}
                    className={l.key === layer.key ? "on" : ""} onClick={() => { onActive(l.key); setScrollTop(0); }}>
              <span className="lp-swatch" style={{ borderColor: l.color }} aria-hidden="true" />{l.name}<span className="mono">{l.shown.length}</span>
            </button>
          ))}
        </div>
        <input className="attr-search" type="search" placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search attributes" />
        <button type="button" className={`attr-tool ${extentOnly ? "on" : ""}`} aria-pressed={extentOnly} onClick={() => setExtentOnly((v) => !v)}>Map extent only</button>
        <button type="button" className={`attr-tool ${selectedOnly ? "on" : ""}`} aria-pressed={selectedOnly} disabled={!sel.size && !selectedOnly}
                onClick={() => setSelectedOnly((v) => !v)}>Selected only</button>
        <button type="button" className="attr-tool" disabled={!sel.size} onClick={() => onZoom(layer.shown.filter((f) => sel.has(f.id)))}>
          <IconZoomTo />Zoom to {sel.size || ""}</button>
        <button type="button" className="attr-tool" disabled={!sel.size} onClick={() => onSelect(layer.key, new Set())}>Clear</button>
        <button type="button" className="attr-tool" onClick={() => downloadGeojson(`${layer.name}${sel.size ? "_selected" : ""}`, sel.size ? layer.shown.filter((f) => sel.has(f.id)) : rows)}>
          <IconDownload />Export</button>
        <button type="button" className="lp-icon" onClick={onClose} aria-label="Close table"><IconClose /></button>
      </header>
      <div className="attr-scroll" onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
        <table>
          <thead>
            <tr>
              <th className={`left ${sort.key === "__oid" ? "sorted" : ""}`} onClick={() => setSort((s) => ({ key: "__oid", dir: s.key === "__oid" ? -s.dir : 1 }))}>OID</th>
              {fields.map((k) => (
                <th key={k} className={`${textCols.has(k) ? "left" : ""} ${sort.key === k ? "sorted" : ""}`}
                    aria-sort={sort.key === k ? (sort.dir > 0 ? "ascending" : "descending") : "none"}
                    onClick={() => setSort((s) => ({ key: k, dir: s.key === k ? -s.dir : (k === "class" ? 1 : -1) }))}>
                  {k}{sort.key === k ? (sort.dir > 0 ? " ↑" : " ↓") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {first > 0 && <tr style={{ height: first * ROW }} aria-hidden="true"><td colSpan={fields.length + 1} /></tr>}
            {rows.slice(first, last).map((f, j) => {
              const i = first + j;
              return (
                <tr key={f.id} className={sel.has(f.id) ? "sel" : ""} onClick={(e) => pick(e, f, i)} onDoubleClick={() => onZoom([f])}>
                  <td className="left mono">{f.id}</td>
                  {fields.map((k) => <td key={k} className={textCols.has(k) ? "left" : "mono"}>{fmtValue(k, f.properties?.[k])}</td>)}
                </tr>
              );
            })}
            {last < rows.length && <tr style={{ height: (rows.length - last) * ROW }} aria-hidden="true"><td colSpan={fields.length + 1} /></tr>}
          </tbody>
        </table>
        {rows.length === 0 && <p className="attr-empty">No rows match</p>}
      </div>
      <footer className="attr-foot mono">
        <span><b>{sel.size}</b> of {layer.shown.length} selected</span>
        {inViewCount != null && <span>in view {inViewCount}</span>}
        {rows.length !== layer.shown.length && <span>showing {rows.length}</span>}
        {statKey && nums.length > 0 && (
          <span className="attr-stat">{statKey} · {sel.size ? "selected" : "rows"} · sum <b>{fmtValue(statKey, sum)}</b> · mean <b>{fmtValue(statKey, sum / nums.length)}</b></span>
        )}
      </footer>
    </section>
  );
}
