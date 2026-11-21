import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api/client.js";
import { IconClose, IconTrash } from "./Icons.jsx";
import { fmtAreaKm2 } from "../prefs.js";

// Every GeoTIFF this engine has on disk (GET /api/images), as a gallery: the
// image's own preview, a short name read from the file name (sensor and
// product), the capture date when the name carries one, resolution and area.
// Copies of the same file (same name and size) show once with their count.
// The full file name is the card's native tooltip, nothing larger.

const SENSORS = [
  [/^SVN3/i, "SuperView Neo-3"], [/^SVN2/i, "SuperView Neo-2"], [/^SV-?2/i, "SuperView-2"], [/^SV-?1/i, "SuperView-1"],
  [/^WV0?2/i, "WorldView-2"], [/^WV0?3/i, "WorldView-3"], [/^WV0?4/i, "WorldView-4"], [/^GE0?1/i, "GeoEye-1"],
  [/^PNEO/i, "Pléiades Neo"], [/^PHR/i, "Pléiades"], [/^SPOT/i, "SPOT"], [/^S2[AB]/i, "Sentinel-2"],
  [/^JL1/i, "Jilin-1"], [/^GF/i, "Gaofen"], [/^KOMPSAT|^K3/i, "KOMPSAT"],
];
const PRODUCTS = [[/-PAN\b|_PAN\b/i, "PAN"], [/-MUX\b|_MUX\b/i, "MUX"], [/-PMS\b|_PMS\b/i, "Pan-sharpened"],
  [/-NIR\b/i, "NIR"], [/-Cloud\b/i, "Cloud mask"], [/-Snow\b/i, "Snow mask"]];

export function describe(filename) {
  const stem = filename.replace(/\.(tiff?|jp2|img)$/i, "");
  const sensor = SENSORS.find(([re]) => re.test(stem))?.[1] || null;
  const product = PRODUCTS.find(([re]) => re.test(stem))?.[1] || null;
  const m = stem.match(/(?:^|[_-])((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?=[-_T]|$)/);
  const taken = m ? `${m[1]}-${m[2]}-${m[3]}` : null;
  const short = stem.length > 28 ? `${stem.slice(0, 14)}…${stem.slice(-10)}` : stem;
  return { title: sensor || short, product, taken };
}

const fmtBytes = (n) => (!n ? "" : n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : `${(n / 1024 ** 2).toFixed(1)} MB`);
const fmtGsd = (g) => (g == null ? null : g < 1 ? `${Math.round(g * 100)} cm` : `${+g.toFixed(1)} m`);
const day = (iso) => { const d = new Date(iso); return isNaN(d) ? null : d.toISOString().slice(0, 10); };
const noData = (e) => e.health?.findings?.some((f) => f.code === "no_data");
const touches = (b, v) => b && v && !(b.north < v.south || b.south > v.north || b.east < v.west || b.west > v.east);

export default function ImageLibrary({ open, onClose, onSelectImage, currentImageId, view, onImageDeleted }) {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [scope, setScope] = useState("all");               // all | view
  const [sort, setSort] = useState("newest");              // newest | taken | name
  const [merge, setMerge] = useState(true);
  const [mode, setMode] = useState("gallery");             // gallery | list
  const [confirm, setConfirm] = useState(null);            // the card asking "Delete?"
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setError(null);
    api.listImages().then(setEntries).catch((err) => setError(err.message));
  }, [open]);

  const cards = useMemo(() => {
    if (!entries) return [];
    let groups;
    if (merge) {
      const by = new Map();
      for (const e of entries) {
        const k = `${e.filename}|${e.size_bytes || ""}|${e.image_id ? "" : "orphan"}`;
        by.set(k, [...(by.get(k) || []), e]);
      }
      groups = [...by.values()].map((list) => {
        const sorted = [...list].sort((a, b) => String(b.uploaded_at || "").localeCompare(String(a.uploaded_at || "")));
        const withMeta = sorted.find((e) => e.image_id) || sorted[0];
        return { ...withMeta, copies: list, total: list.reduce((n, e) => n + (e.size_bytes || 0), 0) };
      });
    } else groups = entries.map((e) => ({ ...e, copies: [e], total: e.size_bytes || 0 }));
    const q = search.trim().toLowerCase();
    let list = groups.map((e) => ({ ...e, d: describe(e.filename) }));
    if (q) list = list.filter((e) => [e.filename, e.d.title, e.d.product, e.d.taken].some((v) => v && v.toLowerCase().includes(q)));
    if (scope === "view") list = list.filter((e) => touches(e.bounds_wgs84, view));
    list.sort((a, b) => sort === "name" ? a.filename.localeCompare(b.filename)
      : sort === "taken" ? String(b.d.taken || "").localeCompare(String(a.d.taken || ""))
      : String(b.uploaded_at || "").localeCompare(String(a.uploaded_at || "")));
    return list;
  }, [entries, merge, search, scope, view, sort]);

  if (!open) return null;

  // each copy lives in its own library folder (backend/data/images/<folder>)
  const folderOf = (e) => e.path.split(/[\\/]/).slice(-2, -1)[0];
  async function remove(e) {
    setBusy(true); setError(null);
    try {
      for (const c of e.copies) await api.deleteLibraryCopy(folderOf(c));
      if (e.copies.some((c) => c.image_id && c.image_id === currentImageId)) onImageDeleted?.();
      setConfirm(null);
      setEntries(await api.listImages());
    } catch (err) {
      setError(err.detail?.message || err.detail || err.message);
      setEntries(await api.listImages().catch(() => entries));
    } finally { setBusy(false); }
  }

  async function pick(entry) {
    if (!entry.image_id) return;
    try { onSelectImage(await api.getImage(entry.image_id)); } catch (err) { setError(err.message); }
  }

  const files = entries?.length ?? 0;
  return (
    <section className={`image-library-dock gallery-${mode}`} aria-label="Images">
      <header className="gal-head">
        <span className="gal-title">Images</span>
        <span className="gal-count mono">{entries === null ? "…" : merge && cards.length !== files ? `${cards.length} · ${files} files` : files}</span>
        <input className="gal-search" type="search" placeholder="Search name, sensor, date" value={search}
               onChange={(e) => setSearch(e.target.value)} aria-label="Search images" />
        <div className="gal-seg" role="radiogroup" aria-label="Which images">
          <button type="button" role="radio" aria-checked={scope === "all"} className={scope === "all" ? "on" : ""} onClick={() => setScope("all")}>All</button>
          <button type="button" role="radio" aria-checked={scope === "view"} className={scope === "view" ? "on" : ""} onClick={() => setScope("view")}
                  title="Images whose footprint is in the map view">In view</button>
        </div>
        <select className="gal-select" value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort">
          <option value="newest">Newest added</option>
          <option value="taken">Date taken</option>
          <option value="name">Name</option>
        </select>
        <button type="button" className={`gal-chip ${merge ? "on" : ""}`} aria-pressed={merge} onClick={() => setMerge((v) => !v)}>Merge copies</button>
        <div className="gal-seg" role="radiogroup" aria-label="View">
          <button type="button" role="radio" aria-checked={mode === "gallery"} className={mode === "gallery" ? "on" : ""} onClick={() => setMode("gallery")}>Gallery</button>
          <button type="button" role="radio" aria-checked={mode === "list"} className={mode === "list" ? "on" : ""} onClick={() => setMode("list")}>List</button>
        </div>
        <button type="button" className="lp-icon gal-close" onClick={onClose} aria-label="Close images"><IconClose /></button>
      </header>
      {error && <p className="gal-error">{error}</p>}
      <div className="gal-grid">
        {cards.map((e) => {
          const onMap = e.copies.some((c) => c.image_id && c.image_id === currentImageId);
          const empty = noData(e);
          const meta = [e.d.taken || day(e.uploaded_at), fmtGsd(e.gsd_x_m), e.area_km2 ? fmtAreaKm2(e.area_km2, 0) : null].filter(Boolean).join(" · ");
          return (
            <div key={e.path} className="gal-wrap">
            <button type="button" className={`gal-card ${onMap ? "on" : ""} ${!e.image_id || empty ? "dim" : ""}`}
                    disabled={!e.image_id} onClick={() => pick(e)} title={e.filename}>
              {mode === "gallery" && (
                <span className="gal-thumb">
                  {e.image_id && !empty
                    ? <img src={api.imagePreviewUrl(e.image_id)} alt="" loading="lazy" decoding="async" />
                    : <span className="gal-empty">{!e.image_id ? "No metadata · upload again" : "No image data"}</span>}
                  {onMap && <span className="gal-badge">On map</span>}
                  {e.copies.length > 1 && <span className="gal-copies mono" title={`${e.copies.length} copies · ${fmtBytes(e.total)} in all`}>×{e.copies.length}</span>}
                </span>
              )}
              <span className="gal-name">{e.d.title}{e.d.product && <span className="gal-product"> {e.d.product}</span>}</span>
              <span className="gal-meta mono">{e.image_id ? meta : "no footprint"}{mode === "list" ? ` · ${fmtBytes(e.size_bytes)}` : ""}</span>
              {mode === "list" && (onMap || e.copies.length > 1) && (
                <span className="gal-tags mono">{onMap ? "on map" : ""}{onMap && e.copies.length > 1 ? " · " : ""}{e.copies.length > 1 ? `×${e.copies.length}` : ""}</span>
              )}
            </button>
              {confirm === e.path ? (
                <div className="gal-confirm" role="alertdialog" aria-label="Delete image">
                  <span>{e.copies.length > 1 ? `Delete ${e.copies.length} copies?` : "Delete this image?"}</span>
                  <span className="mono gal-confirm-sub">{fmtBytes(e.total) || "0 MB"} · original kept</span>
                  <div>
                    <button type="button" className="gal-del-go" disabled={busy} onClick={() => remove(e)}>{busy ? "Deleting…" : "Delete"}</button>
                    <button type="button" className="gal-del-cancel" disabled={busy} onClick={() => setConfirm(null)}>Cancel</button>
                  </div>
                </div>
              ) : (
                <button type="button" className="gal-del" aria-label={`Delete ${e.d.title}`} title="Delete from Syntra"
                        onClick={(ev) => { ev.stopPropagation(); setConfirm(e.path); }}><IconTrash /></button>
              )}
            </div>
          );
        })}
        {entries && cards.length === 0 && <p className="gal-none">{scope === "view" ? "No image in this map view" : "No images match"}</p>}
      </div>
    </section>
  );
}
