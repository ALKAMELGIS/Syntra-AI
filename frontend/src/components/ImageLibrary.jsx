import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api/client.js";

function formatDate(iso) {
  if (!iso) return "Unknown date";
  const d = new Date(iso);
  if (isNaN(d)) return "Unknown date";
  return d.toISOString().slice(0, 10);
}

function formatBytes(n) {
  if (!n) return "";
  const mb = n / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

// A real library of every GeoTIFF this backend has on disk (driven by
// GET /api/images, which reads the per-upload metadata sidecar written in
// app/core/image_library.py) — not a mockup. Clicking a card fetches that
// image's full record and hands it to the parent exactly like a fresh
// upload would, so it can be run against detection models immediately.
export default function ImageLibrary({ open, onClose, onSelectImage, currentImageId }) {
  const [entries, setEntries] = useState(null); // null = loading
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("none"); // none | with_meta | missing_meta

  useEffect(() => {
    if (!open) return;
    setError(null);
    api.listImages().then(setEntries).catch((err) => setError(err.message));
  }, [open]);

  const filtered = useMemo(() => {
    if (!entries) return [];
    let list = entries;
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((e) => e.filename.toLowerCase().includes(q));
    }
    if (filter === "with_meta") list = list.filter((e) => e.width != null);
    if (filter === "missing_meta") list = list.filter((e) => e.width == null);
    return list;
  }, [entries, search, filter]);

  if (!open) return null;

  async function handlePick(entry) {
    if (!entry.image_id) return; // orphaned file with no sidecar — nothing we can safely load
    try {
      const full = await api.getImage(entry.image_id);
      onSelectImage(full);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="image-library-dock">
      <div className="image-library-header">
        <span className="image-library-count">
          {entries === null ? "Loading…" : `${entries.length} image${entries.length === 1 ? "" : "s"}`}
        </span>
        <label className="image-library-viewport-check" title="Needs a saved geographic bounding box per image — not computed yet for this library, so this stays off.">
          <input type="checkbox" disabled />
          Filter to viewport
        </label>
        <input
          className="image-library-search"
          placeholder="Search image name…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <label className="image-library-filter">
          Filter
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="none">None</option>
            <option value="with_meta">Has metadata</option>
            <option value="missing_meta">Missing metadata</option>
          </select>
        </label>
        <button className="image-library-close" onClick={onClose}>✕</button>
      </div>
      {error && <div className="note" style={{ color: "var(--error)", padding: "0 12px" }}>⚠ {error}</div>}
      <div className="image-library-grid">
        {filtered.map((e) => (
          <div
            key={e.path}
            className={`image-library-card ${e.image_id === currentImageId ? "selected" : ""} ${!e.image_id ? "disabled" : ""}`}
            onClick={() => handlePick(e)}
            title={e.image_id ? "Click to load this image" : "No metadata sidecar — re-upload to use this file"}
          >
            <div className="image-library-card-date">{formatDate(e.uploaded_at)}</div>
            <div className="image-library-card-name">{e.filename}</div>
            <div className="image-library-card-meta">
              {e.width ? `${e.width}×${e.height} · ` : ""}
              {e.gsd_x_m ? `${Math.round(e.gsd_x_m * 100)} cm · ` : ""}
              {e.area_km2 ? `${e.area_km2} km²` : e.raster_meta_note || ""}
            </div>
            <div className="image-library-card-size">{formatBytes(e.size_bytes)}</div>
          </div>
        ))}
        {entries && filtered.length === 0 && (
          <div className="note" style={{ padding: 12 }}>No images match.</div>
        )}
      </div>
    </div>
  );
}
