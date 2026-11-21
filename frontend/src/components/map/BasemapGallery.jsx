// The basemap gallery: opened from the toolbar's basemap button (and Settings > Map).
// Google tiles need a key, kept on the engine; a thumbnail that cannot load shows a quiet tile.
import React, { useEffect, useState } from "react";
import { IconClose } from "../ui/Icons.jsx";
import { BASEMAPS, BASEMAP_ORDER, thumbUrl } from "./basemaps.js";
import { api } from "../../api/client.js";

export default function BasemapGallery({ basemap, onBasemap }) {
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
              <span className="bm-thumb">{src ? <img src={src} alt="" referrerPolicy="no-referrer"
                onError={(e) => { e.currentTarget.style.display = "none"; e.currentTarget.parentElement.classList.add("bm-failed"); }} /> : <span className="bm-none" />}</span>
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
