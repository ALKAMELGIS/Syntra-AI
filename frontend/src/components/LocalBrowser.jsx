import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";
import { IconChevronLeft, IconFolder, IconRaster } from "./Icons.jsx";
import InfoNote from "./InfoNote.jsx";

// Open a GeoTIFF from this computer's folders. The engine runs on this machine,
// so the file is opened in place: no upload, no copy (the server hard-links it
// into the library and never writes into the original).

const LAST_DIR = "syntra_local_dir";

function formatBytes(n) {
  if (n == null) return "";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

// Long Windows paths: keep the end, which is where you are
function shortPath(p) {
  if (!p) return "This PC";
  const parts = p.split(/[\\/]/).filter(Boolean);
  return parts.length > 3 ? `…\\${parts.slice(-3).join("\\")}` : p;
}

// Scene products share one long prefix and differ only at the end (-MUX, -PAN,
// -Cloud): cut the middle, never the end.
function Name({ name }) {
  const tail = name.length > 22 ? name.slice(-12) : "";
  return (
    <span className="local-name">
      <span className="local-name-head">{tail ? name.slice(0, -12) : name}</span>
      {tail && <span className="local-name-tail">{tail}</span>}
    </span>
  );
}

// cloud / snow / quality masks ship next to the scene; they are not imagery
const MASK = /(cloud|snow|mask|cld|udm|shadow)/i;

export default function LocalBrowser({ onImported }) {
  const [listing, setListing] = useState(null);
  const [error, setError] = useState(null);
  const [opening, setOpening] = useState(null);

  function go(path) {
    setError(null);
    api.browseLocal(path)
      .then((l) => {
        setListing(l);
        try { if (l.path) localStorage.setItem(LAST_DIR, l.path); } catch { /* storage unavailable */ }
      })
      .catch((e) => {
        if (path) go(null);                     // a remembered folder that is gone: start from home
        else setError(e.message);
      });
  }

  useEffect(() => {
    let last = null;
    try { last = localStorage.getItem(LAST_DIR); } catch { last = null; }
    go(last);
  }, []);

  async function open(entry) {
    setOpening(entry.path);
    setError(null);
    try {
      onImported(await api.importLocal(entry.path));
    } catch (e) {
      setError(e.message);
    } finally {
      setOpening(null);
    }
  }

  return (
    <div className="local-browser">
      <div className="local-path">
        <button type="button" className="local-up" onClick={() => go(listing?.parent)} disabled={!listing?.parent}
                aria-label="Up one folder" title="Up one folder">
          <IconChevronLeft />
        </button>
        <span className="local-path-text mono" title={listing?.path || ""}>{shortPath(listing?.path)}</span>
      </div>
      {error && <InfoNote level="error" title={error} />}
      <ul className="local-list" aria-busy={!listing}>
        {listing?.entries.map((e) => (
          <li key={e.path}>
            <button type="button" className={`local-row ${e.kind} ${e.kind === "file" && MASK.test(e.name) ? "mask" : ""}`} disabled={!!opening}
                    onClick={() => (e.kind === "dir" ? go(e.path) : open(e))} title={e.name}>
              {e.kind === "dir" ? <IconFolder /> : <IconRaster />}
              <Name name={e.name} />
              <span className="local-meta mono">
                {opening === e.path ? "Opening…" : e.kind === "file" ? (MASK.test(e.name) ? "mask" : formatBytes(e.size_bytes)) : ""}
              </span>
            </button>
          </li>
        ))}
        {listing && listing.entries.length === 0 && <li className="local-empty">No folders or GeoTIFFs here</li>}
      </ul>
    </div>
  );
}
