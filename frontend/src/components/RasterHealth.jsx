import React, { useEffect, useRef, useState } from "react";
import InfoNote from "./InfoNote.jsx";
import { IconCheck } from "./Icons.jsx";

// The raster check as one small icon in the corner of the image box, coloured
// by its result (ready / to review / error). Hovering or clicking it opens a
// small message with the facts an analyst checks before trusting a run (CRS,
// resolution, data, bands, layout) and one line per thing worth knowing.
// Data comes from the backend's check_raster().

const LEVEL_ORDER = { error: 0, warning: 1, info: 2, ok: 3 };
const SHOWN_IN_FACTS = new Set(["crs", "gsd"]);   // "ok" findings the facts already say

function fmtGsd(m) {
  if (m == null) return "—";
  return m < 1 ? `${(m * 100).toFixed(0)} cm` : `${m.toFixed(2)} m`;
}

export default function RasterHealth({ health, optimizing = null }) {
  const [pinned, setPinned] = useState(false);
  const [hover, setHover] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!pinned) return;
    const down = (e) => { if (ref.current && !ref.current.contains(e.target)) setPinned(false); };
    const key = (e) => { if (e.key === "Escape") setPinned(false); };
    document.addEventListener("mousedown", down);
    window.addEventListener("keydown", key);
    return () => { document.removeEventListener("mousedown", down); window.removeEventListener("keydown", key); };
  }, [pinned]);
  if (!health) return null;
  const findings = [...(health.findings || [])]
    .filter((f) => !(f.level === "ok" && SHOWN_IN_FACTS.has(f.code)))
    // being fixed right now: the Layout row already says so
    .filter((f) => !(optimizing?.pct != null && (f.code === "striped" || f.code === "no_overviews")))
    .sort((a, b) => LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level]);
  const issues = findings.filter((f) => f.level === "error" || f.level === "warning").length;

  const level = health.status === "ok" ? "ready" : health.status === "warning" ? "warn" : "error";
  const busy = optimizing?.pct != null;
  const summary = busy ? `Optimising ${optimizing.pct}%` : issues === 0 ? "Ready" : `${issues} to review`;
  const open = pinned || hover;

  return (
    <div className="rh" ref={ref} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
         onClick={(e) => e.stopPropagation()}>
      <button type="button" className={`rh-btn rh-${level} ${busy ? "busy" : ""}`} aria-expanded={open}
              aria-label={`Raster check: ${summary}`} onClick={() => setPinned((v) => !v)}>
        {level === "ready" ? <IconCheck /> : <span className="rh-mark">!</span>}
      </button>
      {open && (
      <div className={`rh-pop raster-health-${health.status}`} role="dialog" aria-label="Raster check">
      <div className="raster-health-head">
        <span className={`raster-health-dot dot-${level === "ready" ? "ready" : level}`} />
        <span className="raster-health-title">Raster check</span>
        <span className="raster-health-summary">{summary}</span>
      </div>

      <dl className="raster-health-facts mono">
        <dt>CRS</dt><dd title={health.crs_name || ""}>{health.crs || "none"}</dd>
        <dt>GSD</dt><dd>{fmtGsd(health.gsd_m)}</dd>
        <dt>Data</dt><dd>{health.dtype} · {health.bands} band{health.bands > 1 ? "s" : ""}</dd>
        {health.rgb_bands && health.bands > 1 && (<><dt>RGB</dt><dd>{health.rgb_bands.join(" · ")}</dd></>)}
        <dt>Layout</dt>
        {optimizing?.pct != null ? (
          <dd className="raster-health-busy" title="Rewriting as tiles with overviews, like a Cloud-Optimized GeoTIFF. You can keep working.">
            tiling · {optimizing.pct}%
          </dd>
        ) : (
          <dd>{health.tiled ? "tiled" : "stripped"}{health.overviews?.length ? ` · ${health.overviews.length} overviews` : " · no overviews"}</dd>
        )}
      </dl>

      {optimizing?.failed && (
        <div className="raster-health-notes">
          <InfoNote level="warning" title="Couldn't optimise for fast reading" detail={optimizing.error} />
        </div>
      )}
      {findings.length > 0 && (
        <div className="raster-health-notes">
          {findings.map((f) => (
            <InfoNote key={f.code} level={f.level} title={f.title} detail={f.detail}
                      fix={f.level !== "ok" ? f.fix : null} />
          ))}
        </div>
      )}
    </div>
      )}
    </div>
  );
}
