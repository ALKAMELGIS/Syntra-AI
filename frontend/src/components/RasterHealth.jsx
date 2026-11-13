import React, { useState } from "react";

// Compact raster check under the loaded image: the CRS / resolution / bit
// depth / layout facts an analyst checks before trusting a run, plus every
// warning with its fix. Data comes from the backend's check_raster().

const LEVEL_ORDER = { error: 0, warning: 1, info: 2, ok: 3 };

function fmtGsd(m) {
  if (m == null) return "—";
  return m < 1 ? `${(m * 100).toFixed(0)} cm` : `${m.toFixed(2)} m`;
}

export default function RasterHealth({ health }) {
  const [open, setOpen] = useState(health?.status !== "ok");
  if (!health) return null;
  const findings = [...(health.findings || [])].sort((a, b) => LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level]);
  const issues = findings.filter((f) => f.level === "error" || f.level === "warning").length;

  return (
    <div className={`raster-health raster-health-${health.status}`}>
      <button type="button" className="raster-health-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className={`raster-health-dot dot-${health.status === "ok" ? "ready" : health.status === "warning" ? "warn" : "error"}`} />
        <span className="raster-health-title">Raster check</span>
        <span className="raster-health-summary">
          {issues === 0 ? "Ready" : `${issues} to review`}
        </span>
        <span className={`env-collapse-caret ${open ? "open" : ""}`}>▶</span>
      </button>

      <dl className="raster-health-facts mono">
        <dt>CRS</dt><dd>{health.crs || "none"}</dd>
        <dt>GSD</dt><dd>{fmtGsd(health.gsd_m)}</dd>
        <dt>Data</dt><dd>{health.dtype} · {health.bands} band{health.bands > 1 ? "s" : ""}</dd>
        {health.rgb_bands && health.bands > 1 && (<><dt>RGB</dt><dd>{health.rgb_bands.join(" · ")}</dd></>)}
        <dt>Layout</dt><dd>{health.tiled ? "tiled" : "stripped"}{health.overviews?.length ? ` · ${health.overviews.length} overviews` : " · no overviews"}</dd>
      </dl>

      {open && (
        <ul className="raster-health-list">
          {findings.map((f) => (
            <li key={f.code} className={`raster-finding raster-finding-${f.level}`}>
              <div className="raster-finding-title">{f.title}</div>
              <div className="raster-finding-detail">{f.detail}</div>
              {f.fix && f.level !== "ok" && <code className="raster-finding-fix">{f.fix}</code>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
