import React from "react";
import InfoNote from "./InfoNote.jsx";

// Compact raster check under the loaded image: the facts an analyst checks
// before trusting a run (CRS, resolution, data, bands, layout), then one short
// line per thing worth knowing. Each line's explanation and fix open in place
// from its (i). Data comes from the backend's check_raster().

const LEVEL_ORDER = { error: 0, warning: 1, info: 2, ok: 3 };
const SHOWN_IN_FACTS = new Set(["crs", "gsd"]);   // "ok" findings the facts already say

function fmtGsd(m) {
  if (m == null) return "—";
  return m < 1 ? `${(m * 100).toFixed(0)} cm` : `${m.toFixed(2)} m`;
}

export default function RasterHealth({ health }) {
  if (!health) return null;
  const findings = [...(health.findings || [])]
    .filter((f) => !(f.level === "ok" && SHOWN_IN_FACTS.has(f.code)))
    .sort((a, b) => LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level]);
  const issues = findings.filter((f) => f.level === "error" || f.level === "warning").length;

  return (
    <div className={`raster-health raster-health-${health.status}`}>
      <div className="raster-health-head">
        <span className={`raster-health-dot dot-${health.status === "ok" ? "ready" : health.status === "warning" ? "warn" : "error"}`} />
        <span className="raster-health-title">Raster check</span>
        <span className="raster-health-summary">{issues === 0 ? "Ready" : `${issues} to review`}</span>
      </div>

      <dl className="raster-health-facts mono">
        <dt>CRS</dt><dd title={health.crs_name || ""}>{health.crs || "none"}</dd>
        <dt>GSD</dt><dd>{fmtGsd(health.gsd_m)}</dd>
        <dt>Data</dt><dd>{health.dtype} · {health.bands} band{health.bands > 1 ? "s" : ""}</dd>
        {health.rgb_bands && health.bands > 1 && (<><dt>RGB</dt><dd>{health.rgb_bands.join(" · ")}</dd></>)}
        <dt>Layout</dt><dd>{health.tiled ? "tiled" : "stripped"}{health.overviews?.length ? ` · ${health.overviews.length} overviews` : " · no overviews"}</dd>
      </dl>

      {findings.length > 0 && (
        <div className="raster-health-notes">
          {findings.map((f) => (
            <InfoNote key={f.code} level={f.level} title={f.title} detail={f.detail}
                      fix={f.level !== "ok" ? f.fix : null} />
          ))}
        </div>
      )}
    </div>
  );
}
