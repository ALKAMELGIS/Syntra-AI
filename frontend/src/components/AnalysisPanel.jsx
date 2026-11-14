import React from "react";
import { IconClose } from "./Icons.jsx";

// Floating results panel that opens over the map, to the left of the map
// toolbar. Each run is one compact row: colour, model, count, then a single
// muted line of facts. Long explanations (model notes, full error text) live
// in tooltips, not on the card.

const TERMINAL = new Set(["completed", "error", "interrupted"]);

function progressPct(job) {
  if (job.status === "queued") return 4;
  if (!job.tiles_total) return null;
  return Math.min(100, Math.round((100 * (job.tiles_processed || 0)) / job.tiles_total));
}

function stripUuid(name) {
  return (name || "").replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/, "");
}

function JobRow({ entry, color, shown }) {
  const { job, modelMeta, imageName } = entry;
  const running = !TERMINAL.has(job.status);
  const pct = running ? progressPct(job) : null;
  const unvalidated = job.status === "completed" && modelMeta && modelMeta.status !== "ready";

  let meta;
  if (job.status === "completed" && job.result) meta = `${job.result.duration_seconds} s`;
  else if (job.status === "queued") meta = "Queued";
  else if (running) meta = job.tiles_total ? `${job.tiles_processed ?? 0}/${job.tiles_total}` : "Running";
  else if (job.status === "interrupted") meta = "Interrupted";
  else meta = "Failed";

  return (
    <li className={`analysis-row analysis-row-${job.status}`} title={job.error || modelMeta?.notes || ""}>
      <div className="analysis-row-main">
        <span className="analysis-swatch" style={{ background: color }} />
        <span className="analysis-name">{modelMeta?.name || job.model_id}</span>
        {job.status === "completed" && job.result && (
          <span className="analysis-count mono">
            {job.result.summary ? `${job.result.summary.vegetation_area_ha} ha`
              : shown != null && shown !== job.result.final_detection_count ? `${shown} of ${job.result.final_detection_count}`
              : job.result.final_detection_count}
          </span>
        )}
      </div>
      <div className="analysis-row-meta">
        <span className="analysis-file">{stripUuid(imageName)}</span>
        {job.has_aoi && <span className="analysis-tag">Area</span>}
        {job.result?.summary?.tree_candidates != null && (
          <span className="analysis-meta mono" title="Estimated tree crowns">{job.result.summary.tree_candidates} trees</span>
        )}
        <span className="analysis-meta mono">{meta}</span>
        {unvalidated && <span className="status-pill status-validation">Unvalidated</span>}
      </div>
      {running && (
        <div className="progress-bar analysis-progress">
          <div
            className={`progress-fill${pct === null ? " progress-fill-indeterminate" : ""}`}
            style={pct === null ? undefined : { width: `${pct}%` }}
          />
        </div>
      )}
    </li>
  );
}

export default function AnalysisPanel({ entries, colorFor, onClose, minScore = 0, onMinScore, shownCount }) {
  const hasScores = entries?.some((e) => e.job.status === "completed" && !e.job.result?.summary);
  return (
    <section className="analysis-flyout" aria-label="Analysis results">
      <header className="analysis-head">
        <span className="analysis-title">Analysis</span>
        {entries?.length > 0 && <span className="analysis-total mono">{entries.length}</span>}
        <button className="panel-close-btn" title="Close" onClick={onClose}>
          <IconClose />
        </button>
      </header>
      {hasScores && onMinScore && (
        <label className="analysis-score">
          <span>Min score</span>
          <input type="range" min="0" max="0.95" step="0.05" value={minScore}
                 onChange={(e) => onMinScore(parseFloat(e.target.value))} aria-label="Minimum detection score" />
          <span className="mono">{minScore.toFixed(2)}</span>
        </label>
      )}
      {!entries || entries.length === 0 ? (
        <p className="analysis-empty">No runs yet</p>
      ) : (
        <ul className="analysis-list">
          {entries.map((entry) => (
            <JobRow key={entry.key} entry={entry} shown={shownCount?.(entry.key)} color={colorFor(entry.job.model_id)} />
          ))}
        </ul>
      )}
    </section>
  );
}
