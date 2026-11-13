import React from "react";
import { IconClose } from "./Icons.jsx";

const TERMINAL = new Set(["completed", "error", "interrupted"]);

function progressPct(job) {
  if (job.status === "queued") return 4;
  if (!job.tiles_total) return null; // unknown total: show an indeterminate bar
  return Math.min(100, Math.round((100 * (job.tiles_processed || 0)) / job.tiles_total));
}

function JobCard({ entry, color }) {
  const { job, modelMeta, imageName } = entry;
  return (
    <div className="progress-card">
      <div className="result-row">
        <span>
          <span className="legend-swatch-inline" style={{ background: color }} />
          {modelMeta?.name || job.model_id}
          {imageName ? <span className="note"> · {imageName}</span> : null}
        </span>
        {job.status === "completed" && job.result && (
          <span className="result-count">{job.result.final_detection_count}</span>
        )}
      </div>

      {!TERMINAL.has(job.status) && (() => {
        const pct = progressPct(job);
        return (
          <div>
            <div className="note mono">
              {job.status === "queued"
                ? "Queued"
                : job.tiles_total
                  ? `Tile ${job.tiles_processed ?? 0} of ${job.tiles_total} · ${pct}%`
                  : `Tile ${job.tiles_processed ?? 0}`}
            </div>
            <div className="progress-bar">
              <div
                className={`progress-fill${pct === null ? " progress-fill-indeterminate" : ""}`}
                style={pct === null ? undefined : { width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })()}

      {job.status === "error" && (
        <p className="note" style={{ color: "var(--error)" }}>{job.error}</p>
      )}

      {job.status === "interrupted" && (
        <p className="note" style={{ color: "var(--warn)" }}>{job.error || "Interrupted by a server restart. Run it again."}</p>
      )}

      {job.status === "completed" && job.result && (
        <>
          <p className="note">
            {job.result.tiles_processed} tiles · {job.result.raw_detection_count} raw detections ·{" "}
            {job.result.duration_seconds}s · device: {job.result.device_used}
          </p>
          {modelMeta?.status !== "ready" && (
            <p className="note" style={{ color: "var(--warn)" }}>{modelMeta?.notes}</p>
          )}
        </>
      )}
    </div>
  );
}

export default function AnalysisPanel({ entries, colorFor, onClose }) {
  return (
    <div className="panel right">
      <div className="models-dashboard-header">
        <p className="section-title" style={{ margin: 0 }}>Analysis</p>
        {onClose && (
          <button className="panel-close-btn" title="Close" onClick={onClose}>
            <IconClose />
          </button>
        )}
      </div>
      {!entries || entries.length === 0 ? (
        <p className="note">Run a prediction to see results here.</p>
      ) : (
        entries.map((entry) => (
          <JobCard key={entry.key} entry={entry} color={colorFor(entry.job.model_id)} />
        ))
      )}
    </div>
  );
}
