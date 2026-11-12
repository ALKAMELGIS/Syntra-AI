import React from "react";
import { IconClose } from "./Icons.jsx";

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

      {job.status !== "completed" && job.status !== "error" && (
        <div>
          <div className="note">{job.status === "queued" ? "Queued…" : `Processing — tile ${job.tiles_processed ?? 0}`}</div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: job.status === "queued" ? "5%" : "60%" }} />
          </div>
        </div>
      )}

      {job.status === "error" && (
        <p className="note" style={{ color: "var(--error)" }}>{job.error}</p>
      )}

      {job.status === "completed" && job.result && (
        <>
          <p className="note">
            {job.result.tiles_processed} tiles · {job.result.raw_detection_count} raw detections ·{" "}
            {job.result.duration_seconds}s · device: {job.result.device_used}
          </p>
          {modelMeta?.status !== "ready" && (
            <p className="note" style={{ color: "var(--warn)" }}>⚠ {modelMeta?.notes}</p>
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
