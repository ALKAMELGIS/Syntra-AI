import React from "react";

export default function AnalysisPanel({ job, modelMeta }) {
  if (!job) {
    return (
      <div className="panel right">
        <p className="section-title">Analysis</p>
        <p className="note">Run a prediction to see results here.</p>
      </div>
    );
  }

  return (
    <div className="panel right">
      <p className="section-title">Analysis</p>

      {job.status !== "completed" && job.status !== "error" && (
        <div className="progress-card">
          <div>{job.status === "queued" ? "Queued…" : `Processing — tile ${job.tiles_processed}`}</div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: job.status === "queued" ? "5%" : "60%" }} />
          </div>
        </div>
      )}

      {job.status === "error" && (
        <div className="progress-card" style={{ color: "var(--error)" }}>
          {job.error}
        </div>
      )}

      {job.status === "completed" && job.result && (
        <div className="progress-card">
          <div className="result-row">
            <span>{modelMeta?.name}</span>
            <span className="result-count">{job.result.final_detection_count}</span>
          </div>
          <p className="note">
            {job.result.tiles_processed} tiles · {job.result.raw_detection_count} raw detections ·{" "}
            {job.result.duration_seconds}s · device: {job.result.device_used}
          </p>
          {modelMeta?.status !== "ready" && (
            <p className="note" style={{ color: "var(--warn)" }}>
              ⚠ {modelMeta?.notes}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
