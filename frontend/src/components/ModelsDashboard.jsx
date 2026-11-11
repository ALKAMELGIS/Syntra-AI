import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const STATUS_LABEL = {
  ready: "Ready",
  validation: "Unvalidated",
  not_production_ready: "Not ready",
  loading: "Loading",
  processing: "Processing",
  training: "Training",
  error: "Error",
  disabled: "Disabled",
};

function StatCell({ stats }) {
  if (!stats || stats.run_count === 0) {
    return <span className="note">No runs yet</span>;
  }
  return (
    <div className="note">
      {stats.run_count} run{stats.run_count === 1 ? "" : "s"}
      {stats.error_count > 0 && <span style={{ color: "var(--error)" }}> ({stats.error_count} failed)</span>}
      <br />
      avg {stats.avg_duration_seconds}s · avg {stats.avg_detections_per_run} detections
      <br />
      last: {stats.last_status} {stats.last_run_at ? `(${stats.last_run_at})` : ""}
    </div>
  );
}

export default function ModelsDashboard() {
  const [models, setModels] = useState(null);

  function refresh() {
    api.listModels().then(setModels).catch(() => setModels([]));
  }

  useEffect(refresh, []);

  if (models === null) {
    return <div className="panel" style={{ gridColumn: "1 / -1" }}>Loading model catalog…</div>;
  }

  return (
    <div className="models-dashboard">
      <div className="models-dashboard-header">
        <p className="section-title" style={{ margin: 0 }}>Detection Models — Management</p>
        <button className="run-button" style={{ width: "auto", marginTop: 0, padding: "6px 14px" }} onClick={refresh}>
          Refresh
        </button>
      </div>
      <table className="models-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Version</th>
            <th>Type</th>
            <th>GSD range (m)</th>
            <th>Image sources</th>
            <th>Formats</th>
            <th>Status</th>
            <th>Hardware</th>
            <th>Conf. threshold</th>
            <th>Processing statistics</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model_id}>
              <td>
                <strong>{m.name}</strong>
                <div className="note">{m.model_id}</div>
              </td>
              <td>{m.version}</td>
              <td>{m.model_type}</td>
              <td>
                {m.supported_gsd_m[0]}–{m.supported_gsd_m[1]}
              </td>
              <td>{m.supported_image_sources.join(", ")}</td>
              <td>{m.supported_formats.join(", ")}</td>
              <td>
                <span className={`status-pill status-${m.status}`}>{STATUS_LABEL[m.status] || m.status}</span>
              </td>
              <td>{m.hardware_compatibility.join(", ")}</td>
              <td>{m.default_confidence_threshold}</td>
              <td>
                <StatCell stats={m.processing_statistics} />
              </td>
              <td className="note" style={{ maxWidth: 260 }}>{m.notes}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
