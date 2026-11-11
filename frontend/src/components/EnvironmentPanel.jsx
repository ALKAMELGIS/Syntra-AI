import React, { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";

const BUILD_OPTIONS = [
  { value: "auto", label: "Automatic (recommended)" },
  { value: "cpu", label: "CPU only" },
  { value: "cuda", label: "CUDA (NVIDIA GPU)" },
];

const POLL_MS = 1200;

function LibraryRow({ r }) {
  return (
    <div className="lib-row">
      <span className={`model-status-dot ${r.installed ? "dot-ready" : "dot-error"}`} />
      <div className="lib-row-text">
        <strong>{r.module}</strong>
        <span className="note">
          {" "}
          {r.purpose}
          {r.path ? <> · {r.path}</> : null}
        </span>
      </div>
      <span className={`status-pill ${r.installed ? "status-ready" : "status-not_production_ready"}`}>
        {r.installed ? `Installed${r.detail ? ` · ${r.detail}` : ""}` : "Missing"}
      </span>
    </div>
  );
}

export default function EnvironmentPanel() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [build, setBuild] = useState("auto");
  const [job, setJob] = useState(null);
  const pollRef = useRef(null);

  function refresh() {
    setError(null);
    api.getEnvironment().then(setReport).catch((err) => setError(err.message));
  }

  useEffect(() => {
    refresh();
    return () => clearInterval(pollRef.current);
  }, []);

  function pollJob(jobId) {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const j = await api.environmentActionStatus(jobId);
        setJob(j);
        if (j.status === "completed" || j.status === "error") {
          clearInterval(pollRef.current);
          refresh();
        }
      } catch (err) {
        clearInterval(pollRef.current);
        setJob((prev) => ({ ...prev, status: "error", error: err.message }));
      }
    }, POLL_MS);
  }

  async function runAction(action) {
    if (action === "check") {
      refresh();
      return;
    }
    try {
      const started = await api.startEnvironmentAction(action, build);
      setJob({ status: "queued", action, build, log_tail: [], command: started.command });
      pollJob(started.job_id);
    } catch (err) {
      setJob({ status: "error", action, build, log_tail: [], error: err.message });
    }
  }

  const busy = job && (job.status === "queued" || job.status === "running");

  if (error) {
    return (
      <div className="panel" style={{ gridColumn: "1 / -1" }}>
        <p className="note" style={{ color: "var(--error)" }}>⚠ Could not reach the environment check: {error}</p>
      </div>
    );
  }
  if (report === null) {
    return <div className="panel" style={{ gridColumn: "1 / -1" }}>Checking Python environment…</div>;
  }

  return (
    <div className="environment-panel">
      <div className="models-dashboard-header">
        <p className="section-title" style={{ margin: 0 }}>Library Status</p>
        <span className={`status-pill ${report.production_ready ? "status-ready" : "status-not_production_ready"}`}>
          {report.production_ready
            ? "Ready for training and detection"
            : `${report.missing_count}/${report.total_count} missing`}
        </span>
      </div>
      <p className="note" style={{ marginBottom: 10 }}>
        Python {report.python} · {report.platform}
        {report.cuda.torch_installed
          ? report.cuda.cuda_available
            ? ` · GPU: ${report.cuda.devices.map((d) => d.name).join(", ")}`
            : " · GPU: torch installed, CUDA not available (CPU-only build or no driver)"
          : " · GPU: torch not installed"}
        {report.proj_database.ok === true && " · PROJ database: OK"}
        {report.proj_database.ok === false && " · PROJ database: BROKEN (see below)"}
      </p>

      <div className="lib-row-list">
        {report.results.map((r) => (
          <LibraryRow key={r.module} r={r} />
        ))}
      </div>

      {report.proj_database.ok === false && (
        <p className="note" style={{ color: "var(--error)", marginTop: 8 }}>
          ⚠ PROJ database conflict: {report.proj_database.error} — likely another installed GIS tool
          (PostgreSQL/PostGIS, QGIS, ArcGIS) is shadowing pyproj's own proj.db. See backend/.env.example.
        </p>
      )}

      <div className="models-dashboard-header" style={{ marginTop: 18 }}>
        <p className="section-title" style={{ margin: 0 }}>Installation Options</p>
      </div>
      <div className="install-options-row">
        <label className="build-select">
          Build
          <select value={build} onChange={(e) => setBuild(e.target.value)} disabled={busy}>
            {BUILD_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
        <button className="chip-button" onClick={() => runAction("check")} disabled={busy}>
          Check Dependencies
        </button>
        <button className="chip-button chip-primary" onClick={() => runAction("install")} disabled={busy}>
          Install Libraries
        </button>
        <button className="chip-button" onClick={() => runAction("repair")} disabled={busy}>
          Repair Installation
        </button>
        <button className="chip-button" onClick={() => runAction("update")} disabled={busy}>
          Update Libraries
        </button>
      </div>

      {job && (
        <div className="job-log-card">
          <div className="note">
            {job.action && <strong style={{ color: "var(--text)" }}>{job.action}</strong>}
            {job.build ? ` (${job.build})` : ""} — status: {job.status}
            {job.returncode !== undefined && job.returncode !== null && ` · exit code ${job.returncode}`}
          </div>
          {job.command && <div className="note" style={{ fontFamily: "monospace" }}>{job.command}</div>}
          {job.error && <div className="note" style={{ color: "var(--error)" }}>⚠ {job.error}</div>}
          {job.log_tail && job.log_tail.length > 0 && (
            <pre className="job-log-tail">{job.log_tail.join("\n")}</pre>
          )}
        </div>
      )}
    </div>
  );
}
