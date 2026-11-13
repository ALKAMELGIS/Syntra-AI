import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client.js";

const BUILD_OPTIONS = [
  { value: "auto", label: "Automatic (recommended)" },
  { value: "cpu", label: "CPU only" },
  { value: "cuda", label: "CUDA (NVIDIA GPU)" },
];

const POLL_MS = 1200;

function LibraryRow({ r, checked, open, onToggleCheck, onToggleOpen, onInstallOne, rowBusy, disabled }) {
  return (
    <div className={`lib-row ${checked ? "lib-row-selected" : ""}`}>
      <div className="lib-row-main">
        <input
          type="checkbox"
          className="lib-row-checkbox"
          checked={checked}
          onChange={() => onToggleCheck(r.module)}
          disabled={disabled}
        />
        <span className={`model-status-dot ${r.installed ? "dot-ready" : "dot-error"}`} />
        <strong className="lib-row-name">{r.module}</strong>
        <button
          type="button"
          className="lib-row-more"
          title="Details"
          aria-label="Details"
          onClick={() => onToggleOpen(r.module)}
        >
          ⋯
        </button>
        <span className="lib-row-spacer" />
        <span className={`status-pill ${r.installed ? "status-ready" : "status-not_production_ready"}`}>
          {r.installed ? (r.detail ? r.detail : "Installed") : "Missing"}
        </span>
      </div>
      {open && (
        <div className="lib-row-details">
          <span className="note">
            {r.purpose}
            {r.path ? <> · {r.path}</> : null}
          </span>
          {!r.installed && (
            <button
              type="button"
              className="chip-button chip-small chip-primary"
              disabled={disabled}
              onClick={() => onInstallOne(r.module)}
            >
              {rowBusy ? "Installing…" : "Install this library"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function EnvironmentPanel() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [build, setBuild] = useState("auto");
  const [job, setJob] = useState(null);
  const [selected, setSelected] = useState(() => new Set());
  const [openRow, setOpenRow] = useState(null);
  const [folderOpen, setFolderOpen] = useState(false);
  const [copied, setCopied] = useState(null);
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
          if (j.status === "completed") setSelected(new Set());
        }
      } catch (err) {
        clearInterval(pollRef.current);
        setJob((prev) => ({ ...prev, status: "error", error: err.message }));
      }
    }, POLL_MS);
  }

  async function runAction(action, targets) {
    if (action === "check") {
      refresh();
      return;
    }
    try {
      const started = await api.startEnvironmentAction(action, build, targets);
      setJob({ status: "queued", action, build, targets: targets || [], log_tail: [], command: started.command });
      pollJob(started.job_id);
    } catch (err) {
      setJob({ status: "error", action, build, targets: targets || [], log_tail: [], error: err.message });
    }
  }

  const busy = job && (job.status === "queued" || job.status === "running");
  const locked = report?.actions_allowed === false;
  const missing = useMemo(() => (report ? report.results.filter((r) => !r.installed) : []), [report]);
  const selectedList = useMemo(() => Array.from(selected), [selected]);
  const installingModule = busy && job.targets && job.targets.length === 1 ? job.targets[0] : null;

  function toggleCheck(module) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(module)) next.delete(module);
      else next.add(module);
      return next;
    });
  }

  function toggleOpen(module) {
    setOpenRow((prev) => (prev === module ? null : module));
  }

  function selectAllMissing() {
    setSelected(new Set(missing.map((r) => r.module)));
  }

  function clearSelection() {
    setSelected(new Set());
  }

  function installOne(module) {
    runAction("install", [module]);
  }

  function copyPath(label, value) {
    if (!value) return;
    navigator.clipboard?.writeText(value).then(
      () => {
        setCopied(label);
        setTimeout(() => setCopied(null), 1500);
      },
      () => setCopied(null)
    );
  }

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

  const installedCount = report.total_count - report.missing_count;

  return (
    <div className="environment-panel-group">
      {/* --- Card 1: Library Status --- */}
      <div className="env-card">
        <div className="models-dashboard-header">
          <p className="section-title" style={{ margin: 0 }}>Library Status</p>
          <span className={`status-pill ${report.production_ready ? "status-ready" : "status-not_production_ready"}`}>
            {report.production_ready
              ? "Ready for training and detection"
              : `${report.missing_count}/${report.total_count} missing`}
          </span>
        </div>

        <div className="catalog-summary env-summary">
          <span className="catalog-summary-item">
            <span className="model-status-dot dot-ready" /> {installedCount} installed
          </span>
          <span className="catalog-summary-item">
            <span className="model-status-dot dot-error" /> {report.missing_count} missing
          </span>
          <span className="catalog-summary-item">Python {report.python} · {report.platform}</span>
          <span className="catalog-summary-item">
            {report.cuda.torch_installed
              ? report.cuda.cuda_available
                ? `GPU: ${report.cuda.devices.map((d) => d.name).join(", ")}`
                : "GPU: torch installed, CUDA not available (CPU-only build or no driver)"
              : "GPU: torch not installed"}
          </span>
          {report.proj_database.ok === true && (
            <span className="catalog-summary-item" style={{ color: "var(--ready)" }}>PROJ database: OK</span>
          )}
          {report.proj_database.ok === false && (
            <span className="catalog-summary-item" style={{ color: "var(--error)" }}>PROJ database: BROKEN</span>
          )}
        </div>

        <div className="lib-row-toolbar">
          <span className="note">
            {selectedList.length > 0 ? `${selectedList.length} selected` : "Select libraries, or open ⋯ to install one at a time"}
          </span>
          <div className="lib-row-toolbar-actions">
            <button className="chip-button chip-small" onClick={selectAllMissing} disabled={busy || missing.length === 0}>
              Select all missing
            </button>
            <button className="chip-button chip-small" onClick={clearSelection} disabled={busy || selectedList.length === 0}>
              Clear selection
            </button>
          </div>
        </div>

        <div className="lib-row-list">
          {report.results.map((r) => (
            <LibraryRow
              key={r.module}
              r={r}
              checked={selected.has(r.module)}
              open={openRow === r.module}
              onToggleCheck={toggleCheck}
              onToggleOpen={toggleOpen}
              onInstallOne={installOne}
              rowBusy={installingModule === r.module}
              disabled={busy}
            />
          ))}
        </div>

        {report.proj_database.ok === false && (
          <p className="note" style={{ color: "var(--error)", marginTop: 8 }}>
            ⚠ PROJ database conflict: {report.proj_database.error} — likely another installed GIS tool
            (PostgreSQL/PostGIS, QGIS, ArcGIS) is shadowing pyproj's own proj.db. See backend/.env.example.
          </p>
        )}
      </div>

      {/* --- Card 2: Installation Options --- */}
      <div className="env-card">
        <p className="section-title" style={{ margin: "0 0 10px" }}>Installation Options</p>

        <label className="build-select build-select-block">
          Build
          <select value={build} onChange={(e) => setBuild(e.target.value)} disabled={busy}>
            {BUILD_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>

        {locked && (
          <p className="env-locked-note">
            Package installs are disabled for remote browsers. Run them on the server, or set
            {" "}<code>SYNTRA_ALLOW_ENV_ACTIONS=1</code> in <code>backend/.env</code>.
          </p>
        )}

        <div className="install-buttons-grid">
          <button className="chip-button" onClick={() => runAction("check")} disabled={busy}>
            Check Dependencies
          </button>
          <button
            className="chip-button chip-primary"
            onClick={() => runAction("install", selectedList.length ? selectedList : null)}
            disabled={locked || busy || (selectedList.length === 0 && missing.length === 0)}
          >
            {selectedList.length ? `Install Selected (${selectedList.length})` : "Install Libraries"}
          </button>
          <button
            className="chip-button"
            onClick={() => runAction("repair", selectedList.length ? selectedList : null)}
            disabled={locked || busy}
          >
            {selectedList.length ? `Repair Selected (${selectedList.length})` : "Repair Installation"}
          </button>
          <button
            className="chip-button"
            onClick={() => runAction("update", selectedList.length ? selectedList : null)}
            disabled={locked || busy}
          >
            {selectedList.length ? `Update Selected (${selectedList.length})` : "Update Libraries"}
          </button>
        </div>

        <button type="button" className="env-collapse-toggle" onClick={() => setFolderOpen((v) => !v)}>
          <span className={`env-collapse-caret ${folderOpen ? "open" : ""}`}>▶</span> Python and library folder
        </button>
        {folderOpen && (
          <div className="env-folder-info">
            <div className="env-folder-row">
              <span className="env-folder-label">Python</span>
              <span className="env-folder-path" title={report.python_executable || ""}>
                {report.python_executable || "unknown"}
              </span>
              <button
                type="button"
                className="chip-button chip-small"
                onClick={() => copyPath("python", report.python_executable)}
              >
                {copied === "python" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="env-folder-row">
              <span className="env-folder-label">Folder</span>
              <span className="env-folder-path" title={report.site_packages_dir || ""}>
                {report.site_packages_dir || "unknown"}
              </span>
              <button
                type="button"
                className="chip-button chip-small"
                onClick={() => copyPath("folder", report.site_packages_dir)}
              >
                {copied === "folder" ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
        )}

        {job && (
          <div className="job-log-card">
            <div className="note job-log-head">
              {busy && <span className="job-spinner" aria-hidden="true" />}
              {job.action && <strong style={{ color: "var(--text)" }}>{job.action}</strong>}
              {job.build ? ` (${job.build})` : ""} — status: {job.status}
              {job.returncode !== undefined && job.returncode !== null && ` · exit code ${job.returncode}`}
            </div>
            {job.targets && job.targets.length > 0 && (
              <div className="note">Target{job.targets.length > 1 ? "s" : ""}: {job.targets.join(", ")}</div>
            )}
            {job.command && <div className="note" style={{ fontFamily: "monospace" }}>{job.command}</div>}
            {job.error && <div className="note" style={{ color: "var(--error)" }}>⚠ {job.error}</div>}
            {job.log_tail && job.log_tail.length > 0 && (
              <pre className="job-log-tail">{job.log_tail.join("\n")}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
