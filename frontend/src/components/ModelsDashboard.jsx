import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const STATUS_LABEL = {
  ready: "Ready",
  validation: "Unvalidated",
  not_production_ready: "Not Ready",
  loading: "Loading",
  processing: "Processing",
  training: "Training",
  error: "Error",
  disabled: "Disabled",
};

const STATUS_DOT = {
  ready: "dot-ready",
  validation: "dot-warn",
  not_production_ready: "dot-error",
  loading: "dot-info",
  processing: "dot-info",
  training: "dot-info",
  error: "dot-error",
  disabled: "dot-dim",
};

const FORMATS = [
  { value: "yolo", label: "YOLO (images dir + labels dir)" },
  { value: "coco", label: "COCO (JSON + images dir)" },
  { value: "geojson", label: "GeoJSON annotations" },
];

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

function GsdChecker({ modelId }) {
  const [gsd, setGsd] = useState("0.30");
  const [result, setResult] = useState(null);
  const [checking, setChecking] = useState(false);

  async function run() {
    const value = parseFloat(gsd);
    if (Number.isNaN(value) || value <= 0) {
      setResult({ supported: false, message: "Enter a positive GSD value in meters." });
      return;
    }
    setChecking(true);
    try {
      const r = await api.gsdCheck(modelId, value);
      setResult(r);
    } catch (err) {
      setResult({ supported: false, message: err.message });
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="gsd-check">
      <span className="detail-label">GSD compatibility check</span>
      <div className="gsd-check-row">
        <input
          type="number"
          step="0.01"
          min="0"
          value={gsd}
          onChange={(e) => setGsd(e.target.value)}
          className="gsd-input"
        />
        <span className="note">m/px</span>
        <button className="chip-button" onClick={run} disabled={checking}>
          {checking ? "Checking…" : "Check"}
        </button>
      </div>
      {result && (
        <p className={`gsd-result ${result.supported ? "ok" : "warn"}`}>
          {result.supported ? "✓" : "⚠"} {result.message}
        </p>
      )}
    </div>
  );
}

function DatasetVersions({ modelId, reloadToken }) {
  const [versions, setVersions] = useState(null);
  const [error, setError] = useState(null);

  function load() {
    setError(null);
    api.listDatasetVersions(modelId).then(setVersions).catch((err) => setError(err.message));
  }

  useEffect(load, [modelId, reloadToken]);

  if (error) return <p className="note" style={{ color: "var(--error)" }}>⚠ {error}</p>;
  if (versions === null) return <p className="note">Loading dataset versions…</p>;
  if (versions.length === 0) {
    return <p className="note">No dataset versions registered yet for this model.</p>;
  }

  return (
    <div className="dataset-version-list">
      {versions.map((v) => (
        <div key={v.version_id} className="dataset-version-row">
          <div>
            <strong>{v.version_id}</strong>
            <span className="note"> · {v.annotation_format} · {v.sample_count} samples · {v.created_at}</span>
          </div>
          <div className="note">
            train {v.split.train?.length ?? "—"} / val {v.split.val?.length ?? "—"} / test {v.split.test?.length ?? "—"}
            {v.split.geographically_separated ? " · geo-separated" : ""}
          </div>
          {v.class_balance?.underrepresented_classes?.length > 0 && (
            <div className="note" style={{ color: "var(--warn)" }}>
              ⚠ underrepresented: {v.class_balance.underrepresented_classes.join(", ")}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function PrepareDatasetForm({ models, defaultModelId, onDone }) {
  const [form, setForm] = useState({
    model_id: defaultModelId,
    annotation_format: "yolo",
    annotation_path: "",
    images_dir: "",
    labels_dir: "",
    classes_file: "",
    train_frac: 0.7,
    val_frac: 0.15,
    test_frac: 0.15,
  });
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [problems, setProblems] = useState(null);

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function submit() {
    setSubmitting(true);
    setResult(null);
    setProblems(null);
    try {
      const payload = {
        ...form,
        train_frac: parseFloat(form.train_frac),
        val_frac: parseFloat(form.val_frac),
        test_frac: parseFloat(form.test_frac),
      };
      if (form.annotation_format !== "coco") delete payload.images_dir;
      if (form.annotation_format !== "yolo") {
        delete payload.labels_dir;
        delete payload.classes_file;
      }
      const r = await api.prepareDataset(payload);
      setResult(r);
      onDone?.(form.model_id);
    } catch (err) {
      if (err.detail?.problems) {
        setProblems(err.detail.problems);
      } else {
        setProblems([err.message]);
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="prepare-dataset-form">
      <div className="prepare-grid">
        <label>
          Model
          <select value={form.model_id} onChange={(e) => set("model_id", e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>{m.name}</option>
            ))}
          </select>
        </label>
        <label>
          Annotation format
          <select value={form.annotation_format} onChange={(e) => set("annotation_format", e.target.value)}>
            {FORMATS.map((f) => (
              <option key={f.value} value={f.value}>{f.label}</option>
            ))}
          </select>
        </label>
        <label className="prepare-wide">
          {form.annotation_format === "yolo" ? "Images directory" : form.annotation_format === "coco" ? "COCO annotations (.json)" : "GeoJSON file"}
          <input value={form.annotation_path} onChange={(e) => set("annotation_path", e.target.value)} placeholder="/path/to/data" />
        </label>
        {form.annotation_format === "coco" && (
          <label className="prepare-wide">
            Images directory
            <input value={form.images_dir} onChange={(e) => set("images_dir", e.target.value)} placeholder="/path/to/images" />
          </label>
        )}
        {form.annotation_format === "yolo" && (
          <>
            <label>
              Labels directory (optional)
              <input value={form.labels_dir} onChange={(e) => set("labels_dir", e.target.value)} placeholder="defaults to images dir" />
            </label>
            <label>
              classes.txt / data.yaml (optional)
              <input value={form.classes_file} onChange={(e) => set("classes_file", e.target.value)} />
            </label>
          </>
        )}
        <label>
          Train frac
          <input type="number" step="0.05" value={form.train_frac} onChange={(e) => set("train_frac", e.target.value)} />
        </label>
        <label>
          Val frac
          <input type="number" step="0.05" value={form.val_frac} onChange={(e) => set("val_frac", e.target.value)} />
        </label>
        <label>
          Test frac
          <input type="number" step="0.05" value={form.test_frac} onChange={(e) => set("test_frac", e.target.value)} />
        </label>
      </div>
      <button className="run-button" style={{ width: "auto", padding: "8px 18px" }} onClick={submit} disabled={submitting || !form.annotation_path}>
        {submitting ? "Validating & registering…" : "Prepare Dataset"}
      </button>
      {problems && (
        <div className="note" style={{ color: "var(--error)", marginTop: 8 }}>
          ⚠ Dataset failed validation:
          <ul style={{ margin: "4px 0 0 18px" }}>
            {problems.map((p, i) => <li key={i}>{typeof p === "string" ? p : JSON.stringify(p)}</li>)}
          </ul>
        </div>
      )}
      {result && (
        <div className="note" style={{ color: "var(--ready)", marginTop: 8 }}>
          ✓ Registered version {result.version_id} — {result.sample_count} samples
          (train {result.split.train} / val {result.split.val} / test {result.split.test}
          {result.split.geographically_separated ? ", geo-separated" : ""})
        </div>
      )}
    </div>
  );
}

function ModelCard({ model, expanded, onToggle, datasetReloadToken }) {
  const m = model;
  return (
    <div className="model-card">
      <div className="model-card-main" onClick={onToggle}>
        <span className={`model-status-dot ${STATUS_DOT[m.status] || "dot-dim"}`} />
        <div className="model-card-text">
          <strong>{m.name}</strong>
          <div className="note">
            {m.version} · {m.model_type} · {m.supported_image_sources.join(", ")}
          </div>
        </div>
        <span className={`status-pill status-${m.status}`}>{STATUS_LABEL[m.status] || m.status}</span>
        <button className="model-card-toggle">{expanded ? "▲ Hide" : "▼ Details"}</button>
      </div>
      {expanded && (
        <div className="model-card-details">
          <div className="model-detail-grid">
            <div><span className="detail-label">GSD range</span><span>{m.supported_gsd_m[0]}–{m.supported_gsd_m[1]} m</span></div>
            <div><span className="detail-label">Formats</span><span>{m.supported_formats.join(", ")}</span></div>
            <div><span className="detail-label">Hardware</span><span>{m.hardware_compatibility.join(", ")}</span></div>
            <div><span className="detail-label">Conf. threshold</span><span>{m.default_confidence_threshold}</span></div>
            <div><span className="detail-label">Processing stats</span><StatCell stats={m.processing_statistics} /></div>
          </div>
          <p className="note" style={{ marginTop: 8 }}>{m.notes}</p>
          <GsdChecker modelId={m.model_id} />
          <div style={{ marginTop: 14 }}>
            <span className="detail-label">Dataset versions</span>
            <DatasetVersions modelId={m.model_id} reloadToken={datasetReloadToken} />
          </div>
        </div>
      )}
    </div>
  );
}

export default function ModelsDashboard() {
  const [models, setModels] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [showTable, setShowTable] = useState(false);
  const [showPrepare, setShowPrepare] = useState(false);
  const [datasetReloadToken, setDatasetReloadToken] = useState(0);

  function refresh() {
    api.listModels().then(setModels).catch(() => setModels([]));
  }

  useEffect(refresh, []);

  if (models === null) {
    return <div className="panel" style={{ gridColumn: "1 / -1" }}>Loading model catalog…</div>;
  }

  const counts = models.reduce((acc, m) => {
    acc[m.status] = (acc[m.status] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="models-dashboard">
      <div className="models-dashboard-header">
        <p className="section-title" style={{ margin: 0 }}>Detection Models — Status</p>
        <button className="run-button" style={{ width: "auto", marginTop: 0, padding: "6px 14px" }} onClick={refresh}>
          Refresh
        </button>
      </div>

      <div className="catalog-summary">
        <span className="catalog-summary-item"><span className="model-status-dot dot-ready" /> {counts.ready || 0} ready</span>
        <span className="catalog-summary-item"><span className="model-status-dot dot-warn" /> {counts.validation || 0} unvalidated</span>
        <span className="catalog-summary-item"><span className="model-status-dot dot-error" /> {counts.not_production_ready || 0} not ready</span>
      </div>

      <div className="model-card-list">
        {models.map((m) => (
          <ModelCard
            key={m.model_id}
            model={m}
            expanded={expandedId === m.model_id}
            onToggle={() => setExpandedId(expandedId === m.model_id ? null : m.model_id)}
            datasetReloadToken={datasetReloadToken}
          />
        ))}
      </div>

      <div className="models-dashboard-header" style={{ marginTop: 18 }}>
        <p className="section-title" style={{ margin: 0 }}>Models &amp; Datasets</p>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="chip-button" onClick={() => setShowPrepare(!showPrepare)}>
            {showPrepare ? "Close Dataset Manager" : "Model & Dataset Manager…"}
          </button>
          <button className="chip-button" onClick={() => setShowTable(!showTable)}>
            {showTable ? "Hide Full Catalog Table" : "Open Full Catalog Table"}
          </button>
        </div>
      </div>

      {showPrepare && (
        <div className="prepare-dataset-panel">
          <p className="note" style={{ marginBottom: 10 }}>
            Validate an already-annotated dataset on disk, compute class balance and a geographically
            separated train/val/test split, and register it as a new version for a model.
          </p>
          <PrepareDatasetForm
            models={models}
            defaultModelId={models[0]?.model_id}
            onDone={(modelId) => {
              setExpandedId(modelId);
              setDatasetReloadToken((t) => t + 1);
            }}
          />
        </div>
      )}

      {showTable && (
        <table className="models-table" style={{ marginTop: 14 }}>
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
                <td>{m.supported_gsd_m[0]}–{m.supported_gsd_m[1]}</td>
                <td>{m.supported_image_sources.join(", ")}</td>
                <td>{m.supported_formats.join(", ")}</td>
                <td>
                  <span className={`status-pill status-${m.status}`}>{STATUS_LABEL[m.status] || m.status}</span>
                </td>
                <td>{m.hardware_compatibility.join(", ")}</td>
                <td>{m.default_confidence_threshold}</td>
                <td><StatCell stats={m.processing_statistics} /></td>
                <td className="note" style={{ maxWidth: 260 }}>{m.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
