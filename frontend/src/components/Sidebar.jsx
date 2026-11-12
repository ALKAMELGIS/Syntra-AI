import React, { useRef, useState } from "react";
import { api } from "../api/client.js";

function formatBytes(bytes) {
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

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

export default function Sidebar({ models, selectedModels, onToggleModel, image, onImageUploaded, onRun, running }) {
  const fileInput = useRef(null);
  const batchInput = useRef(null);
  const [mode, setMode] = useState("single"); // "single" | "batch"
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [batchQueue, setBatchQueue] = useState([]); // [{filename, status: "uploading"|"done"|"error", image?, error?}]
  const [pendingFiles, setPendingFiles] = useState([]); // picked but not yet uploaded

  async function doUpload(file) {
    if (!file) return;
    if (!/\.tiff?$/i.test(file.name)) {
      setUploadError(`"${file.name}" is not a .tif/.tiff file.`);
      return;
    }
    setUploadError(null);
    setUploading(true);
    try {
      const result = await api.uploadImage(file);
      onImageUploaded(result);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleFile(e) {
    doUpload(e.target.files?.[0]);
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    doUpload(e.dataTransfer.files?.[0]);
  }

  async function handleBatchFiles(fileList) {
    const files = Array.from(fileList || []).filter((f) => /\.tiff?$/i.test(f.name));
    if (files.length === 0) return;
    const initial = files.map((f) => ({ filename: f.name, status: "uploading" }));
    setBatchQueue((q) => [...q, ...initial]);
    for (const file of files) {
      try {
        const result = await api.uploadImage(file);
        setBatchQueue((q) =>
          q.map((item) => (item.filename === file.name && item.status === "uploading" ? { ...item, status: "done", image: result } : item))
        );
      } catch (err) {
        setBatchQueue((q) =>
          q.map((item) => (item.filename === file.name && item.status === "uploading" ? { ...item, status: "error", error: err.message } : item))
        );
      }
    }
  }

  function removeBatchItem(filename) {
    setBatchQueue((q) => q.filter((item) => item.filename !== filename));
  }

  function handlePickFolder(fileList) {
    const files = Array.from(fileList || []).filter((f) => /\.tiff?$/i.test(f.name));
    setPendingFiles(files); // counting only — nothing uploads until "Upload" is clicked
  }

  function confirmUploadPending() {
    handleBatchFiles(pendingFiles);
    setPendingFiles([]);
  }

  function clearPending() {
    setPendingFiles([]);
  }

  const pendingBytes = pendingFiles.reduce((sum, f) => sum + f.size, 0);

  const batchImages = batchQueue.filter((q) => q.status === "done").map((q) => q.image);
  const canRun =
    mode === "single"
      ? !!image && selectedModels.length > 0 && !running
      : batchImages.length > 0 && selectedModels.length > 0 && !running;

  function handleRunClick() {
    if (mode === "single") {
      onRun([image]);
    } else {
      onRun(batchImages);
    }
  }

  return (
    <div className="panel">
      <p className="section-title">Source Image</p>

      <div className="source-mode-tabs">
        <button className={`source-mode-tab ${mode === "single" ? "active" : ""}`} onClick={() => setMode("single")}>
          Single file
        </button>
        <button className={`source-mode-tab ${mode === "batch" ? "active" : ""}`} onClick={() => setMode("batch")}>
          Folder batch
        </button>
      </div>

      {mode === "single" ? (
        <>
          <div
            className={`dropzone ${dragOver ? "drag-over" : ""}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input ref={fileInput} type="file" accept=".tif,.tiff" hidden onChange={handleFile} />
            {uploading ? "Uploading…" : image ? image.filename : "Drop GeoTIFF to begin, or click to browse · .tif · 0.25–0.70 m"}
          </div>
          {uploadError && <p className="note" style={{ color: "var(--error)" }}>⚠ {uploadError}</p>}
          {image && (
            <p className="note">
              {image.size_bytes ? `${formatBytes(image.size_bytes)} · ` : ""}
              {image.width ? `${image.width} × ${image.height} px` : ""}
              {image.gsd_x_m ? ` · ${(image.gsd_x_m * 100).toFixed(0)} cm GSD` : ""}
            </p>
          )}
        </>
      ) : (
        <>
          <input
            ref={batchInput}
            type="file"
            accept=".tif,.tiff"
            multiple
            webkitdirectory=""
            hidden
            onChange={(e) => {
              handlePickFolder(e.target.files);
              e.target.value = "";
            }}
          />

          {pendingFiles.length === 0 ? (
            <div className="folder-pick-zone" onClick={() => batchInput.current?.click()}>
              <span className="folder-pick-icon">📁</span>
              <div className="folder-pick-title">Pick a folder</div>
              <div className="folder-pick-hint">Counts .tif/.tiff files (incl. subdirs). Nothing uploads yet.</div>
            </div>
          ) : (
            <div className="folder-pick-confirm-row">
              <span className="note">
                {pendingFiles.length} .tif/.tiff file{pendingFiles.length === 1 ? "" : "s"} found · {formatBytes(pendingBytes)}
              </span>
              <div style={{ display: "flex", gap: 6 }}>
                <button className="batch-queue-remove" onClick={clearPending} title="Clear selection">✕</button>
                <button className="import-button" style={{ padding: "4px 10px" }} onClick={confirmUploadPending}>
                  Upload {pendingFiles.length}
                </button>
              </div>
            </div>
          )}

          <div className="note" style={{ marginBottom: 8 }}>
            {batchQueue.length === 0
              ? "No files uploaded yet."
              : `${batchImages.length}/${batchQueue.length} uploaded`}
          </div>
          <div className="batch-queue-list">
            {batchQueue.map((item) => (
              <div key={item.filename} className="batch-queue-row">
                <span className={`model-status-dot ${item.status === "done" ? "dot-ready" : item.status === "error" ? "dot-error" : "dot-info"}`} />
                <span className="batch-queue-name">{item.filename}</span>
                {item.status === "error" && <span className="note" style={{ color: "var(--error)" }}>{item.error}</span>}
                <button className="batch-queue-remove" onClick={() => removeBatchItem(item.filename)}>✕</button>
              </div>
            ))}
          </div>
        </>
      )}

      <p className="section-title" style={{ marginTop: 20 }}>
        Detection Models
      </p>
      <div className="detection-model-list">
        {models.map((m) => (
          <div
            key={m.model_id}
            className={`model-row ${selectedModels.includes(m.model_id) ? "selected" : ""}`}
            onClick={() => onToggleModel(m.model_id)}
            title={m.notes}
          >
            <span className="model-name">
              <input type="checkbox" readOnly checked={selectedModels.includes(m.model_id)} />
              <span className="model-name-text">{m.name}</span>
            </span>
            <span className={`status-pill status-${m.status}`}>{STATUS_LABEL[m.status] || m.status}</span>
          </div>
        ))}
      </div>

      <button className="run-button" disabled={!canRun} onClick={handleRunClick}>
        {running ? "Running…" : "Run Prediction"}
      </button>
    </div>
  );
}
