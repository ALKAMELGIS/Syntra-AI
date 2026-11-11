import React, { useEffect, useRef, useState } from "react";
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

export default function Sidebar({ models, selected, onToggle, image, onImageUploaded, onRun, running }) {
  const fileInput = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);

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
    e.target.value = ""; // allow re-selecting the same file after an error
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    doUpload(e.dataTransfer.files?.[0]);
  }

  return (
    <div className="panel">
      <p className="section-title">Source Image</p>
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
        {uploading ? "Uploading…" : image ? image.filename : "Drop GeoTIFF to begin, or click to browse"}
      </div>
      {uploadError && <p className="note" style={{ color: "var(--error)" }}>⚠ {uploadError}</p>}
      {image && (
        <p className="note">
          {image.width ? `${image.width} × ${image.height} px` : ""}
          {image.gsd_x_m ? ` · ${(image.gsd_x_m * 100).toFixed(0)} cm GSD` : ""}
        </p>
      )}

      <p className="section-title" style={{ marginTop: 20 }}>
        Detection Models
      </p>
      {models.map((m) => (
        <div
          key={m.model_id}
          className={`model-row ${selected === m.model_id ? "selected" : ""}`}
          onClick={() => onToggle(m.model_id)}
          title={m.notes}
        >
          <span className="model-name">
            <input type="radio" readOnly checked={selected === m.model_id} />
            {m.name}
          </span>
          <span className={`status-pill status-${m.status}`}>{STATUS_LABEL[m.status] || m.status}</span>
        </div>
      ))}

      <button className="run-button" disabled={!image || !selected || running} onClick={onRun}>
        {running ? "Running…" : "Run Prediction"}
      </button>
    </div>
  );
}
