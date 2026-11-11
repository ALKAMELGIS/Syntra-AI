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

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await api.uploadImage(file);
      onImageUploaded(result);
    } catch (err) {
      alert(err.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="panel">
      <p className="section-title">Source Image</p>
      <div className="dropzone" onClick={() => fileInput.current?.click()}>
        <input ref={fileInput} type="file" accept=".tif,.tiff" hidden onChange={handleFile} />
        {uploading ? "Uploading…" : image ? image.filename : "Drop GeoTIFF to begin, or click to browse"}
      </div>
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
