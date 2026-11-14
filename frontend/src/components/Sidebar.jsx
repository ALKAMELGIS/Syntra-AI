import RasterHealth from "./RasterHealth.jsx";
import ModelList from "./ModelList.jsx";
import LocalBrowser from "./LocalBrowser.jsx";
import InfoNote, { splitMessage } from "./InfoNote.jsx";
import { useOptimizeJob } from "./OptimizeNote.jsx";
import React, { useRef, useState } from "react";
import { api } from "../api/client.js";
import { IconUpload, IconFolder, IconPlay, IconClose, IconDashboard } from "./Icons.jsx";

function formatBytes(bytes) {
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

// Long scene names (SVN3-20241127_L2A0000211147_03_...Cloud.tiff) have no
// spaces to wrap at. Keep them on one line inside the box: cut the middle,
// keep the start and the last characters with the extension; full name on hover.
function FileName({ name }) {
  const tail = name.length > 24 ? name.slice(-14) : "";
  return (
    <span className="dropzone-name mono" title={name}>
      <span className="dropzone-name-head">{tail ? name.slice(0, -14) : name}</span>
      {tail && <span className="dropzone-name-tail">{tail}</span>}
    </span>
  );
}

export default function Sidebar({ models, selectedModels, onToggleModel, onModelsChanged, onTrain, image, onImageUploaded, onRun, running, hasAoi = false, onClearAoi }) {
  const optimizing = useOptimizeJob(image, onImageUploaded);
  const fileInput = useRef(null);
  const batchInput = useRef(null);
  const [mode, setMode] = useState("single"); // "single" | "local" | "batch"
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [batchQueue, setBatchQueue] = useState([]); // [{filename, status: "uploading"|"done"|"error", image?, error?, progress?}]
  const [pendingFiles, setPendingFiles] = useState([]); // picked but not yet uploaded

  // Pick the image and, if present, its product .xml (same name or any one xml).
  function splitPick(fileList) {
    const files = Array.from(fileList || []);
    const tif = files.find((f) => /\.tiff?$/i.test(f.name));
    const stem = tif ? tif.name.replace(/\.tiff?$/i, "") : "";
    const xmls = files.filter((f) => /\.xml$/i.test(f.name));
    const xml = xmls.find((f) => f.name.replace(/\.xml$/i, "") === stem) || (xmls.length === 1 ? xmls[0] : null);
    return { tif, xml };
  }

  async function doUpload(fileList) {
    const { tif: file, xml } = splitPick(fileList);
    if (!file) {
      setUploadError("Choose a .tif or .tiff file.");
      return;
    }
    setUploadError(null);
    setUploading(true);
    setUploadProgress(0);
    try {
      const result = await api.uploadImage(file, setUploadProgress, xml);
      onImageUploaded(result);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  function handleFile(e) {
    doUpload(e.target.files);
    e.target.value = "";
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    doUpload(e.dataTransfer.files);
  }

  async function handleBatchFiles(fileList) {
    const files = Array.from(fileList || []).filter((f) => /\.tiff?$/i.test(f.name));
    if (files.length === 0) return;
    const initial = files.map((f) => ({ filename: f.name, status: "uploading" }));
    setBatchQueue((q) => [...q, ...initial]);
    for (const file of files) {
      try {
        const result = await api.uploadImage(file, (p) => {
          setBatchQueue((q) =>
            q.map((item) => (item.filename === file.name && item.status === "uploading" ? { ...item, progress: p } : item))
          );
        });
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
    mode !== "batch"
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
      <p className="section-title section-title-icon"><IconUpload /> Source image</p>

      <div className="source-mode-tabs">
        <button className={`source-mode-tab ${mode === "single" ? "active" : ""}`} onClick={() => setMode("single")}>
          Upload
        </button>
        <button className={`source-mode-tab ${mode === "local" ? "active" : ""}`} onClick={() => setMode("local")}
                title="Open a GeoTIFF from this computer's folders, without uploading it">
          This PC
        </button>
        <button className={`source-mode-tab ${mode === "batch" ? "active" : ""}`} onClick={() => setMode("batch")}>
          Batch
        </button>
      </div>

      {mode === "local" && (
        <>
          <LocalBrowser onImported={(entry) => { onImageUploaded(entry); setMode("single"); }} />
        </>
      )}
      {mode === "local" ? null : mode === "single" ? (
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
            <input ref={fileInput} type="file" accept=".tif,.tiff,.xml" multiple hidden onChange={handleFile} />
            <IconUpload className="dropzone-icon" />
            {image && !uploading ? (
              <>
                <FileName name={image.filename} />
                <span className="dropzone-meta mono">
                  {[image.size_bytes ? formatBytes(image.size_bytes) : null,
                    image.width ? `${image.width} × ${image.height} px` : null,
                    image.gsd_x_m && !image.health ? `${(image.gsd_x_m * 100).toFixed(0)} cm GSD` : null]
                    .filter(Boolean).join(" · ")}
                </span>
              </>
            ) : (
              <span>
                {uploading
                  ? (uploadProgress >= 1 ? "Reading the image…" : `Uploading… ${Math.round(uploadProgress * 100)}%`)
                  : "Drop a GeoTIFF, with its .xml if you have it"}
              </span>
            )}
          </div>
          {uploading && (
            <div className="upload-progress-bar">
              <div className="upload-progress-fill" style={{ width: `${Math.round(uploadProgress * 100)}%` }} />
            </div>
          )}
          {uploadError && <InfoNote level="error" title={splitMessage(uploadError)[0]} detail={splitMessage(uploadError)[1]} />}
          {image?.health && <RasterHealth health={image.health} optimizing={optimizing} />}
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
              <IconFolder className="folder-pick-icon" />
              <div className="folder-pick-title">Pick a folder</div>
              <div className="folder-pick-hint">Counts .tif/.tiff files (incl. subdirs). Nothing uploads yet.</div>
            </div>
          ) : (
            <div className="folder-pick-confirm-row">
              <span className="note">
                {pendingFiles.length} .tif/.tiff file{pendingFiles.length === 1 ? "" : "s"} found · {formatBytes(pendingBytes)}
              </span>
              <div style={{ display: "flex", gap: 6 }}>
                <button className="batch-queue-remove" onClick={clearPending} title="Clear selection"><IconClose /></button>
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
                {item.status === "uploading" && item.progress != null && (
                  <span className="note" style={{ margin: 0 }}>{Math.round(item.progress * 100)}%</span>
                )}
                {item.status === "error" && <InfoNote level="error" title={splitMessage(item.error)[0]} detail={splitMessage(item.error)[1]} />}
                <button className="batch-queue-remove" onClick={() => removeBatchItem(item.filename)}><IconClose /></button>
              </div>
            ))}
          </div>
        </>
      )}

      <p className="section-title section-title-icon" style={{ marginTop: 20 }}>
        <IconDashboard /> Detection models
      </p>
      <ModelList
        models={models}
        selectedModels={selectedModels}
        onToggleModel={onToggleModel}
        onModelsChanged={onModelsChanged}
        onTrain={onTrain}
        imageGsd={image?.health?.gsd_m ?? null}
      />

      <div className="run-dock">
      {hasAoi && (
        <div className="aoi-chip">
          <span className="aoi-chip-swatch" />
          <span className="aoi-chip-label">Drawn area only</span>
          <button className="aoi-chip-clear" title="Analyse the whole image" onClick={onClearAoi}>
            <IconClose />
          </button>
        </div>
      )}
      <button className="run-button" disabled={!canRun} onClick={handleRunClick}>
        {running ? "Running…" : <><IconPlay /> {hasAoi ? "Run in area" : "Run prediction"}</>}
      </button>
      </div>
    </div>
  );
}
