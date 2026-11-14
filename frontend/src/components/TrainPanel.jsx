import React, { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";
import { classColor, isArea, TRAIN_AREA, CHECK_AREA } from "./AnnotationTool.jsx";
import { IconClose, IconRectangle, IconPolygon, IconTrash, IconUpload } from "./Icons.jsx";

// Train your own model, beside the map, in five steps:
//   Data     the GeoTIFF to learn from (upload, or pick from the library) and its classes
//   Label    draw boxes / polygons on the map for each class
//   Dataset  tile the image and labels into a YOLO dataset
//   Train    fine-tune YOLO11 on it (a background job; the panel can close)
//   Use      name it, add it to the model list and run it on the current image
// The panel stays mounted while closed, so a running job keeps reporting.

export const TRAIN_STEPS = ["Data", "Label", "Dataset", "Train", "Use"];
const TASKS = [
  { id: "detect", label: "Boxes" },
  { id: "obb", label: "Rotated boxes" },
  { id: "segment", label: "Outlines" },
];

function usePoll(fn, active, ms) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => ref.current(), ms);
    return () => clearInterval(t);
  }, [active, ms]);
}

export default function TrainPanel({
  open, onClose, step, onStep, image, onImageChange,
  annotations, classes, onAddClass, activeClass, onActiveClass, mode, onMode, onClearAnnotations, labelError,
  device = "cpu", onModelsChanged, onRunModel, onCheckModel,
}) {
  const [images, setImages] = useState([]);
  const [upload, setUpload] = useState(null);       // null | {pct} | {error}
  const [newClass, setNewClass] = useState("");
  const [task, setTask] = useState("detect");
  const [ds, setDs] = useState(null);               // dataset job
  const [tr, setTr] = useState(null);               // training job
  const [size, setSize] = useState("nano");
  const [epochs, setEpochs] = useState("");
  const [modelName, setModelName] = useState("");
  const [registered, setRegistered] = useState(null);
  const [err, setErr] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => { if (open && step === 0) api.listImages().then(setImages).catch(() => setImages([])); }, [open, step]);

  usePoll(async () => {
    try {
      const st = await api.exportDatasetStatus(ds.job_id);
      setDs((d) => ({ ...d, ...st }));
    } catch { /* transient */ }
  }, ds && !["completed", "error", "canceled", "interrupted"].includes(ds.status), 1200);

  usePoll(async () => {
    try {
      const st = await api.trainingStatus(tr.job_id);
      setTr((t) => ({ ...t, ...st }));
    } catch { /* transient */ }
  }, tr && !["completed", "done", "error", "canceled", "interrupted"].includes(tr.status), 1500);

  const counts = {};
  for (const a of annotations) counts[a.class_name] = (counts[a.class_name] || 0) + 1;
  const objects = annotations.filter((a) => !isArea(a.class_name));
  const drawingArea = isArea(activeClass || "");
  const labelled = objects.length;
  const dsDone = ds?.status === "completed";
  const trDone = tr && ["completed", "done"].includes(tr.status) && tr.result?.best;
  const canNext = [Boolean(image), labelled > 0, dsDone, trDone, false][step];

  async function uploadFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUpload({ pct: 0 });
    try {
      const img = await api.uploadImage(file, (p) => setUpload({ pct: Math.round(p * 100) }));
      setUpload(null);
      onImageChange(img);
    } catch (error) {
      setUpload({ error: error.message });
    }
  }

  function addClass(e) {
    e.preventDefault();
    const name = newClass.trim().toLowerCase().replace(/\s+/g, "_");
    if (!name) return;
    onAddClass(name);
    setNewClass("");
  }

  async function buildDataset() {
    setErr(null);
    try {
      const r = await api.exportDataset({ image_id: image.image_id, task, tile: 640, overlap: 64, val_split: 0.2, neg_ratio: 0.1 });
      setDs({ job_id: r.job_id, status: "queued" });
    } catch (error) { setErr(error.detail?.message || error.message); }
  }

  async function startTraining() {
    setErr(null);
    setRegistered(null);
    try {
      const r = await api.startTraining({
        data_yaml: ds.data_yaml, device, model_size: size,
        epochs: epochs ? parseInt(epochs, 10) : null,
      });
      setTr({ job_id: r.job_id, status: "queued" });
    } catch (error) { setErr(error.detail?.message || error.message); }
  }

  async function register() {
    setErr(null);
    try {
      const name = modelName.trim() || Object.keys(counts).join(", ");
      const entry = await api.registerTrainedModel(tr.job_id, { name });
      setRegistered(entry);
      onModelsChanged?.();
    } catch (error) { setErr(error.detail?.message || error.message); }
  }

  const p = tr?.progress;
  const trRunning = tr && !["completed", "done", "error", "canceled", "interrupted"].includes(tr.status);

  return (
    <section className="train-flyout" aria-label="Train a model" hidden={!open}>
      <header className="analysis-head">
        <span className="analysis-title">Train</span>
        <span className="analysis-count mono">{step + 1}/{TRAIN_STEPS.length}</span>
        <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>

      <ol className="train-steps">
        {TRAIN_STEPS.map((s, i) => (
          <li key={s}>
            <button type="button" className={`train-step ${i === step ? "current" : ""} ${i < step ? "done" : ""}`}
                    onClick={() => onStep(i)} disabled={i > step && !(i === step + 1 && canNext)}>
              <span className="train-step-num mono">{i + 1}</span>{s}
            </button>
          </li>
        ))}
      </ol>

      <div className="train-body">
        {step === 0 && (
          <>
            <p className="train-label">Image to learn from</p>
            {image ? (
              <div className="train-image">
                <span className="train-image-name" title={image.filename}>{image.filename}</span>
                <span className="mono train-dim">
                  {image.health?.gsd_m ? `${Math.round(image.health.gsd_m * 100)} cm` : ""}{image.health?.crs ? ` · ${image.health.crs}` : ""}
                </span>
              </div>
            ) : (
              <p className="train-dim">Drone or satellite GeoTIFF, shown on the base map</p>
            )}
            <div className="train-row">
              <button type="button" className="quiet-button" onClick={() => fileRef.current?.click()} disabled={upload?.pct != null}>
                <IconUpload /> {upload?.pct != null ? (upload.pct >= 100 ? "Reading…" : `${upload.pct}%`) : "Upload GeoTIFF"}
              </button>
              <select className="train-select" value="" onChange={(e) => {
                const img = images.find((i) => i.image_id === e.target.value);
                if (img) onImageChange(img);
              }}>
                <option value="">From library</option>
                {images.map((i) => <option key={i.image_id} value={i.image_id}>{i.filename}</option>)}
              </select>
              <input ref={fileRef} type="file" accept=".tif,.tiff" hidden onChange={uploadFile} />
            </div>
            {upload?.error && <p className="train-error">{upload.error}</p>}

            <p className="train-label">Classes</p>
            <ul className="train-classes">
              {classes.map((c) => (
                <li key={c}><span className="train-swatch" style={{ background: classColor(c) }} />{c}
                  <span className="mono train-dim">{counts[c] || 0}</span></li>
              ))}
            </ul>
            <form className="train-row" onSubmit={addClass}>
              <input className="train-input" value={newClass} onChange={(e) => setNewClass(e.target.value)} placeholder="palm, car, building" />
              <button type="submit" className="quiet-button" disabled={!newClass.trim()}>Add</button>
            </form>
          </>
        )}

        {step === 1 && (
          <>
            <p className="train-label">Class</p>
            <ul className="train-classes train-classes-pick">
              {classes.map((c) => (
                <li key={c}>
                  <button type="button" className={c === activeClass ? "selected" : ""} onClick={() => { onActiveClass(c); if (drawingArea) onMode(null); }}>
                    <span className="train-swatch" style={{ background: classColor(c) }} />{c}
                    <span className="mono train-dim">{counts[c] || 0}</span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="train-row">
              <button type="button" className={`train-tool ${mode === "rect" && !drawingArea ? "active" : ""}`}
                      onClick={() => { if (drawingArea) onActiveClass(classes[0]); onMode(mode === "rect" && !drawingArea ? null : "rect"); }} disabled={!classes.length}>
                <IconRectangle /> Box
              </button>
              <button type="button" className={`train-tool ${mode === "polygon" && !drawingArea ? "active" : ""}`}
                      onClick={() => { if (drawingArea) onActiveClass(classes[0]); onMode(mode === "polygon" && !drawingArea ? null : "polygon"); }} disabled={!classes.length}>
                <IconPolygon /> Polygon
              </button>
            </div>
            <p className="train-label">Areas</p>
            <div className="train-row">
              <button type="button" className={`train-tool train-tool-area ${activeClass === TRAIN_AREA && mode ? "active" : ""}`}
                      onClick={() => { onActiveClass(TRAIN_AREA); onMode(activeClass === TRAIN_AREA && mode ? null : "rect"); }}>
                <span className="train-swatch train-swatch-dash" style={{ borderColor: classColor(TRAIN_AREA) }} /> Training
                <span className="mono train-dim">{counts[TRAIN_AREA] || 0}</span>
              </button>
              <button type="button" className={`train-tool train-tool-area ${activeClass === CHECK_AREA && mode ? "active" : ""}`}
                      onClick={() => { onActiveClass(CHECK_AREA); onMode(activeClass === CHECK_AREA && mode ? null : "rect"); }}>
                <span className="train-swatch train-swatch-dash" style={{ borderColor: classColor(CHECK_AREA) }} /> Accuracy
                <span className="mono train-dim">{counts[CHECK_AREA] || 0}</span>
              </button>
            </div>
            <p className="train-dim">
              {activeClass === TRAIN_AREA && mode ? "Drag a rectangle, then label every object inside it. Only these areas are learned from."
                : activeClass === CHECK_AREA && mode ? "Drag a rectangle and label every object inside it. It is kept out of training and used to score models."
                : mode === "rect" ? "Drag on the map. The tool stays on for the next object."
                : mode === "polygon" ? "Click the corners, double-click to finish."
                : classes.length ? "Pick a tool. Click a saved shape to delete it." : "Add a class in step 1 first."}
            </p>
            {labelError && <p className="train-error">{labelError}</p>}
            <div className="train-row train-row-split">
              <span className="train-dim"><span className="mono">{objects.length}</span> labelled · aim for 50+ per class</span>
              {annotations.length > 0 && (
                <button type="button" className="link-button" onClick={onClearAnnotations}><IconTrash /> Clear</button>
              )}
            </div>
          </>
        )}

        {step === 2 && (
          <>
            <p className="train-label">What the model draws</p>
            <div className="train-seg">
              {TASKS.map((t) => (
                <button type="button" key={t.id} className={task === t.id ? "selected" : ""} onClick={() => setTask(t.id)}>{t.label}</button>
              ))}
            </div>
            <p className="train-dim">640 px tiles with 64 px overlap, 20% held out to measure accuracy</p>
            <button type="button" className="run-button" onClick={buildDataset} disabled={ds && !["completed", "error"].includes(ds.status)}>
              {ds && !["completed", "error"].includes(ds.status) ? `Building… ${ds.progress ?? 0}%` : dsDone ? "Build again" : "Build dataset"}
            </button>
            {dsDone && (
              <p className="train-ok"><span className="mono">{ds.info?.stats?.train_images ?? "–"}</span> training tiles ·{" "}
                <span className="mono">{ds.info?.stats?.val_images ?? "–"}</span> for checking</p>
            )}
            {ds?.status === "error" && <p className="train-error">{ds.error}</p>}
          </>
        )}

        {step === 3 && (
          <>
            <p className="train-label">Model size</p>
            <div className="train-seg">
              {[["nano", "Fast"], ["small", "Balanced"], ["medium", "Accurate"]].map(([id, label]) => (
                <button type="button" key={id} className={size === id ? "selected" : ""} onClick={() => setSize(id)}>{label}</button>
              ))}
            </div>
            <label className="train-field">Epochs
              <input className="train-input mono" type="number" min="1" value={epochs} onChange={(e) => setEpochs(e.target.value)} placeholder="auto" />
            </label>
            <button type="button" className="run-button" onClick={startTraining} disabled={trRunning}>
              {trRunning ? "Training…" : trDone ? "Train again" : "Start training"}
            </button>
            {trRunning && (
              <>
                <div className="pack-bar train-bar"><span className="pack-bar-fill" style={{ width: p ? `${(100 * p.epoch) / p.epochs}%` : "4%" }} /></div>
                <p className="train-dim mono">
                  {p ? `Epoch ${p.epoch}/${p.epochs} · mAP50 ${p.map50?.toFixed?.(2) ?? "–"}` : tr.status}
                </p>
                <button type="button" className="link-button" onClick={() => api.trainingCancel(tr.job_id).catch(() => {})}>Cancel</button>
              </>
            )}
            {trDone && <p className="train-ok">Done · mAP50 <span className="mono">{p?.map50?.toFixed?.(2) ?? "–"}</span> on held-out tiles</p>}
            {tr && ["error", "canceled", "interrupted"].includes(tr.status) && (
              <p className="train-error">{tr.error || tr.result?.message || `Training ${tr.status}`}</p>
            )}
          </>
        )}

        {step === 4 && (
          <>
            <p className="train-label">Name</p>
            <input className="train-input" value={modelName} onChange={(e) => setModelName(e.target.value)}
                   placeholder={Object.keys(counts).join(", ") || "My model"} disabled={!!registered} />
            {!registered ? (
              <button type="button" className="run-button" onClick={register} disabled={!trDone}>Add to models</button>
            ) : (
              <>
                <p className="train-ok">Added to the model list as unvalidated</p>
                <button type="button" className="run-button" onClick={() => onRunModel(registered.model_id)} disabled={!image}>
                  Run on this image
                </button>
                {(counts[CHECK_AREA] || 0) > 0 && (
                  <button type="button" className="quiet-button" onClick={() => onCheckModel?.(registered.model_id)}>
                    Check accuracy on this image
                  </button>
                )}
              </>
            )}
            <p className="train-dim">Runs on any image from the sidebar, like the other models</p>
          </>
        )}
        {err && <p className="train-error">{err}</p>}
      </div>

      <footer className="train-foot">
        <button type="button" className="link-button" onClick={() => onStep(step - 1)} disabled={step === 0}>Back</button>
        <button type="button" className="quiet-button" onClick={() => onStep(step + 1)} disabled={!canNext || step === TRAIN_STEPS.length - 1}>Next</button>
      </footer>
    </section>
  );
}
