import React, { useEffect, useRef, useState } from "react";
import { SaveSamples } from "./SamplesStore.jsx";
import TrainSetup, { placeReady } from "./TrainSetup.jsx";
import BasemapSource from "./BasemapSource.jsx";
import { api } from "../../api/client.js";
import { classColor, isArea, CHECK_AREA } from "../labels/AnnotationTool.jsx";
import LabelTools from "../labels/LabelTools.jsx";
import { IconClose, IconUpload } from "../ui/Icons.jsx";

// Train your own model, beside the map, in five steps:
//   Setup    where training runs: Hugging Face GPU, RunPod GPU or this computer (TrainSetup.jsx)
//   Label    the GeoTIFF to learn from and its classes, then draw on the map (polygon, rectangle, square, circle, freehand) or click
//            an object and let SAM 2.1 outline it (LabelTools.jsx)
//   Dataset  tile the image and labels into a YOLO dataset
//   Train    fine-tune YOLO11 on it (a background job; the panel can close)
//   Use      name it, add it to the model list and run it on the current image
// The panel stays mounted while closed, so a running job keeps reporting.

export const TRAIN_STEPS = ["Setup", "Label", "Dataset", "Train", "Use"];
const PLACES = { hf: "Hugging Face GPU", runpod: "RunPod GPU", local: "This computer" };
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
  outline, onOutline, exact, onExact, segPreview,
  device = "cpu", onModelsChanged, onRunModel, onCheckModel,
  editId, onEditDone, onEditClass, onEditDelete, onImported, aoi, onAddClasses, onDeleteClass,
}) {
  const [fromMap, setFromMap] = useState(false);
  const [images, setImages] = useState([]);
  const [upload, setUpload] = useState(null);       // null | {pct} | {error}
  const [task, setTask] = useState("detect");
  const [ds, setDs] = useState(null);               // dataset job
  const [tr, setTr] = useState(null);               // training job
  const [size, setSize] = useState("nano");
  const [epochs, setEpochs] = useState("");
  const [modelName, setModelName] = useState("");
  const [registered, setRegistered] = useState(null);
  const [where, setWhere] = useState(() => {         // "hf" | "runpod" | "local": Train > Setup
    try { return localStorage.getItem("syntra_train_where") || "local"; } catch { return "local"; }
  });
  const [hosting, setHosting] = useState(null);
  function pickWhere(w) { setWhere(w); try { localStorage.setItem("syntra_train_where", w); } catch { /* private mode */ } }
  const [gpu, setGpu] = useState("rtx4090");
  const [cloud, setCloud] = useState("COMMUNITY");
  const [maxHours, setMaxHours] = useState("2");
  const [pushJob, setPushJob] = useState(null);       // samples saved to Hugging Face
  const [rp, setRp] = useState(null);
  const [err, setErr] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => { if (open && step === 1) api.listImages().then(setImages).catch(() => setImages([])); }, [open, step]);
  const refreshPlaces = () => {
    api.runpodStatus().then(setRp).catch(() => setRp(null));
    api.hosting().then(setHosting).catch(() => setHosting({ error: true }));
  };
  useEffect(() => { if (open && (step === 0 || step === 3)) refreshPlaces(); }, [open, step]);
  useEffect(() => { setPushJob(null); }, [ds?.job_id]);

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
  const labelled = objects.length;
  const dsDone = ds?.status === "completed";
  const trDone = tr && ["completed", "done"].includes(tr.status) && tr.result?.best;
  const canNext = [true, Boolean(image) && labelled > 0, dsDone, trDone, false][step];
  const placeOk = placeReady(where, hosting, rp);

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
      if (where === "runpod") {
        const job = await api.cloudTrain({
          push_job_id: pushJob.job_id, dataset_job_id: ds.job_id, model_size: size, gpu, cloud,
          epochs: epochs ? parseInt(epochs, 10) : null, max_hours: parseFloat(maxHours) || 2,
        });
        setTr(job);
        return;
      }
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
          <TrainSetup open={open && step === 0} where={where} onWhere={pickWhere} hosting={hosting} rp={rp} onRefresh={refreshPlaces} />
        )}

        {step === 1 && (
          <>
            <p className="train-label">Learn from</p>
            {image ? (
              <div className="train-image">
                <span className="train-image-name" title={image.filename}>{image.filename}</span>
                <span className="mono train-dim">
                  {image.health?.gsd_m ? `${Math.round(image.health.gsd_m * 100)} cm` : ""}{image.health?.crs ? ` · ${image.health.crs}` : ""}
                </span>
              </div>
            ) : null}
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
            <button type="button" className={`link-button ${fromMap ? "on" : ""}`} onClick={() => setFromMap((v) => !v)} aria-expanded={fromMap}>
              {fromMap ? "Hide basemap area" : "Or a basemap area"}
            </button>
            {fromMap && <BasemapSource aoi={aoi} onImage={(img) => { onImageChange(img); setFromMap(false); }} />}
            {upload?.error && <p className="train-error">{upload.error}</p>}

          {image && <LabelTools open={open && step === 1} classes={classes} counts={counts} activeClass={activeClass}
                      onActiveClass={onActiveClass} mode={mode} onMode={onMode} annotations={annotations}
                      onClearAnnotations={onClearAnnotations} labelError={labelError}
                      outline={outline} onOutline={onOutline} exact={exact} onExact={onExact} segPreview={segPreview}
                      editId={editId} onEditDone={onEditDone} onEditClass={onEditClass} onEditDelete={onEditDelete}
                      image={image} onImported={onImported} onAddClass={onAddClass}
                      onAddClasses={onAddClasses} onDeleteClass={onDeleteClass} />}
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
              <>
                <p className="train-ok"><span className="mono">{ds.info?.stats?.train_images ?? "–"}</span> training tiles ·{" "}
                  <span className="mono">{ds.info?.stats?.val_images ?? "–"}</span> for checking</p>
                <SaveSamples datasetJobId={ds.job_id} onSaved={setPushJob} />
              </>
            )}
            {ds?.status === "error" && <p className="train-error">{ds.error}</p>}
          </>
        )}

        {step === 3 && (
          <>
            <p className="train-label">Where</p>
            <p className="setup-where">
              <span className={`setup-dot ${placeOk ? "ok" : ""}`} aria-hidden="true" />{PLACES[where]}
              <span className="train-dim">{placeOk ? "ready" : "not ready"}</span>
              <button type="button" className="link-button" onClick={() => onStep(0)} disabled={trRunning}>Change</button>
            </p>
            {where === "runpod" && (
              !rp?.configured ? <p className="train-dim">Add your RunPod API key in Settings</p>
              : !pushJob ? <p className="train-dim">Save the samples to Hugging Face in the Dataset step first: the GPU reads them from there</p>
              : (
                <div className="gpu-opts">
                  <label className="train-field">GPU
                    <select className="train-select" value={gpu} onChange={(e) => setGpu(e.target.value)} disabled={trRunning}>
                      {[["rtx4090", "RTX 4090 · 24 GB"], ["rtx5090", "RTX 5090 · 32 GB"], ["l40s", "L40S · 48 GB"], ["a40", "A40 · 48 GB"]].map(([id, l]) => (
                        <option key={id} value={id}>{l}</option>))}
                    </select>
                  </label>
                  <div className="train-seg">
                    <button type="button" className={cloud === "COMMUNITY" ? "selected" : ""} onClick={() => setCloud("COMMUNITY")} disabled={trRunning}
                            title="Machines run by RunPod's hosts: cheapest">Community</button>
                    <button type="button" className={cloud === "SECURE" ? "selected" : ""} onClick={() => setCloud("SECURE")} disabled={trRunning}
                            title="RunPod's own data centres: for imagery whose licence asks for it">Secure</button>
                  </div>
                  <label className="train-field">Stop after (hours)
                    <input className="train-input mono" type="number" min="0.25" max="24" step="0.25" value={maxHours}
                           onChange={(e) => setMaxHours(e.target.value)} disabled={trRunning} />
                  </label>
                </div>
              )
            )}
            <p className="train-label">Model size</p>
            <div className="train-seg">
              {[["nano", "Fast"], ["small", "Balanced"], ["medium", "Accurate"]].map(([id, label]) => (
                <button type="button" key={id} className={size === id ? "selected" : ""} onClick={() => setSize(id)}>{label}</button>
              ))}
            </div>
            <label className="train-field">Epochs
              <input className="train-input mono" type="number" min="1" value={epochs} onChange={(e) => setEpochs(e.target.value)} placeholder="auto" />
            </label>
            <button type="button" className="run-button" onClick={startTraining}
                    disabled={trRunning || !placeOk || (where === "runpod" && !pushJob)}>
              {trRunning ? "Training…" : trDone ? "Train again" : where === "local" ? "Start training" : "Train on GPU"}
            </button>
            {trRunning && (
              <>
                <div className="pack-bar train-bar"><span className="pack-bar-fill" style={{ width: p ? `${(100 * p.epoch) / p.epochs}%` : "4%" }} /></div>
                <p className="train-dim mono">
                  {p ? `Epoch ${p.epoch}/${p.epochs} · mAP50 ${p.map50?.toFixed?.(2) ?? "–"}` : tr.status}
                </p>
                {tr.where === "runpod" && (
                  <p className="train-dim mono">
                    {[tr.pod_status, tr.cost_per_hour != null ? `$${Number(tr.cost_per_hour).toFixed(2)}/h` : null,
                      tr.cost_usd != null ? `$${Number(tr.cost_usd).toFixed(2)} so far` : null, tr.message].filter(Boolean).join(" · ")}
                  </p>
                )}
                {tr.where === "runpod" ? (
                  <button type="button" className="link-button" onClick={() => api.cloudStop(tr.job_id).then((j) => setTr((t) => ({ ...t, ...j }))).catch((e) => setErr(e.message))}>
                    Stop the GPU
                  </button>
                ) : (
                  <button type="button" className="link-button" onClick={() => api.trainingCancel(tr.job_id).catch(() => {})}>Cancel</button>
                )}
              </>
            )}
            {trDone && <p className="train-ok">Done · mAP50 <span className="mono">{p?.map50?.toFixed?.(2) ?? "–"}</span> on held-out tiles
              {tr.where === "runpod" && tr.cost_usd != null && <> · GPU cost <span className="mono">${Number(tr.cost_usd).toFixed(2)}</span></>}</p>}
            {tr?.warning && <p className="train-error">{tr.warning}</p>}
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
