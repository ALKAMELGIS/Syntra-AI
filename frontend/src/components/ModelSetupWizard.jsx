import React, { useEffect, useRef, useState } from "react";
import ModelsDashboard from "./ModelsDashboard.jsx";
import { api } from "../api/client.js";
import {
  IconGear,
  IconPencil,
  IconDatabase,
  IconPlay,
  IconEye,
} from "./Icons.jsx";

// Mirrors the reference workflow: Setup -> Annotate -> Dataset -> Train -> Detect.
// Each step is a real, honestly-labeled stage of the pipeline — a step that is
// "not built yet" says so plainly instead of faking a working control.
const STEPS = [
  { key: "setup", title: "Setup", icon: IconGear },
  { key: "annotate", title: "Annotate", icon: IconPencil },
  { key: "dataset", title: "Dataset", icon: IconDatabase },
  { key: "train", title: "Train", icon: IconPlay },
  { key: "detect", title: "Detect", icon: IconEye },
];

function StepBadge({ index, state }) {
  // state: "current" | "done" | "pending"
  return (
    <span className={`wizard-step-badge wizard-step-badge-${state}`} aria-hidden="true">
      {state === "done" ? "✓" : index + 1}
    </span>
  );
}

function WizardStepBar({ current, reached, onGo }) {
  return (
    <div className="wizard-steps" role="tablist">
      {STEPS.map((s, i) => {
        const state = i === current ? "current" : i <= reached ? "done" : "pending";
        const Icon = s.icon;
        const clickable = i <= reached;
        return (
          <button
            key={s.key}
            role="tab"
            aria-selected={i === current}
            className={`wizard-step wizard-step-${state}`}
            disabled={!clickable}
            onClick={() => clickable && onGo(i)}
          >
            <StepBadge index={i} state={state} />
            {Icon && <Icon />}
            <span className="wizard-step-title">{s.title}</span>
          </button>
        );
      })}
    </div>
  );
}

function DatasetStep() {
  const [form, setForm] = useState({
    image_path: "",
    class_names: "",
    annotations_json: "",
    tile: 640,
    overlap: 64,
    val_split: 0.2,
    neg_ratio: 0.1,
    task: "detect",
  });
  const [advanced, setAdvanced] = useState(false);
  const [images, setImages] = useState([]);
  const [imageId, setImageId] = useState("");
  // null = nothing picked yet, "loading" = fetching, otherwise {count, class_names}
  const [annPreview, setAnnPreview] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  // Populate the "pick from library" dropdown once — this is the shortcut
  // that closes the loop with the Annotate step (gap ②): no more hand-typed
  // image_path + annotations_json for the common case.
  useEffect(() => {
    api.listImages().then(setImages).catch(() => setImages([]));
  }, []);

  useEffect(() => {
    if (advanced || !imageId) {
      setAnnPreview(null);
      return;
    }
    setAnnPreview("loading");
    let cancelled = false;
    api.listAnnotations(imageId)
      .then((r) => { if (!cancelled) setAnnPreview({ count: r.count, class_names: r.class_names }); })
      .catch(() => { if (!cancelled) setAnnPreview({ count: 0, class_names: [] }); });
    return () => { cancelled = true; };
  }, [advanced, imageId]);

  useEffect(() => () => clearInterval(pollRef.current), []);

  async function submit() {
    setError(null);
    setJob(null);
    let payload;
    if (advanced) {
      let annotations;
      try {
        annotations = JSON.parse(form.annotations_json);
      } catch {
        setError("Annotations must be valid JSON — a list of {\"class_id\": 0, \"points_px\": [[x,y], ...]}.");
        return;
      }
      payload = {
        image_path: form.image_path,
        class_names: form.class_names.split(",").map((s) => s.trim()).filter(Boolean),
        annotations,
      };
    } else {
      payload = { image_id: imageId };
    }
    try {
      payload.tile = parseInt(form.tile, 10);
      payload.overlap = parseInt(form.overlap, 10);
      payload.val_split = parseFloat(form.val_split);
      payload.neg_ratio = parseFloat(form.neg_ratio);
      payload.task = form.task;
      const r = await api.exportDataset(payload);
      setJobId(r.job_id);
      setJob({ status: "queued" });
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const st = await api.exportDatasetStatus(r.job_id);
          setJob(st);
          if (st.status === "completed" || st.status === "error") clearInterval(pollRef.current);
        } catch { /* transient poll error */ }
      }, 1200);
    } catch (err) {
      setError(err.detail?.message || err.message);
    }
  }

  const canSubmit = advanced
    ? Boolean(form.image_path && form.class_names)
    : Boolean(imageId && annPreview && annPreview !== "loading" && annPreview.count > 0);

  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">Dataset</p>
      <p className="note" style={{ maxWidth: 640 }}>
        Tiles the image below into {form.tile}px tiles ({form.overlap}px overlap), converts the
        annotations into YOLO labels, filters black tiles, samples background tiles, and splits
        train/val. Pick an image you've already annotated (Annotate step) below, or fall back to
        pasting pixel-coordinate JSON by hand.
      </p>
      <div className="prepare-grid">
        {!advanced ? (
          <>
            <label className="prepare-wide">
              Image (annotated in the library)
              <select value={imageId} onChange={(e) => setImageId(e.target.value)}>
                <option value="">— select an image —</option>
                {images.map((img) => (
                  <option key={img.image_id} value={img.image_id}>{img.filename}</option>
                ))}
              </select>
            </label>
            <div className="prepare-wide note" style={{ marginTop: -4 }}>
              {!imageId ? (
                "Pick an image you drew annotations on with the Annotate tool on the map."
              ) : annPreview === "loading" ? (
                "Checking annotations…"
              ) : annPreview?.count > 0 ? (
                <span style={{ color: "var(--ready)" }}>
                  ✓ {annPreview.count} annotation(s) — classes: {annPreview.class_names.join(", ")}
                </span>
              ) : (
                <span style={{ color: "var(--error)" }}>
                  ⚠ No annotations saved for this image yet — open it on the map and use the Annotate tool first.
                </span>
              )}
            </div>
          </>
        ) : (
          <>
            <label className="prepare-wide">
              Image path
              <input value={form.image_path} onChange={(e) => set("image_path", e.target.value)} placeholder="/path/to/orthomosaic.tif" />
            </label>
            <label className="prepare-wide">
              Class names (comma-separated)
              <input value={form.class_names} onChange={(e) => set("class_names", e.target.value)} placeholder="tree, building" />
            </label>
            <label className="prepare-wide">
              Annotations (JSON — pixel coordinates)
              <textarea
                value={form.annotations_json}
                onChange={(e) => set("annotations_json", e.target.value)}
                placeholder='[{"class_id": 0, "points_px": [[100,100],[160,100],[160,160],[100,160]]}]'
                rows={4}
                style={{ fontFamily: "monospace", fontSize: 11.5 }}
              />
            </label>
          </>
        )}
        <label>
          Tile size (px)
          <input type="number" value={form.tile} onChange={(e) => set("tile", e.target.value)} />
        </label>
        <label>
          Overlap (px)
          <input type="number" value={form.overlap} onChange={(e) => set("overlap", e.target.value)} />
        </label>
        <label>
          Val split
          <input type="number" step="0.05" value={form.val_split} onChange={(e) => set("val_split", e.target.value)} />
        </label>
        <label>
          Background ratio
          <input type="number" step="0.05" value={form.neg_ratio} onChange={(e) => set("neg_ratio", e.target.value)} />
        </label>
        <label>
          Task
          <select value={form.task} onChange={(e) => set("task", e.target.value)}>
            <option value="detect">detect (boxes)</option>
            <option value="obb">obb (rotated boxes)</option>
            <option value="segment">segment (polygons)</option>
          </select>
        </label>
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <button className="run-button" style={{ width: "auto", padding: "8px 18px" }}
          onClick={submit} disabled={!canSubmit || job?.status === "processing"}>
          Export Dataset
        </button>
        <button className="chip-button" onClick={() => setAdvanced((a) => !a)}>
          {advanced ? "Use an annotated image from the library" : "Advanced: manual path + JSON"}
        </button>
      </div>
      {error && <p className="note" style={{ color: "var(--error)", marginTop: 8 }}>⚠ {error}</p>}
      {job && (
        <div className="note" style={{ marginTop: 10 }}>
          {job.status === "completed" ? (
            <span style={{ color: "var(--ready)" }}>
              ✓ Done — {job.info?.stats?.train_images} train / {job.info?.stats?.val_images} val tiles
              → <code>{job.data_yaml}</code>
            </span>
          ) : job.status === "error" ? (
            <span style={{ color: "var(--error)" }}>⚠ {job.error}</span>
          ) : (
            <span>{job.status} — {job.progress ?? 0}% {job.message ? `(${job.message})` : ""}</span>
          )}
        </div>
      )}
    </div>
  );
}

function TrainStep() {
  const [form, setForm] = useState({
    data_yaml: "", device: "cpu", model_size: "nano",
    epochs: "", batch: "", workers: "", imgsz: "",
  });
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const [modelName, setModelName] = useState("");
  const [registerState, setRegisterState] = useState(null); // null | "saving" | {model_id,...} | {error}

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  useEffect(() => () => clearInterval(pollRef.current), []);

  async function submit() {
    setError(null);
    setJob(null);
    setRegisterState(null);
    try {
      const payload = {
        data_yaml: form.data_yaml,
        device: form.device,
        model_size: form.model_size,
        epochs: form.epochs ? parseInt(form.epochs, 10) : null,
        batch: form.batch ? parseInt(form.batch, 10) : null,
        workers: form.workers ? parseInt(form.workers, 10) : null,
        imgsz: form.imgsz ? parseInt(form.imgsz, 10) : null,
      };
      const r = await api.startTraining(payload);
      setJobId(r.job_id);
      setJob({ status: "queued" });
      clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const st = await api.trainingStatus(r.job_id);
          setJob(st);
          if (["completed", "done", "canceled", "error"].includes(st.status)) clearInterval(pollRef.current);
        } catch { /* transient poll error */ }
      }, 1500);
    } catch (err) {
      setError(err.detail?.message || err.message);
    }
  }

  async function cancel() {
    if (jobId) await api.trainingCancel(jobId).catch(() => {});
  }

  async function registerModel() {
    if (!jobId) return;
    setRegisterState("saving");
    try {
      const entry = await api.registerTrainedModel(jobId, { name: modelName || "Trained model" });
      setRegisterState(entry);
    } catch (err) {
      setRegisterState({ error: err.detail?.message || err.message });
    }
  }

  const p = job?.progress;
  const running = job && ["queued", "processing"].includes(job.status);

  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">Train</p>
      <p className="note" style={{ maxWidth: 640 }}>
        Trains a YOLO model on the dataset above, in a separate process so the app never freezes —
        works on CPU, faster with a GPU. Batch size, worker count and epochs are auto-suggested from
        the dataset size and device; override only what you need.
      </p>
      <div className="prepare-grid">
        <label className="prepare-wide">
          data.yaml path
          <input value={form.data_yaml} onChange={(e) => set("data_yaml", e.target.value)} placeholder="/path/to/dataset/data.yaml" />
        </label>
        <label>
          Device
          <select value={form.device} onChange={(e) => set("device", e.target.value)}>
            <option value="cpu">CPU</option>
            <option value="cuda:0">GPU (cuda:0)</option>
          </select>
        </label>
        <label>
          Model size
          <select value={form.model_size} onChange={(e) => set("model_size", e.target.value)}>
            <option value="nano">nano (fastest, CPU-friendly)</option>
            <option value="small">small</option>
            <option value="medium">medium</option>
          </select>
        </label>
        <label>
          Epochs (auto if blank)
          <input type="number" value={form.epochs} onChange={(e) => set("epochs", e.target.value)} />
        </label>
        <label>
          Batch (auto if blank)
          <input type="number" value={form.batch} onChange={(e) => set("batch", e.target.value)} />
        </label>
        <label>
          Workers (auto if blank)
          <input type="number" value={form.workers} onChange={(e) => set("workers", e.target.value)} />
        </label>
        <label>
          Image size (auto if blank)
          <input type="number" value={form.imgsz} onChange={(e) => set("imgsz", e.target.value)} />
        </label>
      </div>
      <div style={{ display: "flex", gap: 10 }}>
        <button className="run-button" style={{ width: "auto", padding: "8px 18px" }}
          onClick={submit} disabled={!form.data_yaml || running}>
          Start Training
        </button>
        {running && (
          <button className="chip-button" onClick={cancel}>Cancel</button>
        )}
      </div>
      {error && <p className="note" style={{ color: "var(--error)", marginTop: 8 }}>⚠ {error}</p>}
      {job && (
        <div className="note" style={{ marginTop: 10 }}>
          {job.status === "error" ? (
            <span style={{ color: "var(--error)" }}>⚠ {job.error || job.result?.message}</span>
          ) : job.status === "canceled" ? (
            <span style={{ color: "var(--warn)" }}>Canceled at epoch {p?.epoch ?? "?"}.</span>
          ) : job.status === "done" || job.status === "completed" ? (
            <>
              <span style={{ color: "var(--ready)" }}>
                ✓ Done — best weights at <code>{job.result?.best}</code>
              </span>
              {job.result?.best && (
                <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <input
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="Name this model (e.g. Palm Trees — Site A)"
                    style={{ minWidth: 240 }}
                    disabled={registerState?.model_id != null}
                  />
                  <button
                    className="chip-button"
                    onClick={registerModel}
                    disabled={registerState === "saving" || !modelName.trim() || registerState?.model_id != null}
                  >
                    {registerState?.model_id ? "Added to Detect ✓" : "Add to Detect models"}
                  </button>
                </div>
              )}
              {registerState?.model_id && (
                <p className="note" style={{ color: "var(--ready)", marginTop: 6 }}>
                  ✓ Registered as <code>{registerState.model_id}</code> — classes: {Object.values(registerState.class_names || {}).join(", ")}.
                  It's now selectable on the Detect step or the full map, marked "validation" (real weights, not yet benchmarked) until you check its results.
                </p>
              )}
              {registerState?.error && (
                <p className="note" style={{ color: "var(--error)", marginTop: 6 }}>⚠ {registerState.error}</p>
              )}
            </>
          ) : p ? (
            <>
              <span>Epoch {p.epoch}/{p.epochs} — loss {p.box_loss?.toFixed?.(3)} · precision {p.precision?.toFixed?.(2)} ·
                recall {p.recall?.toFixed?.(2)} · mAP50 {p.map50?.toFixed?.(2)}</span>
              <div className="upload-progress-bar" style={{ marginTop: 6 }}>
                <div className="upload-progress-fill" style={{ width: `${(100 * p.epoch) / p.epochs}%` }} />
              </div>
            </>
          ) : (
            <span>{job.status}…</span>
          )}
        </div>
      )}
    </div>
  );
}

function AnnotateStep({ onViewChange }) {
  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">Annotate</p>
      <p className="note" style={{ maxWidth: 640 }}>
        Object annotation happens directly on the map: use the drawing tools (box, circle, polygon)
        on an uploaded image to mark the objects you want the model to learn.
      </p>
      <button className="run-button" style={{ width: "auto", marginTop: 12, padding: "8px 18px" }}
        onClick={() => onViewChange?.("detect")}>
        Open full map to annotate
      </button>
    </div>
  );
}

function DetectStep({ onViewChange }) {
  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">Detect</p>
      <p className="note" style={{ maxWidth: 640 }}>
        Detection runs on the full map: pick an uploaded image and a model, then Run prediction from
        the sidebar. A model you trained in the Train step and added with "Add to Detect models"
        shows up in that same model list, marked "validation" (real weights, no benchmark run yet)
        until you've checked its results — it is not silently promoted to "ready" on its own.
      </p>
      <button className="run-button" style={{ width: "auto", marginTop: 12, padding: "8px 18px" }}
        onClick={() => onViewChange?.("detect")}>
        Open full map to detect
      </button>
    </div>
  );
}

export default function ModelSetupWizard({ onViewChange, initialStep = null }) {
  const start = Math.max(0, STEPS.findIndex((s) => s.key === initialStep));
  const [current, setCurrent] = useState(start);
  const [reached, setReached] = useState(start);

  function go(i) {
    setCurrent(i);
    setReached((r) => Math.max(r, i));
  }

  return (
    <div className="wizard-shell">
      <WizardStepBar current={current} reached={reached} onGo={go} />
      <div className="wizard-body">
        {current === 0 && <ModelsDashboard />}
        {current === 1 && <AnnotateStep onViewChange={onViewChange} />}
        {current === 2 && <DatasetStep />}
        {current === 3 && <TrainStep />}
        {current === 4 && <DetectStep onViewChange={onViewChange} />}
      </div>
      <div className="wizard-nav">
        <button className="chip-button" disabled={current === 0} onClick={() => go(current - 1)}>‹ Back</button>
        <span className="note">Step {current + 1} of {STEPS.length} · {STEPS[current].title}</span>
        <button className="chip-button" disabled={current === STEPS.length - 1} onClick={() => go(current + 1)}>Next ›</button>
      </div>
    </div>
  );
}
