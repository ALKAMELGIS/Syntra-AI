import React, { useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import LabelImport from "./LabelImport.jsx";
import { classColor, isArea, TRAIN_AREA, CHECK_AREA, AREA_LABEL, fmtArea } from "./AnnotationTool.jsx";
import { IconBoxSelect, IconLasso, IconBoxAI, IconCheck, IconClickAI, IconCircle, IconPencil, IconPolygon, IconRectangle, IconSquare, IconTrash } from "../ui/Icons.jsx";

// The Label step of the Train panel: classes (keys 1-9), the tools (one
// letter each) and what the active tool needs. The map layer that does the
// drawing is AnnotationTool.jsx.

export const LABEL_TOOLS = [
  { id: "click", group: "AI", label: "Click object", key: "A", Icon: IconClickAI },
  { id: "box", group: "AI", label: "Box around object", key: "B", Icon: IconBoxAI },
  { id: "boxsel", group: "Select", label: "Box select", key: "X", Icon: IconBoxSelect },
  { id: "lasso", group: "Select", label: "Lasso select", key: "L", Icon: IconLasso },
  { id: "polygon", group: "Draw", label: "Polygon", key: "P", Icon: IconPolygon },
  { id: "rect", group: "Draw", label: "Rectangle", key: "R", Icon: IconRectangle },
  { id: "square", group: "Draw", label: "Square", key: "S", Icon: IconSquare },
  { id: "circle", group: "Draw", label: "Circle", key: "C", Icon: IconCircle },
  { id: "freehand", group: "Draw", label: "Freehand", key: "F", Icon: IconPencil },
];

const SHAPES = [
  ["auto", "By class"], ["polygon", "Traced"], ["right_angles", "Right angles"],
  ["box", "Box"], ["rectangle", "Rotated box"], ["circle", "Circle"], ["smooth", "Smooth"],
];

const HINTS = {
  polygon: "Click the corners. Double-click or Enter closes it, Backspace removes the last corner",
  rect: "Drag corner to corner, Shift for a square. Or click twice along one edge, then a third time for its width",
  square: "Drag from a corner. Alt draws from the centre",
  circle: "Drag from the centre outwards",
  freehand: "Press and draw around the object",
  click: "Click inside the object. Alt+click or right-click removes a part. Enter saves, Esc discards",
  boxsel: "Drag a box over labels to select them. Shift adds to the selection",
  lasso: "Draw around labels to select them. Shift adds to the selection",
  box: "Drag a box around the object. Alt+click or right-click removes a part. Enter saves, Esc discards",
};

function isTyping(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
}

function num(v) {
  const n = parseFloat(String(v).replace(",", "."));
  return Number.isFinite(n) && n > 0 ? n : null;
}

function ExactSize({ mode, exact, onExact }) {
  const fields = mode === "circle" ? [["r", "Radius"]] : mode === "square" ? [["s", "Side"]] : [["w", "Width"], ["h", "Height"]];
  const [vals, setVals] = useState({});
  useEffect(() => { if (!exact) setVals({}); }, [exact, mode]);
  function set(k, v) {
    const next = { ...vals, [k]: v };
    setVals(next);
    const parsed = Object.fromEntries(fields.map(([f]) => [f, num(next[f])]));
    onExact(fields.every(([f]) => parsed[f]) ? parsed : null);
  }
  return (
    <div className="label-exact">
      <span className="train-dim">Exact size</span>
      {fields.map(([k, l]) => (
        <label key={k} className="label-exact-field">
          <span className="sr-only">{l}</span>
          <input className="train-input mono" inputMode="decimal" placeholder={l} value={vals[k] || ""}
                 onChange={(e) => set(k, e.target.value)} aria-label={`${l} in metres`} />
          <span className="train-dim">m</span>
        </label>
      ))}
      {exact && <span className="train-dim">click the map to place it</span>}
    </div>
  );
}

function ClickTool({ outline, onOutline, preview }) {
  const [st, setSt] = useState(null);
  const [job, setJob] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    api.segmentStatus().then(setSt).catch((e) => setSt({ ready: false, reason: e.message }));
  }, []);
  useEffect(() => () => clearInterval(timer.current), []);

  async function install() {
    try {
      const r = await api.segmentInstall();
      setJob({ ...r, progress: 0 });
      clearInterval(timer.current);
      timer.current = setInterval(async () => {
        try {
          const j = await api.segmentInstallStatus(r.job_id);
          setJob(j);
          if (["completed", "error", "interrupted"].includes(j.status)) {
            clearInterval(timer.current);
            if (j.status === "completed") api.segmentStatus().then(setSt);
          }
        } catch { /* transient */ }
      }, 1000);
    } catch (e) { setJob({ status: "error", error: e.detail || e.message }); }
  }

  if (!st) return <p className="train-dim">Checking the click tool…</p>;
  if (!st.ready) {
    const running = job && !["completed", "error", "interrupted"].includes(job.status);
    return (
      <div className="label-ctx">
        <p className="train-dim">{st.reason}</p>
        <button type="button" className="quiet-button" onClick={install} disabled={running}>
          {running ? `Downloading ${Math.round(job.progress || 0)}%` : "Install click tool"}
        </button>
        {job?.status === "error" && <p className="train-error">{job.error}</p>}
      </div>
    );
  }
  const res = preview?.res;
  return (
    <div className="label-ctx">
      <label className="train-field">Outline
        <select className="train-select" value={outline.shape} onChange={(e) => onOutline({ ...outline, shape: e.target.value })}>
          {SHAPES.map(([id, l]) => <option key={id} value={id}>{l}</option>)}
        </select>
      </label>
      <div className="label-exact">
        <label className="label-exact-field">
          <span className="train-dim">Detail</span>
          <input className="train-input mono" inputMode="decimal" value={outline.simplify_m}
                 onChange={(e) => onOutline({ ...outline, simplify_m: parseFloat(e.target.value) || 0 })} aria-label="Simplify in metres" />
          <span className="train-dim">m</span>
        </label>
        <label className="label-exact-field">
          <span className="train-dim">Grow</span>
          <input className="train-input mono" inputMode="decimal" value={outline.grow_m}
                 onChange={(e) => onOutline({ ...outline, grow_m: parseFloat(e.target.value) || 0 })} aria-label="Grow or shrink in metres" />
          <span className="train-dim">m</span>
        </label>
      </div>
      {preview?.busy && <p className="train-dim">Outlining…</p>}
      {preview?.error && <p className="train-error">{preview.error}</p>}
      {res && (
        <div className="label-result">
          <span>{SHAPES.find(([id]) => id === res.shape)?.[1] || res.shape}</span>
          <span className="mono">{res.area_m2 != null ? fmtArea(res.area_m2) : ""} · {res.vertices} pts</span>
          {res.warning && <span className="train-dim label-result-wide">{res.warning}</span>}
          <span className="train-dim mono label-result-wide">
            {res.timings?.cached ? "view reused" : `view ${(res.timings?.encode_ms / 1000).toFixed(1)} s`} · click {Math.round(res.timings?.decode_ms)} ms
          </span>
        </div>
      )}
      <p className="train-dim">{st.label} on this computer</p>
    </div>
  );
}

export default function LabelTools({
  open, classes, counts, activeClass, onActiveClass, mode, onMode, annotations, onClearAnnotations, labelError,
  outline, onOutline, exact, onExact, segPreview, editId, onEditDone, onEditClass, onEditDelete, image, onImported,
}) {
  const drawingArea = isArea(activeClass || "");
  const objects = annotations.filter((a) => !isArea(a.class_name));

  function pickTool(id) {
    if (drawingArea) onActiveClass(classes[0]);
    onMode(mode === id && !drawingArea ? null : id);
  }

  // keys: 1-9 pick a class, a tool's letter toggles it
  const live = useRef();
  live.current = { classes, pickTool, onActiveClass, editId, onEditClass, onMode };
  useEffect(() => {
    if (!open) return;
    function onKey(e) {
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      const { classes: cls, pickTool: pick, onActiveClass: setCls, editId: editing, onEditClass: reclass } = live.current;
      if (/^[1-9]$/.test(e.key) && cls[+e.key - 1]) {
        e.preventDefault();
        if (editing) reclass?.(cls[+e.key - 1]); else setCls(cls[+e.key - 1]);   // while editing, the number changes the label's class
        return;
      }
      if (e.key.toUpperCase() === "V") { e.preventDefault(); live.current.onMode(null); return; }
      const t = LABEL_TOOLS.find((x) => x.key === e.key.toUpperCase());
      if (t && cls.length) { e.preventDefault(); pick(t.id); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const editing = editId && annotations.find((a) => a.id === editId);
  return (
    <>
      <p className="train-label">Class</p>
      <ul className="train-classes train-classes-pick">
        {classes.map((c, i) => (
          <li key={c}>
            <button type="button" className={c === (editing ? editing.class_name : activeClass) ? "selected" : ""}
                    onClick={() => { if (editing) { onEditClass?.(c); return; } onActiveClass(c); if (drawingArea) onMode(null); }}>
              {i < 9 && <kbd className="label-key">{i + 1}</kbd>}
              <span className="train-swatch" style={{ background: classColor(c) }} />{c}
              <span className="mono train-dim">{counts[c] || 0}</span>
            </button>
          </li>
        ))}
      </ul>

      <p className="train-label">Tool</p>
      {(() => {
        const t = !drawingArea && LABEL_TOOLS.find((x) => x.id === mode);
        return t ? (
          <p className="label-current"><t.Icon /><span>{t.label}</span><kbd className="label-key">{t.key}</kbd>
            <span className="train-dim label-current-hint">{HINTS[t.id]}</span></p>
        ) : <p className="train-dim">The tools are on the map's left edge. Point at one to see its group</p>;
      })()}
      {(mode === "click" || mode === "box") && !drawingArea && <ClickTool outline={outline} onOutline={onOutline} preview={segPreview} />}
      {["rect", "square", "circle"].includes(mode) && !drawingArea && <ExactSize mode={mode} exact={exact} onExact={onExact} />}

      <p className="train-label">Areas</p>
      <div className="label-areas">
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
      {drawingArea && mode && (
        <p className="train-dim">{activeClass === TRAIN_AREA ? "Label everything inside it" : "Label everything inside it · kept for scoring"}</p>
      )}
      <p className="train-label">Import</p>
      <LabelImport image={image} onImported={onImported} />
      {editing ? (
        <div className="label-edit"
             title={`Drag corners or midpoints · Alt+click removes a corner · drag to move · round handle rotates${isArea(editing.class_name) ? "" : " · 1–9 class"} · Delete · Esc`}>
          <span className="train-swatch" style={{ background: classColor(editing.class_name) }} />
          <span className="label-edit-name">{AREA_LABEL[editing.class_name] || editing.class_name}</span>
          <span className="mono train-dim">{editing.points_lonlat.length} pts</span>
          <button type="button" className="label-btn danger" onClick={onEditDelete} aria-label="Delete label" title="Delete · Del"><IconTrash /></button>
          <button type="button" className="label-btn" onClick={onEditDone} aria-label="Done" title="Done · Esc"><IconCheck /></button>
        </div>
      ) : (!mode && classes.length > 0 && <p className="train-dim">Pick a tool, or click a label to edit</p>)}
      {labelError && <p className="train-error">{labelError}</p>}
      <div className="train-row train-row-split">
        <span className="train-dim"><span className="mono">{objects.length}</span> labelled · aim for 50+ per class</span>
        {annotations.length > 0 && (
          <button type="button" className="link-button" onClick={onClearAnnotations}><IconTrash /> Clear</button>
        )}
      </div>
    </>
  );
}
