import React, { useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import LabelImport from "./LabelImport.jsx";
import ClassLibrary from "./ClassLibrary.jsx";
import { classColor, isArea, TRAIN_AREA, CHECK_AREA, AREA_LABEL, fmtArea } from "./AnnotationTool.jsx";
import { IconDatabase, IconPlus, IconGridDots, IconBoxSelect, IconLasso, IconBoxAI, IconCheck, IconClickAI, IconCircle, IconPencil, IconPolygon, IconRectangle, IconSquare, IconTrash } from "../ui/Icons.jsx";

// The Label step of the Train panel: classes (keys 1-9), the tools (one
// letter each) and what the active tool needs. The map layer that does the
// drawing is AnnotationTool.jsx.

export const LABEL_TOOLS = [
  { id: "click", group: "AI", label: "Click object", key: "A", Icon: IconClickAI },
  { id: "box", group: "AI", label: "Box around object", key: "B", Icon: IconBoxAI },
  { id: "everything", group: "AI", label: "Everything in view", key: "E", Icon: IconGridDots },
  { id: "boxsel", group: "Select", label: "Box select", key: "X", Icon: IconBoxSelect },
  { id: "lasso", group: "Select", label: "Lasso select", key: "L", Icon: IconLasso },
  { id: "polygon", group: "Draw", label: "Polygon", key: "P", Icon: IconPolygon },
  { id: "rect", group: "Draw", label: "Rectangle", key: "R", Icon: IconRectangle },
  { id: "square", group: "Draw", label: "Square", key: "S", Icon: IconSquare },
  { id: "circle", group: "Draw", label: "Circle", key: "C", Icon: IconCircle },
  { id: "freehand", group: "Draw", label: "Freehand", key: "F", Icon: IconPencil },
  { id: "curve", group: "Draw", label: "Curve", key: "U", Icon: IconPolygon },
  { id: "trace", group: "Draw", label: "Trace", key: "T", Icon: IconPolygon },
  { id: "split", group: "Edit", label: "Split", key: "K", Icon: IconPolygon },
  { id: "reshape", group: "Edit", label: "Reshape with a line", key: "H", Icon: IconPolygon },
  { id: "cut", group: "Edit", label: "Cut", key: "Q", Icon: IconPolygon },
  { id: "ellipse", group: "Draw", label: "Ellipse", key: "O", Icon: IconPolygon },
  { id: "similar", group: "AI", label: "Similar to selected", key: "G", Icon: IconPolygon },
];

// What is being outlined: how SAM's answer is chosen and cleaned, and the output it suits
export const TARGETS = [
  ["auto", "Auto (by class)", { shape: "auto", pick: "large", simplify_m: 0.6 }],
  ["building", "Building boundaries", { shape: "right_angles", pick: "large", simplify_m: 0.5 }],
  ["tree", "Tree canopies", { shape: "smooth", pick: "best", simplify_m: 0.3 }],
  ["natural", "Natural features", { shape: "smooth", pick: "large", simplify_m: 1.0 }],
  ["object", "Object outlines", { shape: "polygon", pick: "best", simplify_m: 0.3 }],
];
const SHAPES = [
  ["auto", "By class"], ["polygon", "Polygon outline"], ["right_angles", "Right angles"], ["smooth", "Smooth"],
  ["rectangle", "Oriented box"], ["square", "Square box"], ["box", "Box"], ["circle", "Circle"],
];

export const HINTS = {
  polygon: "Click the corners. Double-click or Enter closes it, Backspace removes the last corner",
  rect: "Drag corner to corner, Shift for a square. Or click twice along one edge, then a third time for its width",
  square: "Drag from a corner. Alt draws from the centre",
  circle: "Drag from the centre outwards",
  freehand: "Press and draw around the object",
  click: "Click an object. Click the next one to keep it. Shift+click adds a part, Alt+click removes one. Enter keeps, Del discards",
  everything: "Finds every object in the view. Enter keeps them all, Esc drops them. Pick it again after moving the map",
  boxsel: "Drag a box over labels to select them. Shift adds to the selection",
  lasso: "Draw around labels to select them. Shift adds to the selection",
  curve: "Click points: the outline bends smoothly through them. Double-click or Enter closes it",
  trace: "Click on another label's edge, then further along it: the outline follows that edge. Ctrl+click places a free point",
  split: "Draw a line across a label and double-click: it becomes two labels",
  reshape: "Draw a line that crosses the outline twice and double-click: outside adds area, inside takes it away",
  cut: "Click out the area to remove and double-click: it is taken out of the labels it covers",
  ellipse: "Drag the ellipse's box. Shift keeps it round",
  similar: "More like the selected labels in the view: Enter keeps them all, Esc drops them",
  box: "Drag a box around the object. Alt+click or right-click removes a part. Enter saves, Esc discards",
};

function isTyping(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
}

function num(v) {
  const n = parseFloat(String(v).replace(",", "."));
  return Number.isFinite(n) && n > 0 ? n : null;
}

function ExactSize({ mode, exact, onExact, gsd }) {
  const fields = mode === "circle" ? [["r", "Radius"]] : mode === "square" ? [["s", "Side"]] : [["w", "Width"], ["h", "Height"]];
  const [vals, setVals] = useState({});
  const [unit, setUnit] = useState("m");             // "px": the image's pixels, sent in metres (× GSD)
  useEffect(() => { if (!exact) setVals({}); }, [exact, mode]);
  function emit(next, u) {
    const k = u === "px" && gsd ? gsd : 1;
    const parsed = Object.fromEntries(fields.map(([f]) => [f, num(next[f]) && num(next[f]) * k]));
    onExact(fields.every(([f]) => parsed[f]) ? parsed : null);
  }
  function set(k, v) {
    const next = { ...vals, [k]: v };
    setVals(next);
    emit(next, unit);
  }
  return (
    <div className="label-exact">
      <span className="train-dim">Size</span>
      {fields.map(([k, l]) => (
        <label key={k} className="label-exact-field">
          <span className="sr-only">{l}</span>
          <input className="train-input mono" inputMode="decimal" placeholder={l} value={vals[k] || ""}
                 onChange={(e) => set(k, e.target.value)} aria-label={`${l} in ${unit === "px" ? "pixels" : "metres"}`} />
        </label>
      ))}
      <select className="train-select label-unit" value={unit} aria-label="Unit"
              onChange={(e) => { setUnit(e.target.value); emit(vals, e.target.value); }} disabled={!gsd}>
        <option value="m">m</option><option value="px">px</option>
      </select>
    </div>
  );
}

function ClickTool({ outline, onOutline, preview, onAutoAll }) {
  const [st, setSt] = useState(null);
  const [job, setJob] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    api.segmentStatus().then(setSt).catch((e) => setSt({ ready: false, reason: e.message }));
  }, []);
  useEffect(() => () => clearInterval(timer.current), []);

  async function install(model) {
    try {
      const r = await api.segmentInstall(typeof model === "string" ? model : undefined);
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
  const missing = !st.ready || /install/i.test(preview?.error || "");
  if (missing) {
    const running = job && !["completed", "error", "interrupted"].includes(job.status);
    return (
      <div className="label-ctx">
        <p className="train-dim">{st.reason || preview?.error}</p>
        <button type="button" className="quiet-button" onClick={install} disabled={running}>
          {running ? `Downloading ${Math.round(job.progress || 0)}%` : "Install click tool"}
        </button>
        {job?.status === "error" && <p className="train-error">{job.error}</p>}
      </div>
    );
  }
  const res = preview?.res;
  return (
    <div className="ai">
      <div className="ai-grid">
        <label className="ai-field"><span>Object</span>
          <select className="train-select" value={outline.target || "auto"}
                  onChange={(e) => { const t = TARGETS.find(([id]) => id === e.target.value); onOutline({ ...outline, target: t[0], ...t[2] }); }}>
            {TARGETS.map(([id, l]) => <option key={id} value={id}>{l}</option>)}
          </select>
        </label>
        <label className="ai-field"><span>Output</span>
          <select className="train-select" value={outline.shape} onChange={(e) => onOutline({ ...outline, shape: e.target.value })}>
            {SHAPES.map(([id, l]) => <option key={id} value={id}>{l}</option>)}
          </select>
        </label>
        <label className="ai-field"><span>Detail</span>
          <span className="ai-num"><input className="mono" inputMode="decimal" value={outline.simplify_m} aria-label="Detail in metres"
                 onChange={(e) => onOutline({ ...outline, simplify_m: parseFloat(e.target.value) || 0 })} /><i>m</i></span>
        </label>
        <label className="ai-field"><span>Grow</span>
          <span className="ai-num"><input className="mono" inputMode="decimal" value={outline.grow_m} aria-label="Grow in metres"
                 onChange={(e) => onOutline({ ...outline, grow_m: parseFloat(e.target.value) || 0 })} /><i>m</i></span>
        </label>
      </div>
      <button type="button" className="ai-auto" onClick={() => onAutoAll?.()}>
        <IconGridDots /> Auto segment the view <kbd className="mono">E</kbd>
      </button>
      {preview?.busy && <p className="ai-status mono">Outlining…</p>}
      {preview?.error && <p className="train-error">{preview.error}</p>}
      {res && (
        <div className="ai-res">
          <div className="ai-res-head">
            <b>{SHAPES.find(([id]) => id === res.shape)?.[1] || (res.shape === "polygon" ? "Polygon outline" : res.shape)}</b>
            <span className="mono">{res.area_m2 != null ? fmtArea(res.area_m2) : ""} · {res.vertices} pts · {Math.round(res.timings?.decode_ms)} ms</span>
          </div>
          {res.warning && <p className="ai-warn" title={res.warning}>Kept the traced outline</p>}
          <div className="ai-res-acts">
            <button type="button" onClick={() => window.dispatchEvent(new CustomEvent("syntra:preview", { detail: "keep" }))}>
              <IconCheck /> Keep <kbd className="mono">Enter</kbd></button>
            <button type="button" className="danger" onClick={() => window.dispatchEvent(new CustomEvent("syntra:preview", { detail: "discard" }))}>
              <IconTrash /> Discard <kbd className="mono">Del</kbd></button>
          </div>
        </div>
      )}
      {st.models?.["sam2.1_small"] && !st.models["sam2.1_small"].installed && (
        job && !["completed", "error", "interrupted"].includes(job.status)
          ? <p className="ai-status mono">SAM 2.1 small · {Math.round(job.progress || 0)}%</p>
          : <button type="button" className="ai-more" onClick={() => install("sam2.1_small")} title="SAM 2.1 small · 150 MB">More accurate model</button>
      )}
    </div>
  );
}

export default function LabelTools({
  open, classes, counts, activeClass, onActiveClass, mode, onMode, annotations, onClearAnnotations, labelError,
  outline, onOutline, exact, onExact, segPreview, editId, onEditDone, onEditClass, onEditDelete, image, onImported, onAddClass,
  onAddClasses, onDeleteClass,
}) {
  const [libOpen, setLibOpen] = useState(false);
  const [newClass, setNewClass] = useState("");
  function addClass(e) {
    e.preventDefault();
    const name = newClass.trim().toLowerCase().replace(/\s+/g, "_");
    if (!name) return;
    onAddClass?.(name);
    setNewClass("");
    e.currentTarget.querySelector("input")?.blur();
  }
  const drawingArea = isArea(activeClass || "");
  const objects = annotations.filter((a) => !isArea(a.class_name));

  function pickTool(id) {
    if (!classes.length && !["boxsel", "lasso", "split", "reshape", "cut"].includes(id)) onAddClass?.("object");   // the tool starts at once; rename or reclass later
    else if (drawingArea) onActiveClass(classes[0]);
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
      if (t) { e.preventDefault(); pick(t.id); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const editing = editId && annotations.find((a) => a.id === editId);
  return (
    <>
      <p className="train-label lt-head">Classes
        {onAddClasses && (
          <button type="button" className="lp-icon" onClick={() => setLibOpen(true)} aria-label="Class library" title="Class library"><IconDatabase /></button>
        )}
      </p>
      <ul className="train-classes train-classes-pick">
        {classes.map((c, i) => (
          <li key={c} className="class-row">
            <button type="button" className={c === (editing ? editing.class_name : activeClass) ? "selected" : ""}
                    onClick={() => { if (editing) { onEditClass?.(c); return; } onActiveClass(c); if (drawingArea) onMode(null); }}>
              {i < 9 && <kbd className="label-key">{i + 1}</kbd>}
              <span className="train-swatch" style={{ background: classColor(c) }} />{c.replace(/_/g, " ")}
              <span className="mono train-dim">{counts[c] || 0}</span>
            </button>
            {onDeleteClass && (
              <button type="button" className="class-del" onClick={() => onDeleteClass(c)} aria-label={`Delete class ${c}`} title="Delete class"><IconTrash /></button>
            )}
          </li>
        ))}
      </ul>

      {onAddClass && (
        <form className="lt-add" onSubmit={addClass}>
          <IconPlus aria-hidden="true" />
          <input value={newClass} onChange={(e) => setNewClass(e.target.value)} placeholder="New class" aria-label="New class" />
        </form>
      )}
      <ClassLibrary open={libOpen} onClose={() => setLibOpen(false)} existing={classes} onAdd={(list) => onAddClasses?.(list)} />

      {["click", "box", "everything"].includes(mode) && !drawingArea && <ClickTool outline={outline} onOutline={onOutline} preview={segPreview} onAutoAll={() => onMode("everything")} />}
      {["rect", "square", "circle"].includes(mode) && !drawingArea && <ExactSize mode={mode} exact={exact} onExact={onExact} gsd={image?.health?.gsd_m} />}

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
      {editing ? (
        <div className="label-edit"
             title={`Drag corners or midpoints · Alt+click removes a corner · drag to move · round handle rotates${isArea(editing.class_name) ? "" : " · 1–9 class"} · Delete · Esc`}>
          <span className="train-swatch" style={{ background: classColor(editing.class_name) }} />
          <span className="label-edit-name">{AREA_LABEL[editing.class_name] || editing.class_name}</span>
          <span className="mono train-dim">{editing.points_lonlat.length} pts</span>
          <button type="button" className="label-btn danger" onClick={onEditDelete} aria-label="Delete label" title="Delete · Del"><IconTrash /></button>
          <button type="button" className="label-btn" onClick={onEditDone} aria-label="Done" title="Done · Esc"><IconCheck /></button>
        </div>
      ) : null}
      {labelError && <p className="train-error">{labelError}</p>}
      <div className="lt-foot">
        <span><span className="mono">{objects.length}</span> labelled</span>
        <LabelImport image={image} onImported={onImported} compact />
        <button type="button" className="lp-icon" onClick={onClearAnnotations} disabled={!annotations.length}
                aria-label="Clear this image's labels" title="Clear this image's labels"><IconTrash /></button>
      </div>
    </>
  );
}
