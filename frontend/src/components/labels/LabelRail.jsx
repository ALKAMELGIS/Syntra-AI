// The label tools as a rail on the map's left edge, shown while labelling
// (Train > Label). One button per group; the icons stay visible, and pointing
// at a group opens its tools beside the rail. A group's button shows the tool
// last used in it and clicking it picks that tool again. The keys are the same
// as in the Train panel (LabelTools.jsx handles them).
import React, { useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import { classColor, TRAIN_AREA, CHECK_AREA } from "./AnnotationTool.jsx";
import {
  IconAreaDashed, IconCheck, IconGridDots, IconBoxAI, IconBoxSelect, IconClose, IconLasso, IconCircle, IconClickAI, IconPencil, IconPointer, IconPolygon,
  IconRectangle, IconSparkle, IconSquare, IconTag, IconTrash, IconUndo,
} from "../ui/Icons.jsx";

const SELECT = "select";
export const DEFAULT_CLASS = "object";
const SHAPE_LABEL = { polygon: "Outline", right_angles: "Right angles", rectangle: "Oriented box", square: "Square box", box: "Box", circle: "Circle", smooth: "Smooth" };      // made when a tool is picked before any class exists
// what the AI tools draw: the Refine `shape` sent to the engine
const DRAWS = [["auto", "Auto"], ["polygon", "Outline"], ["rectangle", "OBB"], ["square", "SBB"]];

const GROUPS = [
  { id: "polygon", tools: [
    { id: "polygon", label: "Polygon", key: "P", Icon: IconPolygon },
    { id: "freehand", label: "Freehand", key: "F", Icon: IconPencil }] },
  { id: "rect", tools: [
    { id: "rect", label: "Rectangle", key: "R", Icon: IconRectangle, note: "Drag, or click 3 times for a rotated one" },
    { id: "square", label: "Square", key: "S", Icon: IconSquare }] },
  { id: "circle", tools: [{ id: "circle", label: "Circle", key: "C", Icon: IconCircle }] },
  { id: "class", title: "Class", Icon: IconTag },
  { sep: true },
  { id: "select", title: "Select", tools: [
    { id: SELECT, label: "Select and edit", key: "V", Icon: IconPointer, note: "Click a label to reshape, move or rotate it. Shift+click adds it to the selection" },
    { id: "boxsel", label: "Box select", key: "X", Icon: IconBoxSelect, note: "Drag over labels. Shift adds" },
    { id: "lasso", label: "Lasso select", key: "L", Icon: IconLasso, note: "Draw around labels. Shift adds" },
    { id: "allclass", label: "All of this class", key: "Ctrl+A", Icon: IconTag, action: "all" }] },
  { sep: true },
  { id: "ai", title: "Auto segment", Icon: IconSparkle, tools: [
    { id: "click", label: "Click object", key: "A", Icon: IconClickAI, note: "Alt+click removes a part" },
    { id: "box", label: "Box around object", key: "B", Icon: IconBoxAI, note: "Drag a box around it" },
    { id: "everything", label: "Everything in view", key: "E", Icon: IconGridDots, note: "Every object in the view at once" }], draws: true },
  { id: "areas", title: "Areas", Icon: IconAreaDashed, tools: [
    { id: TRAIN_AREA, label: "Training area", area: true, Icon: IconAreaDashed, note: "Label everything inside it" },
    { id: CHECK_AREA, label: "Accuracy area", area: true, Icon: IconAreaDashed, note: "Kept out of training, for scoring" }] },
  { sep: true },
  { id: "undo", title: "Undo", key: "Ctrl+Z", Icon: IconUndo, action: "undo" },
  { id: "delete", title: "Delete", Icon: IconTrash, tools: [
    { id: "delete", label: "Delete selected", key: "Del", Icon: IconTrash, action: "delete" },
    { id: "clear", label: "Clear this image's labels", Icon: IconTrash, action: "clear", danger: true }] },
];

function isTyping(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
}

export default function LabelRail({
  classes = [], counts = {}, activeClass, onActiveClass, mode, onMode, editId, onEditClass, onEditDelete,
  onClear, onUndo, canUndo, outline, onOutline, editingClass, onAddClass, segPreview,
  annotations = [], selectedIds = [], onSelect, onDeleteMany, onReclassMany, recent, onDeleteRecent,
}) {
  const [open, setOpen] = useState(null);
  const [last, setLast] = useState({});            // group -> tool id last picked there
  const closeT = useRef(null);
  const drawingArea = activeClass === TRAIN_AREA || activeClass === CHECK_AREA;
  const [newClass, setNewClass] = useState("");

  // the AI tools need the engine's SAM model; say so in the group instead of failing on the map
  const [seg, setSeg] = useState(null);           // {ready, reason} | {down}
  const [install, setInstall] = useState(null);
  const poll = useRef(null);
  const checkSeg = () => Promise.all([
    fetch("/api/health").then((r) => r.json()).catch(() => null),
    api.segmentStatus().catch((e) => ({ ready: false, reason: e.message })),
  ]).then(([h, st]) => setSeg(!h ? { down: "The engine isn't answering. Run start-local.bat" }
    : (h.api || 0) < 4 ? { down: "The engine is out of date: close the Syntra engine window and run update.bat" } : st));
  useEffect(() => { checkSeg(); return () => clearInterval(poll.current); }, []);
  async function installSam() {
    try {
      const r = await api.segmentInstall();
      setInstall({ progress: 0 });
      clearInterval(poll.current);
      poll.current = setInterval(async () => {
        try {
          const j = await api.segmentInstallStatus(r.job_id);
          setInstall(j);
          if (["completed", "error", "interrupted"].includes(j.status)) { clearInterval(poll.current); checkSeg(); }
        } catch { /* transient */ }
      }, 1000);
    } catch (e) { setInstall({ status: "error", error: e.detail || e.message }); }
  }
  function addClass(e) {
    e.preventDefault();
    const c = newClass.trim().toLowerCase();
    if (!c) return;
    onAddClass?.(c);
    setNewClass("");
    e.currentTarget.querySelector("input")?.blur();   // the tool keys (P, R, B…) work straight away
  }

  // Ctrl+Z: the last label added here; Ctrl+A: every label of the current class
  const selectAll = () => onSelect?.(annotations.filter((a) => a.class_name === activeClass).map((a) => a.id));
  const live = useRef();
  live.current = { onUndo, canUndo, selectAll };
  useEffect(() => {
    function key(e) {
      if (!(e.ctrlKey || e.metaKey) || e.shiftKey || isTyping(e.target)) return;
      const k = e.key.toLowerCase();
      if (k === "z" && live.current.canUndo) { e.preventDefault(); live.current.onUndo?.(); }
      if (k === "a") { e.preventDefault(); live.current.selectAll(); }
    }
    window.addEventListener("keydown", key);
    return () => { window.removeEventListener("keydown", key); clearTimeout(closeT.current); };
  }, []);

  const enter = (id) => { clearTimeout(closeT.current); setOpen(id); };
  const leave = () => { clearTimeout(closeT.current); closeT.current = setTimeout(() => setOpen(null), 160); };

  const isOn = (t) => {
    if (t.area) return activeClass === t.id && !!mode;
    if (t.id === SELECT) return !mode;
    return !t.action && mode === t.id && !drawingArea;
  };
  function pick(g, t) {
    setLast((l) => ({ ...l, [g.id]: t.id }));
    if (t.action === "all") { onMode(null); selectAll(); return; }
    if (t.action === "delete") {
      if (segPreview?.res) { preview("discard"); return; }
      if (selectedIds.length) onDeleteMany?.(selectedIds); else if (editId) onEditDelete?.();
      return;
    }
    if (t.action === "clear") { onClear?.(); return; }
    if (t.area) { onActiveClass(t.id); onMode("rect"); return; }
    const draws = !["select"].includes(g.id);
    if (draws && !classes.length) onAddClass?.(DEFAULT_CLASS);   // every tool starts at once: rename or reclass later
    else if (draws && drawingArea) onActiveClass(classes[0]);
    if (t.id === "everything" && mode === "everything") { window.dispatchEvent(new CustomEvent("syntra:suggestions", { detail: "again" })); return; }
    onMode(t.id === SELECT ? null : t.id);
  }
  function main(g) {
    if (g.action === "undo") { onUndo?.(); return; }
    if (!g.tools) { setOpen(open === g.id ? null : g.id); return; }
    const t = g.tools.find((x) => x.id === last[g.id]) || g.tools[0];
    if (t.action) { setOpen(g.id); return; }       // Delete, All of class: choose in the list
    pick(g, t);
  }
  const shown = (g) => (g.tools ? g.tools.find((x) => x.id === last[g.id]) || g.tools[0] : null);


  const n = selectedIds.length;
  const all = mode === "everything" && segPreview?.all ? segPreview : null;
  const preview = (what) => window.dispatchEvent(new CustomEvent("syntra:preview", { detail: what }));
  const ask = (what) => window.dispatchEvent(new CustomEvent("syntra:suggestions", { detail: what }));
  return (
    <>
    {all && (
      <div className="label-selbar" role="status">
        {all.busy ? <span>Finding every object in the view…</span>
          : all.error ? <span className="label-rail-err" style={{ margin: 0 }}>{all.error}</span>
          : all.dropped ? <span>Suggestions dropped</span>
          : <><span className="mono">{all.count}</span><span>{all.count === 1 ? "object found" : "objects found"}</span></>}
        {!all.busy && all.count > 0 && !all.dropped && (
          <button type="button" className="label-selbar-btn" onClick={() => ask("keep")} title="Enter">Keep all</button>
        )}
        {!all.busy && all.count > 0 && !all.dropped && (
          <button type="button" className="label-selbar-btn" onClick={() => ask("drop")} title="Esc">Drop</button>
        )}
        {!all.busy && <button type="button" className="label-selbar-btn" onClick={() => ask("again")}>Run here again</button>}
      </div>
    )}
    {(mode === "click" || mode === "box") && segPreview?.res && (
      <div className="label-selbar" role="status">
        <span>{SHAPE_LABEL[segPreview.res.shape] || segPreview.res.shape}</span>
        <button type="button" className="label-selbar-btn" onClick={() => preview("keep")} title="Enter"><IconCheck /> Keep <kbd className="mono">Enter</kbd></button>
        <button type="button" className="label-selbar-btn danger" onClick={() => preview("discard")} title="Del"><IconTrash /> Discard <kbd className="mono">Del</kbd></button>
      </div>
    )}
    {recent && !all && n === 0 && !segPreview?.res && (
      <div className="label-selbar" role="status">
        <span className="label-saved-dot" aria-hidden="true" /><span>Saved</span>
        <button type="button" className="label-selbar-btn" onClick={() => onUndo?.()} title="Ctrl+Z"><IconUndo /> Undo <kbd className="mono">Ctrl+Z</kbd></button>
        <button type="button" className="label-selbar-btn danger" onClick={() => onDeleteRecent?.()} title="Delete"><IconTrash /> Delete</button>
      </div>
    )}
    {editId && !all && n === 0 && !recent && (
      <div className="label-selbar" role="status">
        <span>{(editingClass || "").replace(/_/g, " ")}</span>
        <button type="button" className="label-selbar-btn danger" onClick={() => onEditDelete?.()} title="Del"><IconTrash /> Delete <kbd className="mono">Del</kbd></button>
        {canUndo && <button type="button" className="label-selbar-btn" onClick={() => onUndo?.()} title="Ctrl+Z"><IconUndo /> Undo</button>}
      </div>
    )}
    {n > 0 && (
      <div className="label-selbar" role="status">
        <span className="mono">{n}</span><span>selected</span>
        <select className="label-selbar-class" value="" aria-label="Change the class of the selected labels"
                onChange={(e) => e.target.value && onReclassMany?.(selectedIds, e.target.value)}>
          <option value="">Change class</option>
          {classes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <button type="button" className="label-selbar-btn danger" onClick={() => onDeleteMany?.(selectedIds)} title="Delete · Del">
          <IconTrash /> Delete
        </button>
        <button type="button" className="label-selbar-btn" onClick={() => onSelect?.([])} aria-label="Clear the selection" title="Esc"><IconClose /></button>
      </div>
    )}
    <div className="label-rail" role="toolbar" aria-label="Label tools" aria-orientation="vertical"
         onMouseDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
      {GROUPS.map((g, i) => {
        if (g.sep) return <span key={`s${i}`} className="label-rail-sep" aria-hidden="true" />;
        const t = shown(g);
        const Icon = g.id === "select" ? t.Icon : g.Icon || t.Icon;
        const on = g.tools ? g.tools.some(isOn) : false;
        const title = g.title || t.label;
        const many = (g.tools?.length || 0) > 1 || g.id === "class" || g.draws;
        const off = g.action === "undo" && !canUndo;
        return (
          <div key={g.id} className="label-rail-item" onMouseEnter={() => enter(g.id)} onMouseLeave={leave}>
            <button type="button" className={`label-rail-btn ${on ? "on" : ""} ${many ? "more" : ""}`}
                    aria-label={title} aria-pressed={on} aria-haspopup={many ? "true" : undefined} aria-expanded={many ? open === g.id : undefined}
                    disabled={off} onClick={() => main(g)} onFocus={() => enter(g.id)}>
              {g.id === "class" && activeClass && !drawingArea
                ? <span className="label-rail-swatch" style={{ background: classColor(editingClass || activeClass) }} />
                : <Icon />}
            </button>
            {open === g.id && (
              <div className="label-rail-fly" role="menu" aria-label={title} onMouseEnter={() => enter(g.id)} onMouseLeave={leave}>
                <p className="label-rail-title">{title}{g.key && <kbd className="mono">{g.key}</kbd>}</p>
                {g.id === "class" && (classes.length ? classes.map((c, k) => (
                  <button key={c} type="button" role="menuitemradio" aria-checked={c === (editingClass || activeClass)}
                          className={`label-rail-row ${c === (editingClass || activeClass) ? "on" : ""}`}
                          onClick={() => (editId ? onEditClass?.(c) : onActiveClass(c))}>
                    <span className="label-rail-swatch" style={{ background: classColor(c) }} />
                    <span className="label-rail-name">{c}</span>
                    <span className="mono label-rail-key">{k < 9 ? k + 1 : ""}</span>
                  </button>
                )) : null)}
                {g.id === "class" && onAddClass && (
                  <form className="label-rail-add" onSubmit={addClass}>
                    <input className="train-input" value={newClass} onChange={(e) => setNewClass(e.target.value)}
                           placeholder="New class: palm, car…" aria-label="New class" />
                  </form>
                )}
                {g.tools?.map((x) => (
                  <button key={x.id} type="button" role="menuitem" disabled={x.action === "delete" && !editId && !selectedIds.length && !segPreview?.res}
                          className={`label-rail-row ${isOn(x) ? "on" : ""} ${x.danger ? "danger" : ""}`}
                          onClick={() => pick(g, x)}
                          title={x.note || undefined}>
                    {x.area ? <span className="label-rail-swatch dash" style={{ borderColor: classColor(x.id) }} /> : <x.Icon />}
                    <span className="label-rail-name">{x.label}</span>
                    {x.area ? <span className="mono label-rail-key">{counts[x.id] || 0}</span>
                      : x.key && <span className="mono label-rail-key">{x.key}</span>}
                  </button>
                ))}
                {g.draws && seg && !seg.ready && (
                  <div className="label-rail-warn">
                    <p>{seg.down || seg.reason}</p>
                    {!seg.down && (install && !["completed", "error", "interrupted"].includes(install.status)
                      ? <p className="mono">Downloading {Math.round(install.progress || 0)}%</p>
                      : <button type="button" className="quiet-button" onClick={installSam}>Install</button>)}
                    {install?.status === "error" && <p className="label-rail-err">{install.error}</p>}
                  </div>
                )}
                {g.draws && segPreview?.error && <p className="label-rail-err">{segPreview.error}</p>}
                {g.draws && (
                  <>
                    <p className="label-rail-title sub">Draws</p>
                    <div className="label-rail-seg" role="radiogroup" aria-label="What the AI tools draw">
                      {DRAWS.map(([id, l]) => (
                        <button key={id} type="button" role="radio" aria-checked={outline?.shape === id}
                                className={outline?.shape === id ? "on" : ""} onClick={() => onOutline?.({ ...outline, shape: id })}>{l}</button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
    </>
  );
}
