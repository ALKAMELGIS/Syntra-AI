// The label tools as a rail on the map's left edge, shown while labelling
// (Train > Label). One button per group; the icons stay visible, and pointing
// at a group opens its tools beside the rail. A group's button shows the tool
// last used in it and clicking it picks that tool again. The keys are the same
// as in the Train panel (LabelTools.jsx handles them).
import React, { useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import { classColor, TRAIN_AREA, CHECK_AREA } from "./AnnotationTool.jsx";
import {
  IconAreaDashed, IconBoxAI, IconCircle, IconClickAI, IconPencil, IconPointer, IconPolygon,
  IconRectangle, IconSparkle, IconSquare, IconTag, IconTrash, IconUndo,
} from "../ui/Icons.jsx";

const SELECT = "select";
// what the AI tools draw: the Refine `shape` sent to the engine
const DRAWS = [["auto", "Auto"], ["box", "Box"], ["rectangle", "Rotated"], ["polygon", "Outline"]];

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
  { id: "select", tools: [{ id: SELECT, label: "Select and edit", key: "V", Icon: IconPointer,
    note: "Click a label: drag corners to reshape, the shape to move it, the round handle to rotate" }] },
  { sep: true },
  { id: "ai", title: "Auto segment", Icon: IconSparkle, tools: [
    { id: "click", label: "Click object", key: "A", Icon: IconClickAI, note: "Alt+click removes a part" },
    { id: "box", label: "Box around object", key: "B", Icon: IconBoxAI, note: "Drag a box around it" }], draws: true },
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
  }

  // Ctrl+Z: the last label added here
  const live = useRef();
  live.current = { onUndo, canUndo };
  useEffect(() => {
    function key(e) {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || e.shiftKey || isTyping(e.target)) return;
      if (live.current.canUndo) { e.preventDefault(); live.current.onUndo?.(); }
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
    if (t.action === "delete") { if (editId) onEditDelete?.(); return; }
    if (t.action === "clear") { onClear?.(); return; }
    if (t.area) { onActiveClass(t.id); onMode("rect"); return; }
    if (drawingArea && classes.length) onActiveClass(classes[0]);
    onMode(t.id === SELECT ? null : t.id);
  }
  function main(g) {
    if (g.action === "undo") { onUndo?.(); return; }
    if (needsClass(g)) { setOpen("class"); return; }
    if (!g.tools) { setOpen(open === g.id ? null : g.id); return; }
    const t = g.tools.find((x) => x.id === last[g.id]) || g.tools[0];
    if (t.action) { setOpen(g.id); return; }       // Delete: choose in the list
    pick(g, t);
  }
  const shown = (g) => (g.tools ? g.tools.find((x) => x.id === last[g.id]) || g.tools[0] : null);
  const needsClass = (g) => !classes.length && !["class", "undo", "delete", "areas"].includes(g.id);

  return (
    <div className="label-rail" role="toolbar" aria-label="Label tools" aria-orientation="vertical"
         onMouseDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
      {GROUPS.map((g, i) => {
        if (g.sep) return <span key={`s${i}`} className="label-rail-sep" aria-hidden="true" />;
        const t = shown(g);
        const Icon = g.Icon || t.Icon;
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
                )) : <p className="label-rail-note">Add the first class: what you'll draw</p>)}
                {g.id === "class" && onAddClass && (
                  <form className="label-rail-add" onSubmit={addClass}>
                    <input className="train-input" value={newClass} onChange={(e) => setNewClass(e.target.value)}
                           placeholder="New class: palm, car…" aria-label="New class" />
                  </form>
                )}
                {g.id === "class" && editId && <p className="label-rail-note">Changes the selected label</p>}
                {g.tools?.map((x) => (
                  <button key={x.id} type="button" role="menuitem" disabled={x.action === "delete" && !editId}
                          className={`label-rail-row ${isOn(x) ? "on" : ""} ${x.danger ? "danger" : ""}`}
                          onClick={() => (needsClass(g) ? setOpen("class") : pick(g, x))}
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
                    <p className="label-rail-note">Enter keeps the outline, Esc drops it</p>
                  </>
                )}
                {!g.draws && g.tools?.length === 1 && g.tools[0].note && <p className="label-rail-note">{g.tools[0].note}</p>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
