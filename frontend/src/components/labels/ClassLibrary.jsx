// Object class library: tick ready-made classes (by theme) or add your own type,
// each with its drawing tool and the outline the AI tools fit to it.
import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { CLASS_GROUPS, CLASS_LIBRARY, SHAPE_NAMES, TOOL_NAMES, classSlug } from "../../lib/classLibrary.js";
import { classColor } from "./AnnotationTool.jsx";
import { IconChevronDown, IconClose } from "../ui/Icons.jsx";

const CUSTOM_KEY = "syntra_class_library_custom";

function loadCustom() {
  try { return JSON.parse(localStorage.getItem(CUSTOM_KEY) || "[]"); } catch { return []; }
}

export default function ClassLibrary({ open, onClose, existing = [], onAdd }) {
  const [q, setQ] = useState("");
  const [picked, setPicked] = useState(new Set());
  const [closed, setClosed] = useState(new Set());
  const [custom, setCustom] = useState(loadCustom);
  const [draft, setDraft] = useState({ name: "", group: CLASS_GROUPS[0], tool: "polygon", shape: "polygon" });

  useEffect(() => { if (open) setPicked(new Set()); }, [open]);
  useEffect(() => {
    if (!open) return undefined;
    const k = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open, onClose]);

  const all = useMemo(() => [...CLASS_LIBRARY, ...custom.map((c) => ({ ...c, custom: true }))], [custom]);
  const have = new Set(existing);
  const shown = all.filter((c) => !q || c.name.toLowerCase().includes(q.toLowerCase()) || c.group.toLowerCase().includes(q.toLowerCase()));
  const groups = [...new Set([...CLASS_GROUPS, ...custom.map((c) => c.group)])];
  if (!open) return null;

  const toggle = (slug) => setPicked((p) => { const n = new Set(p); n.has(slug) ? n.delete(slug) : n.add(slug); return n; });
  const toggleGroup = (rows) => setPicked((p) => {
    const n = new Set(p);
    const free = rows.map((c) => classSlug(c.name)).filter((s) => !have.has(s));
    const allOn = free.every((s) => n.has(s));
    free.forEach((s) => (allOn ? n.delete(s) : n.add(s)));
    return n;
  });
  function saveCustom(list) { setCustom(list); try { localStorage.setItem(CUSTOM_KEY, JSON.stringify(list)); } catch { /* private mode */ } }
  function addType(e) {
    e.preventDefault();
    const name = draft.name.trim();
    if (!name || all.some((c) => classSlug(c.name) === classSlug(name))) return;
    saveCustom([...custom, { ...draft, name }]);
    setPicked((p) => new Set(p).add(classSlug(name)));
    setDraft((d) => ({ ...d, name: "" }));
  }
  const removeType = (name) => saveCustom(custom.filter((c) => c.name !== name));
  function addPicked() {
    onAdd(all.filter((c) => picked.has(classSlug(c.name))).map((c) => ({ name: classSlug(c.name), tool: c.tool, shape: c.shape })));
    onClose();
  }

  return createPortal(
    <div className="set-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="clib" role="dialog" aria-modal="true" aria-label="Class library">
        <header className="set-head clib-head">
          <h2>Class library</h2>
          <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
        </header>
        <form className="clib-new" onSubmit={addType}>
          <input className="train-input" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="New type" aria-label="New type" />
          <select className="train-select" value={draft.group} onChange={(e) => setDraft({ ...draft, group: e.target.value })} aria-label="Theme">
            {CLASS_GROUPS.map((x) => <option key={x}>{x}</option>)}
          </select>
          <select className="train-select" value={draft.tool} onChange={(e) => setDraft({ ...draft, tool: e.target.value })} aria-label="Tool">
            {Object.entries(TOOL_NAMES).map(([id, l]) => <option key={id} value={id}>{l}</option>)}
          </select>
          <select className="train-select" value={draft.shape} onChange={(e) => setDraft({ ...draft, shape: e.target.value })} aria-label="Outline">
            {Object.entries(SHAPE_NAMES).map(([id, l]) => <option key={id} value={id}>{l}</option>)}
          </select>
          <button type="submit" className="quiet-button" disabled={!draft.name.trim()}>Add</button>
        </form>
        <div className="clib-tools">
          <input className="train-input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search" aria-label="Search classes" />
          <button type="button" className="link-button" onClick={() => setClosed(new Set())}>Expand all</button>
          <button type="button" className="link-button" onClick={() => setClosed(new Set(groups))}>Collapse all</button>
        </div>
        <div className="clib-table" role="table">
          <div className="clib-row clib-th" role="row"><span>Class</span><span>Tool</span><span>Outline</span></div>
          {groups.map((grp) => {
            const rows = shown.filter((c) => c.group === grp);
            if (!rows.length) return null;
            const isOpen = !closed.has(grp) || q;
            const free = rows.filter((c) => !have.has(classSlug(c.name)));
            const allOn = free.length > 0 && free.every((c) => picked.has(classSlug(c.name)));
            return (
              <React.Fragment key={grp}>
                <div className="clib-row clib-group" role="row">
                  <span>
                    <button type="button" className={`clib-caret ${isOpen ? "" : "shut"}`} aria-label={isOpen ? "Collapse" : "Expand"}
                            onClick={() => setClosed((s) => { const n = new Set(s); n.has(grp) ? n.delete(grp) : n.add(grp); return n; })}><IconChevronDown /></button>
                    <input type="checkbox" checked={allOn} disabled={!free.length} onChange={() => toggleGroup(rows)} aria-label={`All of ${grp}`} />
                    {grp} <span className="mono train-dim">{rows.length}</span>
                  </span><span /><span />
                </div>
                {isOpen && rows.map((c) => {
                  const slug = classSlug(c.name);
                  const there = have.has(slug);
                  return (
                    <label key={slug} className={`clib-row ${there ? "there" : ""}`} role="row">
                      <span>
                        <input type="checkbox" checked={there || picked.has(slug)} disabled={there} onChange={() => toggle(slug)} />
                        <span className="train-swatch" style={{ background: classColor(slug) }} />{c.name}
                        {c.custom && <button type="button" className="clib-x" onClick={(e) => { e.preventDefault(); removeType(c.name); }} aria-label={`Remove ${c.name}`}><IconClose /></button>}
                      </span>
                      <span>{TOOL_NAMES[c.tool] || c.tool}</span>
                      <span>{SHAPE_NAMES[c.shape] || c.shape}</span>
                    </label>
                  );
                })}
              </React.Fragment>
            );
          })}
        </div>
        <footer className="clib-foot">
          <span className="train-dim mono">{all.length} classes · {existing.filter((x) => all.some((c) => classSlug(c.name) === x)).length} in use</span>
          <button type="button" className="link-button" onClick={onClose}>Cancel</button>
          <button type="button" className="quiet-button" disabled={!picked.size} onClick={addPicked}>Add {picked.size || ""} selected</button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
