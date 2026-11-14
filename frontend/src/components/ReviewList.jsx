import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client.js";
import { IconCheck, IconChevronLeft, IconClose } from "./Icons.jsx";

// Review one run's results: accept or reject each object, least sure first,
// then add the review to training (accepted -> labels, rejected -> "not this").
// Keys: A accept · R reject · ↑ ↓ move. A decision moves to the next object.

const FILTERS = [["open", "Open"], ["accept", "Accepted"], ["reject", "Rejected"], ["all", "All"]];
const LIMIT = 300;

function size(p) {
  if (p.length_m != null && p.width_m != null) return `${Math.round(p.length_m)} × ${Math.round(p.width_m)} m`;
  if (p.area_m2 != null) return `${Math.round(p.area_m2)} m²`;
  return "";
}

export default function ReviewList({ entry, color, onBack, onFocus, onClose }) {
  const { job } = entry;
  const modelId = job.model_id;
  const [features, setFeatures] = useState(null);
  const [decisions, setDecisions] = useState({});
  const [filter, setFilter] = useState("open");
  const [current, setCurrent] = useState(null);       // feature index
  const [applied, setApplied] = useState(null);
  const [err, setErr] = useState(null);
  const listRef = useRef(null);

  useEffect(() => {
    let live = true;
    Promise.all([api.getGeojson(job.job_id, modelId), api.getReview(job.job_id, modelId)])
      .then(([fc, rv]) => { if (live) { setFeatures(fc.features || []); setDecisions(rv.decisions || {}); } })
      .catch((e) => live && setErr(e.message));
    return () => { live = false; onFocus?.(null); };
  }, [job.job_id, modelId]);

  // least sure first: the objects a person most needs to look at
  const order = useMemo(() => (features || [])
    .map((f, i) => ({ i, score: f.properties?.score ?? 1 }))
    .sort((a, b) => a.score - b.score)
    .map((x) => x.i), [features]);

  const visible = useMemo(() => order.filter((i) => {
    const d = decisions[String(i)];
    return filter === "all" || (filter === "open" ? !d : d === filter);
  }), [order, decisions, filter]);

  const counts = useMemo(() => {
    const v = Object.values(decisions);
    const acc = v.filter((x) => x === "accept").length, rej = v.filter((x) => x === "reject").length;
    return { acc, rej, open: (features?.length || 0) - acc - rej };
  }, [decisions, features]);

  const rootRef = useRef(null);
  function select(i) {
    setCurrent(i);
    if (rootRef.current && !rootRef.current.contains(document.activeElement)) rootRef.current.focus({ preventScroll: true });
    if (i != null && features) onFocus?.(features[i]);
    requestAnimationFrame(() => listRef.current?.querySelector(`[data-i="${i}"]`)?.scrollIntoView({ block: "nearest" }));
  }

  function step(dir) {
    if (!visible.length) return;
    const at = visible.indexOf(current);
    const next = at < 0 ? visible[0] : visible[Math.min(visible.length - 1, Math.max(0, at + dir))];
    select(next);
  }

  async function decide(i, d) {
    const key = String(i);
    const value = decisions[key] === d ? null : d;               // the same button again undoes
    const before = decisions;
    const next = { ...decisions };
    if (value) next[key] = value; else delete next[key];
    setDecisions(next);
    setApplied(null);
    if (filter === "open" && value) {                             // keep going through the open ones
      const rest = visible.filter((x) => x !== i);
      const at = visible.indexOf(i);
      select(rest[Math.min(at, rest.length - 1)] ?? null);
    }
    try {
      await api.putReview(job.job_id, modelId, { [key]: value });
    } catch (e) {
      setDecisions(before);
      setErr(e.message);
    }
  }

  function onKey(e) {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); step(1); }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); step(-1); }
    else if ((e.key === "a" || e.key === "A") && current != null) decide(current, "accept");
    else if ((e.key === "r" || e.key === "R") && current != null) decide(current, "reject");
  }

  async function apply() {
    setErr(null);
    try {
      setApplied(await api.applyReview(job.job_id, modelId));
    } catch (e) {
      setErr(e.message);
    }
  }

  return (
    <div className="review" ref={rootRef} tabIndex={0} onKeyDown={onKey} aria-label="Review results" data-own-keys>
      <div className="review-head">
        <button type="button" className="review-back" onClick={onBack} aria-label="Back to runs" title="Back to runs"><IconChevronLeft /></button>
        <span className="analysis-swatch" style={{ background: color }} />
        <span className="review-title">{entry.modelMeta?.name || modelId}</span>
        <span className="review-counts mono" title="Accepted · rejected · open">
          {counts.acc} · {counts.rej} · {counts.open}
        </span>
        {onClose && <button type="button" className="panel-close-btn" title="Close" aria-label="Close" onClick={onClose}><IconClose /></button>}
      </div>
      <div className="review-filters" role="tablist">
        {FILTERS.map(([k, label]) => (
          <button key={k} type="button" role="tab" aria-selected={filter === k} className={filter === k ? "active" : ""} onClick={() => setFilter(k)}>
            {label}
          </button>
        ))}
      </div>
      {err && <p className="review-error">{err}</p>}
      {!features && !err && <p className="analysis-empty">Loading…</p>}
      {features && (
        <ul className="review-list" ref={listRef}>
          {visible.slice(0, LIMIT).map((i) => {
            const p = features[i].properties || {};
            const d = decisions[String(i)];
            return (
              <li key={i} data-i={i} className={`review-row ${current === i ? "current" : ""} ${d || ""}`} onClick={() => select(i)}>
                <span className="review-score mono">{p.score != null ? p.score.toFixed(2) : "–"}</span>
                <span className="review-class">{p.class || "object"}</span>
                <span className="review-size mono">{size(p)}</span>
                <button type="button" className={`review-btn accept ${d === "accept" ? "on" : ""}`} aria-pressed={d === "accept"}
                        title="Accept (A)" aria-label="Accept" onClick={(e) => { e.stopPropagation(); decide(i, "accept"); }}><IconCheck /></button>
                <button type="button" className={`review-btn reject ${d === "reject" ? "on" : ""}`} aria-pressed={d === "reject"}
                        title="Reject (R)" aria-label="Reject" onClick={(e) => { e.stopPropagation(); decide(i, "reject"); }}><IconClose /></button>
              </li>
            );
          })}
          {visible.length === 0 && <li className="analysis-empty">{filter === "open" ? "All reviewed" : "None"}</li>}
          {visible.length > LIMIT && <li className="review-more mono">first {LIMIT} of {visible.length}</li>}
        </ul>
      )}
      <div className="review-foot">
        <span className="review-keys mono">A · R · ↑↓</span>
        <button type="button" className="quiet-button" disabled={!counts.acc && !counts.rej} onClick={apply}
                title="Accepted results become labels, rejected ones teach the model what is not an object">
          Add to training
        </button>
      </div>
      {applied && (
        <p className="review-applied">
          {applied.accepted} label{applied.accepted === 1 ? "" : "s"} and {applied.rejected} negative{applied.rejected === 1 ? "" : "s"} on this image, ready in Train
        </p>
      )}
    </div>
  );
}
