import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../../api/client.js";
import { IconCheck, IconChevronLeft, IconClose, IconEdit } from "../ui/Icons.jsx";
import { fmtAreaM2 } from "../../lib/prefs.js";

// Review one run's results: accept or reject each object, least sure first,
// then add the review to training (accepted -> labels, rejected -> "not this").
// Keys: A accept · R reject · ↑ ↓ move. A decision moves to the next object.

const FILTERS = [["open", "Open"], ["accept", "Accepted"], ["fix", "Fixed"], ["reject", "Rejected"], ["all", "All"]];
const kind = (d) => (d && d.startsWith("fix:") ? "fix" : d);
const LIMIT = 300;

function size(p) {
  if (p.length_m != null && p.width_m != null) return `${Math.round(p.length_m)} × ${Math.round(p.width_m)} m`;
  if (p.area_m2 != null) return fmtAreaM2(p.area_m2);
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
  const [fixing, setFixing] = useState(null);         // {i, value}: the class being typed
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
    return filter === "all" || (filter === "open" ? !d : kind(d) === filter);
  }), [order, decisions, filter]);

  const counts = useMemo(() => {
    const v = Object.values(decisions);
    const acc = v.filter((x) => x === "accept").length, rej = v.filter((x) => x === "reject").length;
    const fix = v.filter((x) => kind(x) === "fix").length;
    return { acc, rej, fix, open: (features?.length || 0) - acc - rej - fix };
  }, [decisions, features]);

  const classes = useMemo(() => [...new Set((features || []).map((f) => f.properties?.class).filter(Boolean))], [features]);
  const rootRef = useRef(null);
  function startFix(i) {
    const d = decisions[String(i)];
    setFixing({ i, value: kind(d) === "fix" ? d.slice(4) : "" });
  }
  function saveFix(e) {
    e.preventDefault();
    const v = fixing?.value.trim();
    const i = fixing?.i;
    setFixing(null);
    rootRef.current?.focus({ preventScroll: true });
    if (v && i != null) decide(i, `fix:${v}`);
  }
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
    else if ((e.key === "r" || e.key === "R" || e.key === "x" || e.key === "X") && current != null) decide(current, "reject");
    else if ((e.key === "e" || e.key === "E") && current != null) { e.preventDefault(); startFix(current); }
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
        <span className="review-counts mono" title="Accepted · fixed · rejected · open">
          {counts.acc} · {counts.fix} · {counts.rej} · {counts.open}
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
              <li key={i} data-i={i} className={`review-row ${current === i ? "current" : ""} ${kind(d) || ""}`} onClick={() => select(i)}>
                <span className="review-score mono">{p.score != null ? p.score.toFixed(2) : "–"}</span>
                {fixing?.i === i ? (
                  <form className="review-fix" onSubmit={saveFix} onClick={(e) => e.stopPropagation()}>
                    <input className="train-input" list="review-classes" autoFocus value={fixing.value} placeholder="Class"
                           aria-label="Correct class" onChange={(e) => setFixing({ i, value: e.target.value })}
                           onKeyDown={(e) => { if (e.key === "Escape") { setFixing(null); rootRef.current?.focus(); } }}
                           onBlur={() => setFixing(null)} />
                  </form>
                ) : (
                  <span className="review-class">{kind(d) === "fix" ? <><s>{p.class || "object"}</s> {d.slice(4)}</> : p.class || "object"}</span>
                )}
                <span className="review-size mono">{size(p)}</span>
                <button type="button" className={`review-btn accept ${d === "accept" ? "on" : ""}`} aria-pressed={d === "accept"}
                        title="Accept (A)" aria-label="Accept" onClick={(e) => { e.stopPropagation(); decide(i, "accept"); }}><IconCheck /></button>
                <button type="button" className={`review-btn reject ${d === "reject" ? "on" : ""}`} aria-pressed={d === "reject"}
                        title="Reject (X)" aria-label="Reject" onClick={(e) => { e.stopPropagation(); decide(i, "reject"); }}><IconClose /></button>
                <button type="button" className={`review-btn fix ${kind(d) === "fix" ? "on" : ""}`} aria-pressed={kind(d) === "fix"}
                        title="Fix class (E)" aria-label="Fix class" onClick={(e) => { e.stopPropagation(); select(i); startFix(i); }}><IconEdit /></button>
              </li>
            );
          })}
          {visible.length === 0 && <li className="analysis-empty">{filter === "open" ? "All reviewed" : "None"}</li>}
          {visible.length > LIMIT && <li className="review-more mono">first {LIMIT} of {visible.length}</li>}
        </ul>
      )}
      <div className="review-foot">
        <span className="review-keys mono">A · X · E · ↑↓</span>
        <datalist id="review-classes">{classes.map((c) => <option key={c} value={c} />)}</datalist>
        <button type="button" className="quiet-button" disabled={!counts.acc && !counts.rej && !counts.fix} onClick={apply}
                title="Accepted results become labels, rejected ones teach the model what is not an object">
          Add to training
        </button>
      </div>
      {applied && (
        <p className="review-applied">
          {applied.accepted + (applied.fixed || 0)} label{applied.accepted + (applied.fixed || 0) === 1 ? "" : "s"} and {applied.rejected} negative{applied.rejected === 1 ? "" : "s"} on this image, ready in Train
        </p>
      )}
    </div>
  );
}
