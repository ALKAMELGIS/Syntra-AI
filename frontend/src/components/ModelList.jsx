import React, { useRef, useState } from "react";
import { api } from "../api/client";

// The sidebar's model list, in three groups instead of one wall of status pills:
//   Ready             runnable now; a small amber dot marks "pretrained, not yet
//                     checked on your imagery" (details in the tooltip)
//   Available to add  open weights exist: one Install action (or From file offline)
//   Needs training    no open model; collapsed by default, muted, no pills

const RUNNABLE = new Set(["ready", "validation"]);

function PackRow({ m, onInstalled }) {
  const [state, setState] = useState({ busy: false, pct: null, error: null });
  const fileRef = useRef(null);
  const pack = m.pack;

  async function install() {
    setState({ busy: true, pct: null, error: null });
    try {
      const started = await api.installPack(pack.pack_id);
      if (started.installed) return onInstalled();
      for (;;) {
        await new Promise((r) => setTimeout(r, 800));
        const j = await api.installStatus(started.job_id);
        if (j.status === "completed") return onInstalled();
        if (j.status === "error" || j.status === "interrupted") {
          const e = new Error(j.error || "Install failed");
          e.detail = j.detail;
          throw e;
        }
        setState((s) => ({ ...s, pct: j.total_bytes ? Math.round((100 * j.bytes) / j.total_bytes) : null }));
      }
    } catch (err) {
      setState({ busy: false, pct: null, error: err.message, detail: err.detail, failed: true });
    }
  }

  async function fromFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setState({ busy: true, pct: null, error: null });
    try {
      await api.installPackFile(pack.pack_id, file);
      onInstalled();
    } catch (err) {
      setState({ busy: false, pct: null, error: err.message });
    }
  }

  return (
    <li className="pack-row" title={m.notes}>
      <div className="pack-row-main">
        <span className="pack-name">{m.name}</span>
        {state.busy ? (
          <span className="pack-progress mono">{state.pct != null ? `${state.pct}%` : "Installing"}</span>
        ) : (
          <button type="button" className="pack-install" onClick={install}>Install</button>
        )}
      </div>
      <div className="pack-meta">
        <span>{pack.licence}</span>
        {pack.commercial_use === false && <span className="pack-nc">Non-commercial</span>}
        {!state.busy && (
          <button type="button" className="pack-file" onClick={() => fileRef.current?.click()}>From file</button>
        )}
        <input ref={fileRef} type="file" accept=".onnx" hidden onChange={fromFile} />
      </div>
      {state.error && (
        <div className="pack-error" title={state.detail || state.error}>
          {state.error}
          {state.failed && pack.url && (
            <> <a href={pack.url} target="_blank" rel="noreferrer">Download</a></>
          )}
        </div>
      )}
    </li>
  );
}

export default function ModelList({ models, selectedModels, onToggleModel, onModelsChanged }) {
  const [showTraining, setShowTraining] = useState(false);
  const ready = models.filter((m) => RUNNABLE.has(m.status));
  const installable = models.filter((m) => m.status === "not_installed" && m.pack);
  const training = models.filter((m) => !RUNNABLE.has(m.status) && m.status !== "not_installed");

  return (
    <div className="model-list">
      <ul className="model-group">
        {ready.map((m) => {
          const on = selectedModels.includes(m.model_id);
          return (
            <li key={m.model_id}>
              <label className={`model-row ${on ? "selected" : ""}`} title={m.notes}>
                <input type="checkbox" checked={on} onChange={() => onToggleModel(m.model_id)} />
                <span className="model-name-text">{m.name}</span>
                {m.status === "validation" && (
                  <span className="model-unchecked" aria-label="Pretrained, not yet checked on your imagery" />
                )}
              </label>
            </li>
          );
        })}
      </ul>
      {ready.some((m) => m.status === "validation") && (
        <p className="model-key"><span className="model-unchecked" /> Not yet checked on your imagery</p>
      )}

      {installable.length > 0 && (
        <>
          <p className="model-group-title">Available to add</p>
          <ul className="model-group">
            {installable.map((m) => <PackRow key={m.model_id} m={m} onInstalled={onModelsChanged} />)}
          </ul>
        </>
      )}

      {training.length > 0 && (
        <>
          <button type="button" className="model-group-toggle" aria-expanded={showTraining}
                  onClick={() => setShowTraining((v) => !v)}>
            <span className={`env-collapse-caret ${showTraining ? "open" : ""}`}>▶</span>
            Needs training data
            <span className="mono model-group-count">{training.length}</span>
          </button>
          {showTraining && (
            <ul className="model-group model-group-muted">
              {training.map((m) => (
                <li key={m.model_id} className="model-row-muted" title={m.notes}>{m.name}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
