import React, { useRef, useState } from "react";
import { api } from "../api/client";

// The sidebar's model list, in three groups:
//   ready             runnable now; an amber dot = pretrained, not yet checked
//                     on your imagery (explained once under the list)
//   available to add  open weights exist: one quiet Install per row; the
//                     offline route (From file) only appears after a failure
//   needs training    one line with a next step; expanded, each row has Train
// One primary action in the sidebar stays "Run prediction": nothing here is filled.

const RUNNABLE = new Set(["ready", "validation"]);

// Short in the row (CC BY-NC 4.0 -> CC BY-NC), full wording in the tooltip.
function licenceShort(pack) {
  return pack.licence.replace(/\s+\d+(\.\d+)*$/, "");
}
function licenceTitle(pack) {
  if (pack.commercial_use === false) return `${pack.licence}: non-commercial use only`;
  if (pack.commercial_use === null) return `${pack.licence}: licence not stated by the publisher`;
  return `${pack.licence}: commercial use allowed`;
}

function PackRow({ m, onInstalled }) {
  const [st, setSt] = useState({ busy: false, pct: null, failed: false, detail: null });
  const fileRef = useRef(null);
  const pack = m.pack;

  async function install() {
    setSt({ busy: true, pct: null, failed: false, detail: null });
    try {
      const started = await api.installPack(pack.pack_id);
      if (started.installed) return onInstalled();
      for (;;) {
        await new Promise((r) => setTimeout(r, 800));
        const j = await api.installStatus(started.job_id);
        if (j.status === "completed") return onInstalled();
        if (j.status === "error" || j.status === "interrupted") {
          setSt({ busy: false, pct: null, failed: true, detail: j.detail || j.error });
          return;
        }
        setSt((s) => ({ ...s, pct: j.total_bytes ? Math.round((100 * j.bytes) / j.total_bytes) : null }));
      }
    } catch (err) {
      setSt({ busy: false, pct: null, failed: true, detail: err.message });
    }
  }

  async function fromFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setSt({ busy: true, pct: null, failed: false, detail: null });
    try {
      await api.installPackFile(pack.pack_id, file);
      onInstalled();
    } catch (err) {
      setSt({ busy: false, pct: null, failed: true, detail: err.message });
    }
  }

  return (
    <li className="pack-row" title={m.notes}>
      <span className="pack-name">{m.name}</span>
      <span className="pack-licence mono" title={licenceTitle(pack)}>{licenceShort(pack)}</span>
      {st.busy ? (
        <span className="pack-progress mono">{st.pct != null ? `${st.pct}%` : "…"}</span>
      ) : (
        <button type="button" className="quiet-button" onClick={install}>Install</button>
      )}
      {st.busy && (
        <span className="pack-bar" aria-hidden="true">
          <span className={st.pct == null ? "pack-bar-fill pack-bar-indeterminate" : "pack-bar-fill"}
                style={st.pct == null ? undefined : { width: `${st.pct}%` }} />
        </span>
      )}
      {st.failed && (
        <span className="pack-fail" title={st.detail || ""}>
          <span className="pack-fail-dot" aria-hidden="true" />
          Couldn't download
          <a href={pack.url} target="_blank" rel="noreferrer">Download</a>
          <button type="button" className="link-button" onClick={() => fileRef.current?.click()}>From file</button>
        </span>
      )}
      <input ref={fileRef} type="file" accept=".onnx" hidden onChange={fromFile} />
    </li>
  );
}

export default function ModelList({ models, selectedModels, onToggleModel, onModelsChanged, onTrain }) {
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
                {m.status === "validation" && <span className="model-unchecked" aria-label="Not yet checked on your imagery" />}
              </label>
            </li>
          );
        })}
      </ul>
      {ready.some((m) => m.status === "validation") && (
        <p className="model-key"><span className="model-unchecked" /> Not yet checked on your imagery</p>
      )}

      {installable.length > 0 && (
        <section className="model-section">
          <p className="model-group-title">Available to add</p>
          <ul className="model-group">
            {installable.map((m) => <PackRow key={m.model_id} m={m} onInstalled={onModelsChanged} />)}
          </ul>
        </section>
      )}

      {training.length > 0 && (
        <section className="model-section">
          <div className="model-train-line">
            <button type="button" className="model-group-toggle" aria-expanded={showTraining}
                    onClick={() => setShowTraining((v) => !v)}>
              <span className={`env-collapse-caret ${showTraining ? "open" : ""}`}>▶</span>
              <span><span className="mono">{training.length}</span> need labelled data</span>
            </button>
            <button type="button" className="link-button" onClick={() => onTrain?.()}>Train one</button>
          </div>
          {showTraining && (
            <ul className="model-group">
              {training.map((m) => (
                <li key={m.model_id} className="model-row-muted" title={m.notes}>
                  <span className="model-name-text">{m.name}</span>
                  <button type="button" className="row-action" onClick={() => onTrain?.(m.model_id)}>Train</button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}
