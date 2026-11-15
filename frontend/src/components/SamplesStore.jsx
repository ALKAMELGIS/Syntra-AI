import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

// Where training samples are kept: a private Hugging Face dataset repo.
// HfSettings sits in Settings; SaveSamples sits under a finished export in
// Train. The token stays on this computer: the engine never sends it back.

export function HfSettings() {
  const [st, setSt] = useState(null);
  const [token, setToken] = useState("");
  const [repo, setRepo] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => { api.hfStatus().then((s) => { setSt(s); setRepo(s.repo || ""); }).catch(() => setSt(null)); }, []);

  async function save(body) {
    setBusy(true); setErr(null);
    try {
      const s = await api.hfSave(body);
      setSt(s); setRepo(s.repo || ""); setToken("");
    } catch (e) { setErr(e.message); }
    setBusy(false);
  }

  if (!st) return null;
  return (
    <div className="hf-settings">
      {st.configured ? (
        <p className="hf-status"><span className="auth-dot up" aria-hidden="true" /> <span className="mono">{st.user}</span>
          <button type="button" className="link-button" onClick={() => save({ token: "" })} disabled={busy}>Remove</button></p>
      ) : (
        <form className="hf-row" onSubmit={(e) => { e.preventDefault(); if (token.trim()) save({ token: token.trim() }); }}>
          <input type="password" className="hf-input" placeholder="Write token (hf_…)" value={token} autoComplete="off"
                 onChange={(e) => setToken(e.target.value)} aria-label="Hugging Face write token" spellCheck={false} />
          <button type="submit" className="quiet-button" disabled={busy || !token.trim()}>{busy ? "Checking…" : "Connect"}</button>
        </form>
      )}
      {st.configured && (
        <form className="hf-row" onSubmit={(e) => { e.preventDefault(); save({ repo }); }}>
          <input className="hf-input mono" value={repo} onChange={(e) => setRepo(e.target.value)} aria-label="Repo for samples" spellCheck={false} />
          <button type="submit" className="quiet-button" disabled={busy || repo === (st.repo || "")}>Save</button>
        </form>
      )}
      {!st.library && <p className="train-error">Needs the huggingface_hub package: pip install huggingface_hub</p>}
      {err && <p className="train-error">{err}</p>}
    </div>
  );
}

export function SaveSamples({ datasetJobId }) {
  const [hf, setHf] = useState(null);
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.hfStatus().then(setHf).catch(() => setHf(null)); }, []);
  useEffect(() => { setJob(null); setErr(null); }, [datasetJobId]);

  useEffect(() => {
    if (!job || ["completed", "error"].includes(job.status)) return;
    const t = setInterval(() => api.pushStatus(job.job_id).then((s) => setJob((j) => ({ ...j, ...s }))).catch(() => {}), 1200);
    return () => clearInterval(t);
  }, [job?.job_id, job?.status]);

  if (!hf) return null;
  if (!hf.configured) return <p className="train-dim">Connect Hugging Face in Settings to keep these samples in a private repo</p>;
  const running = job && !["completed", "error"].includes(job.status);
  return (
    <div className="save-samples">
      <button type="button" className="quiet-button" disabled={running} onClick={async () => {
        setErr(null);
        try { setJob(await api.pushSamples(datasetJobId)); } catch (e) { setErr(e.message); }
      }}>
        {running ? `Saving… ${Math.round(job.progress || 0)}%` : job?.status === "completed" ? "Save again" : "Save to Hugging Face"}
      </button>
      {job?.status === "completed" && job.result && (
        <p className="train-ok">
          <span className="mono">{job.result.files}</span> files in{" "}
          <a href={job.result.url} target="_blank" rel="noreferrer" className="mono">{job.result.repo_id}/{job.result.folder}</a>
        </p>
      )}
      {(err || job?.status === "error") && <p className="train-error">{err || job.error}</p>}
    </div>
  );
}
