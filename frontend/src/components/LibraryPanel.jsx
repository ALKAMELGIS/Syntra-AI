import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";
import { classColor } from "./AnnotationTool.jsx";
import { IconClose } from "./Icons.jsx";

// The sample library, beside the map toolbar. It shows the data folder as it
// is, by class: samples (classes/<class>/labels), areas, dataset versions
// (datasets/<class>/vNNN) and models (models/<class>/vNNN, approved.json).
// Each dataset version has one Send menu: saved here already, back up to
// Hugging Face, train here or on a RunPod GPU.

const ACTIVE = (s) => s && !["completed", "done", "error", "canceled", "interrupted"].includes(s);

function score(m) {
  if (m.test && m.test.f1 != null) return { v: m.test.f1.toFixed(2), l: "F1 test" };
  if (m.val_map50 != null) return { v: m.val_map50.toFixed(2), l: "mAP50 val" };
  return { v: "–", l: m.weights ? "" : "training" };
}

function SendMenu({ ds, onClose, onJob, hf, rp, device }) {
  const [err, setErr] = useState(null);
  const ref = useRef(null);
  useEffect(() => {
    const away = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    const esc = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", away);
    window.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", away); window.removeEventListener("keydown", esc); };
  }, [onClose]);

  async function run(fn) {
    setErr(null);
    try { await fn(); onClose(); } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
  }
  const upload = () => run(async () => {
    const r = await api.pushSamples(ds.export_job_id);
    onJob({ kind: "push", id: r.job_id, label: `Uploading ${ds.version}` });
  });
  const trainHere = () => run(async () => {
    const st = await api.exportDatasetStatus(ds.export_job_id);
    const r = await api.startTraining({ data_yaml: st.data_yaml, device });
    onJob({ kind: "train", id: r.job_id, label: `Training ${ds.version} here` });
  });
  const trainGpu = () => run(async () => {
    onJob({ kind: "gpu", ds, label: ds.pushed ? `Starting a GPU for ${ds.version}` : `Uploading ${ds.version}, then a GPU` });
  });
  const noJob = !ds.export_job_id;
  return (
    <div className="send-menu" ref={ref} role="menu" aria-label={`Send ${ds.version}`}>
      <div className="send-opt">
        <span className="send-ic" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="3" y="4" width="18" height="12" rx="1.5" /><path d="M8 20h8M12 16v4" /></svg>
        </span>
        <div className="send-txt"><span>On this computer</span><span className="mono">datasets/{ds.group}/{ds.version}</span></div>
        <button type="button" className="quiet-button" onClick={trainHere} disabled={noJob}>Train here</button>
      </div>
      <div className="send-opt">
        <span className="send-ic" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M7 18h10a4 4 0 0 0 .5-8A6 6 0 0 0 6 9.5 4.3 4.3 0 0 0 7 18Z" /><path d="M12 15v-5m-2.5 2.5L12 10l2.5 2.5" /></svg>
        </span>
        <div className="send-txt"><span>Back up to Hugging Face</span>
          <span className="mono">{!hf?.configured ? "add your token in Settings" : ds.pushed ? `saved as ${ds.pushed.folder}` : `private · ${hf.repo}`}</span></div>
        <button type="button" className="quiet-button" onClick={upload} disabled={!hf?.configured || noJob}>{ds.pushed ? "Again" : "Upload"}</button>
      </div>
      <div className="send-opt">
        <span className="send-ic" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="5" y="5" width="14" height="14" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /></svg>
        </span>
        <div className="send-txt"><span>Train on a RunPod GPU</span>
          <span className="mono">{!rp?.configured ? "add your RunPod key in Settings" : !hf?.configured ? "needs Hugging Face too" : ds.pushed ? "RTX 4090 · stops by itself" : "backs up first"}</span></div>
        <button type="button" className="run-button send-primary" onClick={trainGpu} disabled={!rp?.configured || !hf?.configured || noJob}>Train</button>
      </div>
      {noJob && <p className="train-dim send-note">Built before this panel existed: build a new version to send it</p>}
      {err && <p className="train-error send-note">{err}</p>}
    </div>
  );
}

export default function LibraryPanel({ open, onClose, device = "cpu" }) {
  const [lib, setLib] = useState(null);
  const [sel, setSel] = useState(null);
  const [hf, setHf] = useState(null);
  const [rp, setRp] = useState(null);
  const [menu, setMenu] = useState(null);
  const [jobs, setJobs] = useState([]);           // {kind, id, label, status, progress, error}
  const [err, setErr] = useState(null);

  const load = useCallback(() => api.library().then((l) => {
    setLib(l);
    setSel((s) => s || l.classes[0]?.slug || null);
  }).catch((e) => setErr(e.message)), []);

  useEffect(() => {
    if (!open) return;
    load();
    api.hfStatus().then(setHf).catch(() => setHf(null));
    api.runpodStatus().then(setRp).catch(() => setRp(null));
  }, [open, load]);

  // follow running jobs; refresh the library when one finishes
  const busy = jobs.some((j) => ACTIVE(j.status) || !j.status);
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(async () => {
      const next = await Promise.all(jobs.map(async (j) => {
        if (j.status && !ACTIVE(j.status)) return j;
        try {
          if (j.kind === "gpu" && !j.id) return j;
          const st = j.kind === "build" ? await api.exportDatasetStatus(j.id)
            : j.kind === "push" ? await api.pushStatus(j.id)
            : await api.trainingStatus(j.id);
          return { ...j, status: st.status, progress: typeof st.progress === "number" ? st.progress : st.progress?.epoch, error: st.error, message: st.message };
        } catch { return j; }
      }));
      setJobs(next);
      if (next.some((j, k) => j.status !== jobs[k].status && !ACTIVE(j.status))) load();
    }, 1500);
    return () => clearInterval(t);
  }, [busy, jobs, load]);

  // "Train on GPU": upload first when needed, then start the pod
  async function startGpu(ds, label) {
    const entry = { kind: "gpu", label };
    setJobs((l) => [entry, ...l]);
    const update = (fields) => setJobs((l) => l.map((j) => (j === entry || j.label === label ? { ...j, ...fields } : j)));
    try {
      let pushId = null;
      if (!ds.pushed) {
        const r = await api.pushSamples(ds.export_job_id);
        pushId = r.job_id;
        for (;;) {
          await new Promise((res) => setTimeout(res, 1500));
          const st = await api.pushStatus(pushId);
          update({ status: "processing", progress: st.progress });
          if (st.status === "completed") break;
          if (["error", "interrupted"].includes(st.status)) throw new Error(st.error || "Upload failed");
        }
      }
      const job = await api.cloudTrain({ push_job_id: pushId, dataset_job_id: ds.export_job_id });  // no push id: the version's pushed.json
      update({ kind: "train", id: job.job_id, status: job.status, label: `Training ${ds.version} on a GPU` });
    } catch (e) {
      update({ status: "error", error: e.detail?.message || e.detail || e.message });
    }
  }

  function onJob(j) {
    if (j.kind === "gpu") { startGpu(j.ds, j.label); return; }
    setJobs((l) => [{ ...j, status: "queued" }, ...l]);
  }

  async function build(c) {
    setErr(null);
    try {
      const r = await api.libraryBuild(c.slug, {});
      setJobs((l) => [{ kind: "build", id: r.job_id, label: `Building ${c.name} from ${r.images} image(s)`, status: "queued" }, ...l]);
    } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
  }
  async function approve(m, version) {
    setErr(null);
    try { await api.libraryApprove(m.group, version); load(); } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
  }
  async function toggleAuto() {
    try { setHf(await api.hfSave({ auto_backup: !hf?.auto_backup })); } catch (e) { setErr(e.detail?.message || e.message); }
  }

  if (!open) return null;
  const c = lib?.classes.find((x) => x.slug === sel);
  const lastPushed = c?.datasets.filter((d) => d.pushed).at(-1);
  const nextV = c ? `v${String((c.datasets.length ? parseInt(c.datasets.at(-1).version.slice(1), 10) : 0) + 1).padStart(3, "0")}` : "";
  const visibleJobs = jobs.slice(0, 3);

  return (
    <section className="lib-flyout" aria-label="Sample library">
      <header className="analysis-head">
        <span className="analysis-title">Library</span>
        {c && <span className="analysis-count lib-sync">
          <span className={`lib-dot ${lastPushed ? "on" : ""}`} />{lastPushed ? `Hugging Face up to ${lastPushed.version}` : "On this computer"}
        </span>}
        <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>
      {!lib && !err && <p className="perf-empty">Reading the library…</p>}
      {lib && !lib.classes.length && <p className="perf-empty">No samples yet. Label objects in Train › Label: each class gets its own folder here.</p>}
      {lib && lib.classes.length > 0 && (
        <div className="lib-body">
          <nav className="lib-classes" aria-label="Classes">
            {lib.classes.map((x) => (
              <button type="button" key={x.slug} className={x.slug === sel ? "selected" : ""} onClick={() => setSel(x.slug)}>
                <span className="train-swatch" style={{ background: classColor(x.name) }} />
                <span className="lib-cname">{x.name}</span><span className="mono train-dim">{x.samples}</span>
              </button>
            ))}
          </nav>
          {c && (
            <div className="lib-detail">
              <div className="lib-sec">
                <div className="lib-sec-head"><span>Samples</span><span className="train-dim">{c.images} image{c.images === 1 ? "" : "s"} · target {c.target.toLocaleString()}</span></div>
                <div className="lib-big"><span className="mono">{c.samples}</span><span className="mono train-dim">/ {c.target.toLocaleString()}</span></div>
                <div className="lib-bar"><div style={{ width: `${Math.min(100, (100 * c.samples) / c.target)}%` }} /></div>
              </div>
              <div className="lib-sec">
                <div className="lib-sec-head"><span>Areas</span><span className="train-dim">where every object is labelled</span></div>
                <div className="lib-cards">
                  <div><b className="mono">{c.areas.complete.count}</b>complete · {c.areas.complete.km2} km²</div>
                  <div><b className="mono">{c.areas.accuracy.count}</b>accuracy · {c.areas.accuracy.km2} km²</div>
                  <div><b className="mono">{c.new_since_last ?? c.samples}</b>{c.datasets.length ? `new since ${c.datasets.at(-1).version}` : "not built yet"}</div>
                </div>
                {!c.areas.accuracy.count && <p className="train-dim">Draw an accuracy area to get a test score</p>}
              </div>
              <div className="lib-sec">
                <div className="lib-sec-head"><span>Datasets</span><span className="train-dim">split by area, 1-tile gap</span></div>
                {c.datasets.length > 0 && (
                  <div className="lib-tbl">
                    <div className="lib-tr lib-th"><span>Version</span><span>train</span><span>val</span><span>test</span><span /></div>
                    {[...c.datasets].reverse().map((d) => (
                      <div className="lib-tr" key={d.group + d.version}>
                        <span className="mono">{d.version}</span>
                        <span className="mono">{d.counts?.train?.images ?? "–"}</span>
                        <span className="mono">{d.counts?.val?.images ?? "–"}</span>
                        <span className="mono">{d.counts?.test?.images ?? "–"}</span>
                        <span className="lib-send-wrap">
                          <button type="button" className="quiet-button lib-send" aria-haspopup="menu" aria-expanded={menu === d.version}
                                  onClick={() => setMenu(menu === d.version ? null : d.version)}>Send</button>
                          {menu === d.version && <SendMenu ds={d} hf={hf} rp={rp} device={device} onClose={() => setMenu(null)} onJob={onJob} />}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {c.datasets.at(-1)?.split?.notes?.length > 0 && (
                  <p className="train-dim">{c.datasets.at(-1).version}: {c.datasets.at(-1).split.notes.join("; ")}</p>
                )}
                <div className="train-row train-row-split">
                  <span className="train-dim">{c.datasets.length ? `${nextV} would add ${c.new_since_last ?? 0} sample${c.new_since_last === 1 ? "" : "s"}` : "Tiles every labelled image of this class"}</span>
                  <button type="button" className="quiet-button" onClick={() => build(c)} disabled={jobs.some((j) => j.kind === "build" && ACTIVE(j.status))}>Build {nextV}</button>
                </div>
                {hf?.configured && (
                  <label className="lib-auto"><input type="checkbox" checked={Boolean(hf.auto_backup)} onChange={toggleAuto} /> Back up every new version to Hugging Face</label>
                )}
              </div>
              <div className="lib-sec">
                <div className="lib-sec-head"><span>Models</span><span className="train-dim">scored on test areas</span></div>
                {c.models.length === 0 ? <p className="train-dim">Send a dataset version to training</p> : (
                  <div className="lib-tbl">
                    <div className="lib-tr lib-th lib-mtr"><span>Version</span><span>score</span><span>dataset</span><span /></div>
                    {[...c.models].reverse().map((m) => {
                      const s = score(m);
                      return (
                        <div className="lib-tr lib-mtr" key={m.group + m.version}>
                          <span className="mono">{m.version}</span>
                          <span className="mono" title={s.l}>{s.v}<span className="train-dim lib-score-l"> {s.l}</span></span>
                          <span className="mono">{m.dataset_version || "–"}</span>
                          {m.approved
                            ? <button type="button" className="lib-pill on" onClick={() => approve(m, null)} title="Stop using it">Approved</button>
                            : <button type="button" className="lib-pill" disabled={!m.weights} onClick={() => approve(m, m.version)}>
                                {c.models.some((x) => x.approved && x.version > m.version) ? "Roll back" : "Approve"}</button>}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              {visibleJobs.length > 0 && (
                <div className="lib-jobs">
                  {visibleJobs.map((j, k) => (
                    <div key={k} className="lib-job">
                      <span>{j.label}</span>
                      <span className={`mono ${j.status === "error" ? "train-error" : "train-dim"}`}>
                        {j.status === "error" ? (j.error || "failed") : ACTIVE(j.status) || !j.status ? (j.progress != null ? `${Math.round(j.progress)}${j.kind === "train" ? "" : "%"}` : "…") : "done"}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {err && <p className="train-error">{err}</p>}
              <p className="train-dim mono lib-path">{lib.data_dir}</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
