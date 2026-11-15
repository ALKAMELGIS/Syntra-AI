import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api/client.js";
import { classColor } from "./AnnotationTool.jsx";
import { IconClose } from "./Icons.jsx";

// The sample library, beside the map toolbar. It shows the data folder as it
// is, by class: samples (classes/<class>/labels), areas, dataset versions
// (datasets/<class>/vNNN) and models (models/<class>/vNNN, approved.json).
// Each dataset version shows three steps: saved here, backed up to Hugging
// Face, trained (here or on a RunPod GPU). Only the owner can act on them.

const ACTIVE = (s) => s && !["completed", "done", "error", "canceled", "interrupted"].includes(s);

function score(m) {
  if (m.test && m.test.f1 != null) return { v: m.test.f1.toFixed(2), l: "F1 test" };
  if (m.val_map50 != null) return { v: m.val_map50.toFixed(2), l: "mAP50 val" };
  return { v: "–", l: m.weights ? "" : "training" };
}

// One dataset version as three steps: saved here -> backed up -> trained.
// Done steps say what happened; the next step is the only button.
function VersionPipeline({ ds, models, hf, rp, jobs, onUpload, onTrainHere, onTrainGpu, owner, open, onToggle }) {
  const model = models.filter((m) => m.group === ds.group && m.dataset_version === ds.version).at(-1);
  const running = jobs.find((j) => j.version === ds.version && j.group === ds.group && (j.status == null || ACTIVE(j.status)));
  const uploading = running && running.kind === "push";
  const training = running && (running.kind === "train" || running.kind === "gpu");
  const noJob = !ds.export_job_id;
  const sc = model ? score(model) : null;
  return (
    <div className="ver">
      <button type="button" className="ver-head" onClick={onToggle} aria-expanded={open}>
        <span className="mono ver-v">{ds.version}</span>
        <span className="mono train-dim">{ds.counts?.train?.images ?? "–"} train · {ds.counts?.val?.images ?? "–"} val · {ds.counts?.test?.images ?? "–"} test</span>
      </button>
      <div className="pipe">
        <div className="step done"><b>Saved</b><span className="mono">on this PC</span></div>
        {ds.pushed ? (
          <div className="step done"><b>Backup</b><span className="mono" title={ds.pushed.url}>{ds.pushed.folder}</span></div>
        ) : (
          <div className={`step ${uploading ? "busy" : "next"}`}>
            <b>Backup</b>
            <span className="mono">{uploading ? `uploading ${Math.round(running.progress || 0)}%` : !hf?.configured ? "token in Settings" : "Hugging Face"}</span>
            {!uploading && owner && <button type="button" className="step-btn" onClick={() => onUpload(ds)} disabled={!hf?.configured || noJob}>Upload</button>}
          </div>
        )}
        {model ? (
          <div className="step done"><b>Trained</b><span className="mono">model {model.version}{sc && sc.v !== "–" ? ` · ${sc.l.split(" ")[0]} ${sc.v}` : ""}</span></div>
        ) : (
          <div className={`step ${training ? "busy" : ds.pushed ? "next" : "wait"}`}>
            <b>Train</b>
            <span className="mono">{training ? (running.label || "training") : ds.pushed ? (rp?.configured ? "here or on a GPU" : "here · GPU: RunPod key") : "GPU: back up first"}</span>
            {!training && owner && (
              <span className="step-btns">
                <button type="button" className="step-btn" onClick={() => onTrainHere(ds)} disabled={noJob}>Here</button>
                <button type="button" className="step-btn" onClick={() => onTrainGpu(ds)} disabled={noJob || !ds.pushed || !rp?.configured}>GPU</button>
              </span>
            )}
          </div>
        )}
      </div>
      {open && (
        <div className="ver-detail">
          <div><span>Images</span><span className="mono">{ds.images}</span></div>
          <div><span>Split</span><span className="mono">{ds.split?.method || "–"}{ds.split?.block_tiles ? ` · ${ds.split.block_tiles}×${ds.split.block_tiles} tiles, gap ${ds.split.buffer_tiles}` : ""}</span></div>
          {ds.split?.notes?.length > 0 && <div><span>Note</span><span>{ds.split.notes.join("; ")}</span></div>}
          <div><span>Folder</span><span className="mono">datasets/{ds.group}/{ds.version}</span></div>
          {ds.pushed && <div><span>Hugging Face</span><a className="mono" href={ds.pushed.url} target="_blank" rel="noreferrer">{ds.pushed.folder}</a></div>}
          {noJob && <div><span /><span className="train-dim">Built before the Library: build a new version to back it up or train it</span></div>}
        </div>
      )}
    </div>
  );
}

export default function LibraryPanel({ open, onClose, device = "cpu", onModelsChanged }) {
  const [lib, setLib] = useState(null);
  const [sel, setSel] = useState(null);
  const [hf, setHf] = useState(null);
  const [rp, setRp] = useState(null);
  const [openVer, setOpenVer] = useState(null);
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
    const entry = { kind: "gpu", label, version: ds.version, group: ds.group };
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
      update({ kind: "train", id: job.job_id, status: job.status, label: "training on a GPU" });
    } catch (e) {
      update({ status: "error", error: e.detail?.message || e.detail || e.message });
    }
  }

  async function upload(ds) {
    setErr(null);
    try {
      const r = await api.pushSamples(ds.export_job_id);
      setJobs((l) => [{ kind: "push", id: r.job_id, label: `Uploading ${ds.version}`, version: ds.version, group: ds.group, status: "queued" }, ...l]);
    } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
  }
  async function trainHere(ds) {
    setErr(null);
    try {
      const st = await api.exportDatasetStatus(ds.export_job_id);
      const r = await api.startTraining({ data_yaml: st.data_yaml, device });
      setJobs((l) => [{ kind: "train", id: r.job_id, label: `training here`, version: ds.version, group: ds.group, status: "queued" }, ...l]);
    } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
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
    try { await api.libraryApprove(m.group, version); load(); onModelsChanged?.(); } catch (e) { setErr(e.detail?.message || e.detail || e.message); }
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
                <div className="lib-sec-head"><span>Datasets</span><span className="train-dim">saved · backed up · trained</span></div>
                {c.datasets.length > 0 && (
                  <div className="vers">
                    {[...c.datasets].reverse().map((d) => (
                      <VersionPipeline key={d.group + d.version} ds={d} models={c.models} hf={hf} rp={rp} jobs={jobs} owner={lib.owner}
                        open={openVer === d.group + d.version} onToggle={() => setOpenVer(openVer === d.group + d.version ? null : d.group + d.version)}
                        onUpload={upload} onTrainHere={trainHere} onTrainGpu={(x) => startGpu(x, `GPU · ${x.version}`)} />
                    ))}
                  </div>
                )}
                <div className="train-row train-row-split">
                  <span className="train-dim">{c.datasets.length ? `${nextV} would add ${c.new_since_last ?? 0} sample${c.new_since_last === 1 ? "" : "s"}` : "Tiles every labelled image of this class"}</span>
                  {lib.owner && <button type="button" className="quiet-button" onClick={() => build(c)} disabled={jobs.some((j) => j.kind === "build" && ACTIVE(j.status))}>Build {nextV}</button>}
                </div>
                {hf?.configured && lib.owner && (
                  <label className="lib-auto"><input type="checkbox" checked={Boolean(hf.auto_backup)} onChange={toggleAuto} /> Back up every new version to Hugging Face</label>
                )}
              </div>
              <div className="lib-sec">
                <div className="lib-sec-head"><span>Models</span><span className="train-dim">scored on test areas</span></div>
                {c.models.length === 0 ? <p className="train-dim">Train a dataset version above</p> : (
                  <div className="lib-tbl">
                    <div className="lib-tr lib-th lib-mtr"><span>Version</span><span>score</span><span>dataset</span><span /></div>
                    {[...c.models].reverse().map((m) => {
                      const s = score(m);
                      return (
                        <div className="lib-tr lib-mtr" key={m.group + m.version}>
                          <span className="mono">{m.version}</span>
                          <span className="mono" title={s.l}>{s.v}<span className="train-dim lib-score-l"> {s.l}</span></span>
                          <span className="mono">{m.dataset_version || "–"}</span>
                          {!lib.owner ? <span className={`lib-pill ${m.approved ? "on" : ""}`}>{m.approved ? "Approved" : ""}</span> : m.approved
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
