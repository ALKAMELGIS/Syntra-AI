// Train > Setup: where training runs, and whether that place is ready.
//   Hugging Face GPU  the engine itself, when it runs in a Space with a real GPU
//                     (ZeroGPU only lends a GPU for under a minute: too short to train)
//   RunPod GPU        a pod started for the run and stopped after it (Settings > Services key)
//   This computer     the engine on the user's own computer: its libraries, and a GPU build
// The Train step uses the choice; labelling doesn't need any of it.
import React, { useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";

const LIBS = [["torch", "PyTorch"], ["torchvision", "TorchVision"], ["ultralytics", "Ultralytics YOLO"], ["onnxruntime", "ONNX Runtime"], ["cv2", "OpenCV"]];
const DONE = ["completed", "error", "interrupted", "canceled"];

export function placeReady(where, hosting, rp) {
  if (where === "hf") return Boolean(hosting?.train_gpu);
  if (where === "runpod") return Boolean(rp?.configured);
  return Boolean(hosting) && !hosting.space_id;
}

function Option({ id, where, onWhere, title, children, ready }) {
  return (
    <button type="button" role="radio" aria-checked={where === id} className={`setup-opt ${where === id ? "on" : ""}`} onClick={() => onWhere(id)}>
      <span className="setup-opt-head">
        <span className={`setup-dot ${ready ? "ok" : ""}`} aria-hidden="true" />
        <span className="setup-opt-title">{title}</span>
        <span className="setup-opt-state">{ready ? "Ready" : "Not ready"}</span>
      </span>
      <span className="setup-opt-body">{children}</span>
    </button>
  );
}

export default function TrainSetup({ open, where, onWhere, hosting, rp, onRefresh }) {
  const [env, setEnv] = useState(null);
  const [job, setJob] = useState(null);
  const timer = useRef(null);
  const local = hosting && !hosting.space_id;

  const check = () => api.getEnvironment().then(setEnv).catch((e) => setEnv({ error: e.message }));
  useEffect(() => { if (open && where === "local" && local) check(); }, [open, where, local]);
  useEffect(() => () => clearInterval(timer.current), []);

  async function run(action, build, targets) {
    try {
      const r = await api.startEnvironmentAction(action, build, targets);
      setJob({ ...r, status: "queued" });
      clearInterval(timer.current);
      timer.current = setInterval(async () => {
        try {
          const j = await api.environmentActionStatus(r.job_id);
          setJob(j);
          if (DONE.includes(j.status)) { clearInterval(timer.current); check(); onRefresh?.(); }
        } catch { /* transient */ }
      }, 1500);
    } catch (e) { setJob({ status: "error", error: e.detail || e.message }); }
  }

  const rows = env?.results ? LIBS.map(([m, l]) => [l, env.results.find((r) => r.module === m)]).filter(([, r]) => r) : [];
  const missing = rows.filter(([, r]) => !r.installed).map(([, r]) => r.module);
  const cuda = env?.cuda;
  const running = job && !DONE.includes(job.status);
  const space = hosting?.space_id;

  return (
    <>
      <p className="train-label">Where training runs</p>
      <div className="setup-opts" role="radiogroup" aria-label="Where training runs">
        <Option id="hf" where={where} onWhere={onWhere} title="Hugging Face GPU" ready={placeReady("hf", hosting, rp)}>
          {!hosting ? "Checking…"
            : hosting.train_gpu ? <>The Space's own GPU, ready to train. <span className="mono">{hosting.gpu_name}</span></>
            : hosting.zero_gpu ? <>This Space is on ZeroGPU: a GPU lent for under a minute at a time, too short to train.
                Upgrade the Space's hardware to a T4 or L4 in its settings{space ? <> (<a href={`https://huggingface.co/spaces/${space}/settings`} target="_blank" rel="noreferrer">open</a>)</> : null}</>
            : space ? "This Space has no GPU. Pick a GPU in the Space's hardware settings"
            : "Trains in Syntra's Hugging Face Space, once its hardware is a GPU. Open Syntra there to use it"}
        </Option>
        <Option id="runpod" where={where} onWhere={onWhere} title="RunPod GPU" ready={placeReady("runpod", hosting, rp)}>
          {rp?.configured ? "A GPU started for the run and stopped after it. Billed by RunPod per hour"
            : "Add the RunPod API key in Settings > Services"}
        </Option>
        <Option id="local" where={where} onWhere={onWhere} title="This computer" ready={placeReady("local", hosting, rp)}>
          {!hosting ? "Checking…"
            : space ? "Not here: this Syntra runs on Hugging Face. Run Syntra on your computer to train on it"
            : cuda?.cuda_available ? <>Your GPU: <span className="mono">{cuda.devices?.[0]?.name || "CUDA"}</span></>
            : "Your computer's CPU, or its NVIDIA GPU once the GPU build is installed"}
        </Option>
      </div>

      {where === "local" && local && (
        <>
          <p className="train-label">Libraries on this computer</p>
          {!env ? <p className="train-dim">Checking…</p> : env.error ? <p className="train-error">{env.error}</p> : (
            <>
              <ul className="setup-libs">
                {rows.map(([l, r]) => (
                  <li key={r.module}><span className={`setup-dot ${r.installed ? "ok" : "bad"}`} aria-hidden="true" />{l}
                    <span className="mono train-dim">{r.installed ? r.detail || "installed" : "missing"}</span></li>
                ))}
                <li><span className={`setup-dot ${cuda?.cuda_available ? "ok" : ""}`} aria-hidden="true" />GPU (CUDA)
                  <span className="mono train-dim">{cuda?.cuda_available ? cuda.devices?.[0]?.name || "yes" : "CPU only"}</span></li>
              </ul>
              {!env.actions_allowed ? <p className="train-dim">Install from the computer running Syntra</p> : (
                <div className="train-row">
                  {missing.length > 0 && (
                    <button type="button" className="quiet-button" disabled={running} onClick={() => run("install", "auto", missing)}>Install missing</button>
                  )}
                  {!cuda?.cuda_available && (
                    <button type="button" className="quiet-button" disabled={running} onClick={() => run("install", "cuda", ["torch", "torchvision"])}
                            title="PyTorch built for NVIDIA GPUs: about 2.5 GB">Install GPU build</button>
                  )}
                  <button type="button" className="link-button" disabled={running} onClick={check}>Check again</button>
                </div>
              )}
              {running && <p className="train-dim mono">Installing… this can take several minutes</p>}
              {job?.status === "completed" && <p className="train-ok">Installed. Close the engine window and run start-local.bat to use it</p>}
              {job?.status === "error" && <p className="train-error">{job.error || "The install failed: see Settings > System"}</p>}
            </>
          )}
        </>
      )}
      <p className="train-dim">Labelling works anywhere: this choice is used when you train</p>
    </>
  );
}
