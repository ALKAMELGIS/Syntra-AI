import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";
import { IconClose } from "./Icons.jsx";

// Model performance, beside the map toolbar. For each model with a training
// history: the number that matters (mAP50 on held-out images), loss per epoch
// (training against validation: a widening gap means the model memorises),
// validation precision / recall / mAP50 per epoch, and mAP50 per class.
// Detection models are not scored with "accuracy"; these are their measures.

const W = 328, H = 132, L = 32, R = 72, T = 8, B = 20;

function LineChart({ epochs, series, yMax, yFmt = (v) => v.toFixed(1), label, xMax }) {
  const [hover, setHover] = useState(null);
  const pts = epochs.filter((e) => series.some((s) => e[s.key] != null));
  if (pts.length < 2) return <p className="perf-dim">{pts.length ? `Epoch ${pts[0].epoch} done · the line starts after the next` : "Waiting for the first epoch"}</p>;
  const x0 = pts[0].epoch, x1 = Math.max(pts.at(-1).epoch, xMax || 0);
  const vals = pts.flatMap((e) => series.map((s) => e[s.key]).filter((v) => v != null));
  const top = yMax ?? Math.max(...vals) * 1.08, bot = yMax != null ? 0 : Math.min(...vals) * 0.92;
  const X = (ep) => L + ((ep - x0) / Math.max(1, x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - (v - bot) / (top - bot || 1)) * (H - T - B);
  const ticks = [bot, (bot + top) / 2, top];
  // end labels: keep at least 12 px apart
  const ends = series
    .map((s) => ({ s, v: pts.filter((e) => e[s.key] != null).at(-1)?.[s.key] }))
    .filter((e) => e.v != null)
    .map((e) => ({ ...e, y: Y(e.v) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 12) ends[i].y = ends[i - 1].y + 12;

  const move = (ev) => {
    const r = ev.currentTarget.getBoundingClientRect();
    const ep = x0 + ((((ev.clientX - r.left) / r.width) * W - L) / (W - L - R)) * Math.max(1, x1 - x0);
    const near = pts.reduce((a, e) => (Math.abs(e.epoch - ep) < Math.abs(a.epoch - ep) ? e : a), pts[0]);
    setHover(near);
  };
  return (
    <div className="perf-chart-wrap">
    <svg className="perf-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}
         onMouseMove={move} onMouseLeave={() => setHover(null)}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} className="perf-grid" />
          <text x={L - 6} y={Y(t) + 4} textAnchor="end" className="perf-tick">{yFmt(t)}</text>
        </g>
      ))}
      {[x0, Math.round((x0 + x1) / 2), x1].map((ep, i) => (
        <text key={i} x={X(ep)} y={H - 4} textAnchor="middle" className="perf-tick">{ep}</text>
      ))}
      {series.map((s) => {
        const p = pts.filter((e) => e[s.key] != null);
        return (
          <path key={s.key} d={p.map((e, i) => `${i ? "L" : "M"}${X(e.epoch).toFixed(1)} ${Y(e[s.key]).toFixed(1)}`).join("")}
                className={`perf-line ${s.strong ? "strong" : ""}`} style={s.strong ? undefined : { opacity: s.opacity ?? 1 }}>
            <title>{s.label}</title>
          </path>
        );
      })}
      {ends.map(({ s, v, y }) => (
        <text key={s.key} x={W - R + 6} y={y + 4} className={`perf-end ${s.strong ? "strong" : ""}`}>
          {s.label} <tspan className="perf-end-v">{yFmt(v)}</tspan>
        </text>
      ))}
      {hover && (<>
        <line x1={X(hover.epoch)} x2={X(hover.epoch)} y1={T} y2={H - B} className="perf-cross" />
        {series.filter((s) => hover[s.key] != null).map((s) => (
          <circle key={s.key} cx={X(hover.epoch)} cy={Y(hover[s.key])} r={2.5} className={`perf-dot ${s.strong ? "strong" : ""}`} />
        ))}
      </>)}
    </svg>
    <p className="perf-readout mono" aria-live="polite">
      {hover ? `Epoch ${hover.epoch} · ${series.filter((s) => hover[s.key] != null).map((s) => `${s.label} ${yFmt(hover[s.key])}`).join(" · ")}` : "\u00a0"}
    </p>
    </div>
  );
}

function CheckResult({ check }) {
  const rows = Object.entries(check.per_class || {});
  return (
    <div className="perf-check">
      <div className="perf-hero">
        <span className="perf-figure mono">{check.overall.f1 != null ? check.overall.f1.toFixed(2) : "–"}</span>
        <span className="perf-hero-text">
          F1 on your accuracy areas
          <span className="perf-dim">{check.image_name} · {check.areas} area{check.areas === 1 ? "" : "s"} · IoU {check.iou} · {check.checked_at.slice(0, 10)}</span>
        </span>
      </div>
      <table className="perf-table mono">
        <thead><tr><th>Class</th><th>Found</th><th>Wrong</th><th>Missed</th><th>P</th><th>R</th><th>F1</th></tr></thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name}>
              <td className="perf-table-name">{name}</td><td>{r.tp}</td><td>{r.fp}</td><td>{r.fn}</td>
              <td>{r.precision ?? "–"}</td><td>{r.recall ?? "–"}</td><td className="strong">{r.f1 ?? "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {check.unmatched_label_classes?.length > 0 && (
        <p className="perf-dim">Not scored: the model has no class named {check.unmatched_label_classes.join(", ")}</p>
      )}
    </div>
  );
}

const STATUS = { done: "Finished", completed: "Finished", canceled: "Stopped", error: "Failed", interrupted: "Interrupted" };
const fmtTime = (s) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : `${Math.max(1, Math.round(s / 60))} min`);

export default function PerformancePanel({ open, onClose, preferModelId, image, models = [], checkAreas = 0, onChecked }) {
  const [list, setList] = useState(null);
  const [sel, setSel] = useState(null);
  const [perf, setPerf] = useState(null);
  const [err, setErr] = useState(null);
  const [checkModel, setCheckModel] = useState("");
  const [checking, setChecking] = useState(null);   // null | {status, pct}
  const [reload, setReload] = useState(0);
  const runnable = models.filter((m) => m.status === "ready" || m.status === "validation");

  useEffect(() => {
    if (!open) return;
    api.listPerformance().then((l) => {
      setList(l);
      const live = l.find((m) => m.live)?.model_id;
      setSel((cur) => (preferModelId && l.some((m) => m.model_id === preferModelId) ? preferModelId
        : cur && l.some((m) => m.model_id === cur) ? cur : live || l[0]?.model_id || null));
    }).catch(() => setList([]));
  }, [open, preferModelId, reload]);

  useEffect(() => {
    if (!sel) return;
    setErr(null);
    api.modelPerformance(sel).then(setPerf).catch((e) => setErr(e.message));
  }, [sel, reload]);

  useEffect(() => { if (preferModelId) setCheckModel(preferModelId); }, [preferModelId]);

  // a run still training: read it again every few seconds, so the charts grow with each epoch
  useEffect(() => {
    if (!open || !perf?.live) return;
    const t = setTimeout(() => setReload((n) => n + 1), 4000);
    return () => clearTimeout(t);
  }, [open, perf]);

  async function runCheck() {
    const id = checkModel || sel;
    if (!id || !image) return;
    setErr(null);
    setChecking({ status: "queued" });
    try {
      const r = await api.startAccuracy({ image_id: image.image_id, model_id: id });
      for (;;) {
        await new Promise((res) => setTimeout(res, 1200));
        const j = await api.accuracyStatus(r.job_id);
        if (j.status === "completed") break;
        if (j.status === "error" || j.status === "interrupted") throw new Error(j.error || "The check did not finish");
        setChecking({ status: j.status });
      }
      setChecking(null);
      setSel(id);
      setReload((n) => n + 1);
      onChecked?.();
    } catch (e) {
      setChecking(null);
      setErr(e.message);
    }
  }

  if (!open) return null;
  const last = perf?.epochs?.at(-1);
  const classes = perf?.per_class ? Object.entries(perf.per_class).sort((a, b) => b[1] - a[1]) : [];

  return (
    <section className="perf-flyout" aria-label="Model performance">
      <header className="analysis-head">
        <span className="analysis-title">Model performance</span>
        <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>

      {list && list.length === 0 && !checkAreas && (
        <p className="perf-empty">Nothing measured yet. Train a model, or draw an accuracy area in the Train panel's Label step and check any model here.</p>
      )}

      {image && checkAreas > 0 && runnable.length > 0 && (
        <div className="perf-checkbar">
          <select className="train-select" value={checkModel || sel || ""} onChange={(e) => setCheckModel(e.target.value)}>
            {runnable.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
          </select>
          <button type="button" className="quiet-button" onClick={runCheck} disabled={!!checking}>
            {checking ? "Checking…" : "Check on this image"}
          </button>
        </div>
      )}

      {list && (list.length > 1 || list[0]?.run_id) && (
        <div className="perf-models">
          {list.map((m) => (
            <button type="button" key={m.model_id} className={m.model_id === sel ? "selected" : ""} onClick={() => setSel(m.model_id)}>
              <span className="perf-model-name">{m.live && <span className="perf-live" aria-label="Training now" />}{m.name}</span>
              <span className="mono perf-dim">{m.live ? `${m.epoch}/${m.planned_epochs ?? "?"}`
                : m.f1 != null ? `F1 ${m.f1.toFixed(2)}` : m.best ? m.best.map50.toFixed(2) : "–"}</span>
            </button>
          ))}
        </div>
      )}

      {err && <p className="perf-empty">{err}</p>}

      {perf && (
        <div className="perf-body">
          {perf.checks?.length > 0 && (
            <>
              <p className="perf-label">On your images</p>
              <CheckResult check={perf.checks[0]} />
            </>
          )}
          {(perf.epochs.length > 0 || perf.live) && (<>
          {perf.checks?.length > 0 && <p className="perf-label">On held-out training images</p>}
          <div className="perf-hero">
            <span className="perf-figure mono">{perf.best ? perf.best.map50.toFixed(2) : "–"}</span>
            <span className="perf-hero-text">
              mAP50 on held-out images{perf.live ? " so far" : ""}
              <span className="perf-dim">
                {perf.run_id
                  ? `${perf.live ? "Training" : STATUS[perf.status] || perf.status} · epoch ${perf.epochs.at(-1)?.epoch ?? 0} of ${perf.planned_epochs ?? "?"}${perf.best ? ` · best ${perf.best.epoch}` : ""}`
                  : `best of ${perf.epochs.length} epochs · epoch ${perf.best?.epoch ?? "–"}`}
              </span>
            </span>
          </div>
          {perf.run_id && perf.planned_epochs > 0 && (
            <div className="perf-progress" role="progressbar" aria-valuemin={0} aria-valuemax={perf.planned_epochs}
                 aria-valuenow={perf.epochs.at(-1)?.epoch ?? 0}>
              <span style={{ width: `${Math.min(100, ((perf.epochs.at(-1)?.epoch ?? 0) / perf.planned_epochs) * 100)}%` }} />
            </div>
          )}
          {perf.error && <p className="perf-dim perf-err">{perf.error}</p>}
          {perf.test && (
            <p className="perf-dim">On the accuracy areas (never trained on): <span className="mono">F1 {perf.test.f1} · mAP50 {perf.test.map50}</span></p>
          )}

          <p className="perf-label">Loss per epoch</p>
          <LineChart epochs={perf.epochs} xMax={perf.planned_epochs} label="Training and validation loss per epoch"
                     series={[{ key: "train_loss", label: "Train", opacity: 0.55 }, { key: "val_loss", label: "Val", strong: true }]} />
          {last && last.val_loss != null && last.train_loss != null && perf.epochs.length >= 3 && (
            <p className="perf-dim">
              {last.val_loss - last.train_loss > 0.25 * last.train_loss
                ? "Validation loss sits well above training: more varied labels would help"
                : "Both losses fall together: no sign of memorising"}
            </p>
          )}

          <p className="perf-label">Validation per epoch</p>
          <LineChart epochs={perf.epochs} xMax={perf.planned_epochs} yMax={1} yFmt={(v) => v.toFixed(2)} label="Validation precision, recall and mAP50 per epoch"
                     series={[{ key: "precision", label: "P", opacity: 0.45 }, { key: "recall", label: "R", opacity: 0.7 }, { key: "map50", label: "mAP50", strong: true }]} />

          </>)}
          {classes.length > 0 && (
            <>
              <p className="perf-label">mAP50 by class</p>
              <ul className="perf-classes">
                {classes.map(([name, v]) => {
                  const hidden = perf.hidden_classes?.includes(name);
                  return (
                    <li key={name} className={hidden ? "hidden-class" : ""} title={hidden ? "Too weak: not returned in results" : ""}>
                      <span className="perf-class-name">{name}</span>
                      <span className="perf-bar"><span style={{ width: `${Math.round(v * 100)}%` }} /></span>
                      <span className="mono">{v.toFixed(2)}</span>
                    </li>
                  );
                })}
              </ul>
              {perf.hidden_classes?.length > 0 && <p className="perf-dim">Dimmed classes score under 0.2 and are not returned</p>}
            </>
          )}

          <dl className="perf-meta">
            {perf.training?.train_images != null && (<><dt>Images</dt><dd className="mono">{perf.training.train_images} train · {perf.training.val_images} held out</dd></>)}
            {perf.training?.gsd_cm != null && (<><dt>Scale</dt><dd className="mono">about {perf.training.gsd_cm} cm per pixel</dd></>)}
            {perf.licence && (<><dt>Licence</dt><dd>{perf.licence}</dd></>)}
            {perf.source && (<><dt>Source</dt><dd>{perf.source}</dd></>)}
            {perf.run_id && (<><dt>Where</dt><dd className="mono">{perf.where}{perf.device ? ` · ${perf.device}` : ""}</dd></>)}
            {perf.run_id && perf.elapsed_s != null && (<><dt>Time</dt><dd className="mono">{fmtTime(perf.elapsed_s)}</dd></>)}
          </dl>
          {perf.training?.note && <p className="perf-dim">{perf.training.note}</p>}
        </div>
      )}
    </section>
  );
}
