// The app: one project (image, models, AOI, results) shared by the sidebar, the map and its panels.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import TopBar from "./components/shell/TopBar.jsx";
import ErrorBoundary from "./components/ui/ErrorBoundary.jsx";
import Sidebar from "./components/shell/Sidebar.jsx";
import MapView from "./components/map/MapView.jsx";
import AnalysisPanel from "./components/detection/AnalysisPanel.jsx";
import TrainPanel from "./components/training/TrainPanel.jsx";
import PerformancePanel from "./components/training/PerformancePanel.jsx";
import LibraryPanel from "./components/training/LibraryPanel.jsx";
import { IconMap, IconSliders } from "./components/ui/Icons.jsx";
import { api } from "./api/client.js";
import { getPref, detectionOptions } from "./lib/prefs.js";
import L from "leaflet";
import { isArea } from "./components/labels/AnnotationTool.jsx";
import { splitRing, mergeRings } from "./lib/geomEdit.js";

const PALETTE = ["#5b8cff", "#b48ead", "#f5a623", "#ff5d6c", "#c678dd", "#56c7d6", "#e8a87c", "#a8e6cf"];

const TERMINAL = new Set(["completed", "error", "interrupted"]);

function jobKey(imageId, modelId) {
  return `${imageId}::${modelId}`;
}

export const PROJECT_KEY = "syntra_project";

function runTime(t) {
  if (t == null) return null;
  const d = new Date(typeof t === "number" ? t * 1000 : t);
  if (isNaN(d)) return null;
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export default function App({ user = null, onSignOut = null }) {
  // phones show one pane at a time: the setup panel or the map
  const [mobilePane, setMobilePane] = useState("setup");
  const [models, setModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [image, setImage] = useState(null);
  const [jobs, setJobs] = useState({}); // key -> job
  const [geojsonByKey, setGeojsonByKey] = useState({});
  const [devices, setDevices] = useState(null);
  const [enabledGpus, setEnabledGpus] = useState({});
  const [aoi, setAoi] = useState(null);
  const [measurement, setMeasurement] = useState(null);
  const [detectionsVisible, setDetectionsVisible] = useState(true);
  // One panel beside the map toolbar at a time: every flyout (analysis, train,
  // performance, library, layers, remote sensing, settings) is a value of `panel`.
  const [panel, setPanel] = useState(() => { try { return JSON.parse(localStorage.getItem("syntra_train_ui") || "{}").open ? "train" : null; } catch { return null; } });
  const [perfModel, setPerfModel] = useState(null);
  // setter for one panel: true opens it (closing any other), false closes it if it is open
  const panelSetter = (name, onOpen, onClose) => (v) => {
    const cur = panel === name;
    const next = typeof v === "function" ? v(cur) : v;
    if (next && !cur) onOpen?.();
    if (!next && cur) onClose?.();
    setPanel((p) => (next ? name : p === name ? null : p));
  };
  const analysisOpen = panel === "analysis", trainOpen = panel === "train", perfOpen = panel === "perf", libOpen = panel === "lib";
  const [trainStep, setTrainStep] = useState(() => { try { return JSON.parse(localStorage.getItem("syntra_train_ui") || "{}").step || 0; } catch { return 0; } });
  const [annotations, setAnnotations] = useState([]);
  const [extraClasses, setExtraClasses] = useState([]);
  const [activeClass, setActiveClass] = useState(null);
  const [outline, setOutline] = useState({ target: "auto", shape: "auto", pick: "large", simplify_m: 0.6, grow_m: 0 });
  const [exactSize, setExactSize] = useState(null);       // {w, h} | {s} | {r} in metres: one click places it
  const [segPreview, setSegPreview] = useState(null);
  const [drawMode, setDrawMode] = useState(null);
  const [editId, setEditId] = useState(null);           // the saved label being edited
  const [labelError, setLabelError] = useState(null);
  const setAnalysisOpen = panelSetter("analysis");
  const setTrainOpen = panelSetter("train", () => setMobilePane("map"), () => setDrawMode(null));
  const setPerfOpen = panelSetter("perf", () => { setDrawMode(null); setMobilePane("map"); });
  const setLibOpen = panelSetter("lib", () => { setDrawMode(null); setMobilePane("map"); });
  function openTrain(step = 0) { setTrainStep(step); setTrainOpen(true); }
  // the Train panel and its step come back after a page reload (the browser may reload a tab left in the background)
  useEffect(() => {
    try { localStorage.setItem("syntra_train_ui", JSON.stringify({ open: trainOpen, step: trainStep })); } catch { /* private mode */ }
  }, [trainOpen, trainStep]);
  const pollRef = useRef(null);

  // The working session survives a refresh or a browser crash. Only small
  // identifiers are kept here (image id, selected models, AOI, filter); the
  // image, its results and running jobs live on the server and are fetched
  // back. The same record is what File > Save project downloads.
  const restoring = useRef(true);
  useEffect(() => {
    let saved = null;
    try { saved = getPref("restore") === "on" ? JSON.parse(localStorage.getItem(PROJECT_KEY) || "null") : null; } catch { saved = null; }
    if (saved?.selectedModels?.length) setSelectedModels(saved.selectedModels);
    if (saved?.aoi) setAoi(saved.aoi);
    if (typeof saved?.minScore === "number") setMinScore(saved.minScore);
    if (saved?.detectionsVisible === false) setDetectionsVisible(false);
    if (saved?.image_id) {
      api.getImage(saved.image_id).then((img) => setImage(img)).catch(() => {}).finally(() => { restoring.current = false; });
    } else {
      restoring.current = false;
    }
  }, []);

  // Restore the image's own runs after a page reload or server restart: the
  // newest finished run per model on the image shown, nothing from other
  // images or older runs, and nothing the person removed (hidden on the server).
  const imageId = image?.image_id || null;
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let recent = [];
      try {
        recent = await api.listJobs("predict", imageId ? 50 : 20, imageId);
      } catch {
        return; // older backend without /api/jobs: start empty, as before
      }
      if (cancelled) return;
      if (!imageId) {                                 // first visit: open the image of the last run still in the library
        if (restoring.current) return;
        for (const id of [...new Set(recent.map((j) => j.image_id).filter(Boolean))]) {
          const img = await api.getImage(id).catch(() => null);
          if (cancelled) return;
          if (img) { setImage((cur) => cur || img); break; }
        }
        return;
      }
      const restored = {};
      for (const j of recent) {
        const key = jobKey(j.image_id, j.model_id);
        if (!restored[key]) restored[key] = j; // list is newest first
      }
      if (Object.keys(restored).length === 0) return;
      setJobs((prev) => ({ ...restored, ...prev }));   // Analysis stays closed: the person opens it
      for (const [key, j] of Object.entries(restored)) {
        if (j.status !== "completed") continue;
        api.getGeojson(j.job_id, j.model_id)
          .then((gj) => !cancelled && setGeojsonByKey((prev) => (prev[key] ? prev : { ...prev, [key]: gj })))
          .catch(() => {});
      }
    })();
    return () => { cancelled = true; };
  }, [imageId]);

  // Remove on a results layer: off the map now and after a reload (its files stay on the server)
  const removeResult = useCallback((key) => {
    const j = jobs[key];
    setJobs((prev) => { const n = { ...prev }; delete n[key]; return n; });
    setGeojsonByKey((prev) => { const n = { ...prev }; delete n[key]; return n; });
    if (j?.job_id) api.hideJob(j.job_id).catch(() => {});
  }, [jobs]);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => setModels([]));
    api.devices().then(setDevices).catch(() => setDevices({ gpu_count: 0, gpu: [], cpu0: {}, cpu1: {} }));
  }, []);

  function toggleModel(modelId) {
    setSelectedModels((sel) => (sel.includes(modelId) ? sel.filter((id) => id !== modelId) : [...sel, modelId]));
  }

  function toggleGpu(index) {
    setEnabledGpus((e) => ({ ...e, [index]: e[index] === false ? true : false }));
  }

  const gpuIndices = useMemo(() => {
    if (!devices?.gpu?.length) return null;
    const allowed = devices.gpu.map((g) => g.index).filter((i) => enabledGpus[i] !== false);
    return allowed.length > 0 ? allowed : null;
  }, [devices, enabledGpus]);

  const colorFor = useMemo(() => {
    const assigned = {};
    let next = 0;
    return (modelId) => {
      if (modelId === "vegetation_index") return "#4cc38a";   // vegetation is always green
      if (!(modelId in assigned)) {
        assigned[modelId] = PALETTE[next % PALETTE.length];
        next += 1;
      }
      return assigned[modelId];
    };
  }, [models]);

  // Poll every in-flight job until it settles.
  useEffect(() => {
    const hasInFlight = Object.values(jobs).some((j) => !TERMINAL.has(j.status));
    if (!hasInFlight) {
      clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(async () => {
      const entries = Object.entries(jobs).filter(
        ([, j]) => j.job_id && !TERMINAL.has(j.status)
      );
      for (const [key, j] of entries) {
        try {
          const latest = await api.jobStatus(j.job_id);
          setJobs((prev) => ({ ...prev, [key]: { ...prev[key], ...latest } }));
          if (latest.status === "completed") {
            const gj = await api.getGeojson(j.job_id, j.model_id);
            setGeojsonByKey((prev) => ({ ...prev, [key]: gj }));
          }
        } catch (err) {
          setJobs((prev) => ({ ...prev, [key]: { ...prev[key], status: "error", error: err.message } }));
        }
      }
    }, 1200);
    return () => clearInterval(pollRef.current);
  }, [jobs]);

  async function handleRun(images, modelIds = selectedModels) {
    setMobilePane("map");
    const validImages = (images || []).filter(Boolean);
    if (validImages.length === 0 || modelIds.length === 0) return;

    for (const img of validImages) {
      for (const modelId of modelIds) {
        const key = jobKey(img.image_id, modelId);
        try {
          const started = await api.startPrediction({
            image_id: img.image_id,
            model_id: modelId,
            gpu_indices: gpuIndices,
            ...detectionOptions(),
            ...(aoi ? { aoi } : {}),
          });
          setJobs((prev) => ({
            ...prev,
            [key]: { ...started, tiles_processed: 0, model_id: modelId, image_id: img.image_id, image_name: img.filename },
          }));
        } catch (err) {
          setJobs((prev) => ({
            ...prev,
            [key]: { status: "error", error: err.message, model_id: modelId, image_id: img.image_id, image_name: img.filename },
          }));
        }
      }
    }
  }

  async function handleExport(format) {
    const completed = Object.values(jobs).filter((j) => j.status === "completed");
    for (const j of completed) {
      try {
        const url = api.resultDownloadUrl(j.job_id, j.model_id, format);
        const res = await fetch(url);
        if (!res.ok) continue;
        const blob = await res.blob();
        const ext = { geojson: "geojson", kml: "kml", gpkg: "gpkg", csv: "csv" }[format] || "zip";
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${j.model_id}_${j.job_id.slice(0, 8)}.${ext}`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch {
        // best-effort — one failed export shouldn't block the rest
      }
    }
  }

  // An image remembered from an earlier session may predate fields the map needs
  // (the exact preview bounds): refresh it once from the library.
  useEffect(() => {
    if (image?.image_id && !image.preview_bounds_wgs84 && image.bounds_wgs84) {
      api.getImage(image.image_id).then((fresh) => fresh && setImage((cur) => (cur?.image_id === fresh.image_id ? { ...cur, ...fresh } : cur))).catch(() => {});
    }
  }, [image?.image_id]);  

  // training labels for the current image
  useEffect(() => {
    if (!image?.image_id) { setAnnotations([]); return; }
    api.listAnnotations(image.image_id).then((r) => setAnnotations(r.annotations || [])).catch(() => setAnnotations([]));
  }, [image?.image_id]);
  const trainClasses = [...new Set([...annotations.map((a) => a.class_name), ...extraClasses])].filter((c) => !c.startsWith("__"));
  useEffect(() => {
    if (!activeClass && trainClasses.length) setActiveClass(trainClasses[0]);
  }, [trainClasses.join("|")]);  

  async function saveLabel(latlngs, meta = {}) {
    const cls = meta.class_name || activeClass;
    if (!image?.image_id || !cls) return null;
    setLabelError(null);
    try {
      const saved = await api.addAnnotation(image.image_id, {
        class_name: cls, points_lonlat: latlngs.map((p) => [p.lng, p.lat]),
        shape: meta.shape || null, tool: meta.tool || null,
      });
      setAnnotations((list) => [...list, saved]);
      if (meta.quiet) return saved;                     // part of a split or merge: undone as one step
      setUndoIds((u) => [...u.slice(-49), saved.id]);
      setRecent({ id: saved.id, n: (meta.count || 1), at: Date.now() });
      return saved;
    } catch (err) {
      setLabelError(err.detail?.message || err.message);
      return null;
    }
  }
  // Split and Merge replace labels: the new ones are saved first, then the old ones
  // go. Undo removes the new ones and brings the old ones back.
  const undoOps = useRef({});
  async function replaceLabels(old, parts, tool) {
    const made = [];
    for (const [latlngs, cls] of parts) {
      const saved = await saveLabel(latlngs, { class_name: cls, tool, shape: "polygon", quiet: true });
      if (!saved) { for (const m of made) await deleteLabel(m.id); return false; }
      made.push(saved);
    }
    for (const a of old) await deleteLabel(a.id);
    const key = `op:${Date.now()}`;
    undoOps.current[key] = { ids: made.map((m) => m.id), restore: old };
    setUndoIds((u) => [...u.slice(-49), key]);
    setSelectedIds([]);
    return true;
  }
  const toXY = (a) => a.points_lonlat.map(([x, y]) => ({ x, y }));
  const toLL = (ring) => ring.map((p) => L.latLng(p.y, p.x));
  async function splitLabels(line) {
    const cut = line.map((p) => ({ x: p.lng, y: p.lat }));
    const pool = selectedIds.length ? annotations.filter((a) => selectedIds.includes(a.id)) : annotations.filter((a) => !isArea(a.class_name));
    const old = [], parts = [];
    for (const a of pool) {
      const got = splitRing(toXY(a), cut);
      if (got.length) { old.push(a); got.forEach((r) => parts.push([toLL(r), a.class_name])); }
    }
    if (!old.length) { setLabelError("The line doesn't cross a label: draw it from outside to outside"); return; }
    await replaceLabels(old, parts, "split");
  }
  async function mergeLabels(ids) {
    const old = annotations.filter((a) => ids.includes(a.id));
    if (old.length < 2) return;
    const got = mergeRings(old.map(toXY));
    if (got.length !== 1) { setLabelError("These labels don't touch: merge needs them to overlap or share an edge"); return; }
    await replaceLabels(old, [[toLL(got[0]), old[0].class_name]], "merge");
  }
  // several labels selected at once (box, lasso, Shift+click): delete or change class together
  const [selectedIds, setSelectedIds] = useState([]);
  // snapping to other labels' corners and edges while drawing and editing (kept in this browser)
  const [snap, setSnap] = useState(() => { try { return localStorage.getItem("syntra_snap") !== "0"; } catch { return true; } });
  useEffect(() => { try { localStorage.setItem("syntra_snap", snap ? "1" : "0"); } catch { /* private mode */ } }, [snap]);
  useEffect(() => setSelectedIds([]), [image?.image_id]);
  async function deleteLabels(ids) {
    if (!ids.length || (ids.length > 1 && !confirm(`Delete ${ids.length} labels?`))) return;
    for (const id of ids) await deleteLabel(id);           // one at a time: each is its own request
    setSelectedIds([]);
  }
  async function reclassLabels(ids, c) {
    for (const id of ids) await updateLabel(id, { class_name: c });
  }
  // the label just added: a chip offers Undo and Delete for a few seconds
  const [recent, setRecent] = useState(null);
  useEffect(() => { if (!recent) return undefined; const t = setTimeout(() => setRecent(null), 6000); return () => clearTimeout(t); }, [recent]);
  const prepareView = (view) => image?.image_id && api.segmentPrepare(image.image_id, { view, device: devices?.gpu_count > 0 ? "cuda:0" : "cpu" }).catch(() => {});
  // Undo (Ctrl+Z or the rail): removes the labels added on this image, newest first
  const [undoIds, setUndoIds] = useState([]);
  useEffect(() => setUndoIds([]), [image?.image_id]);
  const undoable = undoIds.filter((id) => (undoOps.current[id] ? undoOps.current[id].ids.some((x) => annotations.some((a) => a.id === x)) : annotations.some((a) => a.id === id)));
  async function undoLabel() {
    const id = undoable[undoable.length - 1];
    if (!id) return;
    setUndoIds((u) => u.filter((x) => x !== id));
    const op = undoOps.current[id];
    if (!op) { deleteLabel(id); return; }
    delete undoOps.current[id];
    for (const a of op.restore) await saveLabel(a.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon)), { class_name: a.class_name, shape: a.shape, tool: a.tool, quiet: true });
    for (const x of op.ids) await deleteLabel(x);
  }
  // classes picked from the library carry their tool and outline (kept in this browser)
  const [classInfo, setClassInfo] = useState(() => { try { return JSON.parse(localStorage.getItem("syntra_class_info") || "{}"); } catch { return {}; } });
  useEffect(() => { try { localStorage.setItem("syntra_class_info", JSON.stringify(classInfo)); } catch { /* private mode */ } }, [classInfo]);
  const classShape = () => (outline.shape === "auto" ? classInfo[activeClass]?.shape || null : outline.shape);
  function addClasses(list) {
    if (!list.length) return;
    setExtraClasses((l) => [...new Set([...l, ...list.map((c) => c.name)])]);
    setClassInfo((m) => ({ ...m, ...Object.fromEntries(list.map((c) => [c.name, { tool: c.tool, shape: c.shape }])) }));
    setActiveClass(list[0].name);
  }
  async function deleteClass(c) {
    const ids = annotations.filter((a) => a.class_name === c).map((a) => a.id);
    if (ids.length && !confirm(`Delete "${c}" and its ${ids.length} labels?`)) return;
    for (const id of ids) await deleteLabel(id);
    setExtraClasses((l) => l.filter((x) => x !== c));
    setClassInfo((m) => { const n = { ...m }; delete n[c]; return n; });
    setActiveClass((a) => (a === c ? null : a));
  }
  // click to segment: the outline comes from the engine, fitted to the class's shape
  const segmentClick = (clicks, view) => api.segmentClick(image.image_id, {
    clicks, view, class_name: activeClass || "", pick: outline.pick || "large",
    shape: classShape(),
    simplify_m: outline.simplify_m, grow_m: outline.grow_m,
  });
  // every object in the view at once: suggestions the map layer shows dashed until kept
  const segmentAll = (view) => api.segmentAll(image.image_id, {
    view, class_name: activeClass || "", device: devices?.gpu_count > 0 ? "cuda:0" : "cpu",
    shape: classShape(),
    simplify_m: outline.simplify_m, grow_m: outline.grow_m,
  });
  async function saveMany(list) {
    for (const [latlngs, meta] of list) await saveLabel(latlngs, meta);
  }
  const labelling = trainOpen && trainStep === 1;
  useEffect(() => {
    // the map's own letter shortcuts (R, C, F, M, A) pause while label tools own the keys
    document.body.dataset.labelling = labelling ? "1" : "";
  }, [labelling]);

  async function deleteLabel(id) {
    try {
      await api.deleteAnnotation(image.image_id, id);
      setAnnotations((list) => list.filter((a) => a.id !== id));
      setEditId((e) => (e === id ? null : e));
    } catch (err) { setLabelError(err.message); }
  }
  // an edited outline or class, saved at once; the old one comes back if the server refuses
  async function updateLabel(id, patch) {
    if (!image?.image_id) return;
    const before = annotations.find((a) => a.id === id);
    const body = patch.latlngs ? { points_lonlat: patch.latlngs.map((p) => [p.lng, p.lat]) } : {};
    if (patch.class_name) body.class_name = patch.class_name;
    setLabelError(null);
    setAnnotations((list) => list.map((a) => (a.id === id ? { ...a, ...(body.points_lonlat ? { points_lonlat: body.points_lonlat } : {}), ...(body.class_name ? { class_name: body.class_name } : {}) } : a)));
    try {
      const saved = await api.updateAnnotation(image.image_id, id, body);
      setAnnotations((list) => list.map((a) => (a.id === id ? saved : a)));
    } catch (err) {
      if (before) setAnnotations((list) => list.map((a) => (a.id === id ? before : a)));
      setLabelError(err.detail?.message || err.detail || err.message);
    }
  }
  async function clearLabels() {
    if (!image?.image_id || !confirm("Delete every label on this image?")) return;
    try { await api.clearAnnotations(image.image_id); setAnnotations([]); } catch (err) { setLabelError(err.message); }
  }
  const refreshModels = () => api.listModels().then(setModels).catch(() => {});

  const running = Object.values(jobs).some((j) => !TERMINAL.has(j.status));
  const hasCompletedJob = Object.values(jobs).some((j) => j.status === "completed");

  const analysisEntries = Object.entries(jobs)
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([key, job]) => ({
      key,
      job,
      modelMeta: models.find((m) => m.model_id === job.model_id),
      imageName: job.image_name,
    }));

  // Results filtered by the Analysis panel's score slider (no rerun needed).
  const [minScore, setMinScore] = useState(0);
  const [focusFeature, setFocusFeature] = useState(null);    // the result being reviewed, highlighted on the map

  useEffect(() => {
    if (restoring.current) return;          // don't overwrite the saved session before it is read back
    const record = { version: 1, image_id: image?.image_id || null, image_name: image?.filename || null,
                     selectedModels, aoi, minScore, detectionsVisible, saved_at: new Date().toISOString() };
    try { localStorage.setItem(PROJECT_KEY, JSON.stringify(record)); } catch { /* private window: nothing to keep */ }
  }, [image?.image_id, selectedModels, aoi, minScore, detectionsVisible]);
  const filtered = useMemo(() => {
    const out = {};
    for (const [key, gj] of Object.entries(geojsonByKey)) {
      if (!gj?.features) { out[key] = gj; continue; }
      out[key] = minScore <= 0 ? gj : {
        ...gj, features: gj.features.filter((f) => f.properties?.score == null || f.properties.score >= minScore),
      };
    }
    return out;
  }, [geojsonByKey, minScore]);
  const shownCount = (key) => filtered[key]?.features?.length;

  const mapLayers = Object.entries(jobs)
    .filter(([key, job]) => geojsonByKey[key] && job.image_id === imageId)   // only the image on the map
    .map(([key, job]) => ({
      key,
      version: minScore,
      // Legend shows the model only; the source file is listed once under the legend.
      name: models.find((m) => m.model_id === job.model_id)?.name || job.model_id,
      color: colorFor(job.model_id),
      geojson: filtered[key],
      // for the layer's Information page
      modelId: job.model_id,
      jobId: job.job_id,                  // refine / measure on the server
      resultId: job.model_id,
      modelVersion: job.model_version || models.find((m) => m.model_id === job.model_id)?.version || null,
      imageName: job.image_name || image?.filename || null,
      ranAt: runTime(job.updated_at || job.created_at),
      // what the legend shows beside the name: hectares for area layers, else the feature count
      value: job.result?.summary?.vegetation_area_ha != null
        ? `${job.result.summary.vegetation_area_ha} ha`
        : String(filtered[key]?.features?.length ?? 0),
    }));

  return (
    <div className="app-shell">
      <TopBar
        devices={devices}
        enabledGpus={enabledGpus}
        onToggleGpu={toggleGpu}
        hasImage={!!image}
        hasJob={hasCompletedJob}
        onExport={handleExport}
        user={user}
        onSignOut={onSignOut}
      />
      <>
        <div className={`layout pane-${mobilePane}`}>
          <ErrorBoundary label="The sidebar">
          <Sidebar
            models={models}
            selectedModels={selectedModels}
            onToggleModel={toggleModel}
            onModelsChanged={refreshModels}
            onTrain={() => openTrain(0)}
            image={image}
            onImageUploaded={setImage}
            onRun={handleRun}
            running={running}
            hasAoi={!!aoi}
            aoi={aoi}
            onClearAoi={() => setAoi(null)}
          />
          </ErrorBoundary>
          <MapView
            panel={panel} onPanel={setPanel}
            layers={mapLayers}
            onRemoveResult={removeResult}
            onLabelsImported={(list) => { setAnnotations(list); setLabelError(null); }}
            detectionsVisible={detectionsVisible}
            onToggleDetections={() => setDetectionsVisible((v) => !v)}
            aoi={aoi}
            onAoiChange={setAoi}
            measurement={measurement}
            onMeasurementChange={setMeasurement}
            trainOpen={trainOpen}
            onToggleTrain={() => setTrainOpen((v) => !v)}
            labelling={labelling}
            onAnnotate={() => (labelling ? setTrainOpen(false) : openTrain(1))}   // Label: the rail takes a new class too
            labels={{
              visible: trainOpen,
              mode: trainOpen && trainStep === 1 ? drawMode : null,
              className: activeClass, annotations,
              onDrawn: saveLabel, onShapeClick: deleteLabel,
              editable: trainOpen && trainStep === 1, editId,
              onEdit: setEditId, onUpdate: (id, latlngs) => updateLabel(id, { latlngs }),
              onDeleteSelected: (x) => (Array.isArray(x) ? deleteLabels(x) : deleteLabel(x)),
              selectedIds: labelling ? selectedIds : [], onSelect: setSelectedIds,
              onSegment: segmentClick, onPreview: setSegPreview, exact: exactSize,
              onSegmentAll: segmentAll, onDrawnMany: saveMany, onPrepare: prepareView,
              snap, onSplit: splitLabels,
            }}
            labelRail={labelling ? {
              classes: trainClasses, activeClass, onActiveClass: setActiveClass,
              counts: annotations.reduce((m, a) => ({ ...m, [a.class_name]: (m[a.class_name] || 0) + 1 }), {}),
              mode: drawMode, onMode: (m) => { setDrawMode(m); setExactSize(null); if (m) setEditId(null); },
              editId, editingClass: annotations.find((a) => a.id === editId)?.class_name,
              onEditClass: (c) => editId && updateLabel(editId, { class_name: c }), onEditDelete: () => editId && deleteLabel(editId),
              onClear: clearLabels, onUndo: undoLabel, canUndo: undoable.length > 0,
              outline, onOutline: setOutline, segPreview,
              annotations, selectedIds, onSelect: setSelectedIds,
              onDeleteMany: deleteLabels, onReclassMany: reclassLabels,
              onMerge: mergeLabels, snap, onSnap: setSnap, exact: exactSize, onExact: setExactSize,
              recent: recent && annotations.some((a) => a.id === recent.id) ? recent : null,
              onDeleteRecent: () => { if (recent) { deleteLabel(recent.id); setRecent(null); } },
              onAddClass: (c) => { setExtraClasses((l) => (l.includes(c) ? l : [...l, c])); setActiveClass(c); },
            } : null}
            perfOpen={perfOpen}
            onTogglePerf={() => setPerfOpen((v) => !v)}
            libOpen={libOpen}
            onToggleLib={() => setLibOpen((v) => !v)}
            libPanel={<LibraryPanel open={libOpen} onClose={() => setLibOpen(false)} device={devices?.gpu_count > 0 ? "cuda:0" : "cpu"} onModelsChanged={refreshModels} />}
            perfPanel={<PerformancePanel open={perfOpen} onClose={() => setPerfOpen(false)} preferModelId={perfModel}
              image={image} models={models} checkAreas={annotations.filter((a) => a.class_name === "__check_area__").length}
              onChecked={refreshModels} />}
            trainPanel={
              <TrainPanel
                open={trainOpen} onClose={() => setTrainOpen(false)}
                step={trainStep} onStep={(i) => { setDrawMode(null); setEditId(null); setTrainStep(Math.max(0, Math.min(4, i))); }}
                image={image} onImageChange={setImage}
                annotations={annotations} classes={trainClasses}
                onAddClass={(c) => { setExtraClasses((l) => (l.includes(c) ? l : [...l, c])); setActiveClass(c); }}
                activeClass={activeClass} onActiveClass={setActiveClass}
                mode={drawMode} onMode={(m) => { setDrawMode(m); setExactSize(null); if (m) setEditId(null); }}
                editId={editId} onEditDone={() => setEditId(null)}
                onEditClass={(c) => editId && updateLabel(editId, { class_name: c })} onEditDelete={() => editId && deleteLabel(editId)}
                outline={outline} onOutline={setOutline}
                exact={exactSize} onExact={setExactSize} segPreview={segPreview}
                onClearAnnotations={clearLabels} labelError={labelError}
                onImported={(list) => { setAnnotations(list); setLabelError(null); }}
                aoi={aoi} onAddClasses={addClasses} onDeleteClass={deleteClass}
                device={devices?.gpu_count > 0 ? "cuda:0" : "cpu"}
                onModelsChanged={refreshModels}
                onCheckModel={(id) => { refreshModels(); setPerfModel(id); setPerfOpen(true); }}
                onRunModel={(id) => { refreshModels(); setSelectedModels((s) => (s.includes(id) ? s : [...s, id])); handleRun([image], [id]); }}
              />
            }
            onSelectLibraryImage={setImage}
            currentImageId={image?.image_id}
            sourceImage={image}
            onRemoveImage={() => { setImage(null); setEditId(null); setDrawMode(null); }}
            analysisOpen={analysisOpen}
            onToggleAnalysis={() => setAnalysisOpen((v) => !v)}
            analysisCount={analysisEntries.length}
            analysisRunning={Object.values(jobs).some((j) => j && ["queued", "processing", "starting"].includes(j.status))}
            focusFeature={focusFeature}
            analysisPanel={
              <AnalysisPanel entries={analysisEntries} colorFor={colorFor} onClose={() => setAnalysisOpen(false)}
                minScore={minScore} onMinScore={setMinScore} shownCount={shownCount} onFocus={setFocusFeature} />
            }
          />
        </div>
        <nav className="mobile-switch" aria-label="Panels">
          <button type="button" className={mobilePane === "setup" ? "active" : ""} onClick={() => setMobilePane("setup")}>
            <IconSliders /> Setup
          </button>
          <button type="button" className={mobilePane === "map" ? "active" : ""} onClick={() => setMobilePane("map")}>
            <IconMap /> Map
            {analysisEntries.length > 0 && <span className="mono mobile-switch-count">{analysisEntries.length}</span>}
          </button>
        </nav>
      </>
    </div>
  );
}
