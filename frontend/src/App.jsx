import React, { useEffect, useMemo, useRef, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import MapView from "./components/MapView.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import PerformancePanel from "./components/PerformancePanel.jsx";
import { IconMap, IconSliders } from "./components/Icons.jsx";
import { api } from "./api/client.js";

const PALETTE = ["#5b8cff", "#b48ead", "#f5a623", "#ff5d6c", "#c678dd", "#56c7d6", "#e8a87c", "#a8e6cf"];

const TERMINAL = new Set(["completed", "error", "interrupted"]);

function jobKey(imageId, modelId) {
  return `${imageId}::${modelId}`;
}

export const PROJECT_KEY = "syntra_project";

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
  const [analysisOpen, setAnalysisOpenRaw] = useState(false);
  // Train panel: one flyout beside the map toolbar at a time (Analysis or Train)
  const [trainOpen, setTrainOpenRaw] = useState(false);
  const [perfOpen, setPerfOpenRaw] = useState(false);
  const [perfModel, setPerfModel] = useState(null);
  function setPerfOpen(v) {
    setPerfOpenRaw((prev) => {
      const next = typeof v === "function" ? v(prev) : v;
      if (next) { setAnalysisOpenRaw(false); setTrainOpenRaw(false); setDrawMode(null); setMobilePane("map"); }
      return next;
    });
  }
  const [trainStep, setTrainStep] = useState(0);
  const [annotations, setAnnotations] = useState([]);
  const [extraClasses, setExtraClasses] = useState([]);
  const [activeClass, setActiveClass] = useState(null);
  const [drawMode, setDrawMode] = useState(null);
  const [labelError, setLabelError] = useState(null);
  function setAnalysisOpen(v) {
    setAnalysisOpenRaw((prev) => {
      const next = typeof v === "function" ? v(prev) : v;
      if (next) { setTrainOpenRaw(false); setPerfOpenRaw(false); }
      return next;
    });
  }
  function setTrainOpen(v) {
    setTrainOpenRaw((prev) => {
      const next = typeof v === "function" ? v(prev) : v;
      if (next) { setAnalysisOpenRaw(false); setPerfOpenRaw(false); setMobilePane("map"); }
      if (!next) setDrawMode(null);
      return next;
    });
  }
  function openTrain(step = 0) { setTrainStep(step); setTrainOpen(true); }
  const pollRef = useRef(null);

  // The working session survives a refresh or a browser crash. Only small
  // identifiers are kept here (image id, selected models, AOI, filter); the
  // image, its results and running jobs live on the server and are fetched
  // back. The same record is what File > Save project downloads.
  const restoring = useRef(true);
  useEffect(() => {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(PROJECT_KEY) || "null"); } catch { saved = null; }
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

  // Restore recent prediction jobs after a page reload or server restart.
  // The server keeps them in its job store; the newest job per
  // (image, model) pair wins, and finished ones get their layer back.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let recent = [];
      try {
        recent = await api.listJobs("predict", 20);
      } catch {
        return; // older backend without /api/jobs: start empty, as before
      }
      const restored = {};
      for (const j of recent) {
        if (!j.image_id) continue;
        const key = jobKey(j.image_id, j.model_id);
        if (!restored[key]) restored[key] = j; // list is newest first
      }
      if (cancelled || Object.keys(restored).length === 0) return;
      setJobs((prev) => ({ ...restored, ...prev }));
      setAnalysisOpen(true);

      const newest = recent.find((j) => j.image_id);
      api.getImage(newest.image_id).then((img) => !cancelled && setImage((cur) => cur || img)).catch(() => {});

      for (const [key, j] of Object.entries(restored)) {
        if (j.status !== "completed") continue;
        api.getGeojson(j.job_id, j.model_id)
          .then((gj) => !cancelled && setGeojsonByKey((prev) => ({ ...prev, [key]: gj })))
          .catch(() => {});
      }
    })();
    return () => { cancelled = true; };
  }, []);

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

    setAnalysisOpen(true);
    for (const img of validImages) {
      for (const modelId of modelIds) {
        const key = jobKey(img.image_id, modelId);
        try {
          const started = await api.startPrediction({
            image_id: img.image_id,
            model_id: modelId,
            gpu_indices: gpuIndices,
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
  }, [image?.image_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // training labels for the current image
  useEffect(() => {
    if (!image?.image_id) { setAnnotations([]); return; }
    api.listAnnotations(image.image_id).then((r) => setAnnotations(r.annotations || [])).catch(() => setAnnotations([]));
  }, [image?.image_id]);
  const trainClasses = [...new Set([...annotations.map((a) => a.class_name), ...extraClasses])].filter((c) => !c.startsWith("__"));
  useEffect(() => {
    if (!activeClass && trainClasses.length) setActiveClass(trainClasses[0]);
  }, [trainClasses.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps

  async function saveLabel(latlngs) {
    if (!image?.image_id || !activeClass) return;
    setLabelError(null);
    try {
      const saved = await api.addAnnotation(image.image_id, {
        class_name: activeClass, points_lonlat: latlngs.map((p) => [p.lng, p.lat]),
      });
      setAnnotations((list) => [...list, saved]);
    } catch (err) {
      setLabelError(err.detail?.message || err.message);
    }
  }
  async function deleteLabel(id) {
    try {
      await api.deleteAnnotation(image.image_id, id);
      setAnnotations((list) => list.filter((a) => a.id !== id));
    } catch (err) { setLabelError(err.message); }
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
    .filter(([key]) => geojsonByKey[key])
    .map(([key, job]) => ({
      key,
      version: minScore,
      // Legend shows the model only; the source file is listed once under the legend.
      name: models.find((m) => m.model_id === job.model_id)?.name || job.model_id,
      color: colorFor(job.model_id),
      geojson: filtered[key],
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
            onClearAoi={() => setAoi(null)}
          />
          <MapView
            layers={mapLayers}
            detectionsVisible={detectionsVisible}
            onToggleDetections={() => setDetectionsVisible((v) => !v)}
            aoi={aoi}
            onAoiChange={setAoi}
            measurement={measurement}
            onMeasurementChange={setMeasurement}
            trainOpen={trainOpen}
            onToggleTrain={() => setTrainOpen((v) => !v)}
            onAnnotate={() => openTrain(trainClasses.length ? 1 : 0)}
            labels={{
              visible: trainOpen,
              mode: trainOpen && trainStep === 1 ? drawMode : null,
              className: activeClass, annotations,
              onDrawn: saveLabel, onShapeClick: deleteLabel,
            }}
            perfOpen={perfOpen}
            onTogglePerf={() => setPerfOpen((v) => !v)}
            perfPanel={<PerformancePanel open={perfOpen} onClose={() => setPerfOpen(false)} preferModelId={perfModel}
              image={image} models={models} checkAreas={annotations.filter((a) => a.class_name === "__check_area__").length}
              onChecked={refreshModels} />}
            trainPanel={
              <TrainPanel
                open={trainOpen} onClose={() => setTrainOpen(false)}
                step={trainStep} onStep={(i) => { setDrawMode(null); setTrainStep(Math.max(0, Math.min(4, i))); }}
                image={image} onImageChange={setImage}
                annotations={annotations} classes={trainClasses}
                onAddClass={(c) => { setExtraClasses((l) => (l.includes(c) ? l : [...l, c])); setActiveClass(c); }}
                activeClass={activeClass} onActiveClass={setActiveClass}
                mode={drawMode} onMode={setDrawMode}
                onClearAnnotations={clearLabels} labelError={labelError}
                device={devices?.gpu_count > 0 ? "cuda:0" : "cpu"}
                onModelsChanged={refreshModels}
                onCheckModel={(id) => { refreshModels(); setPerfModel(id); setPerfOpen(true); }}
                onRunModel={(id) => { refreshModels(); setSelectedModels((s) => (s.includes(id) ? s : [...s, id])); handleRun([image], [id]); }}
              />
            }
            onSelectLibraryImage={setImage}
            currentImageId={image?.image_id}
            sourceImage={image}
            analysisOpen={analysisOpen}
            onToggleAnalysis={() => setAnalysisOpen((v) => !v)}
            analysisCount={analysisEntries.length}
            analysisPanel={
              <AnalysisPanel entries={analysisEntries} colorFor={colorFor} onClose={() => setAnalysisOpen(false)}
                minScore={minScore} onMinScore={setMinScore} shownCount={shownCount} />
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
