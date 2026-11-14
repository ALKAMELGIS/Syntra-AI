import React, { useEffect, useMemo, useRef, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import MapView from "./components/MapView.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import ModelSetupWizard from "./components/ModelSetupWizard.jsx";
import { IconMap, IconSliders } from "./components/Icons.jsx";
import { api } from "./api/client.js";

const PALETTE = ["#5b8cff", "#b48ead", "#f5a623", "#ff5d6c", "#c678dd", "#56c7d6", "#e8a87c", "#a8e6cf"];

const TERMINAL = new Set(["completed", "error", "interrupted"]);

function jobKey(imageId, modelId) {
  return `${imageId}::${modelId}`;
}

export default function App() {
  const [view, setView] = useState("detect"); // "detect" | "models"
  // phones show one pane at a time: the setup panel or the map
  const [mobilePane, setMobilePane] = useState("setup");
  const [wizardStep, setWizardStep] = useState(null); // open the wizard on a given step
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
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const pollRef = useRef(null);

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

  async function handleRun(images) {
    setMobilePane("map");
    const validImages = (images || []).filter(Boolean);
    if (validImages.length === 0 || selectedModels.length === 0) return;

    setAnalysisOpen(true);
    for (const img of validImages) {
      for (const modelId of selectedModels) {
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
        const ext = format === "geojson" ? "geojson" : format === "kml" ? "kml" : "zip";
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

  const mapLayers = Object.entries(jobs)
    .filter(([key]) => geojsonByKey[key])
    .map(([key, job]) => ({
      key,
      // Legend shows the model only; the source file is listed once under the legend.
      name: models.find((m) => m.model_id === job.model_id)?.name || job.model_id,
      color: colorFor(job.model_id),
      geojson: geojsonByKey[key],
      // what the legend shows beside the name: hectares for area layers, else the feature count
      value: job.result?.summary?.vegetation_area_ha != null
        ? `${job.result.summary.vegetation_area_ha} ha`
        : String(geojsonByKey[key]?.features?.length ?? 0),
    }));

  return (
    <div className="app-shell">
      <TopBar
        view={view}
        onViewChange={(v) => { setWizardStep(null); setView(v); }}
        devices={devices}
        enabledGpus={enabledGpus}
        onToggleGpu={toggleGpu}
        hasImage={!!image}
        hasJob={hasCompletedJob}
        onExport={handleExport}
      />
      {view === "models" ? (
        <ModelSetupWizard onViewChange={(v) => { setWizardStep(null); setView(v); }} initialStep={wizardStep} />
      ) : (
        <>
        <div className={`layout pane-${mobilePane}`}>
          <Sidebar
            models={models}
            selectedModels={selectedModels}
            onToggleModel={toggleModel}
            onModelsChanged={() => api.listModels().then(setModels).catch(() => {})}
            onTrain={() => { setWizardStep("annotate"); setView("models"); }}
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
            onOpenDatasetManager={() => setView("models")}
            onSelectLibraryImage={setImage}
            currentImageId={image?.image_id}
            sourceImage={image}
            analysisOpen={analysisOpen}
            onToggleAnalysis={() => setAnalysisOpen((v) => !v)}
            analysisCount={analysisEntries.length}
            analysisPanel={
              <AnalysisPanel entries={analysisEntries} colorFor={colorFor} onClose={() => setAnalysisOpen(false)} />
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
      )}
    </div>
  );
}
