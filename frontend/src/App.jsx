import React, { useEffect, useState } from "react";
import TopBar from "./components/TopBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import MapView from "./components/MapView.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import ModelsDashboard from "./components/ModelsDashboard.jsx";
import { api } from "./api/client.js";

export default function App() {
  const [view, setView] = useState("detect"); // "detect" | "models"
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState(null);
  const [image, setImage] = useState(null);
  const [job, setJob] = useState(null);
  const [geojson, setGeojson] = useState(null);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => setModels([]));
  }, []);

  // Poll the active job until it settles.
  useEffect(() => {
    if (!job?.job_id || job.status === "completed" || job.status === "error") return;
    const timer = setInterval(async () => {
      try {
        const latest = await api.jobStatus(job.job_id);
        setJob({ ...latest, job_id: job.job_id });
        if (latest.status === "completed") {
          const gj = await api.getGeojson(job.job_id, selected);
          setGeojson(gj);
        }
      } catch {
        clearInterval(timer);
      }
    }, 1200);
    return () => clearInterval(timer);
  }, [job, selected]);

  async function handleRun() {
    setGeojson(null);
    try {
      const started = await api.startPrediction({ image_path: image.path, model_id: selected });
      setJob({ ...started, tiles_processed: 0 });
    } catch (err) {
      // Most common cause: a 409 refusal because the model is not
      // production-ready (see app/api/predict.py). Surface it in the
      // Analysis panel's existing error styling rather than a jarring
      // native alert() -- reuses AnalysisPanel's job.status === "error"
      // rendering instead of adding a parallel error-display path.
      setJob({ status: "error", error: err.message });
    }
  }

  const running = job && job.status !== "completed" && job.status !== "error";
  const modelMeta = models.find((m) => m.model_id === selected);

  return (
    <div className="app-shell">
      <TopBar view={view} onViewChange={setView} />
      {view === "models" ? (
        <ModelsDashboard />
      ) : (
        <div className="layout">
          <Sidebar
            models={models}
            selected={selected}
            onToggle={setSelected}
            image={image}
            onImageUploaded={setImage}
            onRun={handleRun}
            running={!!running}
          />
          <MapView geojson={geojson} />
          <AnalysisPanel job={job} modelMeta={modelMeta} />
        </div>
      )}
    </div>
  );
}
