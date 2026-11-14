import { useEffect, useState } from "react";
import { api } from "../api/client.js";

// A large stripped GeoTIFF is rewritten as tiles with overviews in the background
// after upload (backend app/engine/optimize.py). This hook follows that job; the
// raster check shows it in its own Layout row, so the state lives in one place.
// When it finishes, the image is fetched again so the check shows the new layout.
export function useOptimizeJob(image, onRefreshed) {
  const [job, setJob] = useState(null);
  const id = image?.optimize_job_id;

  useEffect(() => {
    if (!id) { setJob(null); return undefined; }
    let stop = false;
    const tick = async () => {
      try {
        const j = await api.job(id);
        if (stop) return;
        setJob(j);
        if (j.status === "completed") {
          const fresh = await api.getImage(image.image_id);
          onRefreshed?.({ ...fresh, optimize_job_id: undefined });
          return;
        }
        if (j.status === "error" || j.status === "interrupted") return;
      } catch { /* try again */ }
      if (!stop) setTimeout(tick, 1500);
    };
    tick();
    return () => { stop = true; };
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!id || !job || job.status === "completed") return null;
  if (job.status === "error" || job.status === "interrupted") return { failed: true, error: job.error || "" };
  return { pct: Math.round(job.progress || 0) };
}
