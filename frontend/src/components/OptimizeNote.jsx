import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";
import InfoNote from "./InfoNote.jsx";

// A large stripped GeoTIFF is rewritten as tiles with overviews in the background
// after upload (backend app/engine/optimize.py). One line while it runs; when it
// is done the image is fetched again so the raster check shows the new layout.
export default function OptimizeNote({ image, onRefreshed }) {
  const [job, setJob] = useState(null);
  const id = image?.optimize_job_id;

  useEffect(() => {
    if (!id) { setJob(null); return; }
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
  if (job.status === "error" || job.status === "interrupted") {
    return <InfoNote level="warning" title="Couldn't optimise this file for fast reading" detail={job.error || ""} />;
  }
  return (
    <InfoNote level="info" title={`Optimising for fast reading · ${Math.round(job.progress || 0)}%`}
      detail="The file is being rewritten as tiles with overviews, like a Cloud-Optimized GeoTIFF, so the map and the models read it quickly. You can keep working meanwhile." />
  );
}
