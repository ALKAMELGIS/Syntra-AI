import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

export default function TopBar() {
  const [devices, setDevices] = useState(null);

  useEffect(() => {
    api.devices().then(setDevices).catch(() => setDevices({ gpu_count: 0, gpu: [] }));
  }, []);

  const gpuCount = devices?.gpu_count ?? 0;

  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-mark" />
        <span className="brand-name">SYNTRA AI</span>
        <span className="brand-sub">Detection Engine</span>
      </div>
      <div className="device-badge">
        <span className="device-dot" />
        {devices === null
          ? "Checking compute..."
          : gpuCount > 0
          ? `${gpuCount} GPU${gpuCount > 1 ? "s" : ""} detected`
          : "CPU only (no CUDA device)"}
      </div>
    </div>
  );
}
