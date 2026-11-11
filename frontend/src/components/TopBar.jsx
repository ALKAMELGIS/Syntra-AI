import React, { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";

export default function TopBar({ view, onViewChange }) {
  const [devices, setDevices] = useState(null);
  const [open, setOpen] = useState(false);
  const badgeRef = useRef(null);

  useEffect(() => {
    api.devices().then(setDevices).catch(() => setDevices({ gpu_count: 0, gpu: [], cpu0: {}, cpu1: {} }));
  }, []);

  // Close the dropdown on an outside click.
  useEffect(() => {
    function onDocClick(e) {
      if (badgeRef.current && !badgeRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const gpuCount = devices?.gpu_count ?? 0;

  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-mark" />
        <span className="brand-name">SYNTRA AI</span>
        <span className="brand-sub">Detection Engine</span>
      </div>
      <div className="topbar-tabs">
        <button className={`tab-button ${view === "detect" ? "active" : ""}`} onClick={() => onViewChange("detect")}>
          Detect
        </button>
        <button className={`tab-button ${view === "models" ? "active" : ""}`} onClick={() => onViewChange("models")}>
          Detection Models
        </button>
      </div>
      <div className="device-badge-wrap" ref={badgeRef}>
        <button className="device-badge" onClick={() => setOpen((o) => !o)}>
          <span className="device-dot" />
          {devices === null
            ? "Checking compute..."
            : gpuCount > 0
            ? `${gpuCount} GPU${gpuCount > 1 ? "s" : ""} detected`
            : "CPU only (no CUDA device)"}
          <span className="device-caret">▾</span>
        </button>
        {open && devices && (
          <div className="device-dropdown">
            <div className="device-dropdown-title">Compute devices</div>
            <div className="device-row">
              <span>CPU0</span>
              <span className="device-tag">{devices.cpu0?.threads ?? "–"} threads</span>
            </div>
            <div className="device-row">
              <span>CPU1</span>
              <span className="device-tag">{devices.cpu1?.threads ?? "–"} threads</span>
            </div>
            {gpuCount > 0 ? (
              devices.gpu.map((g) => (
                <div className="device-row" key={g.index}>
                  <span>GPU {g.index} — {g.name}</span>
                  <span className="device-tag cuda">{g.free_vram_mb} MB free</span>
                </div>
              ))
            ) : (
              <div className="device-row dim">
                <span>No CUDA device detected on this host — AUTO falls back to CPU.</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
