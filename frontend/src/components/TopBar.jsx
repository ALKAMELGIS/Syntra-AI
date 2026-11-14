import React, { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";
import { IconFolder, IconClose, IconUpload, IconSun, IconMoon, IconSignOut } from "./Icons.jsx";

function useOutsideClose(ref, close) {
  useEffect(() => {
    function onDocClick(e) {
      if (ref.current && !ref.current.contains(e.target)) close();
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [ref, close]);
}

function FileMenu({ hasImage, hasJob, onExport }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutsideClose(ref, () => setOpen(false));

  function createNewProject() {
    if (confirm("Start a new project? This clears the current image and results from view.")) {
      localStorage.removeItem("syntra_project");
      window.location.reload();
    }
    setOpen(false);
  }

  function closeProject() {
    if (confirm("Close the current project? Unsaved work in this view will be lost.")) {
      window.location.reload();
    }
    setOpen(false);
  }

  function saveProject() {
    const raw = localStorage.getItem("syntra_project") || "{}";
    const blob = new Blob([raw], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "syntra-project.json";
    a.click();
    URL.revokeObjectURL(url);
    setOpen(false);
  }

  function openProjectFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        localStorage.setItem("syntra_project", reader.result);
        window.location.reload();
      } catch {
        alert("That file isn't a valid Syntra AI project.");
      }
    };
    reader.readAsText(file);
    setOpen(false);
  }

  return (
    <div className="file-menu-wrap" ref={ref}>
      <button className="file-menu-trigger" onClick={() => setOpen((o) => !o)}>
        File <span className="device-caret">▾</span>
      </button>
      {open && (
        <div className="file-menu-dropdown">
          <button className="file-menu-item" onClick={createNewProject}><IconFolder /> New project</button>
          <label className="file-menu-item">
            <IconFolder /> Open project
            <input type="file" accept="application/json" hidden onChange={openProjectFile} />
          </label>
          <div className="file-menu-divider" />
          <button className="file-menu-item" disabled={!hasImage} onClick={saveProject}><IconUpload style={{ transform: "rotate(180deg)" }} /> Save project</button>
          <button className="file-menu-item" disabled={!hasImage} onClick={closeProject}><IconClose /> Close project</button>
          <div className="file-menu-divider" />
          <button className="file-menu-item" disabled={!hasJob} onClick={() => onExport("geojson")}>
            Export GeoJSON
          </button>
          <button className="file-menu-item" disabled={!hasJob} onClick={() => onExport("kml")}>
            Export KML
          </button>
          <button className="file-menu-item" disabled={!hasJob} onClick={() => onExport("shapefile")}>
            Export Shapefile
          </button>
          <button className="file-menu-item" disabled={!hasJob} onClick={() => onExport("gpkg")}>
            Export GeoPackage
          </button>
          <button className="file-menu-item" disabled={!hasJob} onClick={() => onExport("csv")}>
            Export CSV
          </button>
        </div>
      )}
    </div>
  );
}

function GpuMenu({ devices, enabledGpus, onToggleGpu }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutsideClose(ref, () => setOpen(false));

  const gpuCount = devices?.gpu_count ?? 0;
  const activeCount = devices?.gpu?.filter((g) => enabledGpus[g.index] !== false).length ?? 0;

  return (
    <div className="device-badge-wrap" ref={ref}>
      <button className="device-badge" onClick={() => setOpen((o) => !o)}
              title={gpuCount > 0 ? `${activeCount} of ${gpuCount} GPUs in use` : "No CUDA device: runs on CPU"}>
        <span className="device-dot" />
        <span className="device-label">
          {devices === null ? "Checking…" : gpuCount > 0 ? `${activeCount}/${gpuCount} GPU` : "CPU only"}
        </span>
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
              <label className="device-row gpu-checkbox-row" key={g.index}>
                <span>
                  <input
                    type="checkbox"
                    checked={enabledGpus[g.index] !== false}
                    onChange={() => onToggleGpu(g.index)}
                  />
                  {" "}GPU {g.index} — {g.name}
                </span>
                <span className="device-tag cuda">CUDA</span>
              </label>
            ))
          ) : (
            <div className="device-row dim">
              <span>No CUDA device detected on this host — AUTO/GPU runs fall back to CPU.</span>
            </div>
          )}
          {gpuCount > 0 && (
            <div className="note" style={{ marginTop: 6 }}>
              Unchecked GPUs are excluded from AUTO/GPU device selection for new prediction runs.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function initials(user) {
  const src = (user?.name || user?.email || "S").trim();
  const parts = src.split(/[\s._@-]+/).filter(Boolean);
  return ((parts[0]?.[0] || "S") + (parts[1]?.[0] || "")).toUpperCase();
}

// Who is signed in, and the run counter. One badge; the menu holds sign out.
function AccountMenu({ user, usage, onSignOut }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutsideClose(ref, () => setOpen(false));
  const runs = usage ? `${usage.total_runs} runs` : "…";

  if (!user) {
    return (
      <div className="usage-badge" title="Real usage counter, not a payment balance">
        <span className="usage-badge-avatar" aria-hidden="true">S</span>
        <span className="usage-badge-text"><span className="usage-badge-value">{runs}</span></span>
      </div>
    );
  }
  return (
    <div className="account-wrap" ref={ref}>
      <button type="button" className="usage-badge account-trigger" onClick={() => setOpen((o) => !o)}
              aria-haspopup="menu" aria-expanded={open} aria-label={`Account: ${user.name}`}>
        <span className="usage-badge-avatar" aria-hidden="true">{initials(user)}</span>
        <span className="usage-badge-text">
          <span className="usage-badge-label">{user.name}</span>
          <span className="usage-badge-value">{runs}</span>
        </span>
      </button>
      {open && (
        <div className="account-menu" role="menu">
          <div className="account-menu-who">
            <span className="account-menu-name">{user.name}</span>
            <span className="account-menu-email mono">{user.email}</span>
            <span className="account-menu-role">{user.is_admin ? "Admin" : "Member"}</span>
          </div>
          <div className="file-menu-divider" />
          <button type="button" role="menuitem" className="file-menu-item" onClick={() => { setOpen(false); onSignOut?.(); }}>
            <IconSignOut /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}

export default function TopBar({ devices, enabledGpus, onToggleGpu, hasImage, hasJob, onExport, user = null, onSignOut = null }) {
  const [theme, setTheme] = useState(() => localStorage.getItem("syntra_theme") || "dark");
  const [usage, setUsage] = useState(null);

  useEffect(() => {
    document.documentElement.classList.toggle("theme-light", theme === "light");
    localStorage.setItem("syntra_theme", theme);
  }, [theme]);

  useEffect(() => {
    api.usageSummary().then(setUsage).catch(() => setUsage(null));
  }, [hasJob]);

  return (
    <div className="topbar">
      <div className="topbar-left">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true" />
          <span className="brand-name">Syntra AI</span>
          <span className="brand-sub">Detection Engine</span>
        </div>
        <FileMenu hasImage={hasImage} hasJob={hasJob} onExport={onExport} />
      </div>

      <div className="topbar-right">
        <button
          className="theme-toggle-btn"
          onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          title={theme === "dark" ? "Light theme" : "Dark theme"}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <IconSun /> : <IconMoon />}
        </button>
        <GpuMenu devices={devices} enabledGpus={enabledGpus} onToggleGpu={onToggleGpu} />
        <AccountMenu user={user} usage={usage} onSignOut={onSignOut} />
      </div>
    </div>
  );
}
