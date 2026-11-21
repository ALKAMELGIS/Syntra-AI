// The top bar: brand, File menu (open, save, export), theme toggle, compute devices and the account menu.
import React, { useEffect, useRef, useState } from "react";
import { api, NEED_CREDITS, CREDITS_CHANGED } from "../../api/client.js";
import CreditsDialog from "../billing/CreditsDialog.jsx";
import { IconFolder, IconClose, IconUpload, IconSun, IconMoon, IconSignOut, IconDatabase } from "../ui/Icons.jsx";
import { usePref, resolvedTheme } from "../../lib/prefs.js";

function useOutsideClose(ref, close) {
  useEffect(() => {
    function onDocClick(e) {
      if (ref.current && !ref.current.contains(e.target)) close();
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [ref, close]);
}

export const EXPORTS = [["geojson", "GeoJSON"], ["kml", "KML"], ["shapefile", "Shapefile"], ["gpkg", "GeoPackage"], ["csv", "CSV"]];

function FileMenu({ hasImage, hasJob, onExport }) {
  const [preferred] = usePref("export");
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
          {EXPORTS.slice().sort((a, b) => (b[0] === preferred) - (a[0] === preferred)).map(([k, label]) => (
            <button key={k} className="file-menu-item" disabled={!hasJob} onClick={() => onExport(k)}>
              Export {label}{k === preferred && <span className="file-menu-hint">Default</span>}
            </button>
          ))}
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
                <span title={`GPU ${g.index} — ${g.name}`}>
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
            <div className="device-note">New runs skip unchecked GPUs</div>
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
function AccountMenu({ user, usage, onSignOut, onCredits }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useOutsideClose(ref, () => setOpen(false));
  const runs = usage ? `${usage.total_runs} runs` : "…";
  const owner = !user || user.role === "owner" || user.is_admin;
  const badge = owner ? runs : `${Number(user.credits || 0).toLocaleString("en-US")} credits`;

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
          <span className="usage-badge-value">{badge}</span>
        </span>
      </button>
      {open && (
        <div className="account-menu" role="menu">
          <div className="account-menu-who">
            <span className="account-menu-name">{user.name}</span>
            <span className="account-menu-email mono">{user.email}</span>
            <span className="account-menu-role">{owner ? "Owner" : "Member"}</span>
          </div>
          <div className="file-menu-divider" />
          <button type="button" role="menuitem" className="file-menu-item" onClick={() => { setOpen(false); onCredits?.(); }}>
            <IconDatabase /> Credits{!owner && <span className="file-menu-hint mono">{Number(user.credits || 0).toLocaleString("en-US")}</span>}
          </button>
          <button type="button" role="menuitem" className="file-menu-item" onClick={() => { setOpen(false); onSignOut?.(); }}>
            <IconSignOut /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}

export default function TopBar({ devices, enabledGpus, onToggleGpu, hasImage, hasJob, onExport, user = null, onSignOut = null }) {
  const [themePref, setTheme] = usePref("theme");
  const theme = themePref === "system" ? resolvedTheme() : themePref;
  const [usage, setUsage] = useState(null);
  const [credits, setCredits] = useState(null);       // null | {need}
  const [me, setMe] = useState(user);

  useEffect(() => {
    api.usageSummary().then(setUsage).catch(() => setUsage(null));
  }, [hasJob]);
  // the balance on the badge: re-read after a Run starts, credits are bought or the window closes
  useEffect(() => {
    if (!user) return undefined;
    const read = () => api.billingMe().then((b) => setMe({ ...user, credits: b.credits ?? user.credits })).catch(() => {});
    read();
    const need = (e) => setCredits({ need: e.detail || {} });
    window.addEventListener(CREDITS_CHANGED, read);
    window.addEventListener(NEED_CREDITS, need);
    const q = new URLSearchParams(window.location.search).get("credits");
    if (q) {                                             // back from the checkout page
      window.history.replaceState(null, "", window.location.pathname);
      setCredits({ need: null });
    }
    return () => { window.removeEventListener(CREDITS_CHANGED, read); window.removeEventListener(NEED_CREDITS, need); };
  }, [user, hasJob]);

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
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Light theme" : "Dark theme"}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <IconSun /> : <IconMoon />}
        </button>
        <GpuMenu devices={devices} enabledGpus={enabledGpus} onToggleGpu={onToggleGpu} />
        <AccountMenu user={me || user} usage={usage} onSignOut={onSignOut} onCredits={() => setCredits({ need: null })} />
        <CreditsDialog open={!!credits} need={credits?.need} onClose={() => { setCredits(null); window.dispatchEvent(new Event(CREDITS_CHANGED)); }} />
      </div>
    </div>
  );
}
