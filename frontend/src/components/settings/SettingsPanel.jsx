import React, { useEffect, useState } from "react";
import EnvironmentPanel from "./EnvironmentPanel.jsx";
import { HfSettings, RunPodSettings } from "../training/SamplesStore.jsx";
import { BasemapGallery } from "../map/LayersPanel.jsx";
import { IconClose, IconGear, IconSun, IconMap, IconDatabase, IconRaster, IconClickAI } from "../ui/Icons.jsx";
import { EXPORTS } from "../shell/TopBar.jsx";
import { api, auth } from "../../api/client.js";
import { ACCENTS, usePref } from "../../lib/prefs.js";

// Settings: one window, a section list on the left, rows of label + control.
// Every choice applies at once and is remembered in this browser; keys and
// tokens go to the engine and are never shown again.
const SECTIONS = [
  { id: "general", label: "General", icon: IconGear },
  { id: "theme", label: "Theme", icon: IconSun },
  { id: "detection", label: "Detection", icon: IconClickAI },
  { id: "map", label: "Map", icon: IconMap },
  { id: "services", label: "Services", icon: IconDatabase },
  { id: "system", label: "System", icon: IconRaster },
];

function Seg({ value, onChange, options, label }) {
  return (
    <div className="set-seg" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} type="button" role="radio" aria-checked={value === v}
                className={value === v ? "on" : ""} onClick={() => onChange(v)}>{text}</button>
      ))}
    </div>
  );
}

function Row({ label, hint, children }) {
  return (
    <div className="set-row">
      <div className="set-row-text">
        <span className="set-row-label">{label}</span>
        {hint && <span className="set-row-hint">{hint}</span>}
      </div>
      <div className="set-row-control">{children}</div>
    </div>
  );
}

function Group({ title, children }) {
  return (
    <section className="set-group">
      {title && <h3 className="set-group-title">{title}</h3>}
      <div className="set-card">{children}</div>
    </section>
  );
}

function General() {
  const [restore, setRestore] = usePref("restore");
  const [area, setArea] = usePref("area");
  const [about, setAbout] = useState({ up: null, dataDir: null });
  useEffect(() => {
    auth.engineUp().then((up) => setAbout((a) => ({ ...a, up })));
    api.library().then((l) => setAbout((a) => ({ ...a, dataDir: l.data_dir }))).catch(() => {});
  }, []);
  return (
    <>
      <Group title="Startup">
        <Row label="Reopen last project" hint="Image, models, area and map view">
          <Seg label="Reopen last project" value={restore} onChange={setRestore} options={[["on", "On"], ["off", "Off"]]} />
        </Row>
      </Group>
      <Group title="Units">
        <Row label="Area">
          <Seg label="Area unit" value={area} onChange={setArea} options={[["metric", "km² · m²"], ["ha", "ha"]]} />
        </Row>
      </Group>
      <Group title="About">
        <Row label="Engine">
          <span className="set-readout">
            <span className={`auth-dot ${about.up ? "up" : about.up === false ? "down" : ""}`} aria-hidden="true" />
            {about.up == null ? "…" : about.up ? "Online" : "Offline"}
          </span>
        </Row>
        <Row label="Data folder"><span className="set-readout mono set-path" title={about.dataDir || ""}>{about.dataDir || "—"}</span></Row>
        <Row label="Version"><span className="set-readout mono">0.1.0</span></Row>
      </Group>
    </>
  );
}

function Theme() {
  const [theme, setTheme] = usePref("theme");
  const [accent, setAccent] = usePref("accent");
  const [motion, setMotion] = usePref("motion");
  return (
    <>
      <Group title="Appearance">
        <Row label="Mode">
          <Seg label="Theme mode" value={theme} onChange={setTheme} options={[["dark", "Dark"], ["light", "Light"], ["system", "System"]]} />
        </Row>
        <Row label="Accent" hint="Selection and key figures">
          <div className="set-swatches" role="radiogroup" aria-label="Accent colour">
            {Object.entries(ACCENTS).map(([k, a]) => (
              <button key={k} type="button" role="radio" aria-checked={accent === k} title={a.label} aria-label={a.label}
                      className={`set-swatch ${accent === k ? "on" : ""}`} onClick={() => setAccent(k)} style={{ "--sw-dark": a.dark, "--sw-light": a.light }} />
            ))}
          </div>
        </Row>
      </Group>
      <Group title="Interface">
        <Row label="Motion">
          <Seg label="Motion" value={motion} onChange={setMotion} options={[["full", "Full"], ["reduced", "Reduced"]]} />
        </Row>
      </Group>
      <div className="set-preview" aria-hidden="true">
        <span className="set-preview-bar"><span /><span /><span /></span>
        <span className="set-preview-row on"><span className="mono">214</span> ships</span>
        <span className="set-preview-row"><span className="mono">0.31 m</span> GSD</span>
        <span className="set-preview-btn">Run</span>
      </div>
    </>
  );
}

function Detection() {
  const [conf, setConf] = usePref("conf");
  const [tile, setTile] = usePref("tile");
  const [overlap, setOverlap] = usePref("overlap");
  const [gsd, setGsd] = usePref("gsdmatch");
  const [exp, setExp] = usePref("export");
  return (
    <>
      <Group title="Run">
        <Row label="Minimum score" hint="Model uses each model's own default">
          <select className="set-select mono" value={conf} onChange={(e) => setConf(e.target.value)} aria-label="Minimum score">
            <option value="model">Model</option>
            {["0.1", "0.2", "0.25", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </Row>
        <Row label="Match training scale" hint="Tiles are read at the GSD the model learned, when known">
          <Seg label="Match training scale" value={gsd} onChange={setGsd} options={[["on", "On"], ["off", "Off"]]} />
        </Row>
        <Row label="Tile size" hint="When the scale is not matched">
          <Seg label="Tile size" value={tile} onChange={setTile} options={[["auto", "Auto"], ["512", "512"], ["640", "640"], ["1024", "1024"], ["2048", "2048"]]} />
        </Row>
        <Row label="Tile overlap">
          <Seg label="Tile overlap" value={overlap} onChange={setOverlap} options={[["0.1", "10%"], ["0.2", "20%"], ["0.3", "30%"]]} />
        </Row>
      </Group>
      <Group title="Export">
        <Row label="Default format" hint="Listed first in File">
          <select className="set-select" value={exp} onChange={(e) => setExp(e.target.value)} aria-label="Default export format">
            {EXPORTS.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
          </select>
        </Row>
      </Group>
    </>
  );
}

function MapSection() {
  const [basemap, setBasemap] = usePref("basemap");
  return (
    <Group title="Basemap">
      <div className="set-pad"><BasemapGallery basemap={basemap} onBasemap={setBasemap} /></div>
    </Group>
  );
}

function Services() {
  return (
    <>
      <Group title="Hugging Face">
        <div className="set-pad"><HfSettings /></div>
      </Group>
      <Group title="RunPod">
        <div className="set-pad"><RunPodSettings /></div>
      </Group>
    </>
  );
}

function System() {
  return (
    <Group title="Python environment">
      <div className="set-env"><EnvironmentPanel /></div>
    </Group>
  );
}

const BODY = { general: General, theme: Theme, detection: Detection, map: MapSection, services: Services, system: System };

export default function SettingsPanel({ open, onClose }) {
  const [tab, setTab] = useState(() => { try { return sessionStorage.getItem("syntra_settings_tab") || "general"; } catch { return "general"; } });
  useEffect(() => { try { sessionStorage.setItem("syntra_settings_tab", tab); } catch { /* storage unavailable */ } }, [tab]);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  const Body = BODY[tab] || General;
  const current = SECTIONS.find((s) => s.id === tab) || SECTIONS[0];

  return (
    <div className="set-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="set-window" role="dialog" aria-modal="true" aria-label="Settings">
        <nav className="set-nav" aria-label="Settings sections">
          <span className="set-nav-title">Settings</span>
          {SECTIONS.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" className={`set-nav-item ${tab === id ? "on" : ""}`}
                    aria-current={tab === id ? "page" : undefined} onClick={() => setTab(id)}>
              <Icon /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="set-main">
          <header className="set-head">
            <h2>{current.label}</h2>
            <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
          </header>
          <div className="set-body"><Body /></div>
        </div>
      </div>
    </div>
  );
}
