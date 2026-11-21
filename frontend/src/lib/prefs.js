// Per-browser preferences: one place to read, write and apply them.
// Older keys (syntra_theme, syntra_icon_style, syntra_basemap) are kept so
// existing choices carry over. Every change fires "syntra-prefs" so open
// screens update at once.
import { useEffect, useState } from "react";

const KEYS = {
  theme: ["syntra_theme", "dark"],          // dark | light | system
  accent: ["syntra_accent", "graphite"],   // quiet by default: no hue in the chrome
  motion: ["syntra_motion", "full"],        // full | reduced
  basemap: ["syntra_basemap", "satellite"],
  restore: ["syntra_restore", "on"],        // on | off: reopen the last project
  area: ["syntra_area_unit", "metric"],     // metric | ha
  conf: ["syntra_conf", "model"],           // model | 0.1 … 0.9: minimum score a detection must reach
  tile: ["syntra_tile", "auto"],            // auto (engine default) | 512 | 640 | 1024 | 2048 px
  overlap: ["syntra_overlap", "0.2"],       // share of a tile that neighbours overlap
  gsdmatch: ["syntra_gsd_match", "on"],     // read tiles at the model's training scale when known
  export: ["syntra_export", "geojson"],     // the format File lists first
};

// Options for a detection run, from Settings > Detection
export function detectionOptions() {
  const o = { match_gsd: read("gsdmatch") !== "off", overlap: Number(read("overlap")) || 0.2 };
  const c = read("conf");
  if (c !== "model" && isFinite(Number(c))) o.confidence_threshold = Number(c);
  const t = read("tile");
  if (t !== "auto" && isFinite(Number(t))) o.tile_size = Number(t);
  return o;
}

// Each accent has a dark-theme and a light-theme shade, both 4.5:1 on their surface
export const ACCENTS = {
  graphite: { label: "Graphite", dark: "#c3c8d0", light: "#33383f" },
  blue: { label: "Blue", dark: "#6b9bff", light: "#2f62d0" },
  teal: { label: "Teal", dark: "#4cc0b0", light: "#0f7c70" },
  amber: { label: "Amber", dark: "#e0a43b", light: "#9a6300" },
  rose: { label: "Rose", dark: "#ee7f95", light: "#b8344f" },
};

export function getPref(name) {
  const [key, def] = KEYS[name];
  try { return localStorage.getItem(key) || def; } catch { return def; }
}

export function setPref(name, value) {
  const [key] = KEYS[name];
  try { localStorage.setItem(key, value); } catch { /* private window: this session only */ }
  memory[name] = value;
  applyAppearance();
  window.dispatchEvent(new CustomEvent("syntra-prefs", { detail: { name, value } }));
}

const memory = {};
const read = (name) => memory[name] ?? getPref(name);

export function usePref(name) {
  const [value, setValue] = useState(() => read(name));
  useEffect(() => {
    const on = (e) => { if (e.detail?.name === name) setValue(e.detail.value); };
    window.addEventListener("syntra-prefs", on);
    return () => window.removeEventListener("syntra-prefs", on);
  }, [name]);
  return [value, (v) => setPref(name, v)];
}

const darkQuery = () => window.matchMedia?.("(prefers-color-scheme: dark)");

export function resolvedTheme() {
  const t = read("theme");
  if (t === "system") return darkQuery()?.matches === false ? "light" : "dark";
  return t === "light" ? "light" : "dark";
}

export function applyAppearance() {
  const root = document.documentElement;
  const mode = resolvedTheme();
  root.classList.toggle("theme-light", mode === "light");
  root.classList.toggle("reduce-motion", read("motion") === "reduced");
  const a = ACCENTS[read("accent")] || ACCENTS.graphite;
  const hex = a[mode];
  root.style.setProperty("--accent", hex);
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  root.style.setProperty("--accent-soft", `rgba(${r}, ${g}, ${b}, ${mode === "light" ? 0.10 : 0.12})`);
}

let watching = false;
export function watchSystemTheme() {
  if (watching) return;
  watching = true;
  darkQuery()?.addEventListener?.("change", () => { if (read("theme") === "system") applyAppearance(); });
}

// Areas in the chosen unit
export function fmtAreaM2(m2) {
  if (m2 == null || !isFinite(m2)) return "";
  if (read("area") === "ha") return `${(m2 / 1e4).toLocaleString("en-US", { maximumFractionDigits: m2 < 1e4 ? 3 : 1 })} ha`;
  return m2 >= 1e6 ? `${(m2 / 1e6).toFixed(2)} km²` : `${Math.round(m2).toLocaleString("en-US")} m²`;
}
export function fmtAreaKm2(km2, digits = 2) {
  if (km2 == null || !isFinite(km2)) return "";
  if (read("area") === "ha") return `${Math.round(km2 * 100).toLocaleString("en-US")} ha`;
  return `${Number(km2).toLocaleString("en-US", { maximumFractionDigits: digits })} km²`;
}
