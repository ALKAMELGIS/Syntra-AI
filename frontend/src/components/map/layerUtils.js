// Shared by the Layers panel, the object pop-up and the attribute table.
// Field names are the ones the engine writes (app/engine/postprocess.py,
// measure.py, segmentation_run.py); nothing here invents an attribute.

export const FIELD_ORDER = [
  "class", "score", "area_m2", "area_sqm", "area_ha", "perimeter_m",
  "length_m", "width_m", "orientation_deg", "tree_candidates", "lat", "lon",
];

const UNITS = {
  score: "0–1", area_m2: "m²", area_sqm: "m²", area_ha: "ha", perimeter_m: "m",
  length_m: "m", width_m: "m", orientation_deg: "° from north", lat: "WGS84", lon: "WGS84",
};

export const LAYER_COLOURS = ["#6b9bff", "#4cc38a", "#e0a43b", "#f06a6a", "#c084fc", "#2dd4bf", "#f472b6", "#e8ebf0"];

// Every field present on any feature, known ones first in a fixed order.
export function fieldsOf(features) {
  const seen = new Set();
  for (const f of features) for (const k of Object.keys(f.properties || {})) if (!k.startsWith("_")) seen.add(k);
  const known = FIELD_ORDER.filter((k) => seen.has(k));
  const rest = [...seen].filter((k) => !FIELD_ORDER.includes(k)).sort();
  return [...known, ...rest];
}

export function fieldType(features, key) {
  for (const f of features) {
    const v = f.properties?.[key];
    if (v == null) continue;
    return typeof v === "number" ? "number" : typeof v === "boolean" ? "yes / no" : "text";
  }
  return "empty";
}

export const unitOf = (key) => UNITS[key] || "";

export function fmtValue(key, v) {
  if (v == null || v === "") return "—";
  if (typeof v !== "number") return String(v);
  if (key === "lat" || key === "lon") return v.toFixed(5);
  if (key === "score") return v.toFixed(2);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  return v.toLocaleString("en-US", { maximumFractionDigits: Math.abs(v) >= 100 ? 1 : 2 });
}

// [[south, west], [north, east]] for any GeoJSON geometry, or null.
export function geometryBounds(geom) {
  let s = Infinity, w = Infinity, n = -Infinity, e = -Infinity;
  const visit = (c) => {
    if (typeof c[0] === "number") {
      const [x, y] = c;
      if (y < s) s = y; if (y > n) n = y; if (x < w) w = x; if (x > e) e = x;
    } else c.forEach(visit);
  };
  if (geom?.type === "GeometryCollection") geom.geometries.forEach((g) => g && visit(g.coordinates));
  else if (geom?.coordinates) visit(geom.coordinates);
  return isFinite(s) ? [[s, w], [n, e]] : null;
}

export function featuresBounds(features) {
  let b = null;
  for (const f of features) {
    const g = geometryBounds(f.geometry);
    if (!g) continue;
    b = b ? [[Math.min(b[0][0], g[0][0]), Math.min(b[0][1], g[0][1])], [Math.max(b[1][0], g[1][0]), Math.max(b[1][1], g[1][1])]] : g;
  }
  return b;
}

// Whether a feature's bounds touch the map view ({south, west, north, east}).
export function inView(feature, view) {
  if (!view) return true;
  const b = geometryBounds(feature.geometry);
  if (!b) return false;
  return !(b[1][0] < view.south || b[0][0] > view.north || b[1][1] < view.west || b[0][1] > view.east);
}

export function downloadGeojson(name, features) {
  const blob = new Blob([JSON.stringify({ type: "FeatureCollection", features })], { type: "application/geo+json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${name.replace(/[^\w.-]+/g, "_") || "layer"}.geojson`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
