// Layer symbology, like ArcGIS Online's Styles: one symbol (Location), a colour per
// value of a text field (Types), or colour classes of a number field (Counts and
// amounts) with natural breaks, equal interval or quantile. Pure functions: the
// map, the legend and the Symbology page all read the same classes.

// Sequential and diverging ramps (ColorBrewer / viridis), light to dark
export const RAMPS = {
  viridis: { label: "Viridis", colors: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"] },
  greens: { label: "Greens", colors: ["#e5f5e0", "#a1d99b", "#41ab5d", "#238b45", "#00441b"] },
  blues: { label: "Blues", colors: ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08306b"] },
  ylorrd: { label: "Heat", colors: ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"] },
  purples: { label: "Purples", colors: ["#efedf5", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"] },
  rdylgn: { label: "Red to green", colors: ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"] },
};

// Distinct colours for categories (Tableau 10, readable on imagery)
export const CATEGORICAL = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"];
export const OTHER = "#8a8f98";
export const MAX_UNIQUE = 10;

export const METHODS = { jenks: "Natural breaks", equal: "Equal interval", quantile: "Quantile" };

export const DEFAULT_SYMBOL = { fillOpacity: 0.3, outline: null, width: 1.5, size: 5 };

function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rgbToHex([r, g, b]) { return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join(""); }

// n colours spread along a ramp
export function rampColors(name, n, reverse = false) {
  const cs = (RAMPS[name] || RAMPS.viridis).colors.map(hexToRgb);
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0.5 : i / (n - 1), x = t * (cs.length - 1), k = Math.min(cs.length - 2, Math.floor(x)), f = x - k;
    out.push(rgbToHex(cs[k].map((c, j) => c + (cs[k + 1][j] - c) * f)));
  }
  return reverse ? out.reverse() : out;
}

// Class upper bounds (ascending, the last = max) for `values`
export function breaks(values, method, n) {
  const v = values.filter((x) => typeof x === "number" && isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return [];
  const lo = v[0], hi = v[v.length - 1];
  if (lo === hi) return [hi];
  n = Math.max(2, Math.min(n, 9, new Set(v).size));
  if (method === "equal") return Array.from({ length: n }, (_, i) => (i === n - 1 ? hi : lo + ((hi - lo) * (i + 1)) / n));
  if (method === "quantile") {
    const out = [];
    for (let i = 1; i <= n; i++) out.push(v[Math.min(v.length - 1, Math.ceil((i * v.length) / n) - 1)]);
    return [...new Set(out)];
  }
  return jenks(v, n);
}

// Fisher-Jenks natural breaks on a sorted sample (at most 1,000 values, so it stays fast)
function jenks(sorted, n) {
  let data = sorted;
  if (data.length > 1000) { const step = data.length / 1000; data = Array.from({ length: 1000 }, (_, i) => sorted[Math.floor(i * step)]); data[999] = sorted[sorted.length - 1]; }
  const m = data.length;
  const lower = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  const vari = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(Infinity));
  for (let j = 1; j <= n; j++) { lower[1][j] = 1; vari[1][j] = 0; }
  for (let l = 2; l <= m; l++) {
    let s1 = 0, s2 = 0, w = 0, v = 0;
    for (let k = 1; k <= l; k++) {
      const i3 = l - k + 1, val = data[i3 - 1];
      s2 += val * val; s1 += val; w++;
      v = s2 - (s1 * s1) / w;
      const i4 = i3 - 1;
      if (i4 !== 0) for (let j = 2; j <= n; j++) {
        if (vari[l][j] >= v + vari[i4][j - 1]) { lower[l][j] = i3; vari[l][j] = v + vari[i4][j - 1]; }
      }
    }
    lower[l][1] = 1; vari[l][1] = v;
  }
  const out = new Array(n);
  out[n - 1] = data[m - 1];
  let k = m;
  for (let j = n; j >= 2; j--) { const id = lower[k][j] - 2; out[j - 2] = data[Math.max(0, id)]; k = lower[k][j] - 1; }
  return [...new Set(out)];
}

// The classes a style draws, for the map and the legend: [{label, color, count, test(value)}]
export function classesOf(sym, features, baseColor) {
  if (!sym || sym.type === "single" || !sym.field) return [{ label: "All", color: sym?.color || baseColor, count: features.length, test: () => true }];
  const vals = features.map((f) => f.properties?.[sym.field]);
  if (sym.type === "unique") {
    const counts = new Map();
    for (const v of vals) { const k = v == null ? "(empty)" : String(v); counts.set(k, (counts.get(k) || 0) + 1); }
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const shown = top.slice(0, MAX_UNIQUE);
    const rest = top.slice(MAX_UNIQUE);
    const out = shown.map(([k, c], i) => ({ label: k, color: sym.colors?.[k] || CATEGORICAL[i % CATEGORICAL.length], count: c,
      test: (v) => (v == null ? "(empty)" : String(v)) === k }));
    if (rest.length) {
      const keys = new Set(shown.map(([k]) => k));
      out.push({ label: `Other (${rest.length} values)`, color: OTHER, count: rest.reduce((n, [, c]) => n + c, 0),
        test: (v) => !keys.has(v == null ? "(empty)" : String(v)) });
    }
    return out;
  }
  // graduated
  const nums = vals.filter((v) => typeof v === "number" && isFinite(v));
  const b = breaks(nums, sym.method || "jenks", sym.classes || 5);
  if (!b.length) return [{ label: "No numbers", color: OTHER, count: features.length, test: () => true }];
  const colors = rampColors(sym.ramp || "viridis", b.length, sym.reverse);
  const lo = Math.min(...nums);
  const fmt = (x) => (Math.abs(x) >= 1000 ? Math.round(x).toLocaleString("en-US") : Number(x.toPrecision(3)).toString());
  const out = b.map((hi, i) => {
    const from = i === 0 ? lo : b[i - 1];
    return { label: `${fmt(from)} – ${fmt(hi)}`, color: colors[i], count: nums.filter((v) => (i === 0 ? v >= from : v > from) && v <= hi).length,
      test: (v) => typeof v === "number" && (i === 0 ? v <= hi : v > from && v <= hi) };
  });
  const empty = vals.length - nums.length;
  if (empty) out.push({ label: "No value", color: OTHER, count: empty, test: (v) => typeof v !== "number" });
  return out;
}

// Leaflet path style for one feature
export function styleFor(feature, sym, classes, baseColor, opacity, isArea) {
  const s = { ...DEFAULT_SYMBOL, ...(sym || {}) };
  const v = s.field ? feature.properties?.[s.field] : undefined;
  const c = (classes.find((k) => k.test(v)) || classes[0])?.color || baseColor;
  const fillOpacity = (sym ? s.fillOpacity : isArea ? 0.22 : 0.3) * opacity;
  return { color: s.outline || c, weight: s.width, opacity, fillColor: c, fillOpacity, radius: s.size };
}
