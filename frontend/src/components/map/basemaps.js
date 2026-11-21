// Base maps. All keyless except Google, whose tiles come through the engine
// (/api/basemaps/google, key kept on the server, display only: Google's
// policy forbids detection on or training from its tiles).
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";

export const BASEMAPS = {
  satellite: { label: "Imagery", url: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
    attribution: "&copy; Esri, Maxar, Earthstar Geographics", maxNativeZoom: 19 },
  hybrid: { label: "Hybrid", url: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
    overlay: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
    attribution: "&copy; Esri, Maxar, Earthstar Geographics", maxNativeZoom: 19 },
  google: { label: "Google", url: "/api/basemaps/google/tile/{z}/{x}/{y}", attribution: "&copy; Google",
    maxNativeZoom: 21, needsKey: true },
  streets: { label: "OSM", url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors", maxNativeZoom: 19 },
  esriStreets: { label: "Streets", url: `${ESRI}/World_Street_Map/MapServer/tile/{z}/{y}/{x}`,
    attribution: "&copy; Esri, HERE, Garmin, OpenStreetMap", maxNativeZoom: 19 },
  topo: { label: "Topographic", url: `${ESRI}/World_Topo_Map/MapServer/tile/{z}/{y}/{x}`,
    attribution: "&copy; Esri, HERE, Garmin, USGS, OpenStreetMap", maxNativeZoom: 19 },
  dark: { label: "Dark gray", url: `${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    attribution: "&copy; Esri, HERE, Garmin, OpenStreetMap", maxNativeZoom: 16 },
  light: { label: "Light gray", url: `${ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    attribution: "&copy; Esri, HERE, Garmin, OpenStreetMap", maxNativeZoom: 16 },
};

export const BASEMAP_ORDER = ["satellite", "hybrid", "google", "esriStreets", "streets", "topo", "dark", "light", "none"];

// one tile over the Gulf as the thumbnail
export function thumbUrl(k) {
  const b = BASEMAPS[k];
  if (!b) return null;
  return b.url.replace("{s}", "a").replace("{z}", "5").replace("{x}", "20").replace("{y}", "13");
}
