import React, { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { api } from "../api/client";
import { MapContainer, TileLayer, WMSTileLayer, GeoJSON, ImageOverlay, Rectangle, Tooltip, Popup, useMap, useMapEvents } from "react-leaflet";
import MapDrawTools from "./MapDrawTools.jsx";
import AnnotationLayer from "./AnnotationTool.jsx";
import ImageLibrary from "./ImageLibrary.jsx";
import SettingsPanel from "./SettingsPanel.jsx";
import { usePref, getPref } from "../prefs.js";
import { IconExpand, IconBasemap, IconLayers, IconGallery, IconGear, IconAnalysis, IconTrain, IconPerformance, IconDatabase, IconZoomTo, IconTable, IconSatellite } from "./Icons.jsx";
import RemoteSensingPanel from "./RemoteSensingPanel.jsx";
import LayersPanel from "./LayersPanel.jsx";
import AttributeTable from "./AttributeTable.jsx";
import { featuresBounds, fieldsOf, fmtValue } from "./layerUtils.js";
import { BASEMAPS } from "./basemaps.js";

// Hard ceiling for how far the user can zoom in, on either basemap or on an
// uploaded image overlay. 24 matches what the user asked for and is well
// past what raw satellite/orthophoto GSD can usefully show, but Leaflet
// will happily keep upscaling (or, for the ImageOverlay, keep showing the
// full-res source image) past native tile zoom to get there.
const MAX_ZOOM = 24;

// Lives inside <MapContainer> so it can reach the Leaflet map instance via
// react-leaflet's useMap() hook -- these buttons actually drive the map,
// they are not decorative.
function MapToolbar({
  layers, sourceImage, basemap, onBasemapChange, legendOpen, onLegendToggle,
  onOpenLibrary, onOpenSettings, analysisOpen, onToggleAnalysis, analysisCount, analysisRunning,
  detectionsVisible, onToggleDetections, trainOpen, onToggleTrain, perfOpen, onTogglePerf, libOpen, onToggleLib, editMenu,
  rsOpen, onToggleRs,
}) {
  const map = useMap();
  const toolbarRef = useRef(null);

  // Clicks on the toolbar must not fall through to the map (they would pan,
  // or drop a point for an active drawing tool).
  useEffect(() => {
    if (!toolbarRef.current) return;
    L.DomEvent.disableClickPropagation(toolbarRef.current);
    L.DomEvent.disableScrollPropagation(toolbarRef.current);
  }, []);

  function zoomToFit() {
    const combined = { type: "FeatureCollection", features: layers.flatMap((l) => l.geojson?.features || []) };
    if (combined.features.length) {
      const bounds = L_boundsFromGeojson(combined);
      if (bounds) {
        map.fitBounds(bounds, { padding: [40, 40] });
        return;
      }
    }
    const b = sourceImage?.bounds_wgs84;
    if (b) {
      map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [40, 40] });
      return;
    }
    map.setView([24.7, 46.7], 5);
  }

  function toggleBasemap() {
    onBasemapChange(basemap === "none" ? "satellite" : "none");
  }

  return (
    <div className="map-toolbar" ref={toolbarRef}>
      <button className="map-tool-btn" data-tool="fit" title="Zoom to detections / reset view" onClick={zoomToFit}>
        <IconExpand />
      </button>
      <button className="map-tool-btn" data-tool="basemap" title={basemap === "none" ? "Show basemap" : "Hide basemap"} onClick={toggleBasemap}>
        <IconBasemap />
      </button>
      <button
        className={`map-tool-btn ${legendOpen ? "active" : ""}`}
        data-tool="layers" title="Layers" aria-expanded={legendOpen}
        onClick={onLegendToggle}
      >
        <IconLayers />
      </button>
      <div className="map-toolbar-sep" />
      {editMenu}
      <div className="map-toolbar-sep" />
      <button className="map-tool-btn" data-tool="library" title="Image library" onClick={onOpenLibrary}>
        <IconGallery />
      </button>
      <button className={`map-tool-btn ${trainOpen ? "active" : ""}`} data-tool="train" title="Train a model"
              aria-expanded={trainOpen} onClick={onToggleTrain}>
        <IconTrain />
      </button>
      <button className={`map-tool-btn ${libOpen ? "active" : ""}`} data-tool="samples" title="Sample library"
              aria-expanded={libOpen} onClick={onToggleLib}>
        <IconDatabase />
      </button>
      <button className={`map-tool-btn ${rsOpen ? "active" : ""}`} data-tool="rs" title="Remote sensing"
              aria-expanded={rsOpen} onClick={onToggleRs}>
        <IconSatellite />
      </button>
      <button className="map-tool-btn" data-tool="settings" title="Settings" onClick={onOpenSettings}>
        <IconGear />
      </button>
      <div className="map-toolbar-sep" />
      <button
        className={`map-tool-btn ${analysisOpen ? "active" : ""}`}
        data-tool="analysis" title={analysisRunning ? "Analysis results · running" : "Analysis results"}
        aria-expanded={analysisOpen} onClick={onToggleAnalysis}
      >
        <IconAnalysis />
        {analysisRunning ? <span className="map-tool-badge running" aria-label="Running" />
          : analysisCount > 0 && <span className="map-tool-badge">{analysisCount}</span>}
      </button>
      <button className={`map-tool-btn ${perfOpen ? "active" : ""}`} data-tool="performance" title="Model performance"
              aria-expanded={perfOpen} onClick={onTogglePerf}>
        <IconPerformance />
      </button>
    </div>
  );
}

// The GeoTIFF on the map. Zoomed out: one preview picture (at most 2048 px,
// in Web Mercator, exact corners). Zoomed in past the preview's own detail:
// full-resolution tiles cut from the file (/tiles/{z}/{x}/{y}.png), so a
// 20 000 px orthomosaic is as sharp on screen as it is on disk.
const PREVIEW_MAX_PX = 2048;
const EARTH = 2 * Math.PI * 6378137;

function SourceImage({ image, bounds, opacity = 1, display = null }) {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  const lat = (bounds.north + bounds.south) / 2;
  const cosLat = Math.cos((lat * Math.PI) / 180);
  const gsd = image.health?.gsd_m || image.gsd_x_m;
  const sidePx = Math.max(image.width || 0, image.height || 0);
  // ground metres per preview pixel; tiles take over when the map shows finer than that
  const previewRes = gsd && sidePx ? (gsd * sidePx) / Math.min(PREVIEW_MAX_PX, sidePx) : null;
  const tileZoom = previewRes && sidePx > PREVIEW_MAX_PX
    ? Math.ceil(Math.log2((EARTH * cosLat) / 256 / previewRes)) : null;
  const nativeZoom = gsd ? Math.ceil(Math.log2((EARTH * cosLat) / 256 / gsd)) : null;
  // While the background optimise job runs, the file has no overviews yet and a
  // zoomed-in tile can take seconds: stay on the preview until it finishes.
  // A band or stretch choice is drawn by the tile server, so it uses tiles at every zoom.
  const custom = Boolean(display?.bands?.length || display?.stretch);
  const useTiles = !image.optimize_job_id && (custom || (tileZoom != null && zoom >= tileZoom));
  const ll = [[bounds.south, bounds.west], [bounds.north, bounds.east]];
  // The preview stays underneath at every zoom; tiles draw over it when they
  // are finer. A tile that is still loading shows the preview, never a hole,
  // and switching between the two never blanks the image.
  return (
    <>
      <ImageOverlay key={`preview-${image.image_id}`} url={api.imagePreviewUrl(image.image_id)} bounds={ll}
                    opacity={opacity} className="syntra-raster" zIndex={9} />
      {useTiles && (
        <TileLayer
          key={`tiles-${image.image_id}-${display?.bands?.join(",") || ""}-${display?.stretch || ""}`}
          url={api.imageTileUrl(image.image_id, display)}
          bounds={ll}
          zIndex={10}
          maxZoom={MAX_ZOOM}
          maxNativeZoom={Math.min(MAX_ZOOM, nativeZoom ?? MAX_ZOOM)}
          // no requests mid-zoom (they are thrown away and look like flicker);
          // a wider ring of tiles kept around the view for smooth panning
          updateWhenZooming={false}
          keepBuffer={6}
          noWrap
          opacity={opacity}
          className="syntra-raster"
        />
      )}
    </>
  );
}

// The result under review: the map flies to it, close enough to judge it,
// centred in the part of the map the review panel does not cover, and draws
// one light outline around it (the detection layers already carry colour).
function FocusFeature({ feature }) {
  const map = useMap();
  useEffect(() => {
    if (!feature?.geometry) return;
    const halo = L.geoJSON(feature, { interactive: false, style: { className: "focus-halo", weight: 5, fill: false } });
    const line = L.geoJSON(feature, { interactive: false, style: { className: "focus-outline", weight: 2, fill: false } });
    const b = line.getBounds();
    if (b.isValid()) {
      const panel = document.querySelector(".analysis-flyout");
      const right = panel ? panel.offsetWidth + 86 : 16;          // panel + toolbar + gutters
      map.flyToBounds(b.pad(2.5), { paddingTopLeft: [16, 16], paddingBottomRight: [right, 16], maxZoom: 20, duration: 0.3 });
    }
    halo.addTo(map);
    line.addTo(map);
    return () => { map.removeLayer(halo); map.removeLayer(line); };
  }, [feature, map]);
  return null;
}

// Where the file sits on the ground, drawn whatever its pixels hold: a hairline
// frame on the image's WGS84 extent. A file with no image data (a cloud mask
// uploaded by mistake) gets the frame in the error colour and says so, so an
// empty overlay never looks like a placement failure.
function ImageFootprint({ image }) {
  const b = image?.bounds_wgs84;
  if (!b) return null;
  const empty = image.health?.findings?.some((f) => f.code === "no_data");
  const isMask = image.health?.findings?.some((f) => f.code === "no_data" && /mask/.test(f.title));
  return (
    <Rectangle
      key={`fp-${image.image_id}-${empty}`}
      bounds={[[b.south, b.west], [b.north, b.east]]}
      interactive={false}
      pathOptions={{ className: `image-footprint ${empty ? "empty" : ""}`, weight: 1, fill: false, dashArray: "6 4" }}
    >
      {empty && (
        <Tooltip permanent direction="center" className="footprint-label">
          {isMask ? "No image data · this file is a mask" : "No image data"}
        </Tooltip>
      )}
    </Rectangle>
  );
}

// Frames the newly loaded source image automatically, the same way the
// reference app snaps to a freshly imported scene — real geographic bounds
// from the backend (app.engine.raster_io.compute_wgs84_bounds), not a guess.
const VIEW_KEY = "syntra_view";
const LAYER_KEY = "syntra_image_layer";

function AutoFitSourceImage({ sourceImage }) {
  const map = useMap();
  useEffect(() => {
    const b = sourceImage?.bounds_wgs84;
    if (!b) return;
    // after a refresh, go back to where the user was on this image
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(VIEW_KEY) || "null"); } catch { saved = null; }
    if (getPref("restore") === "on" && saved?.image_id === sourceImage.image_id && Array.isArray(saved.center)) {
      map.setView(saved.center, saved.zoom, { animate: false });
    } else {
      map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [40, 40] });
    }
  }, [sourceImage?.image_id, map]);
  useMapEvents({
    moveend: () => {
      if (!sourceImage?.image_id) return;
      const c = map.getCenter();
      try {
        localStorage.setItem(VIEW_KEY, JSON.stringify({ image_id: sourceImage.image_id, center: [+c.lat.toFixed(6), +c.lng.toFixed(6)], zoom: map.getZoom() }));
      } catch { /* storage unavailable */ }
    },
  });
  return null;
}

// Minimal bounds helper so we don't need to pull in the `leaflet` module
// directly just for L.geoJSON(...).getBounds().
function L_boundsFromGeojson(geojson) {
  let minLat = Infinity, minLng = Infinity, maxLat = -Infinity, maxLng = -Infinity;
  const visit = (coords, depth) => {
    if (depth === 0) {
      const [lng, lat] = coords;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
    } else {
      coords.forEach((c) => visit(c, depth - 1));
    }
  };
  for (const f of geojson.features || []) {
    const geom = f.geometry;
    if (!geom) continue;
    const depth = geom.type === "Point" ? 0 : geom.type === "Polygon" ? 2 : 1;
    visit(geom.coordinates, depth);
  }
  if (!isFinite(minLat)) return null;
  return [[minLat, minLng], [maxLat, maxLng]];
}

const ADDED_COLOURS = ["#2dd4bf", "#f472b6", "#c084fc", "#f06a6a", "#e8ebf0"];

// One results layer on the map: its own colour, transparency and score
// filter; optional labels; a click opens the object's pop-up and selects it
// in the attribute table (unless a drawing or label tool owns the clicks).
const LABEL_LIMIT = 400;
function VectorLayer({ layer, onPick }) {
  const pick = useRef(onPick);
  pick.current = onPick;
  const a = layer.opacity;
  return (
    <GeoJSON
      key={`${layer.key}-${layer.version}-${layer.color}-${a}-${layer.minScore}-${layer.labels}`}
      data={{ type: "FeatureCollection", features: layer.shown }}
      style={() => ({ color: layer.color, weight: 1.5, opacity: a, fillColor: layer.color, fillOpacity: (layer.isArea ? 0.22 : 0.3) * a })}
      pointToLayer={(f, ll) => L.circleMarker(ll, { radius: 5, color: layer.color, weight: 1.5, opacity: a, fillColor: layer.color, fillOpacity: 0.5 * a })}
      onEachFeature={(f, lyr) => {
        if (layer.labels && layer.shown.length <= LABEL_LIMIT) {
          const p = f.properties || {};
          lyr.bindTooltip(`${p.class ?? ""}${typeof p.score === "number" ? ` ${p.score.toFixed(2)}` : ""}`,
                          { permanent: true, direction: "center", className: "map-label" });
        }
        lyr.on("click", (e) => {
          const b = document.body.dataset;
          if (b.drawing === "1" || b.labelling === "1") return;
          L.DomEvent.stopPropagation(e);
          pick.current(layer.key, f.id, e.latlng);
        });
      }}
    />
  );
}

// The rows selected in the attribute table, outlined on the map.
function SelectionLayer({ features }) {
  if (!features.length) return null;
  return (
    <GeoJSON key={features.map((f) => `${f._layer}:${f.id}`).join(",")}
             data={{ type: "FeatureCollection", features }} interactive={false}
             style={() => ({ color: "#ffffff", weight: 2.5, fill: false })}
             pointToLayer={(f, ll) => L.circleMarker(ll, { radius: 7, color: "#ffffff", weight: 2.5, fill: false, interactive: false })} />
  );
}

// Keeps the visible map extent, for "Map extent only" and "in view".
function ViewTracker({ onView }) {
  const map = useMap();
  const push = () => { const b = map.getBounds(); onView({ south: b.getSouth(), west: b.getWest(), north: b.getNorth(), east: b.getEast(), zoom: map.getZoom() }); };
  useMapEvents({ moveend: push });
  useEffect(push, []); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

export default function MapView({
  onRemoveImage, onRemoveResult, onLabelsImported,
  layers, detectionsVisible = true, onToggleDetections,
  aoi, onAoiChange, measurement, onMeasurementChange,
  trainOpen, onToggleTrain, onAnnotate, labels, trainPanel, perfOpen, onTogglePerf, perfPanel, libOpen, onToggleLib, libPanel,
  onSelectLibraryImage, currentImageId, sourceImage,
  analysisOpen, onToggleAnalysis, analysisCount, analysisRunning, analysisPanel, focusFeature = null,
}) {
  const [basemap, setBasemap] = usePref("basemap");
  const [legendOpen, setLegendOpen] = useState(false);
  const [imageLayer, setImageLayerRaw] = useState(() => {
    try { return { visible: true, opacity: 1, ...JSON.parse(localStorage.getItem(LAYER_KEY) || "{}") }; } catch { return { visible: true, opacity: 1 }; }
  });
  const setImageLayer = (v) => {
    setImageLayerRaw(v);
    try { localStorage.setItem(LAYER_KEY, JSON.stringify(v)); } catch { /* storage unavailable */ }
  };
  // A newly opened image starts fully visible with no display changes; the
  // settings belong to the image they were made on and come back with it.
  useEffect(() => {
    const id = sourceImage?.image_id;
    if (!id || imageLayer.forImage === id) return;
    setImageLayer({ visible: true, opacity: 1, contrast: 0, brightness: 0, bands: null, stretch: null, displayFor: id, forImage: id });
  }, [sourceImage?.image_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Legend and Analysis both open beside the toolbar: show one at a time.
  useEffect(() => {
    if (analysisOpen || trainOpen || perfOpen || libOpen) setLegendOpen(false);
  }, [analysisOpen, trainOpen, perfOpen, libOpen]);

  useEffect(() => {
    if (!legendOpen) return;
    const onKey = (e) => { if (e.key === "Escape") setLegendOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [legendOpen]);

  function toggleLegend() {
    setLegendOpen((open) => {
      if (!open && analysisOpen) onToggleAnalysis?.();
      if (!open && trainOpen) onToggleTrain?.();
      if (!open && perfOpen) onTogglePerf?.();
      return !open;
    });
  }
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [rsOpen, setRsOpen] = useState(false);
  const [rsOverlay, setRsOverlay] = useState(null);      // {image, bounds, opacity, stats, index}
  const [geoOverride, setGeoOverride] = useState({});    // key -> geojson re-read after values were added

  // --- Layers: per-layer state, order, removed layers, attribute table, pop-up ---
  const [map, setMap] = useState(null);
  const [layerState, setLayerState] = useState({});      // key -> {visible, opacity, color, name, minScore, labels, popups}
  const [order, setOrder] = useState([]);                // keys, top of the list first (drawn last)
  const [removed, setRemoved] = useState({});            // key -> the geojson it was removed with
  const [aoiVisible, setAoiVisible] = useState(true);
  const [tableOpen, setTableOpen] = useState(false);
  const [tableKey, setTableKey] = useState(null);
  const [selection, setSelection] = useState({});        // key -> Set of feature ids
  const [view, setView] = useState(null);
  const [popup, setPopup] = useState(null);              // {key, id, latlng}
  const [tableH, setTableH] = useState(260);
  const [added, setAdded] = useState([]);                // files the user added: {key, name, color, geojson, value, version}
  const [services, setServices] = useState([]);          // XYZ / WMS: {key, kind, url, layers, name, visible, opacity}

  const allLayers = useMemo(() => [...layers, ...added].map((l) => (geoOverride[l.key] ? { ...l, geojson: geoOverride[l.key] } : l)),
    [layers, added, geoOverride]);
  useEffect(() => {
    const keys = allLayers.map((l) => l.key);
    setOrder((o) => [...keys.filter((k) => !o.includes(k)), ...o.filter((k) => keys.includes(k))]);
  }, [allLayers]);

  const featuresByKey = useMemo(() => {
    const out = {};
    for (const l of allLayers) out[l.key] = (l.geojson?.features || []).map((f, i) => ({ ...f, id: i + 1 }));
    return out;
  }, [allLayers]);

  const vlayers = useMemo(() => order
    .map((k) => allLayers.find((l) => l.key === k))
    .filter((l) => l && l.geojson && removed[l.key] !== l.geojson)
    .map((l) => {
      const st = layerState[l.key] || {};
      const features = featuresByKey[l.key] || [];
      const minScore = st.minScore ?? 0;
      const shown = minScore > 0 ? features.filter((f) => typeof f.properties?.score !== "number" || f.properties.score >= minScore) : features;
      return {
        ...l, baseName: l.name, name: st.name || l.name, color: st.color || l.color,
        visible: st.visible ?? true, opacity: st.opacity ?? 1, minScore, labels: st.labels ?? false, popups: st.popups ?? true,
        minZoom: st.minZoom ?? null, maxZoom: st.maxZoom ?? null,
        task: st.task || null, taskError: st.taskError || null, accuracy: st.accuracy || null,
        isArea: l.key.endsWith("vegetation_index"), features, shown, count: shown.length,
        value: /ha$/.test(l.value || "") ? l.value : String(shown.length),
      };
    }), [order, allLayers, layerState, removed, featuresByKey]);

  const patchLayer = (key, p) => setLayerState((s) => ({ ...s, [key]: { ...(s[key] || {}), ...p } }));
  const moveLayer = (key, d) => setOrder((o) => {
    const vis = o.filter((k) => vlayers.some((l) => l.key === k));
    const i = vis.indexOf(key), j = i + d;
    if (i < 0 || j < 0 || j >= vis.length) return o;
    const next = [...vis]; [next[i], next[j]] = [next[j], next[i]];
    return [...next, ...o.filter((k) => !vis.includes(k))];
  });
  const removeLayer = (key) => {
    if (key.startsWith("added:")) setAdded((a) => a.filter((x) => x.key !== key));
    else onRemoveResult?.(key);                          // a run's layer: not restored after a reload either
    const l = allLayers.find((x) => x.key === key);
    if (l) setRemoved((r) => ({ ...r, [key]: l.geojson }));
    setSelection((s) => ({ ...s, [key]: new Set() }));
    if (popup?.key === key) setPopup(null);
  };
  const zoomTo = (features) => {
    const b = featuresBounds(features);
    if (map && b) map.flyToBounds(b, { padding: [48, 48], maxZoom: 19, duration: 0.4 });
  };
  const zoomNow = view?.zoom ?? null;
  const inRange = (l) => zoomNow == null || ((l.minZoom == null || zoomNow >= l.minZoom) && (l.maxZoom == null || zoomNow <= l.maxZoom));
  async function addFile(file) {
    const r = await api.vectorRead(file);
    const n = added.length;
    const key = `added:${Date.now()}`;
    const features = r.geojson?.features || [];
    setAdded((a) => [...a, { key, name: r.name, color: ADDED_COLOURS[n % ADDED_COLOURS.length], geojson: r.geojson,
                             value: String(r.count), version: 0, modelId: null, imageName: file.name,
                             ranAt: null, sourceCrs: r.source_crs, note: r.note }]);
    zoomTo(features);
  }
  // a copy of a layer as it is shown now (filter applied), with its own settings
  const duplicateLayer = (key) => {
    const l = vlayers.find((x) => x.key === key);
    if (!l) return;
    const n = added.length;
    const features = l.shown.map(({ id, ...f }) => f);
    setAdded((a) => [...a, { key: `added:${Date.now()}`, name: `${l.name} copy`, color: ADDED_COLOURS[n % ADDED_COLOURS.length],
                             geojson: { type: "FeatureCollection", features }, value: String(features.length), version: 0,
                             modelId: l.modelId, modelVersion: l.modelVersion, imageName: l.imageName, ranAt: l.ranAt }]);
  };
  // SAM 2.1 traces every object again inside its box; the result is a new layer beside the original
  async function refineLayer(key) {
    const l = vlayers.find((x) => x.key === key);
    if (!l?.jobId) return;
    const say = (task) => patchLayer(key, { task });
    try {
      patchLayer(key, { task: "Refining…", taskError: null });
      const { job_id: rid } = await api.refineResult(l.jobId, l.resultId);
      let j;
      for (;;) {
        await new Promise((r) => setTimeout(r, 800));
        j = await api.job(rid);
        if (["completed", "error", "failed", "interrupted"].includes(j.status)) break;
        if (j.total) say(`Refining ${j.done}/${j.total}`);
      }
      if (j.status !== "completed") throw new Error(j.error || "Refining stopped");
      const gj = await api.getGeojson(l.jobId, j.result_model_id);
      const st = j.stats || {};
      const n = added.length;
      setAdded((a) => [...a, { key: `added:${Date.now()}`, name: `${l.name} · refined`, color: ADDED_COLOURS[n % ADDED_COLOURS.length],
        geojson: gj, value: String(gj.features?.length ?? 0), version: 0, modelId: l.modelId, modelVersion: l.modelVersion,
        imageName: l.imageName, ranAt: l.ranAt, jobId: l.jobId, resultId: j.result_model_id,
        note: `SAM 2.1 outlines: ${st.refined ?? 0} of ${st.objects ?? 0} refined, ${st.kept ?? 0} kept from the detector`
          + (st.reasons && Object.keys(st.reasons).length ? ` (${Object.entries(st.reasons).map(([k, v]) => `${v} ${k}`).join(", ")})` : "") }]);
      say(null);
    } catch (err) {
      say(null);
      patchLayer(key, { taskError: err.detail || err.message });
    }
  }
  // a layer's polygons become training labels of one class on the open image (clipped, duplicates skipped)
  async function toLabels(key, cls) {
    const l = vlayers.find((x) => x.key === key);
    if (!l || !sourceImage?.image_id) { patchLayer(key, { taskError: "Open the image first" }); return; }
    patchLayer(key, { task: "Labelling…", taskError: null });
    try {
      const gj = { type: "FeatureCollection", features: (l.shown || []).map(({ id, ...f }) => f) };
      const file = new File([JSON.stringify(gj)], `${l.name}.geojson`, { type: "application/geo+json" });
      const r = await api.importLabels(sourceImage.image_id, file, { fixed_class: cls });
      onLabelsImported?.(r.annotations || []);
      patchLayer(key, { task: `+${r.added} labels` });
      setTimeout(() => patchLayer(key, { task: null }), 4000);
    } catch (err) {
      patchLayer(key, { task: null, taskError: err.detail || err.message });
    }
  }
  // the layer against the labels drawn inside the image's accuracy areas
  async function scoreLayer(key) {
    const l = vlayers.find((x) => x.key === key);
    if (!l?.jobId) return;
    patchLayer(key, { task: "Measuring…", taskError: null });
    try {
      const res = await api.scoreResult(l.jobId, l.resultId);
      patchLayer(key, { task: null, accuracy: { ...res, at: new Date().toLocaleString("sv").slice(0, 16) } });
    } catch (err) {
      patchLayer(key, { task: null, taskError: err.detail || err.message });
    }
  }
  const addService = (sv) => setServices((s) => [...s, { ...sv, key: `svc:${Date.now()}`, visible: true, opacity: 1 }]);
  const patchService = (key, p) => setServices((s) => s.map((x) => (x.key === key ? { ...x, ...p } : x)));
  const removeService = (key) => setServices((s) => s.filter((x) => x.key !== key));
  const openTable = (key) => { setTableKey(key); setTableOpen(true); };
  const onPick = (key, id, latlng) => {
    const l = vlayers.find((x) => x.key === key);
    setSelection((s) => ({ ...s, [key]: new Set([id]) }));
    if (tableOpen) setTableKey(key);
    if (l?.popups) setPopup({ key, id, latlng });
  };
  const selectedFeatures = useMemo(() => vlayers.filter((l) => l.visible && detectionsVisible).flatMap((l) => {
    const sel = selection[l.key];
    return sel?.size ? l.shown.filter((f) => sel.has(f.id)).map((f) => ({ ...f, _layer: l.key })) : [];
  }), [vlayers, selection, detectionsVisible]);
  const popLayer = popup && vlayers.find((l) => l.key === popup.key);
  const popFeature = popLayer && popLayer.features.find((f) => f.id === popup.id);
  const imgFilter = `contrast(${1 + (imageLayer.contrast || 0) / 100}) brightness(${1 + (imageLayer.brightness || 0) / 100})`;
  // one panel beside the toolbar at a time
  useEffect(() => {
    if (analysisOpen || trainOpen || perfOpen || legendOpen || libOpen) { setSettingsOpen(false); setRsOpen(false); }
  }, [analysisOpen, trainOpen, perfOpen, legendOpen, libOpen]);
  function toggleRs() {
    setRsOpen((o) => {
      if (!o) {
        setLegendOpen(false); setSettingsOpen(false);
        if (analysisOpen) onToggleAnalysis?.();
        if (trainOpen) onToggleTrain?.();
        if (perfOpen) onTogglePerf?.();
      }
      return !o;
    });
  }
  async function reloadLayer(l) {
    try {
      const gj = await api.getGeojson(l.jobId, l.resultId);
      setGeoOverride((g) => ({ ...g, [l.key]: gj }));
    } catch { /* the values are saved; the next refresh shows them */ }
  }
  const [iconStyle] = usePref("icons");
  const bm = BASEMAPS[basemap] || null;
  // Basemap health: tiles that never load (offline, firewall) leave the map
  // black with no explanation; say so in one line.
  const [tiles, setTiles] = useState({ ok: false, failed: 0 });
  useEffect(() => setTiles({ ok: false, failed: 0 }), [basemap]);
  const aoiLabel = measurement != null ? String(measurement) : null;
  // The preview is drawn in Web Mercator; only its own corners place it exactly.
  // (Without them the overlay is not drawn: the old lat/lon box misplaced it.)
  const overlayBounds = sourceImage?.preview_bounds_wgs84 || null;

  return (
    <div className={`map-area icon-style-${iconStyle} ${libraryOpen ? "library-open" : ""} ${tableOpen && vlayers.length ? "table-open" : ""}`}
         style={{ "--raster-filter": imgFilter, ...(tableOpen && vlayers.length ? { "--dock-h": `${tableH}px` } : {}) }}>
      <MapContainer center={[24.7, 46.7]} zoom={5} maxZoom={MAX_ZOOM} scrollWheelZoom ref={setMap}>
        {bm && <TileLayer
          key={basemap}
          attribution={bm.attribution}
          url={bm.url}
          maxZoom={MAX_ZOOM}
          maxNativeZoom={bm.maxNativeZoom}
          eventHandlers={{
            tileload: () => setTiles((t) => (t.ok ? t : { ...t, ok: true })),
            tileerror: () => setTiles((t) => ({ ...t, failed: t.failed + 1 })),
          }}
        />}
        {bm?.overlay && <TileLayer key={`${basemap}-ref`} url={bm.overlay} maxZoom={MAX_ZOOM} maxNativeZoom={bm.maxNativeZoom} zIndex={2} />}
        {overlayBounds && sourceImage?.image_id && imageLayer.visible && <SourceImage image={sourceImage} bounds={overlayBounds} opacity={imageLayer.opacity}
                       display={imageLayer.displayFor === sourceImage.image_id ? { bands: imageLayer.bands || null, stretch: imageLayer.stretch || null } : null} />}
        <ImageFootprint image={sourceImage} />
        <FocusFeature feature={focusFeature} />
        <AutoFitSourceImage sourceImage={sourceImage} />
        {detectionsVisible &&
          [...vlayers].reverse().map((l) => (l.visible && inRange(l) ? <VectorLayer key={l.key} layer={l} onPick={onPick} /> : null))}
        {services.filter((sv) => sv.visible).map((sv, i) => (sv.kind === "wms"
          ? <WMSTileLayer key={sv.key} url={sv.url} params={{ layers: sv.layers, format: "image/png", transparent: true }} opacity={sv.opacity} zIndex={20 + i} maxZoom={MAX_ZOOM} />
          : <TileLayer key={sv.key} url={sv.url} opacity={sv.opacity} zIndex={20 + i} maxZoom={MAX_ZOOM} />))}
        {rsOverlay?.image && aoi && (
          <ImageOverlay key={rsOverlay.image.length + rsOverlay.time + rsOverlay.index?.id} url={rsOverlay.image}
                        bounds={[[rsOverlay.bounds[0], rsOverlay.bounds[1]], [rsOverlay.bounds[2], rsOverlay.bounds[3]]]}
                        opacity={rsOverlay.opacity ?? 0.85} zIndex={15} className="rs-overlay" />
        )}
        <SelectionLayer features={selectedFeatures} />
        <ViewTracker onView={setView} />
        {popFeature && (
          <Popup position={popup.latlng} eventHandlers={{ remove: () => setPopup(null) }} className="feature-popup" maxWidth={320} minWidth={260} autoPanPaddingTopLeft={[64, 64]} autoPanPaddingBottomRight={[64, 64]}>
            <div className="fp-head">
              <span className="lp-swatch" style={{ borderColor: popLayer.color }} aria-hidden="true" />
              <b>{popFeature.properties?.class ?? popLayer.name}</b><span className="mono">OID {popFeature.id}</span>
            </div>
            <dl className="fp-attrs">
              {fieldsOf([popFeature]).filter((k) => k !== "class").map((k) => (
                <React.Fragment key={k}><dt>{k}</dt><dd className="mono">{fmtValue(k, popFeature.properties[k])}</dd></React.Fragment>
              ))}
            </dl>
            <div className="fp-actions">
              <button type="button" onClick={() => zoomTo([popFeature])}><IconZoomTo />Zoom to</button>
              <button type="button" onClick={() => openTable(popLayer.key)}><IconTable />Show in table</button>
            </div>
          </Popup>
        )}
        {aoi && aoiVisible && (
          <GeoJSON
            key="aoi"
            data={aoi}
            style={() => ({ color: "#ffcf5c", weight: 2, fillColor: "#ffcf5c", fillOpacity: 0.08, dashArray: "6 4" })}
          />
        )}
        <MapToolbar
          layers={layers}
          sourceImage={sourceImage}
          basemap={basemap}
          onBasemapChange={setBasemap}
          legendOpen={legendOpen}
          onLegendToggle={toggleLegend}
          onOpenLibrary={() => setLibraryOpen(true)}
          onOpenSettings={() => setSettingsOpen((o) => {
            if (!o) {
              setLegendOpen(false); setRsOpen(false);
              if (analysisOpen) onToggleAnalysis?.();
              if (trainOpen) onToggleTrain?.();
              if (perfOpen) onTogglePerf?.();
            }
            return !o;
          })}
          analysisOpen={analysisOpen}
          onToggleAnalysis={onToggleAnalysis}
          analysisCount={analysisCount}
          analysisRunning={analysisRunning}
          detectionsVisible={detectionsVisible}
          onToggleDetections={onToggleDetections}
          trainOpen={trainOpen}
          onToggleTrain={onToggleTrain}
          perfOpen={perfOpen}
          onTogglePerf={onTogglePerf}
          libOpen={libOpen}
          onToggleLib={onToggleLib}
          rsOpen={rsOpen}
          onToggleRs={toggleRs}
          editMenu={
            <MapDrawTools
              aoi={aoi}
              onAoiChange={onAoiChange}
              measurement={measurement}
              onMeasurementChange={onMeasurementChange}
              annotateOpen={false}
              onToggleAnnotate={onAnnotate}
              canAnnotate={Boolean(sourceImage?.image_id)}
              otherPanelOpen={analysisOpen || trainOpen || perfOpen || libOpen || legendOpen || settingsOpen || rsOpen}
              onMenuOpen={() => {
                setLegendOpen(false); setSettingsOpen(false); setRsOpen(false);
                if (analysisOpen) onToggleAnalysis?.();
                if (trainOpen) onToggleTrain?.();
                if (perfOpen) onTogglePerf?.();
              }}
            />
          }
        />
        <AnnotationLayer {...labels} />
      </MapContainer>
      {bm && !tiles.ok && tiles.failed >= 4 && (
        <p className="basemap-note" role="status">
          {bm.needsKey ? "Google base map needs a working key: Layers > Basemap" : `${bm.label} base map can't load: this computer can't reach its tile server`}
        </p>
      )}
      {analysisOpen && analysisPanel}
      {trainPanel}
      {perfPanel}
      {libPanel}
      <ImageLibrary
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        onSelectImage={(img) => {
          onSelectLibraryImage?.(img);
          setLibraryOpen(false);
        }}
        currentImageId={currentImageId}
        view={view}
        onImageDeleted={onRemoveImage}
      />
      <RemoteSensingPanel
        open={rsOpen} onClose={() => setRsOpen(false)} aoi={aoi} layers={vlayers}
        overlay={rsOverlay} onOverlay={setRsOverlay} onLayerValues={reloadLayer}
        onShowAoi={() => {
          if (!map || !aoi) return;
          const b = L.geoJSON(aoi).getBounds();
          if (b.isValid()) map.flyToBounds(b, { padding: [48, 48], duration: 0.4 });
        }}
        onUseView={() => {
          if (!map) return;
          const b = map.getBounds().pad(-0.08);
          const [w, s, e, n] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
          onAoiChange?.({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } });
        }}
        onZoom={() => { const b = rsOverlay?.bounds; if (map && b) map.flyToBounds([[b[0], b[1]], [b[2], b[3]]], { padding: [40, 40], duration: 0.4 }); }}
      />
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
      <LayersPanel
        open={legendOpen} onClose={() => setLegendOpen(false)}
        layers={vlayers} onPatch={patchLayer} onMove={moveLayer} onRemove={removeLayer} onZoom={zoomTo} onTable={openTable}
        onDuplicate={duplicateLayer} onRefine={refineLayer} onScore={scoreLayer} onToLabels={toLabels} onRemoveImage={onRemoveImage}
        allVisible={detectionsVisible} onToggleAll={onToggleDetections}
        aoi={aoi} aoiVisible={aoiVisible} onAoiVisible={setAoiVisible} aoiLabel={aoiLabel}
        image={sourceImage} imageLayer={imageLayer} onImageLayer={setImageLayer}
        onZoomImage={() => { const b = sourceImage?.bounds_wgs84; if (map && b) map.flyToBounds([[b.south, b.west], [b.north, b.east]], { padding: [40, 40], duration: 0.4 }); }}
        basemap={basemap} onBasemap={setBasemap}
        zoom={zoomNow} onAddFile={addFile} onAddService={addService}
        services={services} onServicePatch={patchService} onServiceRemove={removeService}
      />
      <AttributeTable
        open={tableOpen && vlayers.length > 0} layers={vlayers} activeKey={tableKey} onActive={setTableKey}
        selection={selection} onSelect={(key, set) => setSelection((s) => ({ ...s, [key]: set }))}
        view={view} onZoom={zoomTo} onClose={() => setTableOpen(false)} height={tableH} onHeight={setTableH}
      />
    </div>
  );
}
