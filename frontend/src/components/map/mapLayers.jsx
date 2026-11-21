// What MapView draws inside the Leaflet map: the GeoTIFF, result layers, selection, focus and view tracking.
import React, { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { api } from "../../api/client";
import { TileLayer, GeoJSON, ImageOverlay, Rectangle, Tooltip, useMap, useMapEvents } from "react-leaflet";
import { getPref } from "../../lib/prefs.js";
import { classesOf, styleFor } from "./symbology.js";

// Hard ceiling for how far the user can zoom in, on either basemap or on an
// uploaded image overlay. 24 matches what the user asked for and is well
// past what raw satellite/orthophoto GSD can usefully show, but Leaflet
// will happily keep upscaling (or, for the ImageOverlay, keep showing the
// full-res source image) past native tile zoom to get there.
export const MAX_ZOOM = 24;

// The GeoTIFF on the map. Zoomed out: one preview picture (at most 2048 px,
// in Web Mercator, exact corners). Zoomed in past the preview's own detail:
// full-resolution tiles cut from the file (/tiles/{z}/{x}/{y}.png), so a
// 20 000 px orthomosaic is as sharp on screen as it is on disk.
export const PREVIEW_MAX_PX = 2048;
export const EARTH = 2 * Math.PI * 6378137;

export function SourceImage({ image, bounds, opacity = 1, display = null }) {
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
export function FocusFeature({ feature }) {
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
export function ImageFootprint({ image }) {
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
export const VIEW_KEY = "syntra_view";
export const LAYER_KEY = "syntra_image_layer";

export function AutoFitSourceImage({ sourceImage }) {
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
export function L_boundsFromGeojson(geojson) {
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

export const ADDED_COLOURS = ["#2dd4bf", "#f472b6", "#c084fc", "#f06a6a", "#e8ebf0"];

// One results layer on the map: its own colour, transparency and score
// filter; optional labels; a click opens the object's pop-up and selects it
// in the attribute table (unless a drawing or label tool owns the clicks).
export const LABEL_LIMIT = 400;
export function VectorLayer({ layer, onPick }) {
  const pick = useRef(onPick);
  pick.current = onPick;
  const a = layer.opacity;
  const sym = layer.symbology || null;
  const classes = useMemo(() => classesOf(sym, layer.shown, layer.color), [sym, layer.shown, layer.color]);
  const st = (f) => styleFor(f, sym, classes, layer.color, a, layer.isArea);
  return (
    <GeoJSON
      key={`${layer.key}-${layer.version}-${layer.color}-${a}-${layer.minScore}-${layer.labels}-${JSON.stringify(sym)}`}
      data={{ type: "FeatureCollection", features: layer.shown }}
      style={st}
      pointToLayer={(f, ll) => { const s = st(f); return L.circleMarker(ll, { ...s, fillOpacity: Math.min(1, s.fillOpacity + 0.2) }); }}
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
          const o = e.originalEvent || {};
          pick.current(layer.key, f.id, e.latlng, o.ctrlKey || o.metaKey || o.shiftKey);
        });
      }}
    />
  );
}

// The selected features (from the map or the attribute table), in the accent
// colour with a dark halo so they read on any imagery.
const accent = () => getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#6b9bff";
export function SelectionLayer({ features }) {
  if (!features.length) return null;
  const c = accent();
  const key = features.map((f) => `${f._layer}:${f.id}`).join(",") + c;
  return (
    <>
      <GeoJSON key={`halo-${key}`} data={{ type: "FeatureCollection", features }} interactive={false}
               style={() => ({ color: "#0d0f12", weight: 5, opacity: 0.55, fill: false })}
               pointToLayer={(f, ll) => L.circleMarker(ll, { radius: 8, color: "#0d0f12", weight: 5, opacity: 0.55, fill: false, interactive: false })} />
      <GeoJSON key={key} data={{ type: "FeatureCollection", features }} interactive={false}
               style={() => ({ color: c, weight: 2.5, fill: true, fillColor: c, fillOpacity: 0.18 })}
               pointToLayer={(f, ll) => L.circleMarker(ll, { radius: 8, color: c, weight: 2.5, fillColor: c, fillOpacity: 0.3, interactive: false })} />
    </>
  );
}

// Ctrl + drag draws a box: the features of visible layers inside it are selected
// (Shift as well adds to the selection). A click on empty map clears it.
export function SelectTools({ onBox, onClear }) {
  const map = useMap();
  const box = useRef(null);
  useEffect(() => {
    const el = map.getContainer();
    let start = null, add = false;
    const down = (e) => {
      if (!(e.ctrlKey || e.metaKey) || e.button !== 0 || document.body.dataset.drawing === "1" || document.body.dataset.labelling === "1") return;
      if (e.target.closest(".leaflet-control, .map-toolbar, .leaflet-popup")) return;
      e.preventDefault(); e.stopPropagation();
      map.dragging.disable();
      start = map.mouseEventToContainerPoint(e); add = e.shiftKey;
      box.current = L.DomUtil.create("div", "select-box", el);
      Object.assign(box.current.style, { left: `${start.x}px`, top: `${start.y}px`, width: "0px", height: "0px" });
    };
    const move = (e) => {
      if (!start) return;
      const p = map.mouseEventToContainerPoint(e);
      Object.assign(box.current.style, { left: `${Math.min(p.x, start.x)}px`, top: `${Math.min(p.y, start.y)}px`,
        width: `${Math.abs(p.x - start.x)}px`, height: `${Math.abs(p.y - start.y)}px` });
    };
    const up = (e) => {
      if (!start) return;
      const p = map.mouseEventToContainerPoint(e);
      box.current?.remove(); box.current = null;
      map.dragging.enable();
      const b = L.latLngBounds(map.containerPointToLatLng(start), map.containerPointToLatLng(p));
      const big = Math.abs(p.x - start.x) > 4 && Math.abs(p.y - start.y) > 4;
      start = null;
      if (big) onBox(b, add);
    };
    el.addEventListener("mousedown", down, true);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { el.removeEventListener("mousedown", down, true); window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, [map, onBox]);
  useMapEvents({
    click: (e) => {
      const o = e.originalEvent || {};
      if (o.ctrlKey || o.metaKey || o.shiftKey || document.body.dataset.drawing === "1" || document.body.dataset.labelling === "1") return;
      onClear();
    },
  });
  return null;
}

// Keeps the visible map extent, for "Map extent only" and "in view".
export function ViewTracker({ onView }) {
  const map = useMap();
  const push = () => { const b = map.getBounds(); onView({ south: b.getSouth(), west: b.getWest(), north: b.getNorth(), east: b.getEast(), zoom: map.getZoom() }); };
  useMapEvents({ moveend: push });
  useEffect(push, []);  
  return null;
}
