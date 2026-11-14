import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { api } from "../api/client";
import { MapContainer, TileLayer, GeoJSON, ImageOverlay, Rectangle, Tooltip, useMap, useMapEvents } from "react-leaflet";
import MapDrawTools from "./MapDrawTools.jsx";
import AnnotationLayer from "./AnnotationTool.jsx";
import ImageLibrary from "./ImageLibrary.jsx";
import SettingsPanel from "./SettingsPanel.jsx";
import { IconExpand, IconBasemap, IconLegend, IconGallery, IconGear, IconAnalysis, IconTrain, IconPerformance, IconEye, IconEyeOff } from "./Icons.jsx";

const BASEMAPS = {
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "&copy; Esri, Maxar, Earthstar Geographics",
    // Esri's World_Imagery service only actually renders tiles up to about
    // z19 in most of the world (higher in a handful of flagship cities).
    // Past that there are no native tiles, so maxNativeZoom tells Leaflet
    // to keep serving/upscaling the z19 tile rather than requesting (and
    // failing to get) tiles that don't exist -- the user can still zoom
    // all the way in past this, the image just gets blurrier.
    maxNativeZoom: 19,
  },
  streets: {
    // CARTO's free basemap tiles (basemaps.cartocdn.com) now require an API
    // key -- confirmed by hand (tiles rendered as "API KEY REQUIRED" in the
    // browser) after wiring this up. Standard OpenStreetMap raster tiles
    // remain genuinely keyless and are the correct default for a "streets"
    // toggle with no account/billing setup required.
    label: "Streets",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap contributors",
    // Standard OSM raster tiles stop at z19 natively; same upscale behavior
    // as above beyond that.
    maxNativeZoom: 19,
  },
};

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
  onOpenLibrary, onOpenSettings, analysisOpen, onToggleAnalysis, analysisCount,
  detectionsVisible, onToggleDetections, trainOpen, onToggleTrain, perfOpen, onTogglePerf, editMenu,
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
    onBasemapChange(basemap === "satellite" ? "streets" : "satellite");
  }

  return (
    <div className="map-toolbar" ref={toolbarRef}>
      <button className="map-tool-btn" data-tool="fit" title="Zoom to detections / reset view" onClick={zoomToFit}>
        <IconExpand />
      </button>
      <button className="map-tool-btn" data-tool="basemap" title="Toggle basemap" onClick={toggleBasemap}>
        <IconBasemap />
      </button>
      <button
        className={`map-tool-btn ${legendOpen ? "active" : ""}`}
        data-tool="legend" title="Legend" aria-expanded={legendOpen}
        onClick={onLegendToggle}
      >
        <IconLegend />
      </button>
      <button
        className={`map-tool-btn ${!detectionsVisible ? "active" : ""}`}
        data-tool="visibility" title={detectionsVisible ? "Hide detections" : "Show detections"}
        onClick={onToggleDetections}
      >
        {detectionsVisible ? <IconEye /> : <IconEyeOff />}
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
      <button className="map-tool-btn" data-tool="settings" title="Settings" onClick={onOpenSettings}>
        <IconGear />
      </button>
      <div className="map-toolbar-sep" />
      <button
        className={`map-tool-btn ${analysisOpen ? "active" : ""}`}
        data-tool="analysis" title="Analysis results"
        onClick={onToggleAnalysis}
      >
        <IconAnalysis />
        {analysisCount > 0 && <span className="map-tool-badge">{analysisCount}</span>}
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

function SourceImage({ image, bounds }) {
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
  const useTiles = tileZoom != null && zoom >= tileZoom;
  const ll = [[bounds.south, bounds.west], [bounds.north, bounds.east]];
  return useTiles ? (
    <TileLayer
      key={`tiles-${image.image_id}`}
      url={api.imageTileUrl(image.image_id)}
      bounds={ll}
      zIndex={10}
      maxZoom={MAX_ZOOM}
      maxNativeZoom={Math.min(MAX_ZOOM, nativeZoom ?? MAX_ZOOM)}
      keepBuffer={1}
    />
  ) : (
    <ImageOverlay key={image.image_id} url={api.imagePreviewUrl(image.image_id)} bounds={ll} />
  );
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
function AutoFitSourceImage({ sourceImage }) {
  const map = useMap();
  useEffect(() => {
    const b = sourceImage?.bounds_wgs84;
    if (b) {
      map.fitBounds([[b.south, b.west], [b.north, b.east]], { padding: [40, 40] });
    }
  }, [sourceImage?.image_id, map]);
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

export default function MapView({
  layers, detectionsVisible = true, onToggleDetections,
  aoi, onAoiChange, measurement, onMeasurementChange,
  trainOpen, onToggleTrain, onAnnotate, labels, trainPanel, perfOpen, onTogglePerf, perfPanel,
  onSelectLibraryImage, currentImageId, sourceImage,
  analysisOpen, onToggleAnalysis, analysisCount, analysisPanel,
}) {
  const [basemap, setBasemap] = useState("satellite");
  const [legendOpen, setLegendOpen] = useState(false);

  // Legend and Analysis both open beside the toolbar: show one at a time.
  useEffect(() => {
    if (analysisOpen || trainOpen || perfOpen) setLegendOpen(false);
  }, [analysisOpen, trainOpen, perfOpen]);

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
  // one panel beside the toolbar at a time
  useEffect(() => {
    if (analysisOpen || trainOpen || perfOpen || legendOpen) setSettingsOpen(false);
  }, [analysisOpen, trainOpen, perfOpen, legendOpen]);
  const [iconStyle, setIconStyle] = useState(() => localStorage.getItem("syntra_icon_style") || "mono");
  const bm = BASEMAPS[basemap];
  // Basemap health: tiles that never load (offline, firewall) leave the map
  // black with no explanation; say so in one line.
  const [tiles, setTiles] = useState({ ok: false, failed: 0 });
  useEffect(() => setTiles({ ok: false, failed: 0 }), [basemap]);
  // The preview is drawn in Web Mercator; only its own corners place it exactly.
  // (Without them the overlay is not drawn: the old lat/lon box misplaced it.)
  const overlayBounds = sourceImage?.preview_bounds_wgs84 || null;

  function changeIconStyle(style) {
    setIconStyle(style);
    localStorage.setItem("syntra_icon_style", style);
  }

  return (
    <div className={`map-area icon-style-${iconStyle} ${libraryOpen ? "library-open" : ""}`}>
      <MapContainer center={[24.7, 46.7]} zoom={5} maxZoom={MAX_ZOOM} scrollWheelZoom>
        <TileLayer
          key={basemap}
          attribution={bm.attribution}
          url={bm.url}
          maxZoom={MAX_ZOOM}
          maxNativeZoom={bm.maxNativeZoom}
          eventHandlers={{
            tileload: () => setTiles((t) => (t.ok ? t : { ...t, ok: true })),
            tileerror: () => setTiles((t) => ({ ...t, failed: t.failed + 1 })),
          }}
        />
        {overlayBounds && sourceImage?.image_id && <SourceImage image={sourceImage} bounds={overlayBounds} />}
        <ImageFootprint image={sourceImage} />
        <AutoFitSourceImage sourceImage={sourceImage} />
        {detectionsVisible &&
          layers.map((l) =>
            l.geojson ? (
              <GeoJSON
                key={`${l.key}-${l.version ?? 0}`}
                data={l.geojson}
                style={() => ({
                  color: l.color, weight: 1, fillColor: l.color,
                  // area layers cover a lot of the image; keep the imagery readable under them
                  fillOpacity: l.key.endsWith("vegetation_index") ? 0.22 : 0.4,
                })}
              />
            ) : null
          )}
        {aoi && (
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
              setLegendOpen(false);
              if (analysisOpen) onToggleAnalysis?.();
              if (trainOpen) onToggleTrain?.();
              if (perfOpen) onTogglePerf?.();
            }
            return !o;
          })}
          analysisOpen={analysisOpen}
          onToggleAnalysis={onToggleAnalysis}
          analysisCount={analysisCount}
          detectionsVisible={detectionsVisible}
          onToggleDetections={onToggleDetections}
          trainOpen={trainOpen}
          onToggleTrain={onToggleTrain}
          perfOpen={perfOpen}
          onTogglePerf={onTogglePerf}
          editMenu={
            <MapDrawTools
              aoi={aoi}
              onAoiChange={onAoiChange}
              measurement={measurement}
              onMeasurementChange={onMeasurementChange}
              annotateOpen={false}
              onToggleAnnotate={onAnnotate}
              canAnnotate={Boolean(sourceImage?.image_id)}
              otherPanelOpen={analysisOpen || trainOpen || perfOpen || legendOpen || settingsOpen}
              onMenuOpen={() => {
                setLegendOpen(false); setSettingsOpen(false);
                if (analysisOpen) onToggleAnalysis?.();
                if (trainOpen) onToggleTrain?.();
                if (perfOpen) onTogglePerf?.();
              }}
            />
          }
        />
        <AnnotationLayer {...labels} />
      </MapContainer>
      {!tiles.ok && tiles.failed >= 4 && (
        <p className="basemap-note" role="status">
          {bm.label} base map can't load: this computer can't reach its tile server
        </p>
      )}
      {analysisOpen && analysisPanel}
      {trainPanel}
      {perfPanel}
      <ImageLibrary
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        onSelectImage={(img) => {
          onSelectLibraryImage?.(img);
          setLibraryOpen(false);
        }}
        currentImageId={currentImageId}
      />
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        iconStyle={iconStyle}
        onChangeIconStyle={changeIconStyle}
      />
      {legendOpen && (
        <section className="legend-popover" aria-label="Legend">
          <header className="legend-head">
            <span className="legend-title">Legend</span>
            <span className="legend-total mono">{layers.length + (aoi ? 1 : 0) || ""}</span>
          </header>
          {layers.length === 0 && !aoi ? (
            <p className="legend-empty">No layers yet</p>
          ) : (
            <ul className="legend-list">
              {layers.map((l) => (
                <li className="legend-row" key={l.key}>
                  <span className="legend-swatch" style={{ background: l.color }} />
                  <span className="legend-name">{l.name}</span>
                  <span className="legend-value mono">{l.value}</span>
                </li>
              ))}
              {aoi && (
                <li className="legend-row">
                  <span className="legend-swatch legend-swatch-aoi" />
                  <span className="legend-name">Area of interest</span>
                  {measurement != null && <span className="legend-value mono">{measurement}</span>}
                </li>
              )}
              {!aoi && measurement != null && (
                <li className="legend-row">
                  <span className="legend-swatch legend-swatch-line" />
                  <span className="legend-name">Measured</span>
                  <span className="legend-value mono">{measurement}</span>
                </li>
              )}
            </ul>
          )}
          {sourceImage?.filename && (
            <footer className="legend-foot mono" title={sourceImage.filename}>
              {sourceImage.filename}{!sourceImage.bounds_wgs84 ? " · no CRS" : ""}
            </footer>
          )}
        </section>
      )}
    </div>
  );
}
