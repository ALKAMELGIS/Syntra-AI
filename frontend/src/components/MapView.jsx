import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, ImageOverlay, useMap } from "react-leaflet";
import MapDrawTools from "./MapDrawTools.jsx";
import ImageLibrary from "./ImageLibrary.jsx";
import SettingsPanel from "./SettingsPanel.jsx";
import { IconExpand, IconBasemap, IconLegend, IconGallery, IconGear } from "./Icons.jsx";

const BASEMAPS = {
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "&copy; Esri, Maxar, Earthstar Geographics",
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
  },
};

// Lives inside <MapContainer> so it can reach the Leaflet map instance via
// react-leaflet's useMap() hook -- these buttons actually drive the map,
// they are not decorative.
function MapToolbar({ layers, sourceImage, basemap, onBasemapChange, legendOpen, onLegendToggle, onOpenLibrary, onOpenSettings }) {
  const map = useMap();

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
    <div className="map-toolbar">
      <button className="map-tool-btn" title="Zoom to detections / reset view" onClick={zoomToFit}>
        <IconExpand />
      </button>
      <button className="map-tool-btn" title="Toggle basemap" onClick={toggleBasemap}>
        <IconBasemap />
      </button>
      <button
        className={`map-tool-btn ${legendOpen ? "active" : ""}`}
        title="Toggle legend"
        onClick={onLegendToggle}
      >
        <IconLegend />
      </button>
      <div className="map-toolbar-sep" />
      <button className="map-tool-btn" title="Image library" onClick={onOpenLibrary}>
        <IconGallery />
      </button>
      <button className="map-tool-btn" title="Settings" onClick={onOpenSettings}>
        <IconGear />
      </button>
    </div>
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
  aoi, onAoiChange, measurement, onMeasurementChange, onOpenDatasetManager,
  onSelectLibraryImage, currentImageId, sourceImage,
}) {
  const [basemap, setBasemap] = useState("satellite");
  const [legendOpen, setLegendOpen] = useState(true);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [iconStyle, setIconStyle] = useState(() => localStorage.getItem("syntra_icon_style") || "mono");
  const bm = BASEMAPS[basemap];

  function changeIconStyle(style) {
    setIconStyle(style);
    localStorage.setItem("syntra_icon_style", style);
  }

  return (
    <div className={`map-area icon-style-${iconStyle}`}>
      <MapContainer center={[24.7, 46.7]} zoom={5} scrollWheelZoom>
        <TileLayer attribution={bm.attribution} url={bm.url} />
        {sourceImage?.bounds_wgs84 && sourceImage?.image_id && (
          <ImageOverlay
            key={sourceImage.image_id}
            url={`/api/images/${sourceImage.image_id}/preview.png`}
            bounds={[
              [sourceImage.bounds_wgs84.south, sourceImage.bounds_wgs84.west],
              [sourceImage.bounds_wgs84.north, sourceImage.bounds_wgs84.east],
            ]}
          />
        )}
        <AutoFitSourceImage sourceImage={sourceImage} />
        {detectionsVisible &&
          layers.map((l) =>
            l.geojson ? (
              <GeoJSON
                key={l.key}
                data={l.geojson}
                style={() => ({ color: l.color, weight: 1, fillColor: l.color, fillOpacity: 0.4 })}
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
          onLegendToggle={() => setLegendOpen((o) => !o)}
          onOpenLibrary={() => setLibraryOpen(true)}
          onOpenSettings={() => setSettingsOpen(true)}
        />
        <MapDrawTools
          detectionsVisible={detectionsVisible}
          onToggleDetections={onToggleDetections}
          aoi={aoi}
          onAoiChange={onAoiChange}
          measurement={measurement}
          onMeasurementChange={onMeasurementChange}
          onOpenDatasetManager={onOpenDatasetManager}
        />
      </MapContainer>
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
        <div className="map-legend">
          <div className="map-legend-title">Legend</div>
          {layers.length === 0 && <div className="map-legend-row dim">No results yet</div>}
          {layers.map((l) => (
            <div className="map-legend-row" key={l.key}>
              <span className="map-legend-swatch" style={{ background: l.color }} />
              {l.name}
            </div>
          ))}
          {aoi && (
            <div className="map-legend-row">
              <span className="map-legend-swatch" style={{ background: "#ffcf5c" }} />
              Area of interest
            </div>
          )}
          {measurement != null && (
            <div className="map-legend-row dim">Measured: {measurement}</div>
          )}
          <div className="map-legend-row dim">Basemap: {bm.label}</div>
          {sourceImage?.filename && (
            <div className="map-legend-row dim">
              Source: {sourceImage.filename}{!sourceImage.bounds_wgs84 ? " (no CRS — not shown on map)" : ""}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
