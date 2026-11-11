import React, { useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import MapDrawTools from "./MapDrawTools.jsx";

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
function MapToolbar({ layers, basemap, onBasemapChange, legendOpen, onLegendToggle }) {
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
    map.setView([24.7, 46.7], 5);
  }

  function toggleBasemap() {
    onBasemapChange(basemap === "satellite" ? "streets" : "satellite");
  }

  return (
    <div className="map-toolbar">
      <button className="map-tool-btn" title="Zoom to detections / reset view" onClick={zoomToFit}>
        ⛶
      </button>
      <button className="map-tool-btn" title="Toggle basemap" onClick={toggleBasemap}>
        ◧
      </button>
      <button
        className={`map-tool-btn ${legendOpen ? "active" : ""}`}
        title="Toggle legend"
        onClick={onLegendToggle}
      >
        ☰
      </button>
    </div>
  );
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
}) {
  const [basemap, setBasemap] = useState("satellite");
  const [legendOpen, setLegendOpen] = useState(true);
  const bm = BASEMAPS[basemap];

  return (
    <div className="map-area">
      <MapContainer center={[24.7, 46.7]} zoom={5} scrollWheelZoom>
        <TileLayer attribution={bm.attribution} url={bm.url} />
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
          basemap={basemap}
          onBasemapChange={setBasemap}
          legendOpen={legendOpen}
          onLegendToggle={() => setLegendOpen((o) => !o)}
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
        </div>
      )}
    </div>
  );
}
