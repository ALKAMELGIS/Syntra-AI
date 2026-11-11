import React, { useRef, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";

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
function MapToolbar({ geojson, basemap, onBasemapChange, legendOpen, onLegendToggle }) {
  const map = useMap();

  function zoomToFit() {
    if (geojson && geojson.features?.length) {
      const layer = L_boundsFromGeojson(geojson);
      if (layer) {
        map.fitBounds(layer, { padding: [40, 40] });
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

export default function MapView({ geojson }) {
  const [basemap, setBasemap] = useState("satellite");
  const [legendOpen, setLegendOpen] = useState(true);
  const bm = BASEMAPS[basemap];

  return (
    <div className="map-area">
      <MapContainer center={[24.7, 46.7]} zoom={5} scrollWheelZoom>
        <TileLayer attribution={bm.attribution} url={bm.url} />
        {geojson && (
          <GeoJSON
            data={geojson}
            style={() => ({ color: "#5b8cff", weight: 1, fillColor: "#3ddc97", fillOpacity: 0.5 })}
          />
        )}
        <MapToolbar
          geojson={geojson}
          basemap={basemap}
          onBasemapChange={setBasemap}
          legendOpen={legendOpen}
          onLegendToggle={() => setLegendOpen((o) => !o)}
        />
      </MapContainer>
      {legendOpen && (
        <div className="map-legend">
          <div className="map-legend-title">Legend</div>
          <div className="map-legend-row">
            <span className="map-legend-swatch" />
            Detection footprint
          </div>
          <div className="map-legend-row dim">Basemap: {bm.label}</div>
        </div>
      )}
    </div>
  );
}
