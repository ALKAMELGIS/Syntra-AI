import React from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";

export default function MapView({ geojson }) {
  return (
    <div className="map-area">
      <MapContainer center={[24.7, 46.7]} zoom={5} scrollWheelZoom>
        <TileLayer
          attribution='&copy; Esri, Maxar, Earthstar Geographics'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />
        {geojson && (
          <GeoJSON
            data={geojson}
            style={() => ({ color: "#5b8cff", weight: 1, fillColor: "#3ddc97", fillOpacity: 0.5 })}
          />
        )}
      </MapContainer>
    </div>
  );
}
