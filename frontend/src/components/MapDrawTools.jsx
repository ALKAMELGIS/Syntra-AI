import React, { useEffect, useRef, useState } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import { IconDatabase, IconEye, IconEyeOff, IconRectangle, IconCircle, IconPencil, IconRuler, IconTrash } from "./Icons.jsx";

// A real, working annotation/AOI toolbar living inside <MapContainer> (via
// useMap()) — every tool here actually drives the Leaflet map instance, none
// of it is decorative. Rectangle/circle draws double as an Area of Interest
// (AOI): the resulting polygon is hex-stored as GeoJSON and handed to the
// parent via onAoiChange so a prediction run can restrict tiling to it.

function circleToPolygon(center, radiusMeters, points = 48) {
  const coords = [];
  const lat = center.lat * (Math.PI / 180);
  const metersPerDegLat = 111320;
  const metersPerDegLng = 111320 * Math.cos(lat);
  for (let i = 0; i <= points; i++) {
    const angle = (i / points) * 2 * Math.PI;
    const dLat = (radiusMeters * Math.sin(angle)) / metersPerDegLat;
    const dLng = (radiusMeters * Math.cos(angle)) / metersPerDegLng;
    coords.push([center.lng + dLng, center.lat + dLat]);
  }
  return { type: "Feature", properties: { shape: "circle" }, geometry: { type: "Polygon", coordinates: [coords] } };
}

function rectToPolygon(bounds) {
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  const coords = [
    [sw.lng, sw.lat], [ne.lng, sw.lat], [ne.lng, ne.lat], [sw.lng, ne.lat], [sw.lng, sw.lat],
  ];
  return { type: "Feature", properties: { shape: "rectangle" }, geometry: { type: "Polygon", coordinates: [coords] } };
}

function polylineToPolygon(latlngs) {
  const coords = latlngs.map((p) => [p.lng, p.lat]);
  if (coords.length > 0) coords.push(coords[0]);
  return { type: "Feature", properties: { shape: "freehand" }, geometry: { type: "Polygon", coordinates: [coords] } };
}

export default function MapDrawTools({ detectionsVisible, onToggleDetections, onAoiChange, onMeasurementChange, onOpenDatasetManager, suspended = false }) {
  const map = useMap();
  const [mode, setMode] = useState(null); // null | "rect" | "circle" | "ruler" | "pencil"
  const drawState = useRef({ overlay: null, start: null, points: [] });

  // When the Annotate tool is active it binds its own mousedown/mouseup
  // handlers to the same map to draw class-labeled shapes; running both
  // drawing modes at once would fire both on the same drag. `suspended`
  // just drops our own mode back to idle, it changes none of the AOI/
  // measurement logic itself.
  useEffect(() => {
    if (suspended) setMode(null);
  }, [suspended]);

  function clearOverlay() {
    if (drawState.current.overlay) {
      map.removeLayer(drawState.current.overlay);
      drawState.current.overlay = null;
    }
  }

  function clearAll() {
    clearOverlay();
    onAoiChange?.(null);
    onMeasurementChange?.(null);
    setMode(null);
  }

  useEffect(() => {
    if (!mode || suspended) return;
    map.dragging.disable();
    map.getContainer().style.cursor = "crosshair";

    function onMouseDown(e) {
      if (mode === "rect") {
        drawState.current.start = e.latlng;
        clearOverlay();
        drawState.current.overlay = L.rectangle([e.latlng, e.latlng], { color: "#ffcf5c", weight: 2, dashArray: "4 3" }).addTo(map);
      } else if (mode === "circle") {
        drawState.current.start = e.latlng;
        clearOverlay();
        drawState.current.overlay = L.circle(e.latlng, { radius: 1, color: "#ffcf5c", weight: 2, dashArray: "4 3" }).addTo(map);
      } else if (mode === "ruler") {
        drawState.current.points.push(e.latlng);
        if (drawState.current.points.length === 2) {
          const [a, b] = drawState.current.points;
          const meters = a.distanceTo(b);
          const label = meters >= 1000 ? `${(meters / 1000).toFixed(2)} km` : `${meters.toFixed(0)} m`;
          clearOverlay();
          drawState.current.overlay = L.polyline([a, b], { color: "#5b8cff", weight: 2 }).addTo(map);
          onMeasurementChange?.(label);
          drawState.current.points = [];
          setMode(null);
        }
      } else if (mode === "pencil") {
        drawState.current.points.push(e.latlng);
        clearOverlay();
        if (drawState.current.points.length > 1) {
          drawState.current.overlay = L.polygon(drawState.current.points, { color: "#ffcf5c", weight: 2, fillOpacity: 0.08 }).addTo(map);
        }
      }
    }

    function onMouseMove(e) {
      if (mode === "rect" && drawState.current.start && drawState.current.overlay) {
        drawState.current.overlay.setBounds([drawState.current.start, e.latlng]);
      } else if (mode === "circle" && drawState.current.start && drawState.current.overlay) {
        const r = drawState.current.start.distanceTo(e.latlng);
        drawState.current.overlay.setRadius(r);
      }
    }

    function onMouseUp(e) {
      if (mode === "rect" && drawState.current.start) {
        const bounds = L.latLngBounds(drawState.current.start, e.latlng);
        onAoiChange?.(rectToPolygon(bounds));
        drawState.current.start = null;
        setMode(null);
      } else if (mode === "circle" && drawState.current.start) {
        const r = drawState.current.start.distanceTo(e.latlng);
        onAoiChange?.(circleToPolygon(drawState.current.start, r));
        drawState.current.start = null;
        setMode(null);
      }
    }

    function onDblClick() {
      if (mode === "pencil" && drawState.current.points.length >= 3) {
        onAoiChange?.(polylineToPolygon(drawState.current.points));
        drawState.current.points = [];
        setMode(null);
      }
    }

    map.on("mousedown", onMouseDown);
    map.on("mousemove", onMouseMove);
    map.on("mouseup", onMouseUp);
    map.on("dblclick", onDblClick);

    return () => {
      map.off("mousedown", onMouseDown);
      map.off("mousemove", onMouseMove);
      map.off("mouseup", onMouseUp);
      map.off("dblclick", onDblClick);
      map.dragging.enable();
      map.getContainer().style.cursor = "";
    };
  }, [mode, map, onAoiChange, onMeasurementChange, suspended]);

  function toggle(next) {
    drawState.current.points = [];
    setMode((m) => (m === next ? null : next));
  }

  return (
    <div className="map-toolbar map-toolbar-draw">
      <button className="map-tool-btn" title="Jump to Dataset &amp; Model Manager" onClick={onOpenDatasetManager}><IconDatabase /></button>
      <button
        className={`map-tool-btn ${!detectionsVisible ? "active" : ""}`}
        title="Toggle detections visibility"
        onClick={onToggleDetections}
      >
        {detectionsVisible ? <IconEye /> : <IconEyeOff />}
      </button>
      <div className="map-toolbar-sep" />
      <button disabled={suspended} className={`map-tool-btn ${mode === "rect" ? "active" : ""}`} title="Draw rectangle AOI" onClick={() => toggle("rect")}><IconRectangle /></button>
      <button disabled={suspended} className={`map-tool-btn ${mode === "circle" ? "active" : ""}`} title="Draw circle AOI" onClick={() => toggle("circle")}><IconCircle /></button>
      <button disabled={suspended} className={`map-tool-btn ${mode === "pencil" ? "active" : ""}`} title="Freehand AOI (click to add points, double-click to finish)" onClick={() => toggle("pencil")}><IconPencil /></button>
      <button disabled={suspended} className={`map-tool-btn ${mode === "ruler" ? "active" : ""}`} title="Measure distance (click two points)" onClick={() => toggle("ruler")}><IconRuler /></button>
      <button className="map-tool-btn" title="Clear AOI / measurement" onClick={clearAll}><IconTrash /></button>
    </div>
  );
}
