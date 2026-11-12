import React, { useEffect, useRef, useState } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import { api } from "../api/client.js";
import { IconRectangle, IconPencil, IconTrash, IconClose } from "./Icons.jsx";

// The real "Annotate" step: draw a class-labeled box/polygon on the current
// image and save it — each shape is converted to the raster's pixel grid
// and persisted via /api/images/{id}/annotations (app.annotations.store),
// ready for the Dataset export step. Separate from MapDrawTools (which only
// ever drew an Area-of-Interest / ruler, never a saved, class-labeled
// object), so none of that existing behavior is touched.

const PALETTE = ["#ffcf5c", "#5b8cff", "#5fd98a", "#ff7a7a", "#c792ea", "#4fd1c5", "#f5a623", "#ec6ea0"];

function colorFor(className) {
  let h = 0;
  for (let i = 0; i < className.length; i++) h = (h * 31 + className.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

function rectLatLngs(bounds) {
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  return [sw, L.latLng(ne.lat, sw.lng), ne, L.latLng(sw.lat, ne.lng)];
}

export default function AnnotationTool({ active, sourceImage }) {
  const map = useMap();
  const [mode, setMode] = useState(null); // null | "rect" | "polygon"
  const [className, setClassName] = useState("object");
  const [annotations, setAnnotations] = useState([]);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const drawState = useRef({ overlay: null, start: null, points: [] });
  const savedLayerGroup = useRef(null);
  const imageId = sourceImage?.image_id;

  // load existing annotations for this image whenever it changes / tool opens
  useEffect(() => {
    if (!active || !imageId) return;
    api.listAnnotations(imageId).then((r) => setAnnotations(r.annotations)).catch(() => setAnnotations([]));
  }, [active, imageId]);

  // render saved annotations as a dedicated layer group, redrawn whenever the list changes
  useEffect(() => {
    if (savedLayerGroup.current) {
      map.removeLayer(savedLayerGroup.current);
      savedLayerGroup.current = null;
    }
    if (!active || !annotations.length) return;
    const group = L.layerGroup();
    for (const a of annotations) {
      const latlngs = a.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon));
      const color = colorFor(a.class_name);
      const poly = L.polygon(latlngs, { color, weight: 2, fillOpacity: 0.15 });
      poly.bindTooltip(a.class_name, { permanent: false, direction: "center" });
      poly.on("click", () => removeAnnotation(a.id));
      group.addLayer(poly);
    }
    group.addTo(map);
    savedLayerGroup.current = group;
    return () => { if (savedLayerGroup.current) map.removeLayer(savedLayerGroup.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, annotations, map]);

  function clearOverlay() {
    if (drawState.current.overlay) {
      map.removeLayer(drawState.current.overlay);
      drawState.current.overlay = null;
    }
  }

  async function removeAnnotation(id) {
    if (!imageId) return;
    try {
      await api.deleteAnnotation(imageId, id);
      setAnnotations((list) => list.filter((a) => a.id !== id));
    } catch (err) {
      setError(err.message);
    }
  }

  async function clearAllAnnotations() {
    if (!imageId) return;
    try {
      await api.clearAnnotations(imageId);
      setAnnotations([]);
    } catch (err) {
      setError(err.message);
    }
  }

  async function saveShape(latlngs) {
    if (!imageId) {
      setError("Open an image from the library first.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const points_lonlat = latlngs.map((p) => [p.lng, p.lat]);
      const saved = await api.addAnnotation(imageId, { class_name: className, points_lonlat });
      setAnnotations((list) => [...list, saved]);
    } catch (err) {
      setError(err.detail?.message || err.message);
    } finally {
      setSaving(false);
    }
  }

  useEffect(() => {
    if (!active || !mode) return;
    map.dragging.disable();
    map.getContainer().style.cursor = "crosshair";

    function onMouseDown(e) {
      if (mode === "rect") {
        drawState.current.start = e.latlng;
        clearOverlay();
        drawState.current.overlay = L.rectangle([e.latlng, e.latlng], { color: colorFor(className), weight: 2, dashArray: "4 3" }).addTo(map);
      } else if (mode === "polygon") {
        drawState.current.points.push(e.latlng);
        clearOverlay();
        if (drawState.current.points.length > 1) {
          drawState.current.overlay = L.polygon(drawState.current.points, { color: colorFor(className), weight: 2, fillOpacity: 0.1 }).addTo(map);
        }
      }
    }

    function onMouseMove(e) {
      if (mode === "rect" && drawState.current.start && drawState.current.overlay) {
        drawState.current.overlay.setBounds([drawState.current.start, e.latlng]);
      }
    }

    async function onMouseUp(e) {
      if (mode === "rect" && drawState.current.start) {
        const bounds = L.latLngBounds(drawState.current.start, e.latlng);
        drawState.current.start = null;
        clearOverlay();
        setMode(null);
        await saveShape(rectLatLngs(bounds));
      }
    }

    async function onDblClick() {
      if (mode === "polygon" && drawState.current.points.length >= 3) {
        const pts = drawState.current.points;
        drawState.current.points = [];
        clearOverlay();
        setMode(null);
        await saveShape(pts);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, mode, map, className]);

  if (!active) return null;

  function toggle(next) {
    drawState.current.points = [];
    clearOverlay();
    setMode((m) => (m === next ? null : next));
  }

  return (
    <div className="map-toolbar map-toolbar-annotate">
      <input
        className="annotate-class-input"
        value={className}
        onChange={(e) => setClassName(e.target.value)}
        placeholder="class name"
        title="Class name for the next shape you draw"
      />
      <button className={`map-tool-btn ${mode === "rect" ? "active" : ""}`} title="Draw a labeled box" onClick={() => toggle("rect")}><IconRectangle /></button>
      <button className={`map-tool-btn ${mode === "polygon" ? "active" : ""}`} title="Draw a labeled polygon (click points, double-click to finish)" onClick={() => toggle("polygon")}><IconPencil /></button>
      <div className="map-toolbar-sep" />
      <span className="annotate-count" title="Saved annotations for this image">{annotations.length}</span>
      <button className="map-tool-btn" title="Clear all annotations on this image" onClick={clearAllAnnotations}><IconTrash /></button>
      {saving && <span className="note" style={{ padding: "0 6px" }}>Saving…</span>}
      {error && (
        <span className="note annotate-error" title={error}>
          ⚠ {error.length > 40 ? error.slice(0, 40) + "…" : error}
          <button className="map-tool-btn" onClick={() => setError(null)}><IconClose /></button>
        </span>
      )}
    </div>
  );
}
