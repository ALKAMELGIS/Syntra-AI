import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMap } from "react-leaflet";
import L from "leaflet";
import {
  IconEdit, IconRectangle, IconCircle, IconPolygon, IconRuler, IconTag, IconTrash,
} from "./Icons.jsx";

// The map's drawing tools, gathered behind one "Edit" button in the main
// map toolbar. Every tool here drives the Leaflet map for real: rectangle /
// circle / freehand draw an Area of Interest (handed to the parent as
// GeoJSON via onAoiChange), the ruler measures a real Haversine distance,
// and "Annotate for training" toggles the class-labelled AnnotationTool.
//
// Keyboard: R rectangle, C circle, F freehand, M measure, A annotate,
// Esc cancels the active tool or closes the menu.

const TOOLS = [
  { id: "rect", label: "Rectangle area", key: "R", Icon: IconRectangle, hint: "Drag on the map to draw a rectangle" },
  { id: "circle", label: "Circle area", key: "C", Icon: IconCircle, hint: "Drag from the centre outwards to draw a circle" },
  { id: "pencil", label: "Freehand area", key: "F", Icon: IconPolygon, hint: "Click to add points, double-click to close the shape" },
  { id: "ruler", label: "Measure distance", key: "M", Icon: IconRuler, hint: "Click two points to measure the distance between them" },
];

const AOI_STYLE = { color: "#ffcf5c", weight: 2, dashArray: "4 3" };

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

// Keys pressed while typing, or inside a panel with its own shortcuts
// (data-own-keys, e.g. the review list's A / R), are not map shortcuts.
function isTypingTarget(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable
    || !!el.closest?.("[data-own-keys]"));
}

export default function MapDrawTools({
  aoi, onAoiChange, measurement, onMeasurementChange,
  annotateOpen = false, onToggleAnnotate, canAnnotate = false, otherPanelOpen = false, onMenuOpen,
}) {
  const map = useMap();
  const [mode, setMode] = useState(null); // null | "rect" | "circle" | "pencil" | "ruler"
  const [menuOpen, setMenuOpen] = useState(false);
  // another panel opened beside the toolbar: fold the Edit menu
  useEffect(() => { if (otherPanelOpen) setMenuOpen(false); }, [otherPanelOpen]);
  const drawState = useRef({ overlay: null, start: null, points: [] });
  const wrapRef = useRef(null);

  // Annotate binds its own mouse handlers to the same map; never run both.
  useEffect(() => {
    if (annotateOpen) setMode(null);
  }, [annotateOpen]);

  function clearOverlay() {
    if (drawState.current.overlay) {
      map.removeLayer(drawState.current.overlay);
      drawState.current.overlay = null;
    }
  }

  function clearAll() {
    clearOverlay();
    drawState.current.points = [];
    onAoiChange?.(null);
    onMeasurementChange?.(null);
    setMode(null);
    setMenuOpen(false);
  }

  function pick(next) {
    drawState.current.points = [];
    if (annotateOpen) onToggleAnnotate?.();
    setMode((m) => (m === next ? null : next));
    setMenuOpen(false);
  }

  function pickAnnotate() {
    if (!canAnnotate) return;
    setMode(null);
    onToggleAnnotate?.();
    setMenuOpen(false);
  }

  // Clicks and scrolls on the menu must not reach the map underneath, or a
  // click on a menu item would also drop a point for the active tool.
  useEffect(() => {
    if (!wrapRef.current) return;
    L.DomEvent.disableClickPropagation(wrapRef.current);
    L.DomEvent.disableScrollPropagation(wrapRef.current);
  }, []);

  // Close the menu on an outside click.
  useEffect(() => {
    if (!menuOpen) return;
    function onDoc(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [menuOpen]);

  // Keyboard shortcuts.
  useEffect(() => {
    function onKey(e) {
      if (e.metaKey || e.ctrlKey || e.altKey || isTypingTarget(e.target)) return;
      if (e.key === "Escape") {
        if (menuOpen) setMenuOpen(false);
        else if (mode) {
          clearOverlay();
          drawState.current = { overlay: null, start: null, points: [] };
          setMode(null);
        }
        return;
      }
      const k = e.key.toUpperCase();
      const tool = TOOLS.find((t) => t.key === k);
      if (tool) {
        e.preventDefault();
        pick(tool.id);
      } else if (k === "A" && canAnnotate) {
        e.preventDefault();
        pickAnnotate();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Drawing behaviour for the active tool.
  useEffect(() => {
    if (!mode || annotateOpen) return;
    map.dragging.disable();
    map.doubleClickZoom.disable();
    map.getContainer().style.cursor = "crosshair";
    L.DomUtil.disableTextSelection(); // a drag must draw, not select toolbar text

    function onMouseDown(e) {
      if (mode === "rect") {
        drawState.current.start = e.latlng;
        clearOverlay();
        drawState.current.overlay = L.rectangle([e.latlng, e.latlng], AOI_STYLE).addTo(map);
      } else if (mode === "circle") {
        drawState.current.start = e.latlng;
        clearOverlay();
        drawState.current.overlay = L.circle(e.latlng, { ...AOI_STYLE, radius: 1 }).addTo(map);
      } else if (mode === "ruler") {
        drawState.current.points.push(e.latlng);
        if (drawState.current.points.length === 2) {
          const [a, b] = drawState.current.points;
          const meters = a.distanceTo(b);
          const label = meters >= 1000 ? `${(meters / 1000).toFixed(2)} km` : `${meters.toFixed(0)} m`;
          clearOverlay();
          drawState.current.overlay = L.polyline([a, b], { color: "#6b9bff", weight: 2 }).addTo(map);
          onMeasurementChange?.(label);
          drawState.current.points = [];
          setMode(null);
        }
      } else if (mode === "pencil") {
        drawState.current.points.push(e.latlng);
        clearOverlay();
        if (drawState.current.points.length > 1) {
          drawState.current.overlay = L.polygon(drawState.current.points, { ...AOI_STYLE, fillOpacity: 0.08 }).addTo(map);
        }
      }
    }

    function onMouseMove(e) {
      if (mode === "rect" && drawState.current.start && drawState.current.overlay) {
        drawState.current.overlay.setBounds([drawState.current.start, e.latlng]);
      } else if (mode === "circle" && drawState.current.start && drawState.current.overlay) {
        drawState.current.overlay.setRadius(drawState.current.start.distanceTo(e.latlng));
      }
    }

    function onMouseUp(e) {
      if (mode === "rect" && drawState.current.start) {
        onAoiChange?.(rectToPolygon(L.latLngBounds(drawState.current.start, e.latlng)));
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
      map.doubleClickZoom.enable();
      map.getContainer().style.cursor = "";
      L.DomUtil.enableTextSelection();
    };
  }, [mode, map, onAoiChange, onMeasurementChange, annotateOpen]);

  const activeTool = TOOLS.find((t) => t.id === mode);
  const ButtonIcon = activeTool ? activeTool.Icon : annotateOpen ? IconTag : IconEdit;
  const hasDrawing = Boolean(aoi) || measurement != null;

  return (
    <div className="edit-menu-wrap" ref={wrapRef}>
      <button
        className={`map-tool-btn ${mode || annotateOpen || menuOpen ? "active" : ""}`}
        data-tool="edit"
        title="Edit: draw and measure"
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        onClick={() => setMenuOpen((o) => { if (!o) onMenuOpen?.(); return !o; })}
      >
        <ButtonIcon />
      </button>

      {menuOpen && (
        <div className="edit-menu" role="menu">
          <div className="edit-menu-title">Draw and measure</div>
          {TOOLS.map(({ id, label, key, Icon }) => (
            <button
              key={id}
              role="menuitemradio"
              aria-checked={mode === id}
              className={`edit-menu-item ${mode === id ? "selected" : ""}`}
              onClick={() => pick(id)}
            >
              <Icon className="edit-menu-icon" />
              <span className="edit-menu-label">{label}</span>
              <kbd className="keycap">{key}</kbd>
            </button>
          ))}

          <div className="edit-menu-sep" />
          <button
            role="menuitemcheckbox"
            aria-checked={annotateOpen}
            className={`edit-menu-item ${annotateOpen ? "selected" : ""}`}
            onClick={pickAnnotate}
            disabled={!canAnnotate}
            title={canAnnotate ? undefined : "Load an image from the library first"}
          >
            <IconTag className="edit-menu-icon" />
            <span className="edit-menu-label">Annotate for training</span>
            <kbd className="keycap">A</kbd>
          </button>

          <div className="edit-menu-sep" />
          <button className="edit-menu-item edit-menu-danger" onClick={clearAll} disabled={!hasDrawing && !mode}>
            <IconTrash className="edit-menu-icon" />
            <span className="edit-menu-label">Clear drawings</span>
          </button>
        </div>
      )}

      {activeTool && !menuOpen && createPortal(
        // Rendered into the map container so it can sit top-centre over the
        // map instead of inside the narrow toolbar column.
        <div className="map-draw-hint" role="status">
          <activeTool.Icon />
          <span>{activeTool.hint}</span>
          <kbd className="keycap">Esc</kbd>
        </div>,
        map.getContainer(),
      )}
    </div>
  );
}
