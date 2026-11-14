import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

// Training labels on the map. Lives inside <MapContainer>; it has no UI of
// its own: the Train panel picks the class and the tool (box / polygon) and
// owns the saved list. This layer draws the saved shapes and turns mouse
// gestures into a shape, which it hands back through onDrawn.

const PALETTE = ["#ffcf5c", "#5b8cff", "#5fd98a", "#ff7a7a", "#c792ea", "#4fd1c5", "#f5a623", "#ec6ea0"];

export const TRAIN_AREA = "__train_area__";
export const CHECK_AREA = "__check_area__";
export const isArea = (name) => name === TRAIN_AREA || name === CHECK_AREA;
export const AREA_LABEL = { [TRAIN_AREA]: "Training area", [CHECK_AREA]: "Accuracy area" };

export function classColor(className) {
  if (className === TRAIN_AREA) return "#c8ccd4";
  if (className === CHECK_AREA) return "#f0b54a";
  let h = 0;
  for (let i = 0; i < className.length; i++) h = (h * 31 + className.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

function rectLatLngs(bounds) {
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  return [sw, L.latLng(ne.lat, sw.lng), ne, L.latLng(sw.lat, ne.lng)];
}

export default function AnnotationLayer({ visible, mode, className, annotations = [], onDrawn, onShapeClick }) {
  const map = useMap();
  const draw = useRef({ overlay: null, start: null, points: [] });
  const group = useRef(null);

  useEffect(() => {
    if (group.current) { map.removeLayer(group.current); group.current = null; }
    if (!visible || !annotations.length) return;
    const g = L.layerGroup();
    for (const a of annotations) {
      const color = classColor(a.class_name);
      const area = isArea(a.class_name);
      const poly = L.polygon(a.points_lonlat.map(([lon, lat]) => L.latLng(lat, lon)),
        area ? { color, weight: 2, dashArray: "8 6", fill: false } : { color, weight: 2, fillOpacity: 0.15 });
      poly.bindTooltip(AREA_LABEL[a.class_name] || a.class_name, { direction: "center" });
      poly.on("click", (e) => { L.DomEvent.stopPropagation(e); if (!mode) onShapeClick?.(a.id); });
      g.addLayer(poly);
    }
    g.addTo(map);
    group.current = g;
    return () => { if (group.current) map.removeLayer(group.current); };
  }, [visible, annotations, map, mode, onShapeClick]);

  useEffect(() => {
    const st = draw.current;
    function clear() { if (st.overlay) { map.removeLayer(st.overlay); st.overlay = null; } }
    st.points = []; st.start = null; clear();
    if (!visible || !mode) return;
    const color = classColor(className || "object");
    map.dragging.disable();
    map.doubleClickZoom.disable();
    map.getContainer().style.cursor = "crosshair";

    function down(e) {
      if (mode === "rect") {
        st.start = e.latlng; clear();
        st.overlay = L.rectangle([e.latlng, e.latlng], { color, weight: 2, dashArray: "4 3" }).addTo(map);
      } else {
        st.points.push(e.latlng); clear();
        if (st.points.length > 1) st.overlay = L.polygon(st.points, { color, weight: 2, fillOpacity: 0.1 }).addTo(map);
      }
    }
    function move(e) {
      if (mode === "rect" && st.start && st.overlay) st.overlay.setBounds([st.start, e.latlng]);
    }
    function up(e) {
      if (mode !== "rect" || !st.start) return;
      const b = L.latLngBounds(st.start, e.latlng);
      st.start = null; clear();
      if (map.latLngToContainerPoint(b.getSouthWest()).distanceTo(map.latLngToContainerPoint(b.getNorthEast())) < 6) return;
      onDrawn?.(rectLatLngs(b));
    }
    function dbl() {
      if (mode !== "polygon" || st.points.length < 3) return;
      const pts = st.points; st.points = []; clear();
      onDrawn?.(pts);
    }
    map.on("mousedown", down); map.on("mousemove", move); map.on("mouseup", up); map.on("dblclick", dbl);
    return () => {
      map.off("mousedown", down); map.off("mousemove", move); map.off("mouseup", up); map.off("dblclick", dbl);
      clear();
      map.dragging.enable();
      map.doubleClickZoom.enable();
      map.getContainer().style.cursor = "";
    };
  }, [visible, mode, className, map, onDrawn]);

  return null;
}
