// The map's toolbar (right edge): fit, basemap gallery, layers, the edit menu and the panel buttons.
import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { createPortal } from "react-dom";
import { useMap } from "react-leaflet";
import { IconExpand, IconBasemap, IconLayers, IconGallery, IconGear, IconAnalysis, IconTrain, IconPerformance, IconDatabase, IconSatellite, IconEye, IconEyeOff } from "../ui/Icons.jsx";
import BasemapGallery from "./BasemapGallery.jsx";
import { L_boundsFromGeojson } from "./mapLayers.jsx";

// Lives inside <MapContainer> so it can reach the Leaflet map instance via
// react-leaflet's useMap() hook -- these buttons actually drive the map,
// they are not decorative.
export default function MapToolbar({
  layers, sourceImage, basemap, onBasemapChange, legendOpen, onLegendToggle,
  onOpenLibrary, onOpenSettings, analysisOpen, onToggleAnalysis, analysisCount, analysisRunning,
  detectionsVisible, onToggleDetections, trainOpen, onToggleTrain, perfOpen, onTogglePerf, libOpen, onToggleLib, editMenu,
  rsOpen, onToggleRs, rsLayer, rsVisible, onToggleRsVisible, onMenuOpen,
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

  // the basemap gallery opens beside its own button
  // (portalled: the toolbar scrolls, so anything inside it would be clipped)
  const [bmOpen, setBmOpen] = useState(null);           // null | {top, right} beside the button
  const bmRef = useRef(null), bmPop = useRef(null);
  useEffect(() => {
    if (!bmOpen) return undefined;
    const down = (e) => { if (!bmRef.current?.contains(e.target) && !bmPop.current?.contains(e.target)) setBmOpen(null); };
    const key = (e) => { if (e.key === "Escape") setBmOpen(null); };
    document.addEventListener("mousedown", down);
    window.addEventListener("keydown", key);
    window.addEventListener("resize", () => setBmOpen(null), { once: true });
    return () => { document.removeEventListener("mousedown", down); window.removeEventListener("keydown", key); };
  }, [bmOpen]);
  function toggleBm() {
    if (bmOpen) { setBmOpen(null); return; }
    onMenuOpen?.();
    const r = bmRef.current.getBoundingClientRect(), tb = toolbarRef.current.getBoundingClientRect();
    setBmOpen({ top: r.top, right: window.innerWidth - tb.left + 12 });
  }

  return (
    <div className="map-toolbar" ref={toolbarRef}>
      <button className="map-tool-btn" data-tool="fit" title="Zoom to detections / reset view" onClick={zoomToFit}>
        <IconExpand />
      </button>
      <button ref={bmRef} className={`map-tool-btn ${bmOpen ? "active" : ""}`} data-tool="basemap" title="Basemap"
              aria-haspopup="dialog" aria-expanded={!!bmOpen} onClick={toggleBm}>
        <IconBasemap />
      </button>
      {bmOpen && createPortal(
        <div className="bm-pop" role="dialog" aria-label="Basemap" ref={bmPop} style={{ top: bmOpen.top, right: bmOpen.right }}>
          <div className="bm-pop-head">Basemap</div>
          <BasemapGallery basemap={basemap} onBasemap={(k) => { onBasemapChange(k); setBmOpen(null); }} />
        </div>, document.body)}
      <button
        className={`map-tool-btn ${legendOpen ? "active" : ""}`}
        data-tool="layers" title="Layers" aria-expanded={legendOpen}
        onClick={onLegendToggle}
      >
        <IconLayers />
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
      <button className={`map-tool-btn ${libOpen ? "active" : ""}`} data-tool="samples" title="Sample library"
              aria-expanded={libOpen} onClick={onToggleLib}>
        <IconDatabase />
      </button>
      <button className={`map-tool-btn ${rsOpen ? "active" : ""}`} data-tool="rs" title="Remote sensing"
              aria-expanded={rsOpen} onClick={onToggleRs}>
        <IconSatellite />
      </button>
      {rsLayer && (
        <button className={`map-tool-btn ${rsVisible ? "active" : ""}`} data-tool="rs-layer"
                title={rsVisible ? "Index layer on map · hide" : "Index layer · show on map"}
                aria-pressed={rsVisible} onClick={onToggleRsVisible}>
          {rsVisible ? <IconEye /> : <IconEyeOff />}
        </button>
      )}
      <button className="map-tool-btn" data-tool="settings" title="Settings" onClick={onOpenSettings}>
        <IconGear />
      </button>
      <div className="map-toolbar-sep" />
      <button
        className={`map-tool-btn ${analysisOpen ? "active" : ""}`}
        data-tool="analysis" title={analysisRunning ? "Analysis results · running" : "Analysis results"}
        aria-expanded={analysisOpen} onClick={onToggleAnalysis}
      >
        <IconAnalysis />
        {analysisRunning ? <span className="map-tool-badge running" aria-label="Running" />
          : analysisCount > 0 && <span className="map-tool-badge">{analysisCount}</span>}
      </button>
      <button className={`map-tool-btn ${perfOpen ? "active" : ""}`} data-tool="performance" title="Model performance"
              aria-expanded={perfOpen} onClick={onTogglePerf}>
        <IconPerformance />
      </button>
    </div>
  );
}
