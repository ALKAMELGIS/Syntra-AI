import React from "react";

// Applies immediately (no "Save" button, matching the reference) and is
// persisted per-browser via localStorage, read by MapView to add/remove
// the `.icon-style-colour` class on the toolbar rail.
export default function SettingsPanel({ open, onClose, iconStyle, onChangeIconStyle }) {
  if (!open) return null;

  return (
    <div className="settings-modal-backdrop" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-modal-header">
          <span>⚙ Settings</span>
          <button className="settings-modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="settings-modal-body">
          <p className="settings-section-title">Icon style</p>
          <p className="note">
            How the tools in the right-hand rail are drawn. Applies immediately and is remembered on this browser.
          </p>

          <div
            className={`settings-option-card ${iconStyle === "mono" ? "selected" : ""}`}
            onClick={() => onChangeIconStyle("mono")}
          >
            <div className="settings-option-title">Monochrome</div>
            <div className="settings-option-desc">Black and white, following the theme. The original look.</div>
            {iconStyle === "mono" && <span className="settings-option-check">✓</span>}
          </div>

          <div
            className={`settings-option-card ${iconStyle === "colour" ? "selected" : ""}`}
            onClick={() => onChangeIconStyle("colour")}
          >
            <div className="settings-option-title">Colour</div>
            <div className="settings-option-desc">A hue per tool — basemap, results, visibility, catalogue.</div>
            {iconStyle === "colour" && <span className="settings-option-check">✓</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
