import React, { useState } from "react";
import EnvironmentPanel from "./EnvironmentPanel.jsx";
import { IconClose } from "./Icons.jsx";

// Settings, beside the map toolbar like every other tool. Toolbar icon style
// applies immediately (remembered per browser). The Python environment check
// is long, so it stays folded until asked for.
export default function SettingsPanel({ open, onClose, iconStyle, onChangeIconStyle }) {
  const [envOpen, setEnvOpen] = useState(false);
  if (!open) return null;

  return (
    <section className="settings-flyout" aria-label="Settings">
      <header className="analysis-head">
        <span className="analysis-title">Settings</span>
        <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
      </header>
      <div className="settings-body">
        <p className="perf-label">Toolbar icons</p>
        <div className="train-seg">
          <button type="button" className={iconStyle === "mono" ? "selected" : ""} onClick={() => onChangeIconStyle("mono")}>Monochrome</button>
          <button type="button" className={iconStyle === "colour" ? "selected" : ""} onClick={() => onChangeIconStyle("colour")}>Colour</button>
        </div>

        <button type="button" className="settings-fold" aria-expanded={envOpen} onClick={() => setEnvOpen((o) => !o)}>
          <span className={`env-collapse-caret ${envOpen ? "open" : ""}`}>▶</span>
          Python environment
        </button>
        {envOpen && <div className="settings-env"><EnvironmentPanel /></div>}
      </div>
    </section>
  );
}
