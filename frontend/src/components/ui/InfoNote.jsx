import React, { useState } from "react";
import { IconInfo } from "./Icons.jsx";

// One short line with a small (i): the explanation opens in place beneath it,
// never as a popup. Used for raster-check findings and error messages.
export function splitMessage(text) {
  const t = String(text || "").trim();
  const m = t.match(/^(.{8,90}?[.:])\s+(.+)$/s);
  return m ? [m[1].replace(/[.:]$/, ""), m[2]] : [t, ""];
}

export default function InfoNote({ level = "info", title, detail, fix }) {
  const [open, setOpen] = useState(false);
  const more = Boolean(detail || fix);
  return (
    <div className={`info-note info-note-${level}`}>
      <div className="info-note-line">
        <span className="info-note-dot" aria-hidden="true" />
        <span className="info-note-title">{title}</span>
        {more && (
          <button type="button" className={`info-note-btn ${open ? "open" : ""}`} onClick={() => setOpen((o) => !o)}
                  aria-expanded={open} aria-label={open ? "Hide details" : "Show details"} title={open ? "Hide details" : "Details"}>
            <IconInfo />
          </button>
        )}
      </div>
      {open && more && (
        <div className="info-note-body">
          {detail && <p>{detail}</p>}
          {fix && <code>{fix}</code>}
        </div>
      )}
    </div>
  );
}
