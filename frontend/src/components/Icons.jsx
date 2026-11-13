import React from "react";

/**
 * Small inline-SVG icon set, used in place of raw Unicode emoji across the
 * toolbar/sidebar/topbar. Emoji render differently per OS/browser/font (size,
 * weight, color) so a row of them never reads as one coherent icon language —
 * that inconsistency was the biggest visible gap vs. the reference app's
 * clean, uniform icon set. These are plain stroke-based SVGs that inherit
 * `currentColor`, so they always match the button's text color/size exactly,
 * on every platform.
 */
const base = {
  width: "1em",
  height: "1em",
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export const IconExpand = (p) => (
  <svg {...base} {...p}><path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" /></svg>
);
export const IconBasemap = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3c2.5 2.6 3.8 5.7 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.7-3.8-9S9.5 5.6 12 3Z" /></svg>
);
export const IconLegend = (p) => (
  <svg {...base} {...p}><path d="M4 6h16M4 12h16M4 18h10" /></svg>
);
export const IconGallery = (p) => (
  <svg {...base} {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5-9 9" /></svg>
);
export const IconGear = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" /></svg>
);
export const IconDatabase = (p) => (
  <svg {...base} {...p}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" /><path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" /></svg>
);
export const IconEye = (p) => (
  <svg {...base} {...p}><path d="M1.5 12S5 5 12 5s10.5 7 10.5 7-3.5 7-10.5 7S1.5 12 1.5 12Z" /><circle cx="12" cy="12" r="3" /></svg>
);
export const IconEyeOff = (p) => (
  <svg {...base} {...p}><path d="M3 3l18 18" /><path d="M10.6 5.1A10.9 10.9 0 0 1 12 5c7 0 10.5 7 10.5 7a14.5 14.5 0 0 1-3 3.9M6.6 6.6C3.4 8.7 1.5 12 1.5 12S5 19 12 19a10.6 10.6 0 0 0 4-.76" /><path d="M9.5 9.5a3 3 0 0 0 4.24 4.24" /></svg>
);
export const IconRectangle = (p) => (
  <svg {...base} {...p}><rect x="4" y="6" width="16" height="12" rx="1.5" /></svg>
);
export const IconCircle = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="8" /></svg>
);
export const IconPencil = (p) => (
  <svg {...base} {...p}><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" /></svg>
);
export const IconRuler = (p) => (
  <svg {...base} {...p}><path d="M3 16 16 3l5 5L8 21Z" /><path d="m7.5 11.5 2 2M11 8l2 2M14.5 4.5l2 2" /></svg>
);
export const IconTrash = (p) => (
  <svg {...base} {...p}><path d="M4 7h16" /><path d="M9 7V4h6v3M6 7l1 13h10l1-13" /></svg>
);
export const IconMap = (p) => (
  <svg {...base} {...p}><path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2Z" /><path d="M9 4v14M15 6v14" /></svg>
);
export const IconDashboard = (p) => (
  <svg {...base} {...p}><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></svg>
);
export const IconUpload = (p) => (
  <svg {...base} {...p}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></svg>
);
export const IconFolder = (p) => (
  <svg {...base} {...p}><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" /></svg>
);
export const IconPlay = (p) => (
  <svg {...base} {...p} fill="currentColor" stroke="none"><path d="M7 4.5v15l12-7.5Z" /></svg>
);
export const IconAnalysis = (p) => (
  <svg {...base} {...p}><path d="M4 20V10M11 20V4M18 20v-7" /></svg>
);
export const IconClose = (p) => (
  <svg {...base} {...p}><path d="M5 5l14 14M19 5 5 19" /></svg>
);
// Edit menu: the pen nib used for the single "Edit / draw" toolbar entry.
export const IconEdit = (p) => (
  <svg {...base} {...p}><path d="m12 19 7-7 3 3-7 7-3-3Z" /><path d="m18 13-1.5-7.5L2 2l3.5 14.5L13 18l5-5Z" /><path d="m2 2 7.6 7.6" /><circle cx="11" cy="11" r="2" /></svg>
);
// Freehand polygon: a closed shape through clicked vertices.
export const IconPolygon = (p) => (
  <svg {...base} {...p}><path d="M5 7 12 3l7 5-2 10H7Z" /><circle cx="5" cy="7" r="1.2" /><circle cx="12" cy="3" r="1.2" /><circle cx="19" cy="8" r="1.2" /><circle cx="17" cy="18" r="1.2" /><circle cx="7" cy="18" r="1.2" /></svg>
);
// Annotate for training: a label tag.
export const IconTag = (p) => (
  <svg {...base} {...p}><path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9Z" /><circle cx="8" cy="8" r="1.5" /></svg>
);
export const IconChevronDown = (p) => (
  <svg {...base} {...p}><path d="m6 9 6 6 6-6" /></svg>
);
