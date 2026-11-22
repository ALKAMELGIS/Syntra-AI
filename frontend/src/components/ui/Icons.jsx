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
  <svg {...base} {...p}><rect x="3" y="4" width="5" height="5" rx="1" /><rect x="3" y="15" width="5" height="5" rx="1" /><path d="M12 6.5h9M12 17.5h9" /></svg>
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
export const IconSquare = (p) => (
  <svg {...base} {...p}><rect x="5" y="5" width="14" height="14" rx="1.5" /></svg>
);
export const IconClickAI = (p) => (
  <svg {...base} {...p}><path d="M9 4.5 9 17l3.2-3 2.3 5 2-.9-2.3-5H19Z" /><path d="M4 4l1.5 1.5M3 9h2M9 2v2" /></svg>
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
// Remote sensing: a satellite with two panels, and its signal
export const IconSatellite = (p) => (
  <svg {...base} {...p}><path d="M12 8.5 15.5 12 12 15.5 8.5 12Z M2.5 6.5l4-4 3 3-4 4Z M14.5 17.5l3-3 4 4-3 3Z M7.5 7.5l2 2M14.5 14.5l2 2M3 14a7 7 0 0 0 7 7M3 18a3 3 0 0 0 3 3" /></svg>
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

export const IconSun = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
);
export const IconMoon = (p) => (
  <svg {...base} {...p}><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" /></svg>
);
export const IconSliders = (p) => (
  <svg {...base} {...p}><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></svg>
);
export const IconTrain = (p) => (
  <svg {...base} {...p}><path d="M2 9l10-5 10 5-10 5z" /><path d="M6 11v5c0 1.5 2.7 3 6 3s6-1.5 6-3v-5" /><path d="M22 9v6" /></svg>
);
export const IconInfo = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></svg>
);
export const IconPerformance = (p) => (
  <svg {...base} {...p}><path d="M3 3v18h18" /><path d="M7 15l4-5 3 3 5-7" /></svg>
);

export const IconSignOut = (p) => (
  <svg {...base} {...p}><path d="M15 4h4a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-4M10 17l-5-5 5-5M5 12h11" /></svg>
);

export const IconChevronLeft = (p) => (
  <svg {...base} {...p}><path d="M15 18l-6-6 6-6" /></svg>
);

export const IconRaster = (p) => (
  <svg {...base} {...p}><rect x="4" y="4" width="16" height="16" rx="2" /><path d="M4 15l4.5-4.5 4 4 3-3L20 16" /></svg>
);

export const IconCheck = (p) => (
  <svg {...base} {...p}><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
);
export const IconLayers = (p) => (
  <svg {...base} {...p}><path d="M12 3 3 8l9 5 9-5-9-5Z" /><path d="m3 13 9 5 9-5" /><path d="m3 17.5 9 4.5 9-4.5" /></svg>
);
export const IconTable = (p) => (
  <svg {...base} {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 10v10" /></svg>
);
export const IconMore = (p) => (
  <svg {...base} {...p}><circle cx="5" cy="12" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /></svg>
);
export const IconZoomTo = (p) => (
  <svg {...base} {...p}><circle cx="11" cy="11" r="6" /><path d="m20 20-4.5-4.5" /></svg>
);
export const IconArrowUp = (p) => (
  <svg {...base} {...p}><path d="M12 19V5M6 11l6-6 6 6" /></svg>
);
export const IconArrowDown = (p) => (
  <svg {...base} {...p}><path d="M12 5v14M6 13l6 6 6-6" /></svg>
);
export const IconDownload = (p) => (
  <svg {...base} {...p}><path d="M12 3v12M7 10l5 5 5-5M4 20h16" /></svg>
);
export const IconFilter = (p) => (
  <svg {...base} {...p}><path d="M4 5h16l-6 7v6l-4 2v-8L4 5Z" /></svg>
);
export const IconPlus = (p) => (
  <svg {...base} {...p}><path d="M12 5v14M5 12h14" /></svg>
);
export const IconCopy = (p) => (
  <svg {...base} {...p}><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" /></svg>
);
export const IconPointer = (p) => (
  <svg {...base} {...p}><path d="M6 3.5 6 18l3.8-3.6 2.6 5.6 2.3-1-2.6-5.6H17.5Z" /></svg>
);
export const IconBoxAI = (p) => (
  <svg {...base} {...p}><path d="M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4" /><path d="M12 8.5 13 11l2.5 1-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1Z" /></svg>
);
export const IconSparkle = (p) => (
  <svg {...base} {...p}><path d="M12 3.5 13.8 9 19.5 11 13.8 13 12 18.5 10.2 13 4.5 11 10.2 9Z" /><path d="M18.5 3.5v3M17 5h3" /></svg>
);
export const IconAreaDashed = (p) => (
  <svg {...base} {...p}><path d="M4 7V4h3M10 4h4M17 4h3v3M20 10v4M20 17v3h-3M14 20h-4M7 20H4v-3M4 14v-4" /></svg>
);
export const IconUndo = (p) => (
  <svg {...base} {...p}><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></svg>
);
export const IconBoxSelect = (p) => (
  <svg {...base} {...p}><path d="M4 7V4h3M10 4h2M15 4h1M4 10v2M4 15v1M4 19v1h1M8 20h1" /><path d="M12 12 12 21l2.6-2.4 1.8 3.8 1.7-.8-1.8-3.8H20Z" /></svg>
);
export const IconLasso = (p) => (
  <svg {...base} {...p}><path d="M7.5 17.5C4.8 16.3 3 14.3 3 12c0-4 4-7 9-7s9 3 9 7-4 7-9 7c-.9 0-1.7-.1-2.5-.3" /><circle cx="7.5" cy="18" r="1.6" /><path d="M7 19.6c-.4 1.2-1.3 2-2.5 2.4" /></svg>
);
export const IconGridDots = (p) => (
  <svg {...base} {...p}><circle cx="5" cy="5" r="1.2" /><circle cx="12" cy="5" r="1.2" /><circle cx="19" cy="5" r="1.2" /><circle cx="5" cy="12" r="1.2" /><circle cx="12" cy="12" r="1.2" /><circle cx="19" cy="12" r="1.2" /><circle cx="5" cy="19" r="1.2" /><circle cx="12" cy="19" r="1.2" /><circle cx="19" cy="19" r="1.2" /></svg>
);
// Label editing: curve, trace along an edge, split, merge, snapping, radius.
export const IconCurve = (p) => (
  <svg {...base} {...p}><path d="M4 18C6 8 10 5 13 9s5 4 7-3" /><circle cx="4" cy="18" r="1.2" /><circle cx="20" cy="6" r="1.2" /></svg>
);
export const IconTrace = (p) => (
  <svg {...base} {...p}><path d="M4 20V8l6-4 10 4v12Z" strokeDasharray="2 2.5" /><path d="M4 14V8l6-4 6 2.4" /><circle cx="16" cy="6.4" r="1.4" /></svg>
);
export const IconSplit = (p) => (
  <svg {...base} {...p}><path d="M4 6h16v12H4Z" /><path d="M14 3 10 21" /></svg>
);
export const IconMerge = (p) => (
  <svg {...base} {...p}><path d="M3 9h8v-4h10v10h-8v4H3Z" /></svg>
);
export const IconMagnet = (p) => (
  <svg {...base} {...p}><path d="M6 4v8a6 6 0 0 0 12 0V4" /><path d="M6 8h3M15 8h3M9 4v8a3 3 0 0 0 6 0V4" /></svg>
);
export const IconRadius = (p) => (
  <svg {...base} {...p}><circle cx="12" cy="12" r="8" /><path d="M12 12h8" /><circle cx="12" cy="12" r="1.2" /></svg>
);
export const IconEditShape = (p) => (
  <svg {...base} {...p}><path d="M5 19 19 5" /><rect x="3" y="17" width="4" height="4" /><rect x="10" y="10" width="4" height="4" /><rect x="17" y="3" width="4" height="4" /></svg>
);
