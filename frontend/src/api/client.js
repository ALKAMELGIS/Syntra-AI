const BASE = "/api";

// Optional shared API key (matches SYNTRA_API_KEY on the backend). Set
// VITE_API_KEY in frontend/.env for builds that talk to a protected server.
const API_KEY = import.meta.env.VITE_API_KEY || "";

function request(url, init = {}) {
  const headers = { ...(init.headers || {}) };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  return fetch(url, { ...init, headers });
}

// For URLs the browser loads directly (<img>, <a download>), which cannot
// carry custom headers.
export function withKey(url) {
  if (!API_KEY) return url;
  return `${url}${url.includes("?") ? "&" : "?"}api_key=${encodeURIComponent(API_KEY)}`;
}

// A 401 from any data route means the session ended (signed out elsewhere,
// expired): tell the sign-in gate, which swaps the app for the sign-in page.
export const SIGNED_OUT = "syntra:signed-out";

async function handle(res) {
  if (res.status === 401 && !res.url.includes("/api/auth/")) window.dispatchEvent(new Event(SIGNED_OUT));
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail;
    const message = typeof detail === "string" ? detail : detail?.message || `Request failed: ${res.status}`;
    const err = new Error(message);
    err.detail = detail;
    throw err;
  }
  return res.json();
}

const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const auth = {
  status: () => request(`${BASE}/auth/status`).then(handle),
  signIn: (email, password) => request(`${BASE}/auth/signin`, json({ email, password })).then(handle),
  signUp: (name, email, password) => request(`${BASE}/auth/signup`, json({ name, email, password })).then(handle),
  signOut: () => request(`${BASE}/auth/signout`, { method: "POST" }).then(handle),
  engineUp: () => fetch(`${BASE}/health`).then((r) => r.ok).catch(() => false),
};

export const api = {
  listPacks: () => request(`${BASE}/models/packs`).then(handle),
  installPack: (packId) => request(`${BASE}/models/packs/${packId}/install`, { method: "POST" }).then(handle),
  installStatus: (jobId) => request(`${BASE}/models/packs/install/${jobId}`).then(handle),
  installPackFile: (packId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`${BASE}/models/packs/${packId}/file`, { method: "POST", body: form }).then(handle);
  },
  listModels: () => request(`${BASE}/models`).then(handle),
  listPerformance: () => request(`${BASE}/models/performance`).then(handle),
  startAccuracy: (payload) => request(`${BASE}/predict/accuracy`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }).then(handle),
  accuracyStatus: (jobId) => request(`${BASE}/predict/accuracy/${jobId}`).then(handle),
  modelPerformance: (modelId) => request(`${BASE}/models/${modelId}/performance`).then(handle),
  getModel: (modelId) => request(`${BASE}/models/${modelId}`).then(handle),
  gsdCheck: (modelId, gsd) => request(`${BASE}/models/${modelId}/gsd-check?gsd=${gsd}`).then(handle),
  deleteModel: (modelId) => request(`${BASE}/models/${modelId}`, { method: "DELETE" }).then(handle),
  listImages: () => request(`${BASE}/images`).then(handle),
  getImage: (imageId) => request(`${BASE}/images/${imageId}`).then(handle),
  // Uses XMLHttpRequest (not fetch) because only XHR exposes real upload
  // progress events — needed for honest percentage feedback on multi-GB
  // GeoTIFF uploads, instead of a static "Uploading…" that looks frozen.
  uploadImage: (file, onProgress, metadataFile = null) =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      if (metadataFile) form.append("metadata", metadataFile);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/images`);
      if (API_KEY) xhr.setRequestHeader("X-API-Key", API_KEY);
      let sent = 0;
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        sent = Math.round((100 * e.loaded) / e.total);
        onProgress?.(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error body */ }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body);
        } else {
          if (xhr.status === 401) window.dispatchEvent(new Event(SIGNED_OUT));
          const detail = body.detail;
          const message = typeof detail === "string" ? detail : detail?.message || `Request failed: ${xhr.status}`;
          const err = new Error(message);
          err.detail = detail;
          reject(err);
        }
      };
      // The browser can't tell why a connection dropped; ask the server whether it is up.
      xhr.onerror = () => {
        fetch(`${BASE}/health`).then((r) => r.ok).catch(() => false).then((up) => reject(new Error(up
          ? `The upload was cut off at ${sent}%. The server is running: try again, and for very large files restart with the latest start script (it removes the 5-minute limit).`
          : "Couldn't reach the Syntra AI server. Check that the backend window is still open, then try again.")));
      };
      xhr.send(form);
    }),
  devices: () => request(`${BASE}/compute`).then(handle),
  startPrediction: (payload) =>
    request(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  job: (jobId) => request(`${BASE}/jobs/${jobId}`).then(handle),
  listJobs: (kind, limit = 20) => request(`${BASE}/jobs?kind=${kind}&limit=${limit}`).then(handle),
  jobStatus: (jobId) => request(`${BASE}/predict/${jobId}`).then(handle),
  getGeojson: (jobId, modelId) => request(`${BASE}/results/${jobId}/${modelId}`).then(handle),
  listDatasetVersions: (modelId) => request(`${BASE}/datasets/${modelId}/versions`).then(handle),
  getDatasetVersion: (modelId, versionId) =>
    request(`${BASE}/datasets/${modelId}/versions/${versionId}`).then(handle),
  prepareDataset: (payload) =>
    request(`${BASE}/datasets/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  getEnvironment: () => request(`${BASE}/environment`).then(handle),
  startEnvironmentAction: (action, build, targets) =>
    request(`${BASE}/environment/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, build, targets: targets && targets.length ? targets : null }),
    }).then(handle),
  environmentActionStatus: (jobId) => request(`${BASE}/environment/actions/${jobId}`).then(handle),
  usageSummary: () => request(`${BASE}/usage`).then(handle),
  resultDownloadUrl: (jobId, modelId, format) =>
    withKey(format === "geojson" ? `${BASE}/results/${jobId}/${modelId}` : `${BASE}/results/${jobId}/${modelId}/${format}`),
  imagePreviewUrl: (imageId) => withKey(`${BASE}/images/${imageId}/preview.png`),
  imageTileUrl: (imageId) => withKey(`${BASE}/images/${imageId}/tiles/{z}/{x}/{y}.png`),
  // --- Model Setup wizard: Dataset export + Train steps ---
  exportDataset: (payload) =>
    request(`${BASE}/training/export-dataset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  exportDatasetStatus: (jobId) => request(`${BASE}/training/export-dataset/${jobId}`).then(handle),
  exportDatasetCancel: (jobId) =>
    request(`${BASE}/training/export-dataset/${jobId}/cancel`, { method: "POST" }).then(handle),
  startTraining: (payload) =>
    request(`${BASE}/training/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  trainingStatus: (jobId) => request(`${BASE}/training/train/${jobId}`).then(handle),
  trainingCancel: (jobId) =>
    request(`${BASE}/training/train/${jobId}/cancel`, { method: "POST" }).then(handle),
  // Closes the Train -> Detect loop: turns a finished run's best.pt into a
  // real, selectable model (class names read back from that run's own
  // data.yaml server-side) -- see app.models.custom_registry.
  registerTrainedModel: (jobId, payload) =>
    request(`${BASE}/training/train/${jobId}/register-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  // --- Annotate step: class-labeled boxes/polygons saved per image ---
  listAnnotations: (imageId) => request(`${BASE}/images/${imageId}/annotations`).then(handle),
  addAnnotation: (imageId, payload) =>
    request(`${BASE}/images/${imageId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  deleteAnnotation: (imageId, annotationId) =>
    request(`${BASE}/images/${imageId}/annotations/${annotationId}`, { method: "DELETE" }).then(handle),
  clearAnnotations: (imageId) =>
    request(`${BASE}/images/${imageId}/annotations`, { method: "DELETE" }).then(handle),
};
