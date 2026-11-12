const BASE = "/api";

async function handle(res) {
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

export const api = {
  listModels: () => fetch(`${BASE}/models`).then(handle),
  getModel: (modelId) => fetch(`${BASE}/models/${modelId}`).then(handle),
  gsdCheck: (modelId, gsd) => fetch(`${BASE}/models/${modelId}/gsd-check?gsd=${gsd}`).then(handle),
  listImages: () => fetch(`${BASE}/images`).then(handle),
  getImage: (imageId) => fetch(`${BASE}/images/${imageId}`).then(handle),
  // Uses XMLHttpRequest (not fetch) because only XHR exposes real upload
  // progress events — needed for honest percentage feedback on multi-GB
  // GeoTIFF uploads, instead of a static "Uploading…" that looks frozen.
  uploadImage: (file, onProgress) =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/images`);
      xhr.upload.onprogress = (e) => {
        if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error body */ }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body);
        } else {
          const detail = body.detail;
          const message = typeof detail === "string" ? detail : detail?.message || `Request failed: ${xhr.status}`;
          const err = new Error(message);
          err.detail = detail;
          reject(err);
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload."));
      xhr.send(form);
    }),
  devices: () => fetch(`${BASE}/compute`).then(handle),
  startPrediction: (payload) =>
    fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  jobStatus: (jobId) => fetch(`${BASE}/predict/${jobId}`).then(handle),
  getGeojson: (jobId, modelId) => fetch(`${BASE}/results/${jobId}/${modelId}`).then(handle),
  listDatasetVersions: (modelId) => fetch(`${BASE}/datasets/${modelId}/versions`).then(handle),
  getDatasetVersion: (modelId, versionId) =>
    fetch(`${BASE}/datasets/${modelId}/versions/${versionId}`).then(handle),
  prepareDataset: (payload) =>
    fetch(`${BASE}/datasets/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  getEnvironment: () => fetch(`${BASE}/environment`).then(handle),
  startEnvironmentAction: (action, build, targets) =>
    fetch(`${BASE}/environment/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, build, targets: targets && targets.length ? targets : null }),
    }).then(handle),
  environmentActionStatus: (jobId) => fetch(`${BASE}/environment/actions/${jobId}`).then(handle),
  usageSummary: () => fetch(`${BASE}/usage`).then(handle),
  resultDownloadUrl: (jobId, modelId, format) =>
    format === "geojson" ? `${BASE}/results/${jobId}/${modelId}` : `${BASE}/results/${jobId}/${modelId}/${format}`,
  // --- Model Setup wizard: Dataset export + Train steps ---
  exportDataset: (payload) =>
    fetch(`${BASE}/training/export-dataset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  exportDatasetStatus: (jobId) => fetch(`${BASE}/training/export-dataset/${jobId}`).then(handle),
  exportDatasetCancel: (jobId) =>
    fetch(`${BASE}/training/export-dataset/${jobId}/cancel`, { method: "POST" }).then(handle),
  startTraining: (payload) =>
    fetch(`${BASE}/training/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),
  trainingStatus: (jobId) => fetch(`${BASE}/training/train/${jobId}`).then(handle),
  trainingCancel: (jobId) =>
    fetch(`${BASE}/training/train/${jobId}/cancel`, { method: "POST" }).then(handle),
};
