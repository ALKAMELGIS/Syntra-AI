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
  uploadImage: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/images`, { method: "POST", body: form }).then(handle);
  },
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
  startEnvironmentAction: (action, build) =>
    fetch(`${BASE}/environment/actions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, build }),
    }).then(handle),
  environmentActionStatus: (jobId) => fetch(`${BASE}/environment/actions/${jobId}`).then(handle),
  usageSummary: () => fetch(`${BASE}/usage`).then(handle),
  resultDownloadUrl: (jobId, modelId, format) =>
    format === "geojson" ? `${BASE}/results/${jobId}/${modelId}` : `${BASE}/results/${jobId}/${modelId}/${format}`,
};
