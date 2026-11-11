const BASE = "/api";

async function handle(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  listModels: () => fetch(`${BASE}/models`).then(handle),
  listImages: () => fetch(`${BASE}/images`).then(handle),
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
};
