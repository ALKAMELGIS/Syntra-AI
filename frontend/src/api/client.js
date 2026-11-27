// Every call the interface makes to the engine (/api/...), in one place.
// Same-origin by default. A build served from a different origin than the
// engine (e.g. a static host with no reverse proxy) sets VITE_API_BASE_URL to
// the engine's full URL (e.g. https://syntra-ai-engine.onrender.com/api).
const BASE = import.meta.env.VITE_API_BASE_URL || "/api";
export const API_BASE = BASE;
// the engine level this interface needs (backend/app/main.py API_LEVEL); lower = an engine started before an update
export const ENGINE_API = 5;

// Optional shared API key (matches SYNTRA_API_KEY on the backend). Set
// VITE_API_KEY in frontend/.env for builds that talk to a protected server.
const API_KEY = import.meta.env.VITE_API_KEY || "";

function request(url, init = {}) {
  const headers = { ...(init.headers || {}) };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  // fetch() omits cookies on a cross-origin request unless told otherwise, so the
  // session cookie would never be sent back once the interface and the engine are
  // on different origins (BASE above). Harmless to set when same-origin too.
  return fetch(url, { ...init, headers, credentials: "include" });
}

// For URLs the browser loads directly (<img>, <a download>), which cannot
// carry custom headers.
export function withKey(url) {
  if (!API_KEY) return url;
  return `${url}${url.includes("?") ? "&" : "?"}api_key=${encodeURIComponent(API_KEY)}`;
}

// 45 s · 3 min · 1 h 20 min
export function formatDuration(s) {
  if (s == null) return "";
  if (s < 60) return `${Math.max(1, Math.round(s))} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

// One readable message for every failed request: the server's own explanation
// when it sent one, otherwise what the status code means and what to do. The
// code is always shown, so a screenshot says what happened.
export function failureMessage(status, detail, upload = false) {
  const said = typeof detail === "string" ? detail : detail?.message;
  if (said) return said;
  const tip = upload ? " For files on this computer, use the This PC tab: it opens them without uploading." : "";
  if (status === 413) return `Too large to upload (HTTP 413).${tip}`;
  if (status === 507) return `The disk is full (HTTP 507).${tip}`;
  if (status === 502 || status === 503 || status === 504)
    return `The engine stopped during the request (HTTP ${status}). Check its window, start it again, then retry.${tip}`;
  if (status >= 500) return `The engine failed (HTTP ${status}). Its window shows the reason.${tip}`;
  return `Request failed (HTTP ${status}).${tip}`;
}

// A 401 from any data route means the session ended (signed out elsewhere,
// expired): tell the sign-in gate, which swaps the app for the sign-in page.
export const SIGNED_OUT = "syntra:signed-out";

// 402 from a Run: not enough credits. The top bar opens the credits window.
export const NEED_CREDITS = "syntra:need-credits";
// any change to the balance (a Run started, credits bought): the top bar re-reads it
export const CREDITS_CHANGED = "syntra:credits-changed";

async function handle(res) {
  if (res.status === 401 && !res.url.includes("/api/auth/")) window.dispatchEvent(new Event(SIGNED_OUT));
  if (res.status === 402) {
    const body = await res.clone().json().catch(() => ({}));
    window.dispatchEvent(new CustomEvent(NEED_CREDITS, { detail: body.detail || {} }));
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail;
    const message = res.status === 402 && detail?.message ? detail.message : failureMessage(res.status, detail);
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

const UPLOAD_CHUNK_BYTES = 32 * 1024 * 1024;   // grid cell; small enough to retry cheaply after a blip
const UPLOAD_CHUNK_RETRIES = 5;
const UPLOAD_PARALLEL = 4;   // concurrent PUTs: several chunks in flight beats one at a time over a
                              // single slow or high-latency connection, which is the usual bottleneck
                              // for a large, high-resolution drone GeoTIFF.

// One chunk PUT with real upload-progress events (XHR, not fetch). Resolves
// with { received, complete }; rejects with an Error carrying .status.
function putChunk(url, blob, offset, onChunkProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    if (API_KEY) xhr.setRequestHeader("X-API-Key", API_KEY);
    xhr.setRequestHeader("X-Upload-Offset", String(offset));
    xhr.withCredentials = true;
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onChunkProgress?.(e.loaded); };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error body */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body);
      } else {
        const err = new Error(body.detail || `Upload chunk failed (HTTP ${xhr.status}).`);
        err.status = xhr.status;
        reject(err);
      }
    };
    xhr.onerror = () => { const err = new Error("network"); err.status = 0; reject(err); };
    xhr.send(blob);
  });
}

// Fixed-size grid over the whole file; a range the server already has
// (returned by /resumable on resume) is dropped so only what is missing is
// sent, whole cells at a time.
function _missingChunks(fileSize, ranges) {
  const covered = (pos) => ranges.some(([s, e]) => pos >= s && pos < e);
  const chunks = [];
  for (let start = 0; start < fileSize; start += UPLOAD_CHUNK_BYTES) {
    const end = Math.min(start + UPLOAD_CHUNK_BYTES, fileSize);
    if (!covered(start)) chunks.push([start, end]);
  }
  return chunks;
}

async function uploadResumable(file, onProgress, metadataFile = null) {
  if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
    throw new Error("Only GeoTIFF (.tif/.tiff) uploads are supported.");
  }
  const { key, ranges } = await request(`${BASE}/images/resumable`, json({
    name: file.name, size: file.size, modified: String(file.lastModified || 0),
  })).then(handle);

  const chunkLoaded = new Map();   // offset -> bytes sent so far, for the combined progress bar
  let alreadyDone = ranges.reduce((n, [s, e]) => n + (e - s), 0);
  const report = () => {
    const sent = alreadyDone + [...chunkLoaded.values()].reduce((a, b) => a + b, 0);
    onProgress?.(Math.min(sent / file.size, 1));
  };
  report();

  const queue = _missingChunks(file.size, ranges);
  let cursor = 0;
  let firstError = null;

  async function worker() {
    for (;;) {
      if (firstError) return;
      const my = cursor++;
      if (my >= queue.length) return;
      const [start, end] = queue[my];
      const blob = file.slice(start, end);
      let attempt = 0;
      for (;;) {
        try {
          const result = await putChunk(`${BASE}/images/resumable/${key}`, blob, start, (loaded) => {
            chunkLoaded.set(start, loaded);
            report();
          });
          chunkLoaded.set(start, end - start);
          report();
          if (result.complete) return;
          break;
        } catch (err) {
          if (err.status !== 0 || ++attempt > UPLOAD_CHUNK_RETRIES) {
            firstError = err;
            return;
          }
          await new Promise((r) => setTimeout(r, 500 * 2 ** (attempt - 1)));   // 0.5s, 1s, 2s, 4s, 8s
        }
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(UPLOAD_PARALLEL, Math.max(queue.length, 1)) }, worker));

  if (firstError) {
    const sentFraction = [...chunkLoaded.values()].reduce((a, b) => a + b, 0) + alreadyDone;
    const up = await fetch(`${BASE}/health`).then((r) => r.ok).catch(() => false);
    if (firstError.status === 0) {
      throw new Error(up
        ? `The upload was cut off (at ${Math.round((100 * sentFraction) / file.size)}%). The server kept what arrived — try again with the same file to continue from there, instead of from the start.`
        : "Couldn't reach the Syntra AI server. Check that the backend window is still open, then try again with the same file to continue from where it left off.");
    }
    throw firstError;
  }

  const form = new FormData();
  if (metadataFile) form.append("metadata", metadataFile);
  return request(`${BASE}/images/resumable/${key}/finish`, { method: "POST", body: form }).then(handle);
}

export const api = {
  libraryPushModel: (group, body) => request(`${BASE}/library/${encodeURIComponent(group)}/push`, json(body)).then(handle),
  googleBasemap: () => request(`${BASE}/basemaps/google`).then(handle),
  googleBasemapSave: (key) => request(`${BASE}/basemaps/google`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) }).then(handle),
  basemapSources: () => request(`${BASE}/images/basemap-sources`).then(handle),
  fromBasemap: (body) => request(`${BASE}/images/from-basemap`, json(body)).then(handle),
  hosting: () => request(`${BASE}/compute/hosting`).then(handle),
  runpodStatus: () => request(`${BASE}/cloud/runpod`).then(handle),
  runpodSave: (key) => request(`${BASE}/cloud/runpod`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) }).then(handle),
  cloudTrain: (body) => request(`${BASE}/cloud/train`, json(body)).then(handle),
  cloudStop: (jobId) => request(`${BASE}/cloud/train/${jobId}/stop`, { method: "POST" }).then(handle),
  hfStatus: () => request(`${BASE}/samples/hf`).then(handle),
  hfSave: (body) => request(`${BASE}/samples/hf`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle),
  pushSamples: (dataset_job_id) => request(`${BASE}/samples/push`, json({ dataset_job_id })).then(handle),
  pushStatus: (jobId) => request(`${BASE}/samples/push/${jobId}`).then(handle),
  // --- sample library: classes, dataset versions, models (data/classes, datasets, models) ---
  library: () => request(`${BASE}/library`).then(handle),
  // --- credits: the signed-in account ---
  billingMe: () => request(`${BASE}/billing/me`).then(handle),
  billingCheckout: (packageId) => request(`${BASE}/billing/checkout`, json({ package_id: packageId })).then(handle),
  // --- the owner's console ---
  adminSummary: () => request(`${BASE}/admin/summary`).then(handle),
  adminUsers: () => request(`${BASE}/admin/users`).then(handle),
  adminCredits: (id, body) => request(`${BASE}/admin/users/${id}/credits`, json(body)).then(handle),
  adminPatchUser: (id, body) => request(`${BASE}/admin/users/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle),
  adminDeleteUser: (id) => request(`${BASE}/admin/users/${id}`, { method: "DELETE" }).then(handle),
  adminLedger: (userId) => request(`${BASE}/admin/ledger${userId ? `?user_id=${userId}` : ""}`).then(handle),
  adminPayments: () => request(`${BASE}/admin/payments`).then(handle),
  adminPricing: () => request(`${BASE}/admin/pricing`).then(handle),
  adminSavePricing: (p) => request(`${BASE}/admin/pricing`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) }).then(handle),
  adminStripe: () => request(`${BASE}/admin/stripe`).then(handle),
  adminSaveStripe: (b) => request(`${BASE}/admin/stripe`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then(handle),
  adminSignin: () => request(`${BASE}/admin/signin`).then(handle),
  adminSaveSignin: (b) => request(`${BASE}/admin/signin`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then(handle),

  // --- remote sensing: Sentinel-2 indices over an AOI ---
  rsStatus: () => request(`${BASE}/rs`).then(handle),
  rsScenes: (body) => request(`${BASE}/rs/scenes`, json(body)).then(handle),
  rsRender: (body) => request(`${BASE}/rs/render`, json(body)).then(handle),
  rsAttributes: (body) => request(`${BASE}/rs/attributes`, json(body)).then(handle),
  rsTimeline: (body) => request(`${BASE}/rs/timeline`, json(body)).then(handle),
  rsExport: async (body) => {
    const res = await request(`${BASE}/rs/export`, json(body));
    if (!res.ok) return handle(res);
    return res.blob();
  },
  libraryBuild: (slug, body = {}) => request(`${BASE}/library/${encodeURIComponent(slug)}/build`, json(body)).then(handle),
  libraryApprove: (group, version) => request(`${BASE}/library/${encodeURIComponent(group)}/approve`, json({ version })).then(handle),
  estimate: (body) => request(`${BASE}/predict/estimate`, json(body)).then(handle),
  browseLocal: (path) => request(`${BASE}/images/local/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`).then(handle),
  importLocal: (path) => request(`${BASE}/images/local`, json({ path })).then(handle),
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
  // Sent in chunks, resumable by the file's own identity (name, size, last
  // modified), not by anything the browser remembers: a File object cannot
  // survive a crash or a reload anyway, so re-picking the same file in the
  // chooser after one continues from the byte already on the server instead
  // of starting over. Uses XMLHttpRequest for each chunk PUT (not fetch)
  // because only XHR exposes real upload progress events.
  uploadImage: (file, onProgress, metadataFile = null) => uploadResumable(file, onProgress, metadataFile),
  devices: () => request(`${BASE}/compute`).then(handle),
  startPrediction: (payload) =>
    request(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle).then((r) => { window.dispatchEvent(new Event(CREDITS_CHANGED)); return r; }),
  job: (jobId) => request(`${BASE}/jobs/${jobId}`).then(handle),
  listJobs: (kind, limit = 20, imageId = null) =>
    request(`${BASE}/jobs?kind=${kind}&limit=${limit}${imageId ? `&image_id=${encodeURIComponent(imageId)}` : ""}`).then(handle),
  hideJob: (jobId) => request(`${BASE}/jobs/${jobId}/hide`, { method: "POST" }).then(handle),
  jobStatus: (jobId) => request(`${BASE}/predict/${jobId}`).then(handle),
  getReview: (jobId, modelId) => request(`${BASE}/results/${jobId}/${modelId}/review`).then(handle),
  putReview: (jobId, modelId, changes) => request(`${BASE}/results/${jobId}/${modelId}/review`,
    { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ changes }) }).then(handle),
  applyReview: (jobId, modelId) => request(`${BASE}/results/${jobId}/${modelId}/review/apply`, { method: "POST" }).then(handle),
  refineResult: (jobId, modelId) => request(`${BASE}/results/${jobId}/${modelId}/refine`, { method: "POST" }).then(handle),
  scoreResult: (jobId, modelId, body = {}) => request(`${BASE}/results/${jobId}/${modelId}/score`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle),
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
  // display = {bands: [r, g, b] | null, stretch: "2-98" | null}: the map's own choice, never the detector's
  imageTileUrl: (imageId, display = null) => {
    const q = [];
    if (display?.bands?.length) q.push(`bands=${display.bands.join(",")}`);
    if (display?.stretch) q.push(`stretch=${display.stretch}`);
    return withKey(`${BASE}/images/${imageId}/tiles/{z}/{x}/{y}.png${q.length ? `?${q.join("&")}` : ""}`);
  },
  deleteLibraryCopy: (folder) => request(`${BASE}/images/library/${folder}`, { method: "DELETE" }).then(handle),
  vectorRead: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`${BASE}/vector/read`, { method: "POST", body: form }).then(handle);
  },
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
  // --- click to segment (SAM 2.1, local) and outline fitting ---
  segmentStatus: () => request(`${BASE}/segment/status`).then(handle),
  segmentInstall: (model = "sam2.1_tiny") =>
    request(`${BASE}/segment/install`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }),
    }).then(handle),
  segmentInstallStatus: (jobId) => request(`${BASE}/segment/install/${jobId}`).then(handle),
  segmentPrepare: (imageId, payload) => request(`${BASE}/images/${imageId}/segment/prepare`, json(payload)).then(handle),
  segmentAll: (imageId, payload) => request(`${BASE}/images/${imageId}/segment-all`, json(payload)).then(handle),
  similarObjects: (imageId, payload) => request(`${BASE}/images/${imageId}/similar`, json(payload)).then(handle),
  segmentClick: (imageId, payload) =>
    request(`${BASE}/images/${imageId}/segment`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    }).then(handle),
  updateAnnotation: (imageId, annotationId, patch) =>
    request(`${BASE}/images/${imageId}/annotations/${annotationId}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
    }).then(handle),
  deleteAnnotation: (imageId, annotationId) =>
    request(`${BASE}/images/${imageId}/annotations/${annotationId}`, { method: "DELETE" }).then(handle),
  previewLabelImport: (imageId, file) => {
    const fd = new FormData(); fd.append("file", file);
    return request(`${BASE}/images/${imageId}/annotations/import/preview`, { method: "POST", body: fd }).then(handle);
  },
  importLabels: (imageId, file, fields) => {
    const fd = new FormData(); fd.append("file", file);
    for (const [k, v] of Object.entries(fields)) if (v) fd.append(k, v);
    return request(`${BASE}/images/${imageId}/annotations/import`, { method: "POST", body: fd }).then(handle);
  },
  clearAnnotations: (imageId) =>
    request(`${BASE}/images/${imageId}/annotations`, { method: "DELETE" }).then(handle),
};
