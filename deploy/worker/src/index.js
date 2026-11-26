// Syntra AI on syntra-ai.<account>.workers.dev: a plain redirect to the real
// engine, which runs where Cloudflare Workers cannot (FastAPI + GDAL/rasterio
// need a real Python process -- see backend/requirements-core.txt).
//
// This used to proxy /api/* to the engine and serve the built interface from
// a Workers assets binding, so the whole app looked same-origin. Dropped
// that: Cloudflare's own Git-integrated build for this Worker (Settings >
// Builds) kept uploading the wrong directory's files as assets no matter how
// Root directory was set, serving the raw, unbuilt frontend instead of its
// Vite build. A redirect needs nothing built or uploaded here at all, so
// there is nothing left for that pipeline to get wrong.
const ENGINE_ORIGIN = "https://syntra-ai-engine.onrender.com";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    return Response.redirect(ENGINE_ORIGIN + url.pathname + url.search, 302);
  },
};
