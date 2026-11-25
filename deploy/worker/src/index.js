// Syntra AI on syntra-ai.<account>.workers.dev: a thin edge proxy in front of
// the real engine, which runs where Cloudflare Workers cannot (FastAPI +
// GDAL/rasterio need a real Python process — see backend/requirements-core.txt).
//
// This Worker does two jobs only:
//   1. /api/*  -> forwarded to the engine (BACKEND_ORIGIN below).
//   2. everything else -> the built interface (frontend/dist), served from
//      the assets binding configured in wrangler.toml.
//
// X-Forwarded-Host is set to the Worker's own hostname so the engine's
// same-origin write check (backend/app/main.py: same_origin_writes) sees a
// Host that matches the browser's Origin header and does not refuse the
// request. Everything else (cookies, auth headers, body) passes through
// unchanged so sign-in sessions work exactly as if the engine were local.
const BACKEND_ORIGIN = "https://syntra-ai-engine.onrender.com";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/api/")) {
      const target = new URL(url.pathname + url.search, BACKEND_ORIGIN);
      const headers = new Headers(request.headers);
      headers.set("X-Forwarded-Host", url.host);
      headers.set("X-Forwarded-Proto", "https");

      const init = {
        method: request.method,
        headers,
        body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
        redirect: "manual",
      };
      const response = await fetch(target, init);
      return new Response(response.body, response);
    }

    return env.ASSETS.fetch(request);
  },
};
