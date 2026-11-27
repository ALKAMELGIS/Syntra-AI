// Syntra AI on syntra-ai.<account>.workers.dev: a full transparent reverse
// proxy in front of the real engine, which runs where Cloudflare Workers
// cannot (FastAPI + GDAL/rasterio need a real Python process -- see
// backend/requirements-core.txt). Every path -- the interface's HTML/JS/CSS,
// not just /api/* -- is forwarded and the response streamed straight back,
// so the browser's address bar stays on this domain the whole time; it never
// sees or redirects to the engine's own onrender.com URL.
//
// This used to serve the built interface from a Workers assets binding
// (fast, no round trip for static files) and only proxy /api/*. Dropped that
// for proxying everything instead: Cloudflare's own Git-integrated build for
// this Worker (Settings > Builds) kept uploading the wrong directory's files
// as assets no matter how Root directory was set, serving the raw, unbuilt
// frontend instead of its Vite build. Proxying needs nothing built or
// uploaded here at all -- only this script -- so there is nothing left for
// that pipeline to get wrong. The cost is a small extra hop per request,
// not a difference the user will notice.
//
// X-Forwarded-Host is set to this Worker's own hostname so the engine's
// same-origin write check (backend/app/main.py: same_origin_writes) sees a
// Host that matches the browser's real Origin header and does not refuse
// the request. Because the browser only ever talks to this domain, any
// Set-Cookie the engine sends is scoped to this domain too -- no cross-site
// cookie behaviour to worry about, unlike calling the engine's own origin
// directly from a different site.
const ENGINE_ORIGIN = "https://syntra-ai-engine.onrender.com";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = new URL(url.pathname + url.search, ENGINE_ORIGIN);

    const headers = new Headers(request.headers);
    headers.set("Host", target.host);
    headers.set("X-Forwarded-Host", url.host);
    headers.set("X-Forwarded-Proto", "https");

    const init = {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "manual",
    };

    const response = await fetch(target, init);

    // A redirect from the engine would otherwise point at its own onrender.com
    // URL; rewrite it back to this domain so the browser never sees it.
    if (response.status >= 300 && response.status < 400 && response.headers.has("location")) {
      const loc = new URL(response.headers.get("location"), target);
      if (loc.origin === ENGINE_ORIGIN) {
        const out = new Headers(response.headers);
        out.set("location", url.origin + loc.pathname + loc.search);
        return new Response(response.body, { status: response.status, headers: out });
      }
    }

    return new Response(response.body, response);
  },
};
