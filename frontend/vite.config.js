import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The backend port is configurable because 8000 is blocked by Windows on
// some machines (WinError 10013 -- Hyper-V/WSL reserve certain port ranges).
// Set VITE_API_PORT in frontend/.env (or the shell environment) to match
// whatever port you actually started `uvicorn app.main:app --port <N>` on.

// Node closes any request that takes longer than 5 minutes (requestTimeout
// 300000). A multi-GB GeoTIFF upload, plus reading it on the server, takes
// longer, and the browser only sees "network error". Uploads and long API
// calls get no time limit here; VITE_REQUEST_TIMEOUT_MS overrides it (tests).
function noRequestTimeout(ms) {
  return {
    name: "syntra-no-request-timeout",
    configureServer(server) {
      const apply = () => {
        const s = server.httpServer;
        if (!s) return;
        s.requestTimeout = ms;
        s.headersTimeout = ms ? Math.min(ms, 60000) : 0;
        s.timeout = 0;
      };
      apply();
      server.httpServer?.on("listening", apply);
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiPort = env.VITE_API_PORT || "8000";
  const requestTimeout = Number(env.VITE_REQUEST_TIMEOUT_MS || 0);

  return {
    plugins: [react(), noRequestTimeout(requestTimeout)],
    server: {
      // 5173 is Vite's default and often taken by another project; strictPort
      // fails loudly instead of silently moving to a different port.
      port: Number(env.VITE_PORT || 5291),
      strictPort: true,
      proxy: {
        "/api": { target: `http://localhost:${apiPort}`, timeout: 0, proxyTimeout: 0 },
      },
    },
  };
});
