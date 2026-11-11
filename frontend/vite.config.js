import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The backend port is configurable because 8000 is blocked by Windows on
// some machines (WinError 10013 -- Hyper-V/WSL reserve certain port ranges).
// Set VITE_API_PORT in frontend/.env (or the shell environment) to match
// whatever port you actually started `uvicorn app.main:app --port <N>` on.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiPort = env.VITE_API_PORT || "8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": `http://localhost:${apiPort}`,
      },
    },
  };
});
