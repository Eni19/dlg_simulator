import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig(({ command, mode }) => {
  const publicEnv = loadEnv(mode, fileURLToPath(new URL(".", import.meta.url)), "VITE_");
  // Only used by standalone `npm run dev`. Vercel routes /api to the gateway.
  // APP_URL is deliberately read only by the gateway at function invocation time.
  const devEnv = command === "serve"
    ? loadEnv(mode, fileURLToPath(new URL(".", import.meta.url)), "DLG_DEV_")
    : {};
  return {
  define: {
    "import.meta.env.VITE_LAB_MODE": JSON.stringify(mode === "showcase" || process.env.VERCEL === "1" ? "showcase" : publicEnv.VITE_LAB_MODE || "full"),
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: devEnv.DLG_DEV_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
  };
});
