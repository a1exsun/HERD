import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/hero/",
  plugins: [react()],
  build: {
    emptyOutDir: false,
    outDir: fileURLToPath(new URL("../../packages/hero-dsh-plugin/dist/web", import.meta.url))
  }
});
