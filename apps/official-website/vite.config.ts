import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Relative asset paths so the build works under the GitHub Pages project path (/HERD/).
  base: './',
  plugins: [react()],
})
