import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { readExperiments } from './scripts/experiments.mjs'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: { __EXPERIMENTS__: JSON.stringify(readExperiments()) },
})
