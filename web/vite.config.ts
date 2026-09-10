import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Where /api/* goes in development.
//
// The default is tools/preview_ui.py, deliberately: it serves the real
// compilers and the real shot vocabulary (pulled out of app.py by AST) against
// stubbed jobs and files, so the whole front end is workable with no Modal
// account, no GPU, no deployment and nothing billed. Pointing this at a
// deployed URL is opt-in and one env var, because the moment it is the default
// somebody's CSS change costs a cold start.
//
//     VISIONARY_API=https://…modal.run npm run dev
const API = process.env.VISIONARY_API || 'http://localhost:8791'

export default defineConfig({
  plugins: [react()],
  server: {
    // PORT wins so two sessions can each run a dev server — the launcher
    // assigns one when 5173 is already held by another chat's server.
    port: Number(process.env.PORT) || 5173,
    // Every route the page talks to is under /api/, except the two that serve
    // bytes off the volume by path. Proxying the prefixes rather than listing
    // 33 routes means a new route needs no change here.
    proxy: Object.fromEntries(
      ['/api'].map((p) => [p, { target: API, changeOrigin: true }]),
    ),
  },
  build: {
    // Served from the web container as a static bundle beside app.py, so the
    // paths in index.html have to work from wherever it is mounted.
    outDir: 'dist',
    emptyOutDir: true,
    // Off. The map is 1.4 MB against a 313 kB bundle — four times the thing it
    // describes — and it would be built into the image and served off the same
    // container the UI comes from. Debugging a deployed build happens by
    // reproducing it locally with `npm run dev`, where the map is free.
    sourcemap: false,
  },
})
