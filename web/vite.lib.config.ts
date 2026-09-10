import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/**
 * The library build, beside the app build rather than instead of it.
 *
 * `vite.config.ts` builds the page that Modal serves; this one builds the five
 * primitives as something importable, for the design-system sync. They are two
 * outputs from one source tree, so this file exists rather than a mode flag —
 * `emptyOutDir` on either would otherwise delete the other's output.
 *
 * React is external: the consumer supplies it, and a second copy of React in
 * the bundle is two reconcilers and a hook that throws.
 */
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist-ds',
    emptyOutDir: true,
    sourcemap: false,
    // One stylesheet, not one per chunk — the sync links a single styles.css.
    cssCodeSplit: false,
    lib: {
      entry: 'src/ds/index.ts',
      formats: ['es'],
      fileName: () => 'index.js',
      cssFileName: 'visionary-ui',
    },
    rollupOptions: {
      external: ['react', 'react-dom', 'react/jsx-runtime', 'react-dom/client'],
    },
  },
})
