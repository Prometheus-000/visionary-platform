/**
 * The design-system build, as one command.
 *
 * Three steps that have to happen in this order and cannot be a plain `&&`
 * chain in package.json, because the first one deletes the third one's output:
 * `vite build` runs with `emptyOutDir`, so declarations emitted before it are
 * gone by the time it finishes. Running tsc second is not a preference.
 *
 * The last step exists because tsc faithfully copies the entry's
 * `import '../styles/ui.css'` into the emitted `.d.ts`, where it points at
 * nothing — the declaration tree has no stylesheet beside it, and a type
 * consumer following that import gets an unresolved-module error for a file
 * whose only job is a side effect at bundle time.
 */
import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'

const run = (...a) => execFileSync('npx', a, { stdio: 'inherit', cwd: import.meta.dirname })

run('vite', 'build', '--config', 'vite.lib.config.ts')
run('tsc', '-p', 'tsconfig.ds.json')

const d = new URL('./dist-ds/types/ds/index.d.ts', import.meta.url)
writeFileSync(d, readFileSync(d, 'utf8').replace(/^import '\.\.\/styles\/ui\.css';?\n/m, ''))

// The converter looks for types beside the entry it is given.
writeFileSync(new URL('./dist-ds/index.d.ts', import.meta.url), "export * from './types/ds/index'\n")
