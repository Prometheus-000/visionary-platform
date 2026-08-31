import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App'
// ui.css first: it is the extracted stylesheet and root.css corrects for the
// one thing React changes about the page's geometry, so it has to win.
import './styles/ui.css'
import './styles/root.css'
import './styles/theme.css'
import { keep, restore } from './keep'
import { applyTheme, loadTheme } from './theme/theme'

// Before first paint, so a Polar user never sees a Midnight flash. Midnight
// itself applies nothing — the stylesheet is the theme.
applyTheme(loadTheme())

const root = document.getElementById('root')
if (!root) throw new Error('#root is missing from index.html')

// **Also before first paint, and for the harder version of the same reason.**
// A theme arriving late is a flash; a scene arriving late is the composer
// painting empty and then filling in, which reads as the app having lost your
// work and changed its mind about it. `restore` swallows everything and resolves
// either way, so the render below is not gated on storage working — see
// `keep.ts`, where a browser with no IndexedDB is the session that shipped
// before this existed.
await restore()
keep()

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
