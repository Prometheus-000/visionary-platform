import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App'
// ui.css first: it is the extracted stylesheet and root.css corrects for the
// one thing React changes about the page's geometry, so it has to win.
import './styles/ui.css'
import './styles/root.css'

const root = document.getElementById('root')
if (!root) throw new Error('#root is missing from index.html')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
