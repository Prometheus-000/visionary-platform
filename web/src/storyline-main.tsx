import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { Sandbox } from './storyline/Sandbox'
import './storyline/sandbox.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode><Sandbox /></StrictMode>,
)
