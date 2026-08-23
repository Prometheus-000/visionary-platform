import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { Viewfinder } from './blocking/Viewfinder'
import './blocking/blocking.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode><Viewfinder /></StrictMode>,
)
