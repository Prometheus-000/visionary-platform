import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import { failed } from './api/client'
import { getState } from './api/routes'
import { autoGrow } from './console/fieldMax'
import { Gallery, useGallery } from './gallery/Gallery'
import { IconPanel, IconTrain } from './icons'
import { useStore } from './store'

/**
 * The shell: header, stage, canvas, console.
 *
 * Structure follows UI_HTML's, class for class, because the stylesheet is
 * being kept verbatim and it selects on `.top`, `.views`, `.stage`, `.canvas`,
 * `.console` and `.field`. Renaming any of those would mean rewriting the CSS,
 * which is the one thing the port is not doing.
 *
 * The canvas is the largest thing on screen and the console is a bar beneath
 * it — never a rail beside it. That is measured rather than preferred: fitting
 * each render aspect into a 1512x982 canvas leaves 0px of dead vertical space
 * at every aspect and 152–1068px horizontal, so the picture is height-bound
 * everywhere and the bar always comes out of the picture.
 */
export function App() {
  const { state, stateError, setState, setStateError } = useStore()
  const prompt = useStore((s) => s.prompt)
  const setPrompt = useStore((s) => s.setPrompt)

  const consoleRef = useRef<HTMLDivElement>(null)
  const fieldRef = useRef<HTMLTextAreaElement>(null)

  const { items, reload } = useGallery()
  const [galleryOpen, setGalleryOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(true)

  useEffect(() => {
    let live = true
    void (async () => {
      const r = await getState()
      if (!live) return
      // `{error}` rather than a throw — see api/client.ts. Treating this as an
      // exception is what left the composer looking like a deployment with no
      // weights on it.
      if (failed(r)) setStateError(r.error)
      else setState(r)
    })()
    return () => {
      live = false
    }
  }, [setState, setStateError])

  // The console has to watch itself, because the prompt is not the only thing
  // that grows: arming Regions adds a bar and picking pills adds a rail, and
  // both happen long after the last keystroke. Without this the field keeps
  // whatever height it won when it was the only claimant — measured, a long
  // prompt sat at 30.0% and climbed to 38.1% when the others arrived.
  //
  // It converges in one pass because `fieldMax` subtracts the field's own
  // height, so `other` does not move when the field does.
  useLayoutEffect(() => {
    const con = consoleRef.current
    const field = fieldRef.current
    if (!con || !field) return
    const grow = () => autoGrow(field, con)
    grow()
    const ro = new ResizeObserver(grow)
    ro.observe(con)
    window.addEventListener('resize', grow)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', grow)
    }
  }, [])

  useLayoutEffect(() => {
    autoGrow(fieldRef.current, consoleRef.current)
  }, [prompt])

  return (
    <>
      <header className="top">
        <button className="brand" id="go-home" type="button">
          Visionary
        </button>
        <span className="grow" />
        {/* Train is one door, labelled with where it leads rather than where
            you are, so two things never look equally selected. Generate has no
            nav item because it is not a place you go — it is the page. */}
        <button className="door" id="door" type="button">
          <IconTrain />
          Train
        </button>
        <button className="ico" id="t-drawer" title="Gallery" type="button"
                onClick={() => setDrawerOpen((d) => !d)}>
          <IconPanel />
        </button>
      </header>

      <div className="views">
        <div className={`view studio${drawerOpen ? '' : ' nodrawer'}`} id="v-generate">
          <div className="stage">
            <div className="canvas" id="canvas">
              {stateError ? (
                <div className="blank" id="canvas-empty">
                  <div className="err-box">{stateError}</div>
                </div>
              ) : (
                <div className="blank" id="canvas-empty">
                  {state ? `${state.models.length} models · ${state.loras.length} LoRAs` : 'Loading…'}
                </div>
              )}
            </div>

            <div className="console" ref={consoleRef}>
              <div className="field">
                <textarea
                  id="prompt"
                  ref={fieldRef}
                  rows={1}
                  placeholder="Describe an image…"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>
            </div>
          </div>

          <Gallery
            items={items}
            open={galleryOpen}
            onClose={() => setGalleryOpen(false)}
            onReload={() => {
              setGalleryOpen(true)
              void reload()
            }}
          />
        </div>
      </div>
    </>
  )
}
