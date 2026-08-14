import { useCallback, useEffect, useState } from 'react'

import { failed } from './api/client'
import { fileUrl, getState } from './api/routes'
import { Canvas } from './canvas/Canvas'
import { useGenerate } from './canvas/useGenerate'
import { Console } from './console/Console'
import { Gallery, useGallery } from './gallery/Gallery'
import { LastShot } from './gallery/LastShot'
import { MetaSheet } from './gallery/MetaSheet'
import { Viewer } from './gallery/Viewer'
import type { GalleryItem } from './gallery/types'
import { IconBack, IconGear, IconPanel, IconTrain } from './icons'
import { fileToB64, toB64 } from './media/files'
import { Settings } from './settings/Settings'
import { Train } from './train/Train'
import { supports, useStore } from './store'
import { useVideo } from './video/useVideo'

/**
 * The shell: header, stage, canvas, console.
 *
 * Structure follows UI_HTML's, class for class, because `styles/ui.css` was lifted out of
 * it byte for byte and selects on `.top`, `.views`, `.stage`, `.canvas`, `.console` and
 * `.field`. UI_HTML is deleted and the stylesheet is the source now, so new rules are
 * written here like anywhere else — but renaming one of those classes still costs a
 * rewrite of the block that measures against it, which is the expensive part.
 *
 * **Generate is the page, not a destination.** It has no nav item; the wordmark is how you
 * get back to it. Train is one door, labelled with where it leads rather than where you
 * are, so two things never look equally selected — and it carries the training run's
 * progress, because a run lasts hours and you are meant to leave and keep working.
 */
export function App() {
  const s = useStore()
  const { items, reload, record, drop, total, behind } = useGallery()
  const [galleryOpen, setGalleryOpen] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [shown, setShown] = useState<{ rows: GalleryItem[]; i: number } | null>(null)
  const [meta, setMeta] = useState<GalleryItem | null>(null)

  const landed = useCallback((it: GalleryItem) => {
    // A result landed: the boxes come off the picture, and the gallery has a new newest.
    // Off on *every* land, which is the whole "editing is a choice" rule — the regions
    // themselves are untouched and go with the next run, they are just not drawn.
    useStore.getState().setEdit('off')
    // The new newest, now, from the run itself. The listing is asked afterwards and
    // fills in the sidecar; it is no longer what decides whether the thing you just
    // made is on screen.
    record(it)
    // Deliberately not in this tick. The canvas paints its stills from /api/file in the
    // same moment, and /api/gallery spends up to half a second inside `_reload_insist`
    // taking the volume's reload lock — the one those stills need. Handing them the
    // window first costs the gallery a beat nobody is watching and buys the picture
    // someone is.
    const t = window.setTimeout(() => void reload(), 600)
    return () => window.clearTimeout(t)
  }, [reload, record])

  const gen = useGenerate(landed)
  const vid = useVideo(landed)

  const fire = useCallback(() => {
    if (useStore.getState().kind === 'image') void gen.start()
    else void vid.start()
  }, [gen, vid])

  const stopRun = useCallback(() => {
    if (useStore.getState().kind === 'image') void gen.cancel()
    else void vid.cancel()
  }, [gen, vid])

  /* Clears the canvas back to the frame, where the boxes are drawn and draggable. It is
     no longer the *only* way to reach geometry — ⌘-click over a render does that — but
     it is still where you go to start a set over. `geometry` rather than `off` because
     with no render there is nothing to keep clean. Shared by the canvas ✕ and the ⌫
     shortcut so the two cannot drift. */
  const clearCanvas = useCallback(() => {
    const st = useStore.getState()
    if (st.kind === 'image') gen.clear()
    else vid.clear()
    st.setEdit('geometry')
  }, [gen, vid])

  const reloadState = useCallback(async () => {
    const r = await getState()
    // `{error}` rather than a throw — see api/client.ts. Treating this as an exception is
    // what left the composer looking like a deployment with no weights on it.
    if (failed(r)) useStore.getState().setStateError(r.error)
    else useStore.getState().setState(r)
  }, [])

  useEffect(() => {
    void reloadState()
  }, [reloadState])

  // The GPU pickers are built once from what the deployment offers: the list only changes
  // on redeploy, and rebuilding it would reset a card chosen between two polls.
  useEffect(() => {
    if (!s.state || s.gpu.image) return
    s.setGpu({ image: s.state.gpus.image.default, video: s.state.gpus.video.default })
  }, [s])

  /* A file is over the window: light every place it could go. See the drag-intent block in
     the stylesheet for what that means and why it is the only moment this app is willing to
     spend pixels explaining itself.

     Driven off `dragover` and a timer rather than dragenter/dragleave counting. The counting
     version is the textbook one and is wrong here: every child element the cursor crosses
     fires its own leave, and on a page whose targets contain images, boxes and eight resize
     handles the depth counter drifts and the reveal strobes. `dragover` repeats while the
     drag is alive, so "still dragging" is a fact the browser re-states every few hundred
     milliseconds, and the only thing that needs guessing is when it stopped. */
  useEffect(() => {
    let off: number | undefined
    const over = (e: DragEvent) => {
      // Files only. Dragging a text selection out of the prompt field must not make the
      // page look like it wants to eat it.
      if (![...(e.dataTransfer?.types ?? [])].includes('Files')) return
      document.body.classList.add('dragging')
      // The same fact in the two places that need it. The stylesheet reads the class to
      // light every eligible target; `RegionLayer` reads the flag to bring the boxes
      // back over a finished render, because a drop cannot land on a box that is at
      // `pointer-events:none` — which is the state the port shipped in, and it made the
      // one gesture nobody discovers on their own undiscoverable by construction.
      useStore.getState().setFileOver(true)
      window.clearTimeout(off)
      // `dragend` fires on a drag that started inside the page; `drop` fires on one that
      // came from outside and landed. Neither fires when a drag leaves the window
      // entirely, which is what the timer is for.
      off = window.setTimeout(() => {
        document.body.classList.remove('dragging')
        useStore.getState().setFileOver(false)
      }, 300)
    }
    const stop = () => {
      window.clearTimeout(off)
      document.body.classList.remove('dragging')
      useStore.getState().setFileOver(false)
    }
    window.addEventListener('dragover', over)
    window.addEventListener('drop', stop)
    window.addEventListener('dragend', stop)
    return () => {
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', stop)
      window.removeEventListener('dragend', stop)
      stop()
    }
  }, [])

  /* What is on the canvas right now, if anything — the thing Space acts on. */
  const canvasSrc = s.kind === 'video'
    ? (vid.run.jobId && vid.run.file ? fileUrl(vid.run.jobId, vid.run.file) : null)
    : (gen.run.jobId && gen.run.files[0] ? fileUrl(gen.run.jobId, gen.run.files[0]) : null)

  const lightbox = useCallback((src: string, kind: 'image' | 'video') => {
    setShown({ rows: [{ job_id: '', kind, files: [], src }], i: 0 })
  }, [])

  /* Space, because ⌘Space is Spotlight on a stock Mac and never reaches the page. Guarded
     on where the caret is rather than on a modifier — a space inside the prompt is a
     space. */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key !== ' ' && e.code !== 'Space') return
      if (e.ctrlKey || e.altKey || e.shiftKey || e.metaKey) return
      const t = e.target as HTMLElement | null
      if (t?.matches?.('input,textarea,select') || t?.isContentEditable) return
      if (document.querySelector('.lb,.menu,.pal,.scrim')) return
      if (!canvasSrc) return
      e.preventDefault()
      lightbox(canvasSrc, s.kind)
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [canvasSrc, s.kind, lightbox])

  /* ⌘/Ctrl+Enter generates from anywhere — the point being *from a region's prompt
     field*, which is what turns the edit-and-regenerate loop into one keystroke. The
     main prompt's own textarea already binds it (see Field), so the caret being there
     is skipped to avoid firing twice. */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key !== 'Enter' || !(e.metaKey || e.ctrlKey) || e.isComposing) return
      const t = e.target as HTMLElement | null
      if (t?.closest?.('#prompt,#neg')) return
      if (gen.run.running || vid.run.running) return
      e.preventDefault()
      fire()
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [fire, gen.run.running, vid.run.running])

  /* Escape puts the picture back. The layer has its own Escape — it is what selects the
     frame — but that one is a handler on the layer, so it only fires when focus is
     already inside it, and the gesture that reveals geometry is a ⌘-click that leaves
     focus on the body. So the way *out* of the mode was unreachable from the state the
     way *in* leaves you in, which the region check found on its first run against it.
     Guarded so anything modal still wins: a lightbox or a menu owns Escape while open. */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (document.querySelector('.lb,.menu,.pal,.scrim')) return
      if (useStore.getState().edit === 'off') return
      // Only over a render. On the frame there is nothing to return to, and `off`
      // there would hide the boxes with no way to ask for them back.
      if (!document.querySelector('#gen-out .film-cell')) return
      e.preventDefault()
      useStore.getState().setEdit('off')
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [])

  /* ⌫ / Delete clears the canvas, so starting over is a keystroke rather than a hunt
     for the small ✕. Guarded to the one meaning it can have: not while typing, and not
     while a region box is focused — there the same key deletes the box, and stealing it
     would be the keyboard fault the canvas-native regions redesign set out to fix. */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key !== 'Backspace' && e.key !== 'Delete') return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const t = e.target as HTMLElement | null
      if (t?.matches?.('input,textarea,select') || t?.isContentEditable) return
      if (t?.closest?.('#region-layer')) return
      if (document.querySelector('.lb,.menu,.pal,.scrim')) return
      if (!canvasSrc) return
      e.preventDefault()
      clearCanvas()
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [canvasSrc, clearCanvas])

  /**
   * The hand-off. A still you just made becomes the thing the next clip animates, without a
   * download and a re-upload — which is the whole point of image and video sharing one
   * workspace.
   *
   * A reference moves the model too. Only one family has a reference checkpoint, so
   * switching to it is the useful reading of the button — the alternative is accepting the
   * image and then dropping it the moment the panel redraws.
   *
   * The bytes are fetched rather than reused from a data URL: the canvas stills are a
   * streamed `<img src>`, so the base64 the video side needs does not exist client-side.
   */
  const handoff = useCallback(
    async (jobId: string, file: string, as: 'first' | 'reference' | 'refvideo') => {
      const b64 = await fileToB64(fileUrl(jobId, file))
      if (!b64) {
        alert('Could not read that file.')
        return
      }
      const st = useStore.getState()
      if (as === 'first') {
        st.setKeyframe('first', b64)
      } else {
        if (!supports(st).references) {
          const m = st.state?.video_models.find((x) => x.supports.references && x.ready)
          if (!m) {
            alert('References need MiniMax-H3 — download it under Settings.')
            return
          }
          st.setVid({ model: m.key })
        }
        const img = as === 'reference'
        const max = img ? (st.state?.max_refs ?? 9) : (st.state?.max_ref_videos ?? 3)
        const bucket = img ? st.refs : st.refVids
        if (bucket.length >= max) {
          alert(`${max} references is the model's limit.`)
          return
        }
        if (img) st.setRefs([...bucket, b64])
        else st.setRefVids([...bucket, b64])
      }
      st.setKind('video')
      setGalleryOpen(false)
      setShown(null)
      requestAnimationFrame(() => document.querySelector<HTMLTextAreaElement>('#prompt')?.focus())
    },
    [],
  )

  const train = s.mode === 'train'

  return (
    <>
      <header className="top">
        <button className="brand" id="go-home" type="button"
                onClick={() => { setGalleryOpen(false); s.setMode('generate') }}>
          Visionary
        </button>
        <span className="grow" />
        <Door />
        {/* The drawer toggle is a Generate control; it has nothing to say in Train. */}
        {!train && (
          <button className={`ico${drawerOpen ? ' on' : ''}`} id="t-drawer" title="Gallery"
                  type="button" onClick={() => setDrawerOpen((d) => !d)}>
            <IconPanel />
          </button>
        )}
        <button className="ico" id="t-settings" title="Settings" type="button"
                onClick={() => { setSettingsOpen(true); void reloadState() }}>
          <IconGear />
        </button>
      </header>

      <div className="views">
        {train ? (
          <Train onLightbox={(src) => lightbox(src, 'image')} />
        ) : (
          <div className={`view studio${drawerOpen ? '' : ' nodrawer'}`} id="v-generate">
            <div className="stage">
              <Canvas
                run={gen.run}
                vidRun={vid.run}
                onOpen={(jobId, i) => setShown({
                  rows: gen.run.files.map((f) => ({ job_id: jobId, kind: 'image', files: [f] })),
                  i,
                })}
                onOpenVideo={(src) => lightbox(src, 'video')}
                onHandoff={(jobId, file, as) => void handoff(jobId, file, as)}
                onFirstFrame={async (f) => {
                  const b64 = await toB64(f)
                  if (b64) useStore.getState().setKeyframe('first', b64)
                  else alert('Could not read that image.')
                }}
                onClear={clearCanvas}
                blank={
                  s.stateError ? <div className="err-box">{s.stateError}</div>
                    : s.state ? null
                    : 'Loading…'
                }
              />
              <Console run={gen.run} vidRun={vid.run} onGenerate={fire} onStop={stopRun}
                       lastShot={<LastShot items={items}
                                           onOpen={(rows, i) => setShown({ rows, i })} />} />
            </div>

            <Gallery
              items={items}
              total={total}
              behind={behind}
              open={galleryOpen}
              onClose={() => setGalleryOpen(false)}
              onReload={() => {
                setGalleryOpen(true)
                void reload()
              }}
              onDropped={drop}
              onMeta={setMeta}
              onHandoff={(it, as) => void handoff(it.job_id, it.files[0] ?? '', as)}
            />
          </div>
        )}
      </div>

      {shown && (
        <Viewer
          items={shown.rows}
          index={shown.i}
          onIndex={(i) => setShown((v) => (v ? { ...v, i } : v))}
          onClose={() => setShown(null)}
          onAll={() => { setShown(null); setGalleryOpen(true); void reload() }}
        />
      )}

      {meta && <MetaSheet item={meta} onClose={() => setMeta(null)} />}

      <Settings
        state={s.state}
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onReload={() => void reloadState()}
      />
    </>
  )
}

/**
 * One button instead of two, and no moment where both look equally selectable.
 *
 * In Generate it doubles as the training readout: a run outlives the visit that started it,
 * and the control that takes you back is the honest place to say how far along it is.
 */
function Door() {
  const mode = useStore((st) => st.mode)
  const setMode = useStore((st) => st.setMode)
  const pct = useStore((st) => st.trainPct)
  const c = 2 * Math.PI * 6

  if (mode === 'train') {
    return (
      <button className="door" id="door" type="button" onClick={() => setMode('generate')}>
        <IconBack />
        Generate
      </button>
    )
  }
  if (pct == null) {
    return (
      <button className="door" id="door" type="button" onClick={() => setMode('train')}>
        <IconTrain />
        Train
      </button>
    )
  }
  return (
    <button className="door live" id="door" type="button" onClick={() => setMode('train')}>
      <svg className="ring" viewBox="0 0 16 16">
        <circle className="bg" cx="8" cy="8" r="6" />
        <circle className="fg" cx="8" cy="8" r="6"
                strokeDasharray={c.toFixed(2)}
                strokeDashoffset={(c * (1 - pct / 100)).toFixed(2)} />
      </svg>
      Training {pct}%
    </button>
  )
}
