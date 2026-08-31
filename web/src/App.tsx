import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react'

import { failed } from './api/client'
import { fileUrl, getState, warm } from './api/routes'
import { Canvas } from './canvas/Canvas'
import { useGenerate } from './canvas/useGenerate'
import { Console } from './console/Console'
import { videoReady } from './console/resolve'
import { ErrorNote } from './ui/ErrorNote'
import { Gallery, useGallery } from './gallery/Gallery'
import { LastShot } from './gallery/LastShot'
import { MetaSheet } from './gallery/MetaSheet'
import { Viewer } from './gallery/Viewer'
import type { Session } from './api/types'
import type { GalleryItem } from './gallery/types'
import { IconBack, IconCube, IconNodes, IconPanel, IconPhoto, IconStack, IconTrain } from './icons'

// Split, not imported: the graph surface is the one dependency-heavy room in
// the product, and the page must pay for it when the door is opened, never on
// load. The chunk is cached after the first visit.
const Playground = lazy(() =>
  import('./playground/Playground').then((m) => ({ default: m.Playground })))
import { fileToB64, toB64 } from './media/files'
import { live } from './scene/model'
import { Settings } from './settings/Settings'
import { warmDatasets } from './datasets/useDatasets'
import { Train } from './train/Train'
import { SheetBuilder } from './sheet/SheetBuilder'
import { ThemeDot } from './theme/ThemeDot'
import { isActive, useSessions } from './train/useSessions'
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
  const { items, reload, record, drop, total, behind, more, done } = useGallery()
  const [galleryOpen, setGalleryOpen] = useState(false)
  /* Closed. The canvas is the largest thing on screen, always — and the drawer is
     the one piece of chrome that takes width from it rather than height, 320px of
     the axis a picture wants most. Reaching the gallery is already a utility that
     lives with the controls: the last generation is a thumbnail beside Generate,
     and the header button is here. Opening on load spent the canvas on a grid of
     work you have already seen, before you had made anything this session. */
  const [drawerOpen, setDrawerOpen] = useState(false)
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

  /* Fetched once at mount, and that one answer decides the whole page — which sizes
     exist, which LoRAs are offered, whether Generate is even enabled. So when it comes
     back wrong the app is wrong until you reload it by hand, and the hand-reload people
     actually found was the gear, because opening Settings calls this again. "Click the
     gear and the weights appear" is that bug wearing a workaround.

     What comes back wrong is a listing that says the volume holds nothing while it holds
     26 GB of Krea 2 — seen once against a real deployment, with `/api/state` fetched from
     that same page reporting `turbo.present: true` a moment later, and not reproducible
     on demand. The cause is below us: `_sizes_on_disk` reads a directory rather than
     statting each file *because* a failed lookup is cached under the mount, and a readdir
     that lands before the mount has caught up is the same fault one level up.

     So this re-checks rather than diagnoses. Bounded at three attempts, and only while
     the answer is still "there is no DiT here" — the one state that is both the symptom
     and the thing that disables the button. A genuinely empty volume pays two extra
     requests on a CPU route once per load and then settles, which is the right price for
     an install that otherwise tells you to download weights you already have. */
  useEffect(() => {
    let alive = true
    const hasDit = () => {
      const st = useStore.getState().state
      return !!st?.models.some((m) => (m.key === 'turbo' || m.key === 'raw') && m.present)
    }
    void (async () => {
      for (const wait of [0, 700, 1800]) {
        if (wait) await new Promise((r) => setTimeout(r, wait))
        if (!alive) return
        await reloadState()
        // Stop on a real answer, and on a real error — retrying a 500 three times just
        // makes the same complaint three times slower.
        if (!alive || hasDit() || useStore.getState().stateError) return
      }
    })()
    // Once, beside the state fetch rather than in an effect of its own: they are
    // the same event — the page opened — and a second effect would be a second
    // thing to keep in step with it. The kind rides along because it decides
    // *which* container is warmed, and the page opens on stills.
    void warm(useStore.getState().kind)
    // The sets listing too, and for the same reason the sessions door gets its
    // rows at app level: Sets is a front-door destination now, and the click
    // that opens it should find the rows already waiting rather than start the
    // volume walk it then blanks on.
    warmDatasets()
    return () => { alive = false }
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
      // Files only. Not a text selection dragged out of the prompt field — that
      // must not make the page look like it wants to eat it.
      //
      // **It used to be files *or* a LoRA**, because a LoRA was something you
      // dragged out of the picker at the prompt bar and onto a box. It is picked
      // from the box's own dropdown now, so the only thing that arrives by drag is
      // a photograph — which comes from the Finder, where a drag is the only
      // gesture there is.
      const types = [...(e.dataTransfer?.types ?? [])]
      if (!types.includes('Files')) return
      document.body.classList.add('dragging')
      // `RegionLayer` reads this to bring the boxes back over a finished render:
      // a drop cannot land on a box that is at `pointer-events:none`, which is
      // the state the port shipped in and it made the gesture undiscoverable by
      // construction.
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

  /* Typing with nothing focused lands in the prompt, not in the hotkeys. A click that
     misses the field by a few pixels leaves focus on the body, and the sentence typed
     next used to arrive as a hotkey barrage — every space toggling the viewer, every
     backspace one keystroke from clearing the canvas. Letters only: Space keeps its
     full-screen meaning when it is the *first* thing pressed, because a sentence never
     starts with one — the first letter moves the caret into the field and the spaces
     after it are just spaces. Routed to whichever prompt is showing, with the caret at
     the end, which is where a caret that was never placed belongs. */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return
      if (e.key.length !== 1 || e.key === ' ') return
      const t = e.target as HTMLElement | null
      if (t?.matches?.('input,textarea,select') || t?.isContentEditable) return
      if (t?.closest?.('#region-layer')) return
      if (document.querySelector('.lb,.menu,.pal,.scrim')) return
      const field = document.querySelector<HTMLTextAreaElement>(
        'textarea#prompt:not(.hide), textarea#neg:not(.hide)')
      if (!field || field.readOnly) return
      e.preventDefault()
      // Focus *synchronously*, so this handler only ever routes the first key —
      // everything after it lands in the field natively. Deferred to the next
      // frame (the `applyWrite` pattern), each keystroke of the sentence arrived
      // here as its own stray key while focus was still in flight.
      field.focus()
      const st = useStore.getState()
      // `#prompt` is the image side's box on one kind and the *selected* shot's
      // row on the other — same id because it is the same thing, "the box you
      // write in". The write has to follow it, or a stray letter on the video
      // side lands in a buffer nothing is showing.
      //
      // **The selected row, never `shots[0]`.** It was the first shot, and the
      // value is composed from the field this handler just focused — so with the
      // caret anywhere but shot 1, one stray letter wrote an empty box's contents
      // plus that letter over shot 1, replacing the sentence there. `shotSel`
      // falls back the way `Shots` does, because it can name a row a ⌫ removed.
      const write = field.id === 'neg'
        ? st.setNegative
        : st.kind === 'video'
          ? (v: string) => {
              st.patchShot(st.shotSel || (st.scene.shots[0]?.id ?? ''), { line: v })
            }
          : st.setPrompt
      // Never welded to the last word — the same courtesy `insertLora` extends.
      const sep = field.value && !/\s$/.test(field.value) ? ' ' : ''
      write(field.value + sep + e.key)
      // After the commit, for the reason `applyWrite` waits: the field is controlled,
      // and a range set now would range over text React has not painted yet.
      requestAnimationFrame(() =>
        field.setSelectionRange(field.value.length, field.value.length))
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [])

  /* **View source, and nothing on the page says so.** ⌘⌥U is the browser's own
     chord for exactly this, which is the point — the composer advertises the
     compiled document with no control at all, the way devtools is a chord and a
     context menu rather than a button. The other half is `onContextMenu` on the
     console. See `SourcePane` for why reading the prompt is not a step in
     composing and therefore not a disclosure.

     Video only: the image side's compiled prompt genuinely *is* a fold under the
     sentence above it, and `#shot-peek` is still that. */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== 'u' || !e.altKey || !(e.metaKey || e.ctrlKey)) return
      const st = useStore.getState()
      if (st.mode !== 'generate' || st.kind !== 'video') return
      e.preventDefault()
      st.setDocOpen(!st.docOpen)
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [])

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
        // A modal, still: this is the whole of what the press was for, so it has
        // nothing to fall through to and silence would read as a dead button. What
        // changed is that it names the one thing that can be retried — the bytes are
        // the server's own file, so the failure is the fetch rather than the file.
        alert('Could not read that render back off the server — reload the page and'
              + ' try again.')
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
        // With a composer live the flat trays never travel — the cast's files
        // are what <Picture N> numbers — so pushing here would file the
        // picture somewhere the run cannot see. Said, with the way in named,
        // rather than a chip that renders and uploads nothing.
        if (live(st.scene)) {
          alert('This scene has a cast, so a reference belongs to somebody — '
                + 'drop the picture on a member\u2019s card, or make it its '
                + 'own subject with @.')
          return
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

  /**
   * **Still → 8s does not throw the picture away.** Picking a duration used to swap the
   * canvas for an empty video placeholder, and the still you had just made was gone from
   * the screen with nothing on it pointing anywhere — it survived only in the gallery
   * drawer, which is closed by default and is the one piece of chrome most people never
   * open. Two seconds of work, silently filed.
   *
   * The row below already has the slot it belongs in, so it goes there rather than into a
   * message about where it went: the still becomes the clip's first frame, visible in
   * `#v-drop-first`, clearable with a click, and `VideoNote` says what having one means.
   * That is also the reading the button on the still itself takes — see `handoff` — so
   * this is that gesture made the default of the switch rather than a second behaviour.
   *
   * Three guards, and each is a way this could have been worse than the blanking:
   * - Only on the image→video edge. Re-running it on every video render would re-attach
   *   a still somebody had just cleared.
   * - Only into an empty row. A keyframe or a reference already there is a choice, and
   *   `handoff` itself lands here with one attached — stomping either would be this
   *   feature undoing the very hand-off it copies.
   * - Only if the clip it implies can still run. A first frame moves the task from t2v to
   *   i2v, and on a volume without the i2v pair that turns Generate off; a switch that
   *   disables the button it is walking you toward is worse than one that clears a canvas.
   */
  const wasKind = useRef(s.kind)
  useEffect(() => {
    const from = wasKind.current
    wasKind.current = s.kind
    if (from !== 'image' || s.kind !== 'video') return
    const jobId = gen.run.jobId
    const file = gen.run.files[0]
    if (!jobId || !file) return
    const st = useStore.getState()
    if (st.keyframe.first || st.keyframe.last || st.refs.length || st.refVids.length) return
    // No `alive` flag and no cleanup, which is the opposite of the usual rule and is
    // what makes this survive StrictMode. The edge is detected off a ref, so the
    // double-invoke pairs "ref says image, start the fetch" with a cleanup and then a
    // second pass where the ref already says video — cancel on cleanup and the switch
    // silently never carries anything in development. Safe to leave running because
    // every guard below is re-read from the store after the await: a second run finds
    // the slot filled and writes nothing.
    void (async () => {
      // Fetched, not reused: the canvas stills are a streamed `<img src>`, so the base64
      // the video side needs does not exist client-side. Same as `handoff`.
      const b64 = await fileToB64(fileUrl(jobId, file))
      // Silent when the fetch fails, and deliberately: nobody asked for this, they asked
      // for 8 seconds. A modal explaining that a courtesy did not happen would interrupt
      // a mode switch to report something that was never promised — and the still is
      // still in the gallery and still there on switching back to Still.
      if (!b64) return
      // Re-read across the await, and re-check every guard against what is true *now*:
      // the fetch is a round trip, and switching back to Still or dropping a reference
      // inside it are both things a hand can do in that time.
      const now = useStore.getState()
      if (now.kind !== 'video') return
      if (now.keyframe.first || now.keyframe.last || now.refs.length || now.refVids.length) return
      if (!videoReady({ ...now, keyframe: { first: b64, last: null } }).ready) return
      now.setKeyframe('first', b64)
      // After the write, because setKeyframe clears it: the note line names
      // the actor when the first frame was the app's doing, not a hand's.
      now.setAutoFirst(true)
    })()
  }, [s.kind, gen.run.jobId, gen.run.files])

  const train = s.mode === 'train'
  // Lifted out of Train, because the door reports training from Generate and
  // Train is not mounted there. It polls at the board's rate while Train is
  // open, slowly while something is running elsewhere, and not at all
  // otherwise — see `useSessions`.
  const sess = useSessions(train)
  // Lifted out of Train for the same reason the sessions were: the header's
  // Sets door has to land on the sets screen directly, and Train is not
  // mounted from Generate to be told which screen to open on.
  const [trainScreen, setTrainScreen] = useState<'board' | 'sets'>('board')

  return (
    <>
      <header className="top">
        <button className="brand" id="go-home" type="button"
                onClick={() => { setGalleryOpen(false); s.setMode('generate') }}>
          Visionary
        </button>
        {/* Its own element, not part of the wordmark: the wordmark is how you get
            back to Generate, and a door that sometimes means "appearance" is the
            header lying the way a mislabelled door does. */}
        <ThemeDot />
        <span className="grow" />
        {/* Sets is a front-door destination, not a room inside Training. It was
            two levels deep — Training, then a text button on the board — which
            made the half of the product you build datasets in read as a detail
            of the half you train them in. A door, so it takes a word; hidden in
            Train because there the header's job is the way back, and moving
            between the board and the sets stays on the stage where it happens. */}
        {/* The sheet composer's door. Its own surface rather than a popover
            off a cast card — composing a reference sheet is neither generating
            nor training, and the label follows Train's rule: named for where
            it leads. Inside, the same door is the way back. */}
        {/* Each word rides a `.door-word` span, which the phone drops the way
            the training door's live label already does: at 375px the row
            overflowed and the Models button sat at x=385 — reachable on no
            phone at all. The icons have to differ once they stand alone, which
            is why Sets stopped sharing Sheet's photograph glyph. */}
        {!train && (
          <button className="door" id="door-sheet" type="button"
                  title="Compose a character reference sheet"
                  onClick={() => { s.setMode(s.mode === 'sheet' ? 'generate' : 'sheet') }}>
            <IconPhoto />
            <span className="door-word">{s.mode === 'sheet' ? 'Done' : 'Sheet'}</span>
          </button>
        )}
        {!train && (
          <button className="door" id="door-sets" type="button"
                  title="Datasets — the folders training reads"
                  onClick={() => { setTrainScreen('sets'); s.setMode('train') }}>
            <IconStack />
            <span className="door-word">Sets</span>
          </button>
        )}
        {/* The node room. A door on the Sheet precedent — experimenting on
            the engine is neither generating nor training — and the same
            Done rule, so two things never look equally selected. */}
        {!train && (
          <button className="door" id="door-playground" type="button"
                  title="The Playground — the backend's own graph, rewired by hand"
                  onClick={() => {
                    s.setMode(s.mode === 'playground' ? 'generate' : 'playground')
                  }}>
            <IconNodes />
            <span className="door-word">
              {s.mode === 'playground' ? 'Done' : 'Playground'}
            </span>
          </button>
        )}
        {/* Each door names its own landing: Training opens on the board even if
            Sets was the last screen open in there — a door that lands you
            somewhere other than what it says is the header lying. */}
        <Door running={sess.rows.filter(isActive)} onOpen={() => setTrainScreen('board')} />
        {/* The drawer toggle is a Generate control; it has nothing to say in Train. */}
        {!train && (
          <button className={`ico${drawerOpen ? ' on' : ''}`} id="t-drawer" title="Gallery"
                  type="button" onClick={() => setDrawerOpen((d) => !d)}>
            <IconPanel />
          </button>
        )}
        {/* "Models", not "Settings": the sheet holds weights — checkpoints,
            LoRAs, caption models — plus the GPU and token that serve them.
            There is no setting in it. The id stays `t-settings` because
            check_settings.py reaches the sheet through it. */}
        <button className="ico" id="t-settings" title="Models" type="button"
                onClick={() => { setSettingsOpen(true); void reloadState() }}>
          <IconCube />
        </button>
      </header>

      <div className="views">
        {s.mode === 'playground' ? (
          <Suspense fallback={null}>
            <Playground />
          </Suspense>
        ) : s.mode === 'sheet' ? (
          <SheetBuilder />
        ) : train ? (
          // A set can hold clips now, and the viewer needs to know which it is being
          // handed. The route is the tell — `/api/clip/` is the only thing that serves
          // one — which keeps the Editor's callback one argument wide rather than making
          // every caller classify a file for it.
          <Train sess={sess} screen={trainScreen} setScreen={setTrainScreen}
                 onLightbox={(src) => lightbox(src, src.startsWith('/api/clip/') ? 'video' : 'image')} />
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
                onChain={() => void vid.chain()}
                chaining={vid.linking}
                onExport={() => void vid.exportScene()}
                exporting={vid.exporting}
                onHandoff={(jobId, file, as) => void handoff(jobId, file, as)}
                onFirstFrame={async (f) => {
                  const b64 = await toB64(f)
                  if (b64) useStore.getState().setKeyframe('first', b64)
                  // A file the browser cannot decode is worth interrupting for — the
                  // drop is over, the canvas took nothing, and there is no quieter
                  // place on this surface to say so. It names the file and a format
                  // that works, which is the half "Could not read that image." was
                  // missing: same words, no next step.
                  else alert(`The browser could not read ${f.name || 'that image'} —`
                             + ' save it as a PNG or JPEG and drop it again.')
                }}
                onClear={clearCanvas}
                blank={
                  s.stateError ? <ErrorNote err={s.stateError} />
                    : s.state ? null
                    : 'Loading…'
                }
              />
              <Console run={gen.run} vidRun={vid.run} onGenerate={fire} onStop={stopRun}
                       lastShot={<LastShot items={items}
                                           onOpen={(rows, i) => setShown({ rows, i })} />} />
            </div>

            {/* `onReload` returns rather than voids. The gallery awaits it to hold its
                delete button busy until the refreshed listing lands; voided, the await
                resolves in the same tick, so the button would go quiet at the reply and
                the card would sit there until the refetch caught up — which is the half
                of the wait the finding was about. */}
            <Gallery
              items={items}
              total={total}
              behind={behind}
              done={done}
              open={galleryOpen}
              drawerOpen={drawerOpen}
              onClose={() => setGalleryOpen(false)}
              onReload={() => {
                setGalleryOpen(true)
                return reload()
              }}
              onMore={more}
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
 *
 * With more than one run going it says the count instead of a percent, and the
 * ring averages them. Naming one of four runs would be naming whichever the
 * server listed first, which is a number about a card you cannot see from here.
 */
function Door({ running, onOpen }: { running: Session[]; onOpen: () => void }) {
  const mode = useStore((st) => st.mode)
  const setMode = useStore((st) => st.setMode)
  const enter = () => { onOpen(); setMode('train') }
  const pct = running.length
    ? Math.round(running.reduce((a, r) => a + Number(r.percent ?? 0), 0) / running.length)
    : null
  const c = 2 * Math.PI * 6

  if (mode === 'train') {
    return (
      <button className="door" id="door" type="button" title="Back to the canvas"
              onClick={() => setMode('generate')}>
        <IconBack />
        <span className="door-word">Generate</span>
      </button>
    )
  }
  if (pct == null) {
    return (
      <button className="door" id="door" type="button" title="Training sessions"
              onClick={enter}>
        <IconTrain />
        <span className="door-word">Train</span>
      </button>
    )
  }
  return (
    <button className="door live" id="door" type="button" onClick={enter}>
      <svg className="ring" viewBox="0 0 16 16">
        <circle className="bg" cx="8" cy="8" r="6" />
        <circle className="fg" cx="8" cy="8" r="6"
                strokeDasharray={c.toFixed(2)}
                strokeDashoffset={(c * (1 - pct / 100)).toFixed(2)} />
      </svg>
      {/* The word is in its own span so the phone can drop it. The header is a fixed
          56px bar, and "Training 50%" beside the brand, Sets and two icons does not fit
          375px: it broke onto a second line inside a bar that cannot have one. Setting
          `nowrap` instead only moves the damage — measured, it pushes the document 10px
          wider than the viewport, trading a wrapped label for a page that scrolls
          sideways. What actually goes is the word, because the ring beside it is already
          saying "training" and a number is what you came to read. Idle says `Train` and
          fits, so this is the live label's problem alone.

          `sessions`, not `runs`: the board this door opens counts sessions, names them
          sessions and says "No sessions yet." when it has none, so the door onto it was
          the one surface calling them something else. */}
      <span className="door-word">Training{running.length > 1 ? ' · ' : ' '}</span>
      {running.length > 1 ? `${running.length} sessions` : `${pct}%`}
    </button>
  )
}
