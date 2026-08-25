import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import { fileUrl } from '../api/routes'
import { RetryImg } from '../media/thumb'
import { IconClose, IconExpand, IconPhoto, IconPlay, IconPlus } from '../icons'
import { Frame } from '../regions/Frame'
import { RegionLayer } from '../regions/RegionLayer'
import { attached, regionsLive, useStore } from '../store'
import { layoutShots } from './layoutShots'
import type { RunState } from './useGenerate'
import type { VideoRun } from '../video/useVideo'

/**
 * The largest thing on screen, always.
 *
 * Options live in a bar under it, never a rail beside it: a settings column costs the
 * picture 384px of the one dimension it cannot get back. That is measured — fitting
 * each render aspect into a 1512x982 canvas leaves 0px of dead vertical space at
 * *every* aspect and 152–1068px horizontal, so the picture is height-bound everywhere
 * and the bar always comes out of it.
 *
 * **One canvas at a time is the law.** A batch used to be a two-up contact sheet at half
 * height, and that is the arrangement that broke the regions: one set of boxes applies to
 * the whole batch, so there was no single surface to draw them on, and at four or eight
 * they landed on the first cell of the grid you were trying to judge. Now each result
 * fills the canvas and the strip slides between them like frames of film — `‹ 1 / 4 ›` is
 * the whole control. The one on screen *is* the canvas, which is also the surface an
 * inpaint would act on when there is one.
 *
 * **Where the boxes are drawn is decided here.** The current still wins over the frame,
 * because adjusting boxes against the picture you actually got is the point of them still
 * being reachable after a render; a plate wins over the still, because a plate is what
 * you are composing *into*, so the last render is no longer the subject. What is *drawn*
 * on that surface is `store.edit`'s business, not this file's — after a render the answer
 * is nothing at all.
 */
export function Canvas({
  run,
  vidRun,
  onOpen,
  onOpenVideo,
  onHandoff,
  onFirstFrame,
  onClear,
  onChain,
  chaining,
  blank,
}: {
  run: RunState
  vidRun: VideoRun
  onOpen: (jobId: string, i: number) => void
  onOpenVideo: (src: string) => void
  onHandoff: (jobId: string, file: string, as: 'first' | 'reference') => void
  /** A picture dropped on the video canvas is the frame the clip starts on. */
  onFirstFrame: (f: File) => Promise<void> | void
  onClear: () => void
  /** The next generation of the same scene — see `useVideo.chain`. */
  onChain: () => void
  chaining: boolean
  blank: React.ReactNode
}) {
  const s = useStore()
  const canvasRef = useRef<HTMLDivElement>(null)
  const gridRef = useRef<HTMLDivElement>(null)
  const capRef = useRef<HTMLParagraphElement>(null)
  const navRef = useRef<HTMLDivElement>(null)

  const image = s.kind === 'image'
  const n = image ? run.files.length : 0
  const vidSrc = !image && vidRun.jobId && vidRun.file
    ? fileUrl(vidRun.jobId, vidRun.file)
    : null

  /** Which frame of the strip is the canvas. Local, because nothing outside this
   *  component has an opinion about it and it changes at click rate. */
  const [cur, setCur] = useState(0)
  // A new render is its own strip, so it starts at its first frame. Keyed on the job
  // rather than on `n` — two runs of four would otherwise leave you on frame 3 of a
  // batch you have not seen.
  useEffect(() => { setCur(0) }, [run.jobId])
  const at = Math.min(cur, Math.max(0, n - 1))

  // The canvas changes height whenever the console does, so the fit is recomputed on
  // that signal rather than set once at generation time.
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const fit = () => layoutShots(gridRef.current, canvas, capRef.current, n, navRef.current)
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [n])

  // The boxes are an image-side thing and the canvas is the same element, so switching
  // kinds has to take the layer with it — otherwise the rectangles sit over a clip they
  // mean nothing to.
  const running = image ? run.running : vidRun.running
  const shown = image ? n > 0 : !!vidSrc
  const ready = !!s.state && !s.stateError

  // Whenever a render is up, not only when boxes exist. `regionsLive` used to gate
  // this, which read as the obvious economy and deleted a gesture: with zero regions
  // there was no layer over the render at all, so ⌘-drag — "a new box, here",
  // everywhere else — hit bare picture and did nothing, and the only way to draw a
  // first box was to clear the render you wanted to draw against. The layer draws
  // nothing in `off` mode, so an empty one over a render costs no paint.
  const boxesOnShot = image && n > 0
    && !attached(s.frame, 'scene') && !attached(s.frame, 'outfit')
  // The frame is the region surface, and now the empty-canvas invitation too. With no
  // render it carries either the boxes you placed or — when there are none — the cold
  // "drag to place a character" prompt, which is the whole reason `RegionLayer` has an
  // invite state. It still replaces the placeholder glyph rather than sitting beside
  // it: an empty frame and an empty-state glyph are two answers to one question.
  const emptyImage = image && n === 0 && !running && ready
  const boxesOnFrame = image && !boxesOnShot && (regionsLive(s) || emptyImage)

  // ← / → step the strip. Guarded exactly like the Space binding in App: not while
  // typing, not behind a lightbox or a menu, and not while a box is selected — there
  // the arrows nudge the rectangle, which is the layer's own binding and the more
  // specific one.
  useEffect(() => {
    if (n < 2) return
    const key = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return
      const t = e.target as HTMLElement | null
      if (t?.matches?.('input,textarea,select') || t?.isContentEditable) return
      if (t?.closest?.('#region-layer')) return
      if (document.querySelector('.lb,.menu,.pal,.scrim')) return
      e.preventDefault()
      setCur((p) => Math.min(n - 1, Math.max(0, p + (e.key === 'ArrowRight' ? 1 : -1))))
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [n])

  return (
    <div className="canvas" id="canvas" ref={canvasRef}>
      {/* No copy beyond the one line the caller passes. An empty frame above a focused
          prompt field is already the whole instruction, and a sentence telling you to
          type is a sentence that will be read on every visit forever to be useful once.
          The glyph names the kind. */}
      {!shown && !running && !boxesOnFrame && (
        <div className="blank" id="canvas-empty">
          <div>
            <div className="glyph" id="canvas-glyph">{image ? <IconPhoto /> : <IconPlay />}</div>
            {blank}
          </div>
        </div>
      )}

      {/* The full placeholder is for a *cold* run — nothing on screen to keep. Once
          there is a render, a new one replaces it when it lands, not when it is asked
          for, so the old picture stays and the run reports itself as a hairline along
          the top edge rather than by blanking what you were judging. */}
      {/* The bar only. The phase used to sit under it and now lives in
          `#gen-meta` with the card beside it, so repeating it here would be the
          same words twice on one screen — which is what the console block below
          the canvas was, and what came out. */}
      {running && !shown && (
        <div className="blank" id="canvas-empty">
          <div style={{ minWidth: 260 }}>
            <div className="bar">
              <i style={{ width: `${image ? run.percent : vidRun.percent}%` }} />
            </div>
          </div>
        </div>
      )}
      {running && shown && (
        <div className="prog-top" id="canvas-prog" aria-hidden="true">
          <i style={{ width: `${image ? run.percent : vidRun.percent}%` }} />
        </div>
      )}

      {/* The frame regional mode draws on: the render's own aspect, at the size the
          canvas can give it. It replaces the placeholder rather than sitting beside it,
          because an empty frame and an empty-state glyph are two answers to the same
          question. */}
      {boxesOnFrame && !running && <Frame />}

      {/* Clear, and full screen. Both belong to the result rather than to the composer,
          so they live on the canvas and not in the strip — and both are quiet at rest
          for the reason `.shot .acts` is: a control on top of the picture must not
          compete with it. */}
      {shown && (
        <div id="canvas-acts">
          <button className="ico" id="canvas-full" title="Full screen — Space" type="button"
                  onClick={() => {
                    if (vidSrc) onOpenVideo(vidSrc)
                    else if (run.jobId) onOpen(run.jobId, at)
                  }}>
            <IconExpand />
          </button>
          {/* **A scene is longer than a generation.** H3 tops out around 14.4
              seconds, so anything with more than one beat in it is several runs,
              and what makes them one scene rather than unrelated clips is that
              the cast, the look and the last frame survive between them. This is
              the whole of chaining and none of it is a model capability — it is
              context the person should never have to rebuild.

              A plus sign, by the owner's call — and the word had stopped being
              one anyway: `.wide` lost its width to `#canvas-acts .ico` on
              cascade order, so "Continue" shipped clipped to "Con" at every
              viewport. The count of takes stays in the title, where the mark's
              one sentence lives. */}
          {!image && (
            <button className="ico" id="canvas-chain" type="button"
                    disabled={chaining}
                    title={chaining ? 'Linking…'
                      : 'Write the next beat. The cast, the look and this last '
                        + 'frame carry over.'
                        + (s.takes.length > 1 ? ` Take ${String(s.takes.length)}.` : '')}
                    onClick={onChain}>
              <IconPlus />
            </button>
          )}
          <button className="ico" id="canvas-clear" title="Clear the canvas — ⌫" type="button"
                  onClick={onClear}>
            <IconClose />
          </button>
        </div>
      )}

      {/* Streamed by filename, not fetched as base64 first. The status record already
          carries `files`, so an extra round trip buys nothing except a JSON body with
          megabytes of base64 in it — which has to arrive whole, and be decoded, before
          any of the stills can paint. Each <img> paints as its own bytes land, off the
          same route and the same cache the gallery uses — and every frame of the strip
          is mounted, so stepping to the next one is a transform rather than a fetch. */}
      {image && (
        <div className="film" id="gen-out" ref={gridRef} hidden={!n}>
          <div className="film-track" style={{ transform: `translateX(-${at * 100}%)` }}>
            {run.jobId && run.files.map((f, i) => (
              <div className="film-cell" key={f} aria-hidden={i !== at}>
                <figure className={`shot${regionsLive(s) ? ' can-drop' : ''}`}>
                  {/* No click-to-zoom. It used to open the lightbox, which was right
                      when a batch was four thumbnails and the canvas was only
                      pretending to show you the picture — the strip shows it at full
                      canvas size now. What replaced it is one rule with no geography
                      in it: a click on the canvas addresses whatever is under it, so
                      over a region it opens that region and over bare picture it does
                      nothing. Full screen is the expand button and Space. */}
                  {/* `width`/`height` are the reservation, not a size: `.shot img` is
                      `width:auto;height:auto`, so they never set the box — the browser
                      takes the ratio from them and holds the space before a byte
                      arrives. Without it the column reflowed as each render landed,
                      under a hand already over the strip. Omitted rather than zeroed
                      when the run has no request behind it, since `width="0"` would
                      reserve a ratio of nothing. */}
                  {/* RetryImg, not <img>: a still that failed to load once is a
                      finished render showing nothing, and the element never
                      asks again on its own. */}
                  <RetryImg src={fileUrl(run.jobId!, f)} alt="" decoding="async"
                            width={run.w || undefined} height={run.h || undefined}
                            fetchPriority={i === at ? 'high' : 'auto'} />
                  {/* Each still carries its own way into video. Two, because they are
                      genuinely different jobs: a first frame is the shot the clip starts
                      on, a reference is a subject the clip is about. */}
                  <span className="acts">
                    <button type="button" title="Animate — use as the first frame of a clip"
                            onClick={() => onHandoff(run.jobId!, f, 'first')}>
                      <IconPlay />
                    </button>
                    <button type="button" title="Use as a reference image"
                            onClick={() => onHandoff(run.jobId!, f, 'reference')}>
                      <IconPhoto />
                    </button>
                  </span>
                  {/* Inside the still, so the boxes land on the picture with no
                      measurement — and with no reparenting, which is what used to make
                      this element deletable by an innerHTML write. On the shown frame
                      only: one set of boxes applies to the whole batch, and mounting a
                      layer per frame would be four cards claiming the same regions. */}
                  {boxesOnShot && i === at && <RegionLayer over="render" />}
                </figure>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* `‹ 2 / 4 ›`. The whole batch control, and it only exists when there is a batch
          — one result is one canvas and has nothing to step through. Quiet at rest like
          everything else that sits near the picture. */}
      {image && n > 1 && (
        <div className="film-nav" id="gen-nav" ref={navRef}>
          <button type="button" className="ico" id="gen-prev" title="Previous — ←"
                  disabled={at === 0} onClick={() => setCur(at - 1)}>‹</button>
          <span className="count">{at + 1} / {n}</span>
          <button type="button" className="ico" id="gen-next" title="Next — →"
                  disabled={at >= n - 1} onClick={() => setCur(at + 1)}>›</button>
        </div>
      )}

      {/* **One line under the picture, and the live phase takes it while there
          is one.** The summary is what the last render *was* — sampler, steps,
          CFG, seconds — and it is worth exactly nothing while a new one is
          running, so a run's status replaces it rather than opening a second
          place to look. It comes back when the run lands, describing the render
          that is now on screen.

          The GPU rides along at the front because it is the one fact here that
          is about the *next* run rather than the last, and the one that explains
          a wait: `H100` beside a phase reading "loading" is the whole of why a
          first press is slow, said where somebody already looks and nowhere
          else. */}
      {image && (
        <p className="muted" id="gen-meta" ref={capRef} style={{ margin: '12px 2px' }}
           aria-live="polite">
          {run.running
            ? [s.gpu.image, run.phase || 'Working…'].filter(Boolean).join(' · ')
            : run.meta.join(' · ')}
          {/* Not a fact about the render but a report that something you asked for did
              not happen, so it carries `.warn` — the same amber the LoRA note uses for
              a name that resolves to no file. Set in the same grey as "6.2s" it was a
              caption the eye reads past, which is the wrong place to hide "your LoRA
              did nothing". */}
          {!run.running && run.skipped.length > 0 && (
            <>
              {run.meta.length ? ' · ' : ''}
              {/* Semicolons, not commas: each entry is now a clause with its own
                  comma in it — see `skipNote` — so a comma-joined list of two ran
                  the reason for one LoRA straight into the name of the next. */}
              <span className="warn">not applied: {run.skipped.join('; ')}</span>
            </>
          )}
        </p>
      )}

      {/* On the video side a dropped picture is the frame the clip starts on, which is
          the one reading that needs no mode: it is what the tile two rows down would
          have done, done on the largest target on screen.

          Rendered whenever the video side is showing rather than only when there is a
          clip, because the element *is* the handler — `check_drop.py` found this
          missing, and what was missing was not a class on a div, it was the gesture. */}
      {!image && (
        <div id="vid-out" className={`can-drop${vidSrc ? '' : ' hide'}`} data-drop="First frame"
             onDragOver={(e) => {
               if (![...(e.dataTransfer?.types ?? [])].includes('Files')) return
               e.preventDefault()
               e.currentTarget.classList.add('hot')
             }}
             onDragLeave={(e) => {
               if (!e.currentTarget.contains(e.relatedTarget as Node))
                 e.currentTarget.classList.remove('hot')
             }}
             onDrop={(e) => {
               e.preventDefault()
               e.currentTarget.classList.remove('hot')
               const f = [...(e.dataTransfer?.files ?? [])].find((x) => x.type.startsWith('image/'))
               // Said, not swallowed: a file of the wrong kind landing on a target that
               // just lit up for it has to say why nothing happened.
               if (!f) {
                 alert('The canvas takes an image — it becomes the first frame.')
                 return
               }
               void onFirstFrame(f)
             }}>
          {vidSrc && (
            <>
              {/* `muted`, or `autoPlay` is a word rather than a behaviour: every
                  browser refuses unmuted autoplay without a gesture, so a 200s
                  render used to land as a strip parked at 0:00 — the one moment
                  the product has to sell itself, spent asking for a click. The
                  clip plays silent and looping the instant it lands, which is
                  what "the result on screen is the canvas" means for video; the
                  native controls are right there for the soundtrack. */}
              <video controls autoPlay muted loop playsInline src={vidSrc} />
              <button className="zoom" title="Full screen" type="button"
                      onClick={() => onOpenVideo(vidSrc)}>
                <IconExpand />
              </button>
            </>
          )}
        </div>
      )}
      {!image && (
        <p className="muted" id="vid-meta" style={{ margin: '12px 2px' }}
           aria-live="polite">
          {vidRun.running
            ? [s.gpu.video, vidRun.phase || 'Working…'].filter(Boolean).join(' · ')
            : vidRun.meta.join(' · ')}
        </p>
      )}
    </div>
  )
}
