import { useLayoutEffect, useRef } from 'react'

import { fileUrl } from '../api/routes'
import { IconClose, IconExpand, IconPhoto, IconPlay } from '../icons'
import { Frame } from '../regions/Frame'
import { RegionLayer } from '../regions/RegionLayer'
import { useStore } from '../store'
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
 * **Where the boxes are drawn is decided here.** A still wins over the frame, because
 * adjusting boxes against the picture you actually got is the whole point of them still
 * being there after a render; a plate wins over the still, because a plate is what you
 * are composing *into*, so the last render is no longer the subject. With a batch it is
 * the first still only: one set of boxes applies to the whole batch, and drawing them
 * four times would say otherwise.
 */
export function Canvas({
  run,
  vidRun,
  onOpen,
  onOpenVideo,
  onHandoff,
  onFirstFrame,
  onClear,
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
  blank: React.ReactNode
}) {
  const s = useStore()
  const canvasRef = useRef<HTMLDivElement>(null)
  const gridRef = useRef<HTMLDivElement>(null)
  const capRef = useRef<HTMLParagraphElement>(null)

  const image = s.kind === 'image'
  const n = image ? run.files.length : 0
  const vidSrc = !image && vidRun.jobId && vidRun.file
    ? fileUrl(vidRun.jobId, vidRun.file)
    : null

  // The canvas changes height whenever the console does, so the fit is recomputed on
  // that signal rather than set once at generation time.
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const fit = () => layoutShots(gridRef.current, canvas, capRef.current, n)
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [n])

  // The boxes are an image-side thing and the canvas is the same element, so switching
  // kinds has to take the layer with it — otherwise the rectangles sit over a clip they
  // mean nothing to.
  const boxesOnShot = image && s.regional && n > 0 && !s.plate.scene && !s.plate.outfit
  const boxesOnFrame = image && s.regional && !boxesOnShot
  const running = image ? run.running : vidRun.running
  const shown = image ? n > 0 : !!vidSrc

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

      {running && (
        <div className="blank" id="canvas-empty">
          <div style={{ minWidth: 260 }}>
            <div className="bar">
              <i style={{ width: `${image ? run.percent : vidRun.percent}%` }} />
            </div>
            <p className="muted" style={{ marginTop: 8 }}>{image ? run.phase : vidRun.phase}</p>
          </div>
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
                    else if (run.jobId) onOpen(run.jobId, 0)
                  }}>
            <IconExpand />
          </button>
          <button className="ico" id="canvas-clear" title="Clear the canvas" type="button"
                  onClick={onClear}>
            <IconClose />
          </button>
        </div>
      )}

      {/* Streamed by filename, not fetched as base64 first. The status record already
          carries `files`, so an extra round trip buys nothing except a JSON body with
          megabytes of base64 in it — which has to arrive whole, and be decoded, before
          any of the four stills can paint. Each <img> paints as its own bytes land, off
          the same route and the same cache the gallery uses. */}
      {image && (
        <div className="shots" id="gen-out" ref={gridRef} hidden={!n}>
          {run.jobId && run.files.map((f, i) => (
            <figure className={`shot${s.regional ? ' can-drop' : ''}`} key={f}>
              <img src={fileUrl(run.jobId!, f)} alt="" decoding="async" fetchPriority="high"
                   // The same viewer the gallery card opens. A still on the canvas is
                   // fitted to whatever the console left it — at four-up, half of that
                   // — so seeing it at size should not cost a trip through the gallery
                   // to a copy of the image already on screen.
                   onClick={() => onOpen(run.jobId!, i)} />
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
                  this element deletable by an innerHTML write. */}
              {boxesOnShot && i === 0 && <RegionLayer />}
            </figure>
          ))}
        </div>
      )}

      {image && (
        <p className="muted" id="gen-meta" ref={capRef} style={{ margin: '12px 2px' }}>
          {run.meta.join(' · ')}
          {/* Not a fact about the render but a report that something you asked for did
              not happen, so it carries `.warn` — the same amber the LoRA note uses for
              a name that resolves to no file. Set in the same grey as "6.2s" it was a
              caption the eye reads past, which is the wrong place to hide "your LoRA
              did nothing". */}
          {run.skipped.length > 0 && (
            <>
              {run.meta.length ? ' · ' : ''}
              <span className="warn">not applied: {run.skipped.join(', ')}</span>
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
              <video controls autoPlay loop playsInline src={vidSrc} />
              <button className="zoom" title="Full screen" type="button"
                      onClick={() => onOpenVideo(vidSrc)}>
                <IconExpand />
              </button>
            </>
          )}
        </div>
      )}
      {!image && (
        <p className="muted" id="vid-meta" style={{ margin: '12px 2px' }}>
          {vidRun.meta.join(' · ')}
        </p>
      )}
    </div>
  )
}
