import { useLayoutEffect, useRef } from 'react'

import { fileUrl } from '../api/routes'
import { IconPhoto, IconPlay } from '../icons'
import { layoutShots } from './layoutShots'
import type { RunState } from './useGenerate'

/**
 * The largest thing on screen, always.
 *
 * Options live in a bar under it, never a rail beside it: a settings column
 * costs the picture 384px of the one dimension it cannot get back. That is
 * measured — fitting each render aspect into a 1512x982 canvas leaves 0px of
 * dead vertical space at *every* aspect and 152–1068px horizontal, so the
 * picture is height-bound everywhere and the bar always comes out of it.
 */
export function Canvas({
  run,
  onOpen,
  onHandoff,
  blank,
}: {
  run: RunState
  onOpen: (jobId: string, file: string, index: number) => void
  onHandoff: (jobId: string, file: string, as: 'first' | 'reference') => void
  blank: React.ReactNode
}) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const gridRef = useRef<HTMLDivElement>(null)
  const capRef = useRef<HTMLParagraphElement>(null)

  const n = run.files.length

  // The canvas changes height whenever the console does, so the fit is
  // recomputed on that signal rather than set once at generation time.
  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const fit = () => layoutShots(gridRef.current, canvas, capRef.current, n)
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [n])

  return (
    <div className="canvas" id="canvas" ref={canvasRef}>
      {!n && !run.running && (
        <div className="blank" id="canvas-empty">{blank}</div>
      )}

      {run.running && (
        <div className="blank" id="canvas-empty">
          <div style={{ minWidth: 260 }}>
            <div className="bar"><i style={{ width: `${run.percent}%` }} /></div>
            <p className="muted" style={{ marginTop: 8 }}>{run.phase}</p>
          </div>
        </div>
      )}

      {/* Streamed by filename, not fetched as base64 first. The status record
          already carries `files`, so an extra round trip buys nothing except a
          JSON body with megabytes of base64 in it — which has to arrive whole,
          and be decoded, before any of the four stills can paint. Each <img>
          paints as its own bytes land, off the same route and the same cache
          the gallery uses. */}
      <div className="shots" id="gen-out" ref={gridRef} hidden={!n}>
        {run.jobId && run.files.map((f, i) => (
          <figure className="shot" key={f}>
            <img src={fileUrl(run.jobId!, f)} alt="" decoding="async"
                 fetchPriority="high"
                 // The same viewer the gallery card opens. A still on the canvas
                 // is fitted to whatever the console left it — at four-up, half
                 // of that — so seeing it at size should not cost a trip through
                 // the gallery to a copy of the image already on screen.
                 onClick={() => onOpen(run.jobId!, f, i)} />
            {/* Each still carries its own way into video. Two, because they are
                genuinely different jobs: a first frame is the shot the clip
                starts on, a reference is a subject the clip is about. */}
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
          </figure>
        ))}
      </div>

      <p className="muted" id="gen-meta" ref={capRef}>
        {run.meta.join(' · ')}
        {/* Not a fact about the render but a report that something you asked
            for did not happen, so it carries `.warn` — the same amber the LoRA
            note uses for a name that resolves to no file. Set in the same grey
            as "6.2s" it was a caption the eye reads past, which is the wrong
            place to hide "your LoRA did nothing". */}
        {run.skipped.length > 0 && (
          <>
            {run.meta.length ? ' · ' : ''}
            <span className="warn">not applied: {run.skipped.join(', ')}</span>
          </>
        )}
      </p>
    </div>
  )
}
