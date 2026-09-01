/**
 * The layer on top of a frame: the camera's stencil and the subjects' arrows.
 *
 * Two arrow kinds, told apart by fill and by nothing else — solid is the
 * camera, hollow is a subject — which is the storyboard convention with its
 * colour scheme taken out and its logic kept. A director reads the frame
 * without a legend, and a hollow arrow over a face with a name on its tail
 * is the whole of what "Maya crosses to the window" needs to be drawn as.
 *
 * The layer paints; the panel listens. Every pointer gesture on the frame is
 * handled one level up, where the picture, the crop and the layer share one
 * press — and what is under the press decides what the drag does. This file
 * only has to make the things it draws hit-testable, which the `data-` marks
 * are for.
 */
import type { CameraAmp, ShotPill } from '../api/types'
import { arrowPath, type Px } from './arrows'
import type { Motion, Pt } from './model'
import { Stencil } from './stencils'

const move = (k: string) => k.split('.').slice(1).join('.')

export function MotionLayer({ w, h, camera, motion, draft, sel, frame }: {
  w: number
  h: number
  camera: ShotPill | null
  motion: Motion[]
  /** The arrow being drawn, until the pointer lifts. */
  draft: [Pt, Pt] | null
  sel: string | null
  /** Where the camera's frame sits on the picture, in pixels — the whole
   *  picture when it is cropped, the aspect box when it is shown whole. The
   *  stencil is the camera's, so it is drawn on the camera's frame; a
   *  subject's arrow is on the picture, so it is not. */
  frame: { x: number; y: number; w: number; h: number }
}) {
  if (!w || !h) return null
  const px = (p: Pt): Px => [p[0] * w, p[1] * h]
  const t = Math.max(3, Math.min(w, h) * 0.034)
  const shaft = (pts: [Pt, Pt]) => arrowPath([px(pts[0]), px(pts[1])], t, t * 2.8, t * 2.6)
  return (
    <svg className="sblayer" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      {camera && (
        <svg x={frame.x} y={frame.y} width={frame.w} height={frame.h}
             viewBox={`0 0 ${frame.w} ${frame.h}`} overflow="visible">
          <Stencil move={move(camera.key)} amp={(camera.amp ?? 'medium') as CameraAmp}
                   w={frame.w} h={frame.h} />
        </svg>
      )}
      {motion.map((m) => (
        <path key={m.id} data-arrow={m.id}
              className={`sbarrow${sel === m.id ? ' sel' : ''}`} d={shaft(m.pts)} />
      ))}
      {draft && <path className="sbarrow draft" d={shaft(draft)} />}
    </svg>
  )
}
