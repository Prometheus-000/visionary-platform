/**
 * The camera's language, drawn.
 *
 * A storyboard artist does not invent an arrow per shot; there is a stencil
 * per move, and a director reads it without a legend: arrows at the top and
 * bottom edges are a pan, curved arrows at the sides are a tilt, four corner
 * arrows are a zoom, a frame inside the frame is a push or a pull, a frame
 * beside the frame is a truck or a pedestal, a ring is an orbit or a roll,
 * brackets in the corners are a locked-off camera. That is what this file
 * holds, one function per move in the served vocabulary's own keys, drawn
 * *solid* because solid is the camera. Every stencil is sized by the frame
 * and by amplitude, so "a large pan" is a long pair of arrows and not a word.
 *
 * Pixel coordinates in, because arrowheads must not stretch with the frame's
 * aspect: the layer measures the frame and passes its size.
 */
import type { CameraAmp } from '../api/types'
import { arc, arrowPath, type Px } from './arrows'

const AMP: Record<CameraAmp, number> = { small: 0.62, medium: 1, large: 1.45 }

type Draw = { paths: string[]; frames: { x: number; y: number; w: number; h: number; ghost?: boolean }[]
              marks: string[]; labels?: { x: number; y: number; t: string }[] }

export function stencil(move: string, amp: CameraAmp, w: number, h: number): Draw | null {
  const A = AMP[amp] ?? 1
  const m = Math.min(w, h)
  const t = Math.max(3, m * 0.032)          // shaft
  const hw = t * 2.7                        // head width
  const hl = t * 2.4                        // head length
  const pad = m * 0.06
  const ar = (pts: Px[]) => arrowPath(pts, t, hw, hl)
  const out: Draw = { paths: [], frames: [], marks: [] }
  const cx = w / 2
  const cy = h / 2

  switch (move) {
    case 'panl':
    case 'panr': {
      const d = move === 'panr' ? 1 : -1
      const len = w * 0.26 * A
      for (const y of [h * 0.13, h * 0.87]) {
        out.paths.push(ar([[cx - (d * len) / 2, y], [cx + (d * len) / 2, y]]))
      }
      break
    }
    case 'tiltu':
    case 'tiltd': {
      const up = move === 'tiltu'
      const r = h * 0.2 * A
      const yc = up ? h * 0.58 : h * 0.42
      out.paths.push(ar(arc(pad + r, yc, r, 180, up ? 270 : 90)))
      out.paths.push(ar(arc(w - pad - r, yc, r, 0, up ? -90 : 90)))
      break
    }
    case 'pushin':
    case 'pullout': {
      const f = 0.18 * A
      const inner = { x: w * f, y: h * f, w: w * (1 - 2 * f), h: h * (1 - 2 * f) }
      out.frames.push(inner)
      const corners: [Px, Px][] = [
        [[pad, pad], [inner.x, inner.y]],
        [[w - pad, pad], [inner.x + inner.w, inner.y]],
        [[pad, h - pad], [inner.x, inner.y + inner.h]],
        [[w - pad, h - pad], [inner.x + inner.w, inner.y + inner.h]],
      ]
      for (const [o, i] of corners) {
        // Stop short of the inner frame's corner so the head does not sit on
        // the line it points at.
        const [dx, dy] = [i[0] - o[0], i[1] - o[1]]
        const l = Math.hypot(dx, dy) || 1
        const gap = t * 1.2
        const near: Px = [i[0] - (dx / l) * gap, i[1] - (dy / l) * gap]
        const far: Px = [o[0] + (dx / l) * gap, o[1] + (dy / l) * gap]
        out.paths.push(move === 'pushin' ? ar([far, near]) : ar([near, far]))
      }
      break
    }
    case 'zoom':
    case 'zoomout': {
      const len = m * 0.17 * A
      const s = Math.SQRT1_2
      const corners: [Px, Px][] = [
        [[pad, pad], [1, 1]], [[w - pad, pad], [-1, 1]],
        [[pad, h - pad], [1, -1]], [[w - pad, h - pad], [-1, -1]],
      ]
      for (const [c, d] of corners) {
        const inner: Px = [c[0] + d[0] * s * len, c[1] + d[1] * s * len]
        out.paths.push(move === 'zoom' ? ar([c, inner]) : ar([inner, c]))
      }
      break
    }
    case 'truckl':
    case 'truckr':
    case 'pedu':
    case 'pedd':
    case 'craneu':
    case 'craned': {
      const crane = move.startsWith('crane')
      const horiz = move.startsWith('truck')
      const d = move === 'truckr' || move === 'pedd' || move === 'craned' ? 1 : -1
      const shift = (horiz ? w : h) * (crane ? 0.24 : 0.18) * A
      const grow = crane ? 1.14 : 1
      const gw = w * grow
      const gh = h * grow
      out.frames.push({
        x: (w - gw) / 2 + (horiz ? d * shift : 0),
        y: (h - gh) / 2 + (horiz ? 0 : d * shift),
        w: gw, h: gh, ghost: true,
      })
      const from: Px = horiz ? [cx - (d * shift) / 2, h * 0.9] : [w * 0.09, cy - (d * shift) / 2]
      const to: Px = horiz ? [cx + (d * shift) / 2, h * 0.9] : [w * 0.09, cy + (d * shift) / 2]
      out.paths.push(ar([from, to]))
      if (!horiz) out.paths.push(ar([[w - w * 0.09, from[1]], [w - w * 0.09, to[1]]]))
      break
    }
    case 'orbit': {
      const r = m * 0.34 * A
      out.paths.push(ar(arc(cx, cy, r, 250, 250 + 300, 40)))
      break
    }
    case 'arc': {
      const r = Math.min(w * 0.36, h * 0.44) * A
      out.paths.push(ar(arc(cx, cy - h * 0.05, r, 160, 20, 30)))
      break
    }
    case 'rollcw':
    case 'rollccw': {
      const r = m * 0.4 * A
      const cw = move === 'rollcw'
      out.paths.push(ar(arc(cx, cy, r, cw ? 215 : 325, cw ? 325 : 215, 24)))
      out.paths.push(ar(arc(cx, cy, r, cw ? 35 : 145, cw ? 145 : 35, 24)))
      break
    }
    case 'trackside': {
      const len = w * 0.5 * A
      const y = h * 0.86
      out.paths.push(ar([[cx - len / 2, y], [cx + len / 2, y]]))
      // The rail it runs on.
      out.marks.push(`M${pad} ${h - pad * 0.9}H${w - pad}`)
      break
    }
    case 'trackrear': {
      const tw = w * 0.1 * A
      const tip = h * 0.5
      const shoulder = h * 0.64
      out.paths.push(
        `M${cx - tw} ${h - pad}L${cx - tw * 0.45} ${shoulder}L${cx - tw * 0.95} ${shoulder}`
        + `L${cx} ${tip}L${cx + tw * 0.95} ${shoulder}L${cx + tw * 0.45} ${shoulder}`
        + `L${cx + tw} ${h - pad}Z`)
      break
    }
    case 'whip': {
      const len = w * 0.62
      const y = h * 0.14
      out.paths.push(ar([[cx - len / 2 + len * 0.18, y], [cx + len / 2, y]]))
      for (let i = 0; i < 3; i++) {
        const yy = y - t * 1.3 + i * t * 1.3
        out.marks.push(`M${cx - len / 2 - t * 2 * i} ${yy}H${cx - len / 2 + len * 0.12 - t * i}`)
      }
      break
    }
    case 'rack': {
      const r = m * 0.11
      const a: Px = [w * 0.3, h * 0.62]
      const b: Px = [w * 0.68, h * 0.4]
      out.marks.push(`M${a[0] + r} ${a[1]}A${r} ${r} 0 1 0 ${a[0] - r} ${a[1]}A${r} ${r} 0 1 0 ${a[0] + r} ${a[1]}`)
      out.frames.push({ x: b[0] - r, y: b[1] - r, w: r * 2, h: r * 2 })
      const [dx, dy] = [b[0] - a[0], b[1] - a[1]]
      const l = Math.hypot(dx, dy) || 1
      out.paths.push(ar([[a[0] + (dx / l) * (r + t), a[1] + (dy / l) * (r + t)],
                         [b[0] - (dx / l) * (r + t), b[1] - (dy / l) * (r + t)]]))
      out.labels = [{ x: a[0], y: a[1] + t * 1.2, t: 'A' }, { x: b[0], y: b[1] + t * 1.2, t: 'B' }]
      break
    }
    case 'handheld': {
      const s = m * 0.045
      const zig = (x: number, y: number, dx: number, dy: number) =>
        `M${x} ${y}l${dx * s} ${dy * s * 0.5}l${-dx * s * 0.6} ${dy * s * 0.9}l${dx * s} ${dy * s * 0.6}`
      out.marks.push(zig(pad, pad, 1, 1), zig(w - pad, pad, -1, 1),
                     zig(pad, h - pad, 1, -1), zig(w - pad, h - pad, -1, -1))
      break
    }
    case 'static': {
      const l = m * 0.11
      out.marks.push(`M${pad} ${pad + l}V${pad}H${pad + l}`,
                     `M${w - pad - l} ${pad}H${w - pad}V${pad + l}`,
                     `M${pad} ${h - pad - l}V${h - pad}H${pad + l}`,
                     `M${w - pad - l} ${h - pad}H${w - pad}V${h - pad - l}`)
      break
    }
    default:
      return null
  }
  return out
}

/** The stencil as SVG, inside whatever `<svg>` owns the frame's pixels. */
export function Stencil({ move, amp, w, h }: { move: string; amp: CameraAmp; w: number; h: number }) {
  if (!w || !h) return null
  const d = stencil(move, amp, w, h)
  if (!d) return null
  return (
    <g className="sbcam" data-stencil="">
      {d.frames.map((f, i) => (
        <rect key={`f${i}`} className={`camf${f.ghost ? ' ghost' : ''}`}
              x={f.x} y={f.y} width={f.w} height={f.h} rx={Math.max(2, Math.min(w, h) * 0.02)} />
      ))}
      {d.marks.map((p, i) => <path key={`m${i}`} className="camm" d={p} />)}
      {d.paths.map((p, i) => <path key={`p${i}`} className="cama" d={p} />)}
      {d.labels?.map((l) => (
        <text key={l.t} className="caml" x={l.x} y={l.y} textAnchor="middle">{l.t}</text>
      ))}
    </g>
  )
}
