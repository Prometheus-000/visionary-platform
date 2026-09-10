/**
 * Arrow geometry, in pixels.
 *
 * One outliner serves both arrow kinds, because the two are told apart by
 * *fill* and not by shape — solid is the camera, hollow is a subject, which is
 * the storyboard convention with the colour scheme taken out and the logic
 * kept. Every arrow here is a closed polygon: a shaft of constant width along
 * a polyline, ending in a head. A camera stencil fills it; a subject arrow
 * strokes it and leaves the inside translucent so the picture shows through.
 */

export type Px = [number, number]

const norm = (x: number, y: number): Px => {
  const l = Math.hypot(x, y) || 1
  return [x / l, y / l]
}

/** Points along an arc, angles in degrees, screen-space (y down). */
export function arc(cx: number, cy: number, r: number, a0: number, a1: number, n = 18): Px[] {
  const out: Px[] = []
  for (let i = 0; i <= n; i++) {
    const t = ((a0 + ((a1 - a0) * i) / n) * Math.PI) / 180
    out.push([cx + r * Math.cos(t), cy + r * Math.sin(t)])
  }
  return out
}

/**
 * A closed arrow polygon along `pts`: shaft `w` wide, a head `headW` across
 * and `headL` long. The head is never more than half the arrow, so a short
 * arrow is a short arrow rather than a head with no shaft.
 */
export function arrowPath(pts: Px[], w: number, headW: number, headL: number): string {
  if (pts.length < 2) return ''
  const segs: number[] = []
  let total = 0
  for (let i = 1; i < pts.length; i++) {
    const d = Math.hypot(pts[i]![0] - pts[i - 1]![0], pts[i]![1] - pts[i - 1]![1])
    segs.push(d)
    total += d
  }
  if (total < 1) return ''
  const hl = Math.min(headL, total * 0.5)
  // The shaft stops `hl` short of the tip; find where along the polyline.
  const cut = total - hl
  const body: Px[] = [pts[0]!]
  let run = 0
  let base: Px = pts[pts.length - 1]!
  for (let i = 1; i < pts.length; i++) {
    const d = segs[i - 1]!
    if (run + d >= cut) {
      const t = d ? (cut - run) / d : 0
      base = [pts[i - 1]![0] + (pts[i]![0] - pts[i - 1]![0]) * t,
              pts[i - 1]![1] + (pts[i]![1] - pts[i - 1]![1]) * t]
      body.push(base)
      break
    }
    run += d
    body.push(pts[i]!)
  }
  const tip = pts[pts.length - 1]!
  const [dx, dy] = norm(tip[0] - base[0], tip[1] - base[1])
  const headN: Px = [-dy, dx]

  // A normal per shaft point: the segment's own at the ends, the average of
  // the two neighbours inside, miter-limited so a sharp turn does not spike.
  const left: Px[] = []
  const right: Px[] = []
  for (let i = 0; i < body.length; i++) {
    const p = body[i]!
    const prev = body[i - 1]
    const next = body[i + 1]
    let n: Px
    if (i === body.length - 1) n = headN
    else if (!prev) {
      const [ux, uy] = norm(next![0] - p[0], next![1] - p[1])
      n = [-uy, ux]
    } else {
      const [ax, ay] = norm(p[0] - prev[0], p[1] - prev[1])
      const [bx, by] = norm(next![0] - p[0], next![1] - p[1])
      const [mx, my] = norm(-ay - by, ax + bx)
      const cos = mx * -ay + my * ax
      const scale = Math.min(2, 1 / Math.max(0.5, Math.abs(cos)))
      n = [mx * scale, my * scale]
    }
    left.push([p[0] + (n[0] * w) / 2, p[1] + (n[1] * w) / 2])
    right.push([p[0] - (n[0] * w) / 2, p[1] - (n[1] * w) / 2])
  }
  const hw = Math.max(headW, w * 1.6)
  const path = [
    ...left,
    [base[0] + (headN[0] * hw) / 2, base[1] + (headN[1] * hw) / 2],
    tip,
    [base[0] - (headN[0] * hw) / 2, base[1] - (headN[1] * hw) / 2],
    ...right.reverse(),
  ] as Px[]
  return `M${path.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join('L')}Z`
}
