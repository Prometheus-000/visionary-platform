/**
 * The attention field across the prompt, sampled at element centres.
 *
 * **Not one colour per element.** The encoder reads a prompt as one composition
 * whose parts condition each other, so a badge per element would present N
 * independent weights and reintroduce the tag-list model in the picture of the
 * prompt. The stops are samples of a continuous field; the interpolation
 * between them means something.
 */

/**
 * Cool to hot. Hue carries the signal and lightness carries it a second time,
 * because a constant-lightness hue ramp is invisible to red-green colour vision
 * deficiency — the same information twice costs nothing here.
 */
export function heat(share: number, max: number): string {
  const t = Math.max(0, Math.min(1, max > 0 ? share / max : 0))
  return `oklch(${(0.62 + t * 0.18).toFixed(3)} 0.16 ${(250 - t * 232).toFixed(0)})`
}

/**
 * The bar as a **map of the prompt**: each element holds a span of it in order,
 * and the span is its share.
 *
 * The first version sampled each element's measured x-centre, which was right
 * for a horizontal row of chips and meaningless the moment the storyline became
 * a vertical list — every element has the same x-centre, so the gradient said
 * nothing and could disagree with the ticks beside it.
 *
 * Position now means what order means everywhere else in this design: read the
 * bar left to right and you are reading the prompt. Width carries share as well
 * as colour does, which is not redundant so much as legible — a wide red band
 * is an element that owns the picture, and you can point at it.
 *
 * Still interpolated rather than banded. The encoder reads one composition, and
 * hard edges would draw N independent weights, which is the tag-list model
 * wearing a gradient.
 */
export function field(shares: number[], max: number): string {
  if (!shares.length) return 'rgba(255,255,255,.06)'
  const colors = shares.map((s) => heat(s, max))
  const stops: string[] = []
  let at = 0
  shares.forEach((s, i) => {
    const c = colors[i] as string
    const w = s * 100
    // A stop inside each end of the element's own span, so its colour sits over
    // the element it belongs to and the blend happens across the boundary.
    stops.push(`${c} ${(at + w * 0.15).toFixed(2)}%`)
    stops.push(`${c} ${(at + w * 0.85).toFixed(2)}%`)
    at += w
  })
  // Anchored at both ends with the bare colour — never by slicing a stop
  // string, since `oklch(0.8 0.16 18)` contains spaces and splitting on one
  // yields `oklch(0.8`, which is invalid and silently drops the whole gradient.
  return `linear-gradient(90deg, ${colors[0]} 0%, ${stops.join(', ')}, `
       + `${colors[colors.length - 1]} 100%)`
}
