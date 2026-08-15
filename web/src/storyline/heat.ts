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

/** A gradient whose stops sit under the elements they belong to. */
export function field(stops: { pct: number; color: string }[]): string {
  const s = [...stops].sort((a, b) => a.pct - b.pct)
  const first = s[0]
  const last = s[s.length - 1]
  if (!first || !last) return 'rgba(255,255,255,.06)'
  const mid = s.map((x) => `${x.color} ${x.pct.toFixed(1)}%`).join(', ')
  return `linear-gradient(90deg, ${first.color} 0%, ${mid}, ${last.color} 100%)`
}
