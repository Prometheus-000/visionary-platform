/**
 * ⌥← / ⌥→ : move the clause under the caret one slot along.
 *
 * A prompt is written by reordering it. "in soft window light" belongs before the
 * subject as often as after, and moving it there by hand is a select, a cut, a click
 * and a paste — four gestures, each with its own way of eating a comma or leaving a
 * double space behind.
 *
 * **The separators are slots and they do not move.** The commas and line breaks stay
 * exactly where they are and the text between them changes places. That is what keeps
 * a prompt written across two lines at two lines, and one with a single comma at a
 * single comma, however many times you press the chord. Each slot also keeps its own
 * leading and trailing whitespace, so a clause moving into the first position does not
 * drag the space that used to precede it along.
 *
 * Returns null rather than doing nothing quietly when there is nowhere to go — at the
 * ends, or against an empty slot, which is a trailing comma and not a clause. The
 * caller lets the key fall through to the OS's word-jump there, which is the honest
 * answer to ⌥← on the first clause in the box.
 */
export function moveClause(
  value: string,
  at: number,
  dir: 1 | -1,
): { value: string; caret: number } | null {
  const slots: {
    s: number; e: number; sep: string; core: string; lead: string; tail: string
  }[] = []
  let s = 0
  for (let i = 0; i <= value.length; i++) {
    if (i !== value.length && value[i] !== ',' && value[i] !== '\n') continue
    const t = value.slice(s, i)
    const core = t.trim()
    slots.push({
      s, e: i, sep: value[i] ?? '', core,
      // An all-whitespace slot has no core to sit between a lead and a tail, and
      // splitting it into both would write the run out twice.
      lead: core ? t.slice(0, t.length - t.trimStart().length) : t,
      tail: core ? t.slice(t.trimEnd().length) : '',
    })
    s = i + 1
  }

  const i = slots.findIndex((sl) => at >= sl.s && at <= sl.e)
  const j = i + dir
  const from = slots[i]
  const to = slots[j]
  if (i < 0 || !from || !to || !from.core || !to.core) return null

  // Where the caret sits inside the clause it is holding, so a run of presses keeps
  // hold of it rather than moving it once and losing its place.
  const within = Math.min(from.core.length, Math.max(0, at - from.s - from.lead.length))
  const swapped = slots.map((sl, k) =>
    k === i ? { ...sl, core: to.core } : k === j ? { ...sl, core: from.core } : sl)

  let out = ''
  let pos = 0
  swapped.forEach((sl, k) => {
    if (k === j) pos = out.length + sl.lead.length
    out += sl.lead + sl.core + sl.tail + sl.sep
  })
  return { value: out, caret: pos + within }
}
