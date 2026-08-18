import type { ParseElement } from '../api/types'

/** A run of the prompt the model wrote rather than the person. */
export type Mark = [number, number]

/**
 * Touching or overlapping runs become one.
 *
 * The page draws one underline per range, so two of them abutting renders as a
 * seam that means nothing and that the reader has to explain away. The server
 * already does this within an element (`_spans_to_text`); this is the same rule
 * across elements, where it could not have known.
 */
export function merge(marks: Mark[]): Mark[] {
  const sorted = [...marks].filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0])
  const out: Mark[] = []
  for (const [a, b] of sorted) {
    const last = out[out.length - 1]
    if (last && a <= last[1]) last[1] = Math.max(last[1], b)
    else out.push([a, b])
  }
  return out
}

/**
 * Element-local offsets, placed into the prompt they came from.
 *
 * The server answers in offsets into each element's own `text`, because that is
 * the string it built and the only one it can be sure of. Where that clause sits
 * in the prompt is a question only the page can answer, and it answers it by
 * walking forward: elements arrive in order, so a cursor that never goes
 * backwards cannot match the same words twice and hand two elements the same
 * span.
 *
 * A clause that is not found is skipped rather than guessed at. The model may
 * have tidied a clause it was quoting, and an underline placed by approximate
 * search is the failure this whole feature is built to avoid — a mark that looks
 * right, over words nobody chose.
 */
export function documentMarks(prompt: string, elements: ParseElement[]): Marks {
  const invented: Mark[] = []
  const spans: Mark[] = []
  let at = 0
  const walk = (e: ParseElement) => {
    const i = e.text ? prompt.indexOf(e.text, at) : -1
    if (i >= 0) {
      at = i + e.text.length
      spans.push([i, at])
      for (const [a, b] of e.invented ?? []) invented.push([i + a, i + b])
    }
    ;(e.children ?? []).forEach(walk)
  }
  elements.forEach(walk)
  return { invented: merge(invented), spans: merge(spans) }
}

/**
 * Two channels, and they answer different questions.
 *
 * **The underline is reach; the colour is authorship.** An element is a thing
 * the document can act on — reroll it, and later whatever else a span becomes —
 * so every element's own run is underlined whether the words in it are the
 * model's or the person's. Grey is a separate claim laid over the top: *these
 * particular words are mine, not yours*. A clause the person wrote and a clause
 * the model supplied are both addressable and only one of them is grey, which
 * is the arrangement the sketch draws and the reason provenance stays binary
 * rather than becoming a third colour.
 *
 * The connective tissue between elements is neither. Nothing owns it — it is
 * the separator the compiler chose — so there is nothing there to touch.
 */
export type Marks = { invented: Mark[]; spans: Mark[] }

/**
 * The invented run the caret is sitting in, and which element owns it.
 *
 * **A span is an object for exactly one thing: reroll.** That is the only
 * response an invented run has which plain text does not already give — editing
 * one is just typing into it — and it is what makes invention acceptable at
 * all: marked one way, and cheap to replace.
 *
 * Null everywhere else, including inside a *derived* element. Those are the
 * person's own words and there is nothing to propose about them.
 */
export function inventedAt(
  prompt: string, elements: ParseElement[], caret: number,
): { id: string; run: Mark } | null {
  let found: { id: string; run: Mark } | null = null
  let at = 0
  const walk = (e: ParseElement) => {
    const i = e.text ? prompt.indexOf(e.text, at) : -1
    if (i >= 0) {
      at = i + e.text.length
      for (const [a, b] of e.invented ?? []) {
        if (e.id && caret >= i + a && caret <= i + b) found = { id: e.id, run: [i + a, i + b] }
      }
    }
    ;(e.children ?? []).forEach(walk)
  }
  elements.forEach(walk)
  return found
}

/**
 * The person's own words, as the box currently stands.
 *
 * **Once the document's prose is written into the box, "what the user typed" is
 * no longer what the box says** — it says the sentence *plus* whatever the model
 * filled in. Preservation and coverage are both questions about the user's
 * words, so handing them the box would make them vacuous: every derived run
 * trivially appears in a string that contains every derived run.
 *
 * So the record is recovered rather than stored: the user's prose is the box
 * with the invented runs taken out. That is true after the first write, and it
 * stays true after an edit — editing a grey run drops its mark, so those words
 * rejoin this string the moment they become yours, with nothing to keep in step.
 */
export function derivedText(text: string, invented: Mark[]): string {
  return gaps(invented, text.length)
    .map(([a, b]) => text.slice(a, b))
    .join(' ')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .trim()
}

/**
 * May this text be written into the box, or would it take words off the user?
 *
 * The rule is one line — **the model may insert, never revise** — and this is
 * where it is enforced on the page. Every run of `text` the model did *not*
 * claim as its own has to be findable in what the person actually typed; if one
 * is not, the model has rewritten them, and the write simply does not happen.
 * The box keeps what was typed and nothing says so, which is the same silent
 * degrade the server takes when it drops an untrustworthy document.
 *
 * It is the client mirror of the validator's derived-text check and it is
 * deliberately the *same shape*: unordered, consuming. Unordered because
 * `PARSE_RULES` instructs a reorder — light and camera go last — so a check
 * that demanded the user's clauses in their original order would refuse exactly
 * the documents the rules ask for. Consuming because that is what stops a
 * document satisfying this by quoting one phrase of the prose repeatedly.
 *
 * Edges are forgiven and nothing else is. The compiler's whole licence over
 * someone's sentence is that it closes it and chooses the separator in front of
 * it, so a clause arriving with a full stop it did not have, or lower-cased
 * where it now sits mid-sentence, is the compiler doing its stated job. A
 * changed word is not, and lands as a missing run.
 */
export function insertionOnly(typed: string, text: string, marks: Mark[]): boolean {
  const hay = fold(typed)
  const taken: Mark[] = []
  for (const [a, b] of gaps(marks, text.length)) {
    const run = edges(fold(text.slice(a, b)))
    if (!run) continue
    let start = 0
    for (;;) {
      const i = hay.indexOf(run, start)
      if (i < 0) return false
      const end = i + run.length
      if (!taken.some(([x, y]) => x < end && i < y)) { taken.push([i, end]); break }
      start = i + 1
    }
  }
  return true
}

/** Lower-cased, unless lowering moved a length — the offsets index the original. */
function fold(s: string): string {
  const low = s.toLowerCase()
  return low.length === s.length ? low : s
}

const EDGES = /^[\s.,;:!?]+|[\s.,;:!?]+$/g
const edges = (s: string): string => s.replace(EDGES, '')

/** The complement of a merged mark list — the runs nobody claimed. */
function gaps(marks: Mark[], len: number): Mark[] {
  const out: Mark[] = []
  let at = 0
  for (const [a, b] of marks) {
    if (a > at) out.push([at, a])
    at = Math.max(at, b)
  }
  if (at < len) out.push([at, len])
  return out
}

/**
 * Marks carried across an edit — and **dropped where the edit landed on them.**
 *
 * That drop is the whole of "editing an invented span makes it derived": you
 * touched those words, so they are yours now, and nothing has to be clicked to
 * say so. It falls out of the bookkeeping rather than being a feature.
 *
 * Carrying rather than re-asking, because the model cannot tell the difference.
 * It is handed one string and has no way to know which half of it was its own
 * suggestion from a moment ago — so a re-ask would mark text the person accepted
 * as though they had written it, or worse, keep marking words they have since
 * rewritten. Only the page knows what it put there.
 */
/**
 * The one edit both `remap` and `remapCaret` are looking at.
 *
 * `from`/`to` bound the changed region in the *old* string; `delta` is what the
 * length moved by. Everything before `from` is untouched, everything at or after
 * `to` has simply shifted.
 */
function edit(before: string, after: string) {
  const max = Math.min(before.length, after.length)
  let head = 0
  while (head < max && before[head] === after[head]) head++
  let tail = 0
  while (tail < max - head
         && before[before.length - 1 - tail] === after[after.length - 1 - tail]) tail++
  return { from: head, to: before.length - tail, delta: after.length - before.length }
}

/**
 * The caret, carried across text the model replaced under it.
 *
 * Without this the box is unusable the moment filling-in works at all: a
 * suggestion lands mid-sentence, React writes a new value, and the caret goes to
 * the end of the prompt — so the next character you type appears somewhere you
 * were not looking. A caret that jumps is worse than a suggestion that never
 * arrives, because you find out about it by having already typed.
 */
export function remapCaret(pos: number, before: string, after: string): number {
  if (before === after) return pos
  const { from, to, delta } = edit(before, after)
  if (pos <= from) return pos
  if (pos >= to) return Math.max(0, Math.min(after.length, pos + delta))
  // Inside what was replaced. The end of the new run is where attention already
  // is once something has been written for you — and it is the only position
  // that does not sit inside a word nobody typed.
  return Math.max(0, Math.min(after.length, to + delta))
}

/**
 * The document after the person edited one of its clauses — or null, re-read it.
 *
 * **Editing a grey run is the frequent gesture, and it must not cost a
 * re-parse.** Not for speed: a re-read of the whole sentence is free to invent
 * something new somewhere *else*, so touching one assumption would move a part
 * of the picture you did not touch — which is the one thing this whole feature
 * promises not to do. Patching locally keeps the change to what the edit
 * implied.
 *
 * The element that contained the edit becomes `derived` with its marks cleared,
 * which is the same statement `remap` already makes about the underline: you
 * touched those words, so they are yours now, and nothing has to be clicked to
 * say so. Emptying a run deletes its element outright — no confirmation, per
 * the standing rule; `docUndo` is the reversal.
 *
 * Null whenever the edit was not contained in exactly one element — across two
 * clauses, or in the separator between them, which nobody owns. Those are real
 * rewrites of the sentence and the model should read it again.
 */
export function editDocument(
  elements: ParseElement[], before: string, after: string,
): ParseElement[] | null {
  if (before === after) return elements
  const { from, to, delta } = edit(before, after)

  // Spans are disjoint rather than nested: a child's clause is joined *after*
  // its parent's, so the walk lays every element out end to end and "the
  // element containing this edit" is one element or none.
  let hit: ParseElement | null = null
  let span: Mark | null = null
  let at = 0
  const walk = (e: ParseElement) => {
    const i = e.text ? before.indexOf(e.text, at) : -1
    if (i >= 0) {
      at = i + e.text.length
      if (i <= from && to <= at) { hit = e; span = [i, at] }
    }
    ;(e.children ?? []).forEach(walk)
  }
  elements.forEach(walk)
  if (!hit || !span) return null

  const [a, b] = span as Mark
  const text = after.slice(a, b + delta).trim()
  const swap = (list: ParseElement[]): ParseElement[] =>
    list.flatMap((e) => {
      const kids = e.children ? swap(e.children) : undefined
      if (e !== hit) return [{ ...e, ...(kids ? { children: kids } : {}) }]
      // Emptied, so it is gone — and its children with it, because they were
      // properties of a thing that is no longer in the sentence.
      if (!text) return []
      const { invented: _drop, ...rest } = e
      return [{ ...rest, text, origin: 'derived' as const,
                ...(kids ? { children: kids } : {}) }]
    })
  return swap(elements)
}

export function remap(marks: Mark[], before: string, after: string): Mark[] {
  if (before === after || !marks.length) return marks
  const { from, to, delta } = edit(before, after)

  const out: Mark[] = []
  for (const [a, b] of marks) {
    if (b <= from) out.push([a, b])
    else if (a >= to) out.push([a + delta, b + delta])
  }
  return merge(out).filter(([a, b]) => a >= 0 && b <= after.length && b > a)
}
