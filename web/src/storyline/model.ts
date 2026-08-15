/**
 * The storyline's data model, mirroring `_validate_modules` / `_module_clause`
 * / `_prominence` in app.py.
 *
 * It is a mirror on purpose and the duplication is the point of failure to
 * watch: a preview with its own implementation is a preview that can disagree
 * with the run, which is worse than no preview. `/api/compile` is the authority
 * and this exists so the field can respond to a keystroke without a round trip.
 * If the two ever diverge, this one is wrong.
 */

export type Origin = 'derived' | 'invented'

/**
 * An anchor plus everything hanging off it — **one** thing in the picture.
 *
 * There is no `order` field: array order is the order, because carrying both
 * would be two sources of truth for one fact and the failure is silent. Order
 * is placement, so getting it wrong moves people around the frame.
 */
export type Module = {
  id: string
  text: string
  origin: Origin
  /** Containment — dependents fold into this element's clause. */
  children: Module[]
  /**
   * **Peer edges.** Ids of elements this one stands in relation to.
   *
   * Merge covers containment and nothing else: light *is* a property of the
   * woman, so nesting it is true. Two small girls *beside* her are not part of
   * her, so nesting them would be a lie — and with nowhere structural to put
   * the relation it ends up in the prose as "beside her", where reordering
   * silently breaks it and the prompt introduces a `her` nobody has met.
   *
   * A tie is the same relation held as a link instead. It survives a reorder
   * because it is not a word, and it constrains one: an element cannot precede
   * something it is tied to.
   */
  ties: string[]
}

let seq = 0
export const mod = (
  text: string, origin: Origin = 'derived', ties: string[] = [],
): Module => ({ id: `m${++seq}`, text, origin, children: [], ties })

/** Extent, summed over the subtree — merging folds a child's words into its anchor. */
export function words(m: Module): number {
  const own = m.text.trim() ? m.text.trim().split(/\s+/).length : 0
  return own + m.children.reduce((n, k) => n + words(k), 0)
}

/**
 * Each top-level element's share of the picture.
 *
 * Extent drives prominence, not position. Rearranging carries a share with the
 * element; only editing changes one. That is why the heat travels on a drag —
 * it falls out of the arithmetic rather than being animated on top of it.
 */
export function shares(mods: Module[]): number[] {
  const counts = mods.map(words)
  const total = counts.reduce((a, b) => a + b, 0) || 1
  return counts.map((c) => c / total)
}

// ── emission ────────────────────────────────────────────────────────────────

/** An anchor, then its dependents as one comma list, as one clause. */
function clause(m: Module): string {
  if (!m.children.length) return m.text
  const kids = m.children.map(clause).filter(Boolean).join(', ')
  return `${m.text.replace(/\.$/, '')} ${kids}`
}

const close = (t: string) => {
  const s = t.trim()
  return s && !/[.!?…"']$/.test(s) ? `${s}.` : s
}

/**
 * Sentences joined so a lowercase fragment does not follow a full stop.
 *
 * No character of the user's text is touched — only the separator in front of
 * it is chosen. Capitalising would silently turn a `k3nan` trigger word into
 * `K3nan` and weaken the LoRA it was trained for.
 */
export function compile(mods: Module[]): string {
  const parts = mods.map(clause).map(close).filter(Boolean)
  let out = ''
  for (const p of parts) {
    if (!out) out = p
    else if (/^[a-z]/.test(p) && out.endsWith('.')) out = `${out.slice(0, -1)}, ${p}`
    else out += ` ${p}`
  }
  return out
}

/**
 * Does this element's text lean on something before it?
 *
 * "Beside her, to her right, stand two small girls" is not free to move — the
 * reference dangles the moment it precedes the woman it refers to, and the
 * prompt introduces a `her` nobody has met. That is a **peer edge**: the girls
 * are adjacent to the woman, not part of her, so merging them under her would
 * be wrong and the relation has nowhere structural to live. It stays in the
 * prose, and prose that points backwards constrains the order.
 *
 * Only a *leading* reference counts. "A woman in her forties" contains `her`
 * and is self-contained; the test is whether the element opens by pointing at
 * something it has not introduced — the same reason `_looks_like_refusal` is
 * prefix-anchored rather than a substring search.
 */
const BACKREF =
  /^(beside|next to|behind|in front of|opposite|across from|between|with (his|her|their)|to (his|her|their)|she|he|they|her|him|his|them|their)\b/i

export const dependsOnPrior = (m: Module): boolean => BACKREF.test(m.text.trim())

// ── structure ───────────────────────────────────────────────────────────────

export type Row = { m: Module; depth: number; path: number[] }

/** Depth-first, for rendering — the tree as the rows you actually see. */
export function rows(mods: Module[], depth = 0, path: number[] = []): Row[] {
  return mods.flatMap((m, i) => [
    { m, depth, path: [...path, i] },
    ...rows(m.children, depth + 1, [...path, i]),
  ])
}

/** The sibling array a path points into, and the index within it. */
function at(mods: Module[], path: number[]): Module[] {
  const head = path[0]
  if (head === undefined) return mods
  const child = mods[head]
  return child ? at(child.children, path.slice(1)) : mods
}

/** The last index of a path — every caller has already established it is here. */
const tip = (path: number[]): number => path[path.length - 1] ?? 0

const clone = (mods: Module[]): Module[] =>
  mods.map((m) => ({ ...m, ties: [...m.ties], children: clone(m.children) }))

/** Every element, flat, so a tie can be resolved wherever it lives. */
export const all = (mods: Module[]): Module[] =>
  mods.flatMap((m) => [m, ...all(m.children)])

/** Where an element sits among its siblings — a tie is satisfied by preceding. */
const indexOf = (mods: Module[], id: string): number =>
  mods.findIndex((m) => m.id === id)

/**
 * Would this order leave a tie pointing forwards?
 *
 * A relation is only readable once both ends exist, so an element has to follow
 * everything it is tied to. This is the whole constraint, and it is why a tie
 * is worth holding as a link: the rule is checkable, where "beside her" is not.
 */
export function danglingTies(mods: Module[]): Set<string> {
  const bad = new Set<string>()
  mods.forEach((m, i) => {
    for (const t of m.ties) {
      const j = indexOf(mods, t)
      if (j >= 0 && j > i) bad.add(m.id)
    }
  })
  return bad
}

export function tie(mods: Module[], from: string, to: string): Module[] {
  const next = clone(mods)
  const m = all(next).find((x) => x.id === from)
  if (m && from !== to && !m.ties.includes(to)) m.ties.push(to)
  return next
}

export function untie(mods: Module[], from: string, to: string): Module[] {
  const next = clone(mods)
  const m = all(next).find((x) => x.id === from)
  if (m) m.ties = m.ties.filter((t) => t !== to)
  return next
}

/**
 * **Merge.** The element becomes a dependent of the sibling above it.
 *
 * This is not a display convention — it changes the emitted grammar from two
 * clauses to one, which is the difference between a light that is its own
 * character and a light that falls on somebody.
 */
export function indent(mods: Module[], path: number[]): Module[] {
  const next = clone(mods)
  const sibs = at(next, path.slice(0, -1))
  const i = tip(path)
  const anchor = sibs[i - 1]
  const moved = sibs[i]
  if (i === 0 || !anchor || !moved) return mods  // nothing above to hang off
  sibs.splice(i, 1)
  anchor.children.push(moved)
  return next
}

/** **Split.** The element leaves its anchor and becomes one of its own. */
export function outdent(mods: Module[], path: number[]): Module[] {
  if (path.length < 2) return mods               // already top level
  const next = clone(mods)
  const sibs = at(next, path.slice(0, -1))
  const grand = at(next, path.slice(0, -2))
  const moved = sibs[tip(path)]
  if (!moved) return mods
  sibs.splice(tip(path), 1)
  grand.splice((path[path.length - 2] ?? 0) + 1, 0, moved)
  return next
}

/**
 * Reorder among siblings — which for subjects is left-to-right in the frame.
 *
 * **Things that are together travel together.** Move a woman two girls are
 * standing beside and the girls come with her, because that is what anyone
 * expects of two things in a relation and it needs no explaining. The earlier
 * version refused the move instead, which is mechanism-thinking: correct about
 * the constraint, and asking the user to understand why.
 *
 * A refusal survives only for the move that no rearrangement can satisfy —
 * pushing a dependent above the thing it depends on. Everything else resolves
 * by carrying the group.
 */
export function move(mods: Module[], path: number[], dir: -1 | 1): Module[] {
  const next = clone(mods)
  const sibs = at(next, path.slice(0, -1))
  const i = tip(path)
  const a = sibs[i]
  if (!a) return mods

  // The element and everything standing in relation to it, in order.
  const groupIds = new Set([a.id, ...sibs.filter((m) => m.ties.includes(a.id)).map((m) => m.id)])
  const group = sibs.filter((m) => groupIds.has(m.id))
  const rest = sibs.filter((m) => !groupIds.has(m.id))
  const first = sibs.findIndex((m) => groupIds.has(m.id))
  const target = dir === -1 ? first - 1 : first + 1
  if (target < 0 || target > rest.length) return mods

  const reordered = [...rest.slice(0, target), ...group, ...rest.slice(target)]
  if (danglingTies(reordered).size) return mods
  sibs.length = 0
  sibs.push(...reordered)
  return next
}

/**
 * Write text into an element — **splitting it if line breaks arrived.**
 *
 * A line break already means "separate thought" to whoever typed it, and the
 * compiler flattens whitespace on the way to the encoder, so an element holding
 * three lines is three thoughts the storyline is not showing and the prompt
 * will not keep apart. Pasting a scene is the common case: without this it
 * lands as one element at 100% of the picture, which is both wrong and the
 * least useful thing the display could say.
 */
export function setText(mods: Module[], path: number[], text: string): Module[] {
  const next = clone(mods)
  const sibs = at(next, path.slice(0, -1))
  const target = sibs[tip(path)]
  if (!target) return next

  const lines = text.split(/\n+/).map((l) => l.trim()).filter(Boolean)
  if (lines.length <= 1) {
    target.text = text
    return next
  }
  target.text = lines[0] ?? ''
  sibs.splice(tip(path) + 1, 0,
              ...lines.slice(1).map((l) => ({ ...mod(l), origin: target.origin })))
  return next
}

export function insertAfter(mods: Module[], path: number[]): [Module[], string] {
  const next = clone(mods)
  const fresh = mod('')
  at(next, path.slice(0, -1)).splice(tip(path) + 1, 0, fresh)
  return [next, fresh.id]
}

export function remove(mods: Module[], path: number[]): Module[] {
  const next = clone(mods)
  const sibs = at(next, path.slice(0, -1))
  const gone = sibs[tip(path)]
  if (!gone) return mods
  sibs.splice(tip(path), 1)
  // Orphans are promoted rather than deleted: removing an anchor should not
  // silently take everything hanging off it.
  sibs.splice(tip(path), 0, ...gone.children)
  return next
}
