/**
 * The note under the prompt, which only ever says what is wrong.
 *
 * What is loaded is legible in the prompt itself; a line confirming the LoRAs you
 * can already read above it would be the page telling you what you can see. What
 * the prompt cannot show — a name that resolves to no file, a stack past
 * `max_loras`, a model that reads no LoRAs at all, the same LoRA in the prompt
 * *and* in a box — is the only thing this ever says.
 *
 * A pure function of the store, deliberately: in the vanilla page this was
 * `syncLoraNote()`, wired to five separate `input` listeners, and the one that
 * mattered most was easy to forget — the region weight, which can silence every
 * box without touching a token.
 */
import type { Store } from '../store'
import { attached, regionsLive, supports, videoModel } from '../store'
import { stripLoras } from './tokens'

export function loraNote(s: Store): string {
  const max = s.state?.max_loras ?? 6
  const bits: string[] = []

  // **Three notes are gone, and they are gone because they became impossible.**
  // A typed name could resolve to nothing (`No LoRA named "x"`) or to two files
  // (`"high" names 2 LoRAs`), and a LoRA bound to a trigger phrase could sit in
  // the stack doing almost nothing while the phrase was missing from the prose.
  // A chip is picked from a list, so the first two cannot happen. The third can,
  // and is deliberately not said: it is managed by hand. `/api/state` still
  // carries `trigger_word` per entry, which is where a check or an agent reads
  // the fact — what went is the nagging, not the information.
  if (s.loras.length > max) bits.push(`Only the first ${max} LoRAs are applied.`)

  if (regionsLive(s)) {
    // The same LoRA on the canvas and in a box is the one combination that
    // quietly undoes the feature: the canvas copy goes onto the global
    // LoraLoader chain and patches everything, so the box's mask is still there
    // and no longer separating anything. It looks like regional bleeding rather
    // than like two copies of one LoRA, which is why it has to be named.
    const boxed = new Set(s.regions.flatMap((r) => (r.lora ? [r.lora.path] : [])))
    for (const c of s.loras) {
      if (boxed.has(c.path)) {
        bits.push(`"${c.rel}" is on the canvas and in a box — the canvas copy applies `
          + 'to the whole frame and cancels the masking.')
      }
    }
    // Region weight multiplies every box's own strength, so a zero here is not a
    // weak render — it is every boxed LoRA switched off. The node answers that by
    // returning the model unpatched, and a picture still comes back, placed by
    // the caption alone. That is what earns the line: nothing else on the page
    // tells that render apart from one the LoRAs actually ran in.
    if (parseFloat(s.regionWeight) === 0 && boxed.size)
      bits.push('Region weight is 0 — every box\u2019s LoRA is switched off.')
  }

  // **The composer is per-kind now, so this is no longer about a stack left
  // behind by the other side** — a Krea 2 chip cannot follow you to video at
  // all. What it still catches is a stack under a model that reads none.
  if (s.kind === 'video' && !supports(s).loras && s.loras.length) {
    bits.push(`${videoModel(s)?.label ?? 'This model'} takes no LoRAs — `
      + 'the ones on the canvas are ignored.')
  }
  return bits.join(' ')
}

/**
 * What the boxes cannot show: which engine this run is about to take.
 *
 * They are genuinely different — one sampling pass against masked LoRA deltas, or
 * a krea2edit compose that regenerates the whole frame around the plate — and the
 * second is several times slower, which is worth knowing before you press
 * Generate rather than after. A region's own photo is neither: it is a latent
 * mold on the fast path, so it never moves the run onto the slow one.
 */
export function regionNote(s: Store, live: number): string {
  if (!regionsLive(s) || !live) return ''
  const molds = s.regions.filter((r) => attached(r, 'identity')).length
  const tail = molds ? ` ${molds} with a reference photo.` : ''
  if (attached(s.frame, 'scene') || attached(s.frame, 'outfit')) {
    return `${live} region${live > 1 ? 's' : ''} composed into the reference — `
      + `slower, and it re-renders the whole frame.${tail}`
  }
  // A box with words but no identity is placed by the description alone — there
  // is no LoRA delta to mask, so it is a soft placement rather than a guaranteed
  // one. Worth saying, because the two kinds of box look identical on the canvas
  // and do not hold their ground equally.
  const soft = s.regions.filter((r) =>
    stripLoras(r.prompt || '').trim() && !attached(r, 'identity') && !r.lora).length
  const softNote = soft ? ` ${soft} described only — placed by the words, not held by a mask.` : ''
  return `${live} region${live > 1 ? 's' : ''}, one pass. `
    + `Each LoRA is masked to its box.${tail}${softNote}`
}

export const NEED_EDIT_LORA =
  'Scene and outfit transfer need the Krea 2 identity-edit LoRA — download it under Settings.'
