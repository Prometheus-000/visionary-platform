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
import { supports, videoModel } from '../store'
import { loraAlternatives, loraIndex, parseLoras, stripLoras } from './tokens'

export function loraNote(s: Store): string {
  const index = loraIndex(s.state)
  const max = s.state?.max_loras ?? 6
  const toks = parseLoras(index, s.prompt)
  const regionToks = s.regions.flatMap((r) => parseLoras(index, r.prompt || ''))
  const bad = [...new Set([...toks, ...regionToks].filter((t) => !t.hit).map((t) => t.name))]
  const bits: string[] = []

  // Ambiguous first and named individually: the fix is to copy one of the
  // alternatives, so listing them is the whole message.
  for (const b of bad) {
    const alts = loraAlternatives(index, b)
    if (alts.length > 1) bits.push(`"${b}" names ${alts.length} LoRAs — use ${alts.join(' or ')}.`)
  }
  const gone = bad.filter((b) => !loraAlternatives(index, b).length)
  if (gone.length) {
    bits.push(gone.length === 1
      ? `No LoRA named "${gone[0]}" on the volume.`
      : `No LoRAs named ${gone.map((b) => `"${b}"`).join(', ')} on the volume.`)
  }
  if (toks.filter((t) => t.hit).length > max) bits.push(`Only the first ${max} LoRAs are applied.`)

  // One per box is the node's shape, and the backend rejects the rest rather than
  // applying the first — so say it here, while the second token is still under
  // the caret, instead of after a round trip.
  s.regions.forEach((r, i) => {
    if (parseLoras(index, r.prompt || '').filter((t) => t.hit).length > 1)
      bits.push(`Region ${i + 1} names more than one LoRA — a region takes one.`)
  })

  if (s.regional) {
    // The same LoRA in the prompt and in a box is the one combination that
    // quietly undoes the feature: the prompt copy goes onto the global
    // LoraLoader chain and patches the whole canvas, so the box's mask is still
    // there and no longer separating anything. It looks like regional bleeding
    // rather than like two copies of one LoRA, which is why it has to be named.
    const boxed = new Set(
      s.regions.flatMap((r) => parseLoras(index, r.prompt || '').filter((t) => t.hit)
        .map((t) => t.hit!.path)),
    )
    const both = [...new Set(toks.filter((t) => t.hit && boxed.has(t.hit.path))
      .map((t) => t.hit!.token))]
    for (const n of both) {
      bits.push(`"${n}" is in the prompt and in a box — the prompt copy applies to `
        + 'the whole canvas and cancels the masking.')
    }
    // Region weight multiplies every box's own strength, so a zero here is not a
    // weak render — it is every boxed LoRA switched off. The node answers that by
    // returning the model unpatched, and a picture still comes back, placed by
    // the caption alone. That is what earns the line: nothing else on the page
    // tells that render apart from one the LoRAs actually ran in.
    if (parseFloat(s.regionWeight) === 0 && boxed.size)
      bits.push('Region weight is 0 — every box’s LoRA is switched off.')
  }

  // The prompt is shared by image and video, so a stack typed for one is still
  // sitting there when you switch to the other. Saying that beats letting a
  // guidance-distilled checkpoint quietly ignore four of them.
  if (s.kind === 'video' && !supports(s).loras && toks.some((t) => t.hit)) {
    bits.push(`${videoModel(s)?.label ?? 'This model'} takes no LoRAs — `
      + 'the ones in the prompt are ignored.')
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
  if (!s.regional || !live) return ''
  const molds = s.regions.filter((r) => r.ref).length
  const tail = molds ? ` ${molds} with a reference photo.` : ''
  if (s.plate.scene || s.plate.outfit) {
    return `${live} region${live > 1 ? 's' : ''} composed into the reference — `
      + `slower, and it re-renders the whole frame.${tail}`
  }
  // A box with words but no identity is placed by the description alone — there
  // is no LoRA delta to mask, so it is a soft placement rather than a guaranteed
  // one. Worth saying, because the two kinds of box look identical on the canvas
  // and do not hold their ground equally.
  const index = loraIndex(s.state)
  const soft = s.regions.filter((r) =>
    stripLoras(r.prompt || '').trim() && !r.ref
    && !parseLoras(index, r.prompt || '').some((t) => t.hit)).length
  const softNote = soft ? ` ${soft} described only — placed by the words, not held by a mask.` : ''
  return `${live} region${live > 1 ? 's' : ''}, one pass. `
    + `Each LoRA is masked to its box.${tail}${softNote}`
}

export const NEED_EDIT_LORA =
  'Scene and outfit transfer need the Krea 2 identity-edit LoRA — download it under Settings.'
