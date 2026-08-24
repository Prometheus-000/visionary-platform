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
import { readRegions, regionArmed } from '../regions/geometry'
import { stripLoras } from './tokens'

/** Every frame attachment that rides V12's extra_ref sockets. */
const PLATE_SLOTS = ['scene', 'outfit', 'object1', 'object2'] as const
/** What the warnings call a slot — the object roles are one word to a user. */
const plateName = (slot: string) => (slot.startsWith('object') ? 'object' : slot)

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

  // The plates have a home at rest now (`PlateRow`), so a photo can arrive
  // before any box exists — and with zero boxes the payload sends it as null.
  // The note teaches the gesture that is missing, because the double-click is
  // the least discoverable thing on the page and this is the one moment
  // someone is looking for it.
  if (s.kind === 'image' && !s.regions.length) {
    for (const slot of PLATE_SLOTS) {
      if (attached(s.frame, slot))
        bits.push(`The ${plateName(slot)} photo waits on a region — double-click the `
          + 'canvas to place someone in it.')
    }
  }

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

    // The payload drops boxes with no words, no LoRA and no photo, and the
    // plates ride on what survives — `imageBody` sends `outfit: null` the
    // moment the filtered list is empty. Found by driving the page: an outfit
    // tile showing its jacket, a rectangle on the canvas, and a request
    // carrying no outfit and no error, because the backend's own "needs at
    // least one region" answer only fires on a plate it is actually sent.
    if (!readRegions([], s.regions, true).length) {
      for (const slot of PLATE_SLOTS) {
        if (attached(s.frame, slot))
          bits.push(`The ${plateName(slot)} photo is not sent — every box is empty. `
            + 'Give a box a sentence, a photo or a LoRA.')
      }
    } else if (!s.regions.some((r) => regionArmed([], r))) {
      // The edit path only arms boxes holding a LoRA or a photo, so a plate
      // over described-only boxes is a rejected run — the backend answers with
      // the same sentence. Found live: "V12 unified attention was not armed",
      // nine seconds after a cold model load.
      for (const slot of PLATE_SLOTS) {
        if (attached(s.frame, slot))
          bits.push(`The ${plateName(slot)} compose needs a box holding an identity — `
            + 'a LoRA or a photo. Described-only boxes cannot anchor it.')
      }
    }
  }

  // Style is the whole-frame engine and the boxes are the regional one; the
  // route refuses the pair, so the note names the conflict while both are on
  // screen — and says which one style would keep.
  if (s.kind === 'image' && s.regions.length
      && attached(s.frame, 'style1')) {
    bits.push('Style references and region boxes are two engines — remove '
      + 'one. Style applies to the whole frame.')
  }

  // Zero is not a weak style, it is the mechanism switched off: the LoRA is
  // what lets the model consume the reference latents at all, and a render
  // still comes back, placed by the prompt alone — indistinguishable from one
  // the references ran in. The region-weight-zero lesson, same silence.
  if (s.kind === 'image' && parseFloat(s.styleStrength) === 0
      && attached(s.frame, 'style1')) {
    bits.push('Style strength is 0 — the reference photo is switched off.')
  }

  // The backend refuses an object plate whose note is empty — a reference the
  // prompt never mentions does close to nothing — so the same sentence shows
  // here, next to the field that fixes it.
  if (s.kind === 'image') {
    for (const slot of ['object1', 'object2'] as const) {
      const a = s.frame.attachments.find((x) => x.role === slot)
      if (a && !(a.note ?? '').trim())
        bits.push('The object photo needs a sentence saying what it is — '
          + '\u201ca motorcycle she leans against\u201d.')
    }
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
  if (!regionsLive(s)) return ''
  // Zero live boxes is not "nothing to say" — it is the one arrangement where
  // the tiles above this note lie: they show their photos, and the payload
  // sends neither, because the plates ride on the filtered region list. The
  // frame card is where those tiles are, so the sentence belongs here too, not
  // only under the main prompt.
  if (!live) {
    const held = (['scene', 'outfit'] as const).filter((p) => attached(s.frame, p))
    return held.length
      ? `Every box is empty — the ${held.join(' and ')} photo${held.length > 1 ? 's are' : ' is'} `
        + 'not sent. Give a box a sentence, a photo or a LoRA.'
      : ''
  }
  const molds = s.regions.filter((r) => attached(r, 'identity')).length
  const tail = molds ? ` ${molds} with a reference photo.` : ''
  if (PLATE_SLOTS.some((slot) => attached(s.frame, slot))) {
    // Promised only when a box can anchor it — the edit path arms boxes
    // holding a LoRA or a photo, and the backend rejects a plate over
    // described-only boxes, so this card saying "composed" while the console
    // says "cannot anchor" was the same state described twice, differently.
    if (!s.regions.some((r) => regionArmed([], r))) {
      return 'The compose needs a box holding an identity — a LoRA or a '
        + 'photo. Described-only boxes cannot anchor it.'
    }
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
