/**
 * What a LoRA is, once it stops pretending to be text.
 *
 * This file used to open by defending `<lora:k3nan:0.4>` in the prompt —
 * Automatic1111's syntax, the notation anyone who has trained these models
 * already types — against a row per LoRA that cost 56px plus a wrapped select.
 * That defence rested on one claim: a row *"could not say the thing that matters
 * most, which is where in the sentence the LoRA applies."*
 *
 * **The claim is true in a region and false in the main prompt**, and this
 * codebase says so itself, in `useDocument.ts`: *"a token's position in the main
 * prompt means nothing to the backend, which reads them into a stack."* So the
 * canvas was paying for a parser, a caret-targeting scheme and a drag subsystem
 * to buy a property only a box has — and what the position means in a box is
 * *which box*, which a control living on that box says without any syntax.
 *
 * What is left here is the index the picker reads, resolution for the one caller
 * that still starts from a name (Reuse), and the two readers that turn chips
 * into the stacks `/api/generate` and `/api/video` take.
 *
 * See `docs/design-notes/loras-are-not-text.md`.
 */
import type { AppState, LoraEntry } from '../api/types'

/**
 * A LoRA plugged into a module.
 *
 * **Not text, and never was.** It was written `<lora:name:1>` in the prompt for
 * exactly one reason — so a strength could be typed beside it — and it paid for
 * that with a parser, a caret-targeting scheme, a drag subsystem and three
 * classes of error message. A chip with a value carries the strength better, so
 * the reason is gone.
 *
 * It never reached the encoder in the first place: `stripLoras` deleted it from
 * the string before `/api/generate` was called and the stack travelled in its
 * own field. A trigger *phrase* is the opposite — it is words, it does reach the
 * encoder, and it stays in the prompt where you put it.
 *
 * See `docs/design-notes/loras-are-not-text.md`.
 */
export type LoraChip = {
  /** The volume path — the only field of a `LoraFile` that is a key, and what
   *  `/api/generate` takes. */
  path: string
  /** The shortest name that still points at one file. What the chip reads. */
  rel: string
  strength: number
  /** Image side. Null leaves it to the backend, which defaults it to the UNet
   *  weight — duplicating that here would freeze today's default into every
   *  chip ever saved. */
  textEncoder: number | null
  /** Video side. Null means read it off the filename, which `vidExpert` gets
   *  right nearly always; this is the override. */
  expert: 'high' | 'low' | 'both' | null
}


/** The one thing here that still knows the syntax exists, and it only ever
 *  deletes. `<lora:…>` left in a prompt is now plain text — nothing parses it,
 *  converts it or migrates it — but text is what the encoder reads, so a reused
 *  prompt carrying one would render the literal word "lora". Stripped on the way
 *  out, never reinterpreted in the box. */
const LORA_RE = /<lora:([^<>]*)>/gi

export type LoraFile = {
  path: string
  /** Relative to `loras/`, derived from the path rather than assembled from the
   *  two labels — the layout allows any nesting under a folder, and
   *  `folder + "/" + name` would quietly lose a level of it. */
  rel: string
  stem: string
  file: string
  /** The shortest name that still points at one file. A volume with one
   *  `k3nan.safetensors` gets `<lora:k3nan:1>`; the matched Wan speed pairs,
   *  whose files are both called `high`, get the folder that tells them apart. */
  token: string
  /** The phrase this LoRA's training bound everything to — from the training
   *  sidecar, or from the catalogue for the Krea style set. Empty when nobody
   *  knows one. */
  trigger: string
  /** The strength the server says this LoRA works at, when it says one. */
  strength: number | null
}

/**
 * Everything a `<lora:…>` token can name.
 *
 * Derived from `/api/state` rather than held separately, so a freshly trained
 * LoRA is typeable without a reload. Memoised on the array identity because
 * every keystroke in the prompt re-parses its tokens and each one resolves
 * through this.
 */
let cacheKey: LoraEntry[] | null = null
let cacheVal: LoraFile[] = []

export function loraIndex(state: AppState | null): LoraFile[] {
  const rows = state?.loras ?? []
  if (rows === cacheKey) return cacheVal
  const flat = rows.flatMap((l) =>
    l.files.map((f) => {
      const path = String(f.path ?? '')
      const rel = (path.split('/loras/').pop() ?? path).replace(/\.safetensors$/i, '')
      return {
        path, rel,
        stem: rel.split('/').pop() ?? rel,
        file: path.split('/').pop() ?? path,
        trigger: l.trigger_word ?? '',
        strength: l.strength ?? null,
      }
    }),
  )
  cacheKey = rows
  cacheVal = flat.map((l) => ({
    ...l,
    token: flat.filter((x) => x.stem === l.stem).length === 1 ? l.stem : l.rel,
  }))
  return cacheVal
}

/**
 * By path under `loras/` first, then by bare filename while that is unambiguous.
 *
 * **Case is part of a filename, not noise.** An earlier version folded it away
 * before comparing, so `K3nan.safetensors` and `k3nan.safetensors` — two real
 * files, two real LoRAs, which is exactly what a volume holds after a Drive pull
 * and a training run disagree about capitalisation — collided into one ambiguous
 * name. The result was not "picked the wrong one", it was worse: neither
 * resolved, so both went untypeable and the note blamed a missing file for a file
 * sitting right there. Nothing in the backend folds case; it addresses LoRAs by
 * exact path and ComfyUI validates them against a directory listing, so this was
 * the only place on the path where two distinct files became one name.
 *
 * Exact first, then case-insensitively and only while that still points at one
 * file: typing lowercase keeps working on every volume that does not hold a
 * collision, and on one that does, the exact spelling always wins.
 */
export function resolveLora(index: LoraFile[], name: string): LoraFile | null {
  const raw = String(name ?? '').trim().replace(/\.safetensors$/i, '')
  if (!raw) return null
  const n = raw.toLowerCase()
  const exactRel = index.filter((l) => l.rel === raw)
  if (exactRel[0]) return exactRel[0]
  const exactStem = index.filter((l) => l.stem === raw)
  if (exactStem.length === 1 && exactStem[0]) return exactStem[0]
  const byRel = index.filter((l) => l.rel.toLowerCase() === n)
  if (byRel.length === 1 && byRel[0]) return byRel[0]
  const byStem = index.filter((l) => l.stem.toLowerCase() === n)
  return byStem.length === 1 ? byStem[0] ?? null : null
}

/** What the encoders must not see. Send path only — see `LORA_RE`. */
export function stripLoras(text: string): string {
  return text.replace(LORA_RE, ' ').replace(/\s+/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim()
}

export const loraNum = (v: string, d: number | null): number | null => {
  const n = parseFloat(v)
  return Number.isFinite(n) ? n : d
}

/** A picked file becomes a chip. The strength is the one the server says this
 *  LoRA works at — a Krea style LoRA at a generic 1 reads as a faint grade over
 *  the picture, which is a LoRA that silently did nothing delivered by the
 *  picker itself. `1.3` in a region, which is the node pack's own guidance for a
 *  character. */
export function chipFrom(l: LoraFile, region = false): LoraChip {
  return {
    path: l.path,
    rel: l.token,
    strength: region ? 1.3 : (l.strength ?? 1),
    textEncoder: null,
    expert: null,
  }
}

/** The matched speed pairs are named `high` and `low` inside one folder, so the
 *  file already says which expert it belongs to. Reading it beats making you
 *  write the same fact twice, and beats the silent quality loss of crossing
 *  them. `chip.expert` is the override; null means read the name. */
export function vidExpert(experts: string[], chip: LoraChip): string {
  if (chip.expert && experts.includes(chip.expert)) return chip.expert
  const n = chip.rel.toLowerCase()
  const read = /(^|\/)high|high_noise/.test(n) ? 'high'
    : /(^|\/)low|low_noise/.test(n) ? 'low' : 'both'
  // Clamped to what this model actually has. The filename read is a heuristic
  // for Wan's matched speed pairs, where `high`/`low` is the whole naming
  // scheme — on a model with one expert it is just a word somebody used, and
  // `high_detail.safetensors` on H3 was being tagged `high` and refused by the
  // route with a message about noise experts that model does not have.
  return experts.includes(read) ? read : 'both'
}

/** The image stack. `text_encoder` is left null when unset on purpose: the
 *  backend defaults it to the UNet weight, so omitting it is a decision the
 *  client does not have to duplicate — and duplicating it would freeze today's
 *  default into every chip ever saved. */
export function readChips(chips: LoraChip[], max: number) {
  return chips.slice(0, max).map((c) => ({
    path: c.path, unet: c.strength, text_encoder: c.textEncoder,
  }))
}

export function readVidChips(chips: LoraChip[], max: number, experts: string[]) {
  return chips.slice(0, max).map((c) => ({
    path: c.path, unet: c.strength, expert: vidExpert(experts, c),
  }))
}

/**
 * A stack off a gallery sidecar, back into chips.
 *
 * The one caller that still starts from a *name* rather than a path, because a
 * sidecar records what ran rather than where it lived. An entry whose file is
 * gone is simply dropped — a chip is picked from a list and always resolves, so
 * there is no such thing as a chip naming nothing.
 */
export function loraChips(
  index: LoraFile[],
  list: { name?: string; unet?: number; expert?: string; text_encoder?: number | null }[] | undefined,
  video: boolean,
): LoraChip[] {
  return (list ?? []).flatMap((l) => {
    // Image records the stem, video the filename; both match the way the two
    // stacks were read back before this.
    const hit = index.find((x) => (video ? x.file === l.name : x.stem === l.name))
      ?? resolveLora(index, l.name ?? '')
    if (!hit) return []
    return [{
      path: hit.path,
      rel: hit.token,
      strength: l.unet ?? 1,
      textEncoder: video ? null : (l.text_encoder ?? null),
      expert: video ? ((l.expert as LoraChip['expert']) ?? null) : null,
    }]
  })
}
