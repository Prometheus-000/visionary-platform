/**
 * LoRAs are written in the prompt, not stacked under it.
 *
 *     a portrait <lora:k3nan:0.4> in soft window light <lora:alxcn:1>
 *
 * Automatic1111's syntax, because it is the notation anyone who has trained these
 * models already types. A row per LoRA cost 56px plus a wrapped select — 380px of
 * canvas for four filenames and eight digits — and it still could not say the
 * thing that matters most, which is *where in the sentence* the LoRA applies.
 *
 * Strength defaults to 1. A second number is the text encoder weight on the image
 * side and the Wan expert on the video side; both are omitted far more often than
 * not.
 */
import type { AppState, LoraEntry } from '../api/types'

/** One capture and a split, rather than three optional groups: the field count
 *  varies by family, and a regex that encodes that is a regex that has to change
 *  every time a family does. */
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
      return { path, rel, stem: rel.split('/').pop() ?? rel, file: path.split('/').pop() ?? path }
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

/**
 * Why a name did not resolve, which is a different question from whether it did.
 *
 * `<lora:high:1>` against a volume holding both Wan speed pairs is not a missing
 * file — it is two files and no way to tell which — and sending you to look for a
 * LoRA that is sitting right there is the worse of the two wrong answers.
 *
 * This one stays case-insensitive, and that is not an oversight left behind by
 * the fix above: it only ever runs on a name that already failed to resolve, so
 * its job is near-misses. A mistyped `<lora:K3NAN:1>` on a volume holding both
 * casings answers "use K3nan or k3nan", which is the message that makes the
 * difference between the two files visible.
 */
export function loraAlternatives(index: LoraFile[], name: string): string[] {
  const n = String(name ?? '').trim().replace(/\.safetensors$/i, '').toLowerCase()
  return index.filter((l) => l.stem.toLowerCase() === n).map((l) => l.token)
}

export type Token = {
  start: number
  end: number
  name: string
  /** The first number: strength. */
  a: string
  /** The second: text encoder weight on the image side, expert on the video. */
  b: string
  hit: LoraFile | null
}

/** Every token in the text, with the offsets ⌘↑/⌘↓ needs to rewrite one in
 *  place. Unresolved names are kept rather than dropped: a LoRA that silently
 *  does nothing is indistinguishable from a LoRA with no effect, which is the
 *  failure this whole file is written to avoid. */
export function parseLoras(index: LoraFile[], text: string): Token[] {
  const out: Token[] = []
  LORA_RE.lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = LORA_RE.exec(text))) {
    const parts = (m[1] ?? '').split(':').map((s) => s.trim())
    out.push({
      start: m.index,
      end: m.index + m[0].length,
      name: parts[0] ?? '',
      a: parts[1] ?? '',
      b: parts[2] ?? '',
      hit: resolveLora(index, parts[0] ?? ''),
    })
  }
  return out
}

/** What the text encoders see. The tokens are markup for this page, not language
 *  — leaving them in would have the model rendering the word "lora". */
export function stripLoras(text: string): string {
  return text.replace(LORA_RE, ' ').replace(/\s+/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim()
}

export const loraNum = (v: string, d: number | null): number | null => {
  const n = parseFloat(v)
  return Number.isFinite(n) ? n : d
}

/** The image stack. `text_encoder` is left null when unwritten on purpose: the
 *  backend defaults it to the UNet weight, so omitting it is a decision the
 *  client does not have to duplicate — and duplicating it would freeze today's
 *  default into every prompt ever saved. */
export function readLoras(index: LoraFile[], text: string, max: number) {
  return parseLoras(index, text)
    .filter((t) => t.hit)
    .slice(0, max)
    .map((t) => ({
      path: t.hit!.path,
      unet: loraNum(t.a, 1),
      text_encoder: t.b === '' ? null : loraNum(t.b, null),
    }))
}

/** The matched speed pairs are named `high` and `low` inside one folder, so the
 *  file already says which expert it belongs to. Reading it beats making you
 *  write the same fact twice, and beats the silent quality loss of crossing
 *  them. */
export function vidExpert(experts: string[], t: Pick<Token, 'b' | 'hit'>): string {
  const b = String(t.b ?? '').toLowerCase()
  if (experts.includes(b)) return b
  const n = (t.hit?.rel ?? '').toLowerCase()
  if (/(^|\/)high|high_noise/.test(n)) return 'high'
  if (/(^|\/)low|low_noise/.test(n)) return 'low'
  return 'both'
}

export function readVidLoras(index: LoraFile[], text: string, max: number, experts: string[]) {
  return parseLoras(index, text)
    .filter((t) => t.hit)
    .slice(0, max)
    .map((t) => ({ path: t.hit!.path, unet: loraNum(t.a, 1), expert: vidExpert(experts, t) }))
}

/**
 * A stack, written back into the prompt — which is now the only place one lives.
 *
 * The canonical form goes in, full path when the bare name is ambiguous, so a
 * reused prompt resolves to the same file the original run used. An entry whose
 * file is gone is simply not written rather than becoming an unresolvable token.
 */
export function loraTokens(
  index: LoraFile[],
  list: { name?: string; unet?: number; expert?: string; text_encoder?: number | null }[] | undefined,
  video: boolean,
  experts: string[] = ['both', 'high', 'low'],
): string {
  return (list ?? [])
    .map((l) => {
      // Image records the stem, video the filename; both match the way the two
      // stacks were read back before this.
      const hit = index.find((x) => (video ? x.file === l.name : x.stem === l.name))
        ?? resolveLora(index, l.name ?? '')
      if (!hit) return ''
      const tail = video
        ? (l.expert && l.expert !== 'both' && l.expert !== vidExpert(experts, { b: '', hit })
            ? `:${l.expert}` : '')
        : (l.text_encoder != null && l.text_encoder !== l.unet ? `:${l.text_encoder}` : '')
      return `<lora:${hit.token}:${l.unet ?? 1}${tail}>`
    })
    .filter(Boolean)
    .join(' ')
}

/* ---- the region inspector's Strength cell ------------------------------ */
/* A *view* of the first number in the box's token, not a second place the number
 * lives — the same relationship the ratio picker has to the width and height
 * boxes. Blank when there is no token to read, because a strength with nothing
 * to apply to is a number that lies. */

export function tokenStrength(index: LoraFile[], text: string): number | null {
  const t = parseLoras(index, text ?? '').find((x) => x.hit)
  return t ? loraNum(t.a, 1) : null
}

export function setTokenStrength(index: LoraFile[], text: string, val: number): string {
  const t = parseLoras(index, text ?? '').find((x) => x.hit)
  if (!t) return text
  const parts = [t.name, String(val)]
  if (t.b !== '') parts.push(t.b)
  return `${text.slice(0, t.start)}<lora:${parts.join(':')}>${text.slice(t.end)}`
}

/* ---- writing one into a field ------------------------------------------ */

export type Written = { value: string; caret: number; select: number }

/**
 * Insert a token at the caret, or take it back out.
 *
 * Picking a LoRA that is already in the prompt removes it rather than writing a
 * second copy: two tokens for one file is not a stack — `apply_stack()` patches
 * both onto the clone chain, so the strengths compound into a number nobody chose
 * and the picker looks like it silently did nothing. There is no prompt that
 * wants the duplicate, so the second pick is free to mean the other thing.
 *
 * No trigger word is written. The picker used to prepend one, and the reason it
 * no longer does is that the right destination stopped being answerable: on the
 * regional path the node encodes the main prompt and nothing else, so a trigger
 * auto-written into a box would silently never reach the encoder, and one written
 * into the main prompt instead would be the picker editing a sentence the caret
 * was nowhere near. Triggers are typed.
 */
export function insertLora(
  index: LoraFile[],
  value: string,
  caret: number,
  l: LoraFile,
  /** 1.3 in a region, 1 everywhere else. The node pack's guidance for a
   *  character LoRA is 1.3–1.4, and a picker that writes the known-weak value
   *  into the one field where it is known to be weak is doing the wrong thing
   *  quietly. The main prompt is a style stack far more often than a character. */
  strength: '1' | '1.3',
): Written {
  const present = parseLoras(index, value).filter((t) => t.hit?.path === l.path)
  if (present.length) return removeLora(value, caret, present)

  let at = Math.min(caret, value.length)
  // Never inside another token. A pick deliberately leaves the caret on its own
  // strength so ⌘↑ can walk it, and clicking + LoRA again does not move it — so
  // two picks in a row wrote `<lora:alxcn: <lora:my_style:1> 1>`. Nothing on the
  // page says that is malformed; it silently loads one LoRA where you asked for
  // two, which is the failure this whole module exists to avoid.
  const inside = parseLoras(index, value).find((t) => at > t.start && at < t.end)
  if (inside) at = inside.end

  // Never welded to the neighbouring word: a token glued to the end of a sentence
  // survives the strip but reads as a typo while you are writing.
  const before = value.slice(0, at)
  const after = value.slice(at)
  const tok = (before && !/\s$/.test(before) ? ' ' : '')
    + `<lora:${l.token}:${strength}>`
    + (after && !/^\s/.test(after) ? ' ' : '')
  // Caret onto the strength, selected, so the next thing you can do is ⌘↑ it or
  // type over it.
  const cur = before.length + tok.indexOf(`:${strength}>`) + 1
  return { value: before + tok + after, caret: cur, select: cur + strength.length }
}

/** Takes the tokens rather than the index, because its caller has already resolved them:
 *  what comes out is decided by the offsets, not by whether the names still name files. */
export function removeLora(value: string, caret: number, toks: Token[]): Written {
  let v = value
  // Back to front, so each splice leaves the earlier offsets valid.
  for (const t of [...toks].sort((a, b) => b.start - a.start)) {
    let s = t.start
    let e = t.end
    // Take the space the token was welded to as well, or removing it leaves a
    // double gap mid-sentence. `[^\S\n]` and not `\s`: a prompt written across
    // several lines should not have its line breaks eaten by a picker click.
    if (/[^\S\n]/.test(v[e] ?? '')) e++
    else if (/[^\S\n]/.test(v[s - 1] ?? '')) s--
    v = v.slice(0, s) + v.slice(e)
  }
  // Only the token. The trigger used to be stripped here as well, on the grounds
  // that this function had put it there — nothing puts it there now, so every
  // trigger in the field is something you typed, and a picker click that deletes
  // a word out of your sentence is the worse surprise.
  const at = Math.min(caret, v.length)
  return { value: v, caret: at, select: at }
}

/**
 * ⌘↑ / ⌘↓ on a strength, the way the same chord nudges attention weights in
 * Automatic. Put the caret anywhere inside the brackets — the token under it is
 * the one that moves, and the selection comes back so a run of presses walks the
 * value instead of moving once and losing its place.
 *
 * Clamped, not unbounded: past ±2 a LoRA stops styling the image and starts
 * destroying it, and the clamp is the cheapest place to say so.
 */
export function nudgeLora(
  value: string,
  caret: number,
  delta: number,
): Written | null {
  // Its own regex pass rather than the resolved index: the token under the caret is the one
  // that moves whether or not it names a file on the volume, and a strength you cannot walk
  // on a mistyped name is a strength you cannot fix.
  const t = parseLoras([], value).find((x) => caret >= x.start && caret <= x.end)
  if (!t) return null
  const next = Math.max(-2, Math.min(2, Math.round(((loraNum(t.a, 1) ?? 1) + delta) * 100) / 100))
  const body = [t.name, String(next)].concat(t.b ? [t.b] : []).join(':')
  const tok = `<lora:${body}>`
  const at = t.start + tok.indexOf(`:${next}`) + 1
  return {
    value: value.slice(0, t.start) + tok + value.slice(t.end),
    caret: at,
    select: at + String(next).length,
  }
}
