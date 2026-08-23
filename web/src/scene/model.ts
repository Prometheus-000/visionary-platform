import type { ShotPill } from '../api/types'

/**
 * The scene's data model, mirroring `_validate_scene` / `_compile_h3_scene`.
 *
 * A mirror on purpose, and the duplication is the point of failure to watch:
 * `/api/compile` is the authority and this exists so the page can hold the
 * shape without a round trip. If the two ever diverge, this one is wrong — the
 * same rule `storyline/model.ts` already records.
 *
 * What it does *not* do is compile. There is no client-side document builder
 * here and there must never be one: a preview with its own implementation is a
 * preview that can disagree with the run.
 */

// ── the pool ────────────────────────────────────────────────────────────────

export type Media = 'image' | 'video' | 'audio'

/**
 * One dropped file, held once however many buckets point at it.
 *
 * Keyed by **content**, not by name: a picture dragged in from two folders is
 * one picture, and a name is the one property two copies of a file are least
 * likely to share. `b64` is what the route takes; `url` is an object URL for
 * the thumbnail and is revoked when the entry goes.
 */
export type PoolFile = {
  id: string
  name: string
  kind: Media
  b64: string
  url: string
}

// ── the cast ────────────────────────────────────────────────────────────────

export type CastKind = 'character' | 'place' | 'thing'

/**
 * Slots per kind, and the media each one takes.
 *
 * This table *is* the file validation — a file over a slot that cannot take it
 * simply does not highlight, so the rejection is the absence of an invitation
 * rather than a toast after the fact. It mirrors `H3_CAST_KINDS` and
 * `H3_SLOT_MEDIA`, and the server asserts it again because a stale tab is a
 * client that can send anything.
 */
export const SLOTS: Record<CastKind, { key: string; label: string; takes: Media }[]> = {
  character: [
    { key: 'face', label: 'Face', takes: 'image' },
    { key: 'wardrobe', label: 'Wardrobe', takes: 'image' },
    { key: 'body', label: 'Body', takes: 'image' },
    { key: 'voice', label: 'Voice', takes: 'audio' },
    { key: 'motion', label: 'Motion', takes: 'video' },
  ],
  place: [
    { key: 'establishing', label: 'Establishing', takes: 'image' },
    { key: 'style', label: 'Style', takes: 'image' },
  ],
  thing: [
    { key: 'object', label: 'Object', takes: 'image' },
    { key: 'style', label: 'Style', takes: 'image' },
  ],
}

/** A pointer plus the roles it plays. One entry per file per bucket: a second
 *  slot on the same file adds a role rather than a second entry, which is what
 *  makes "this photo is both the outfit and the body" one upload. */
export type CastRef = { fileId: string; slots: string[] }

export type CastMember = {
  id: string
  kind: CastKind
  /** The @handle, and the only text anyone types about them. */
  name: string
  note: string
  refs: CastRef[]
}

// ── the timeline ────────────────────────────────────────────────────────────

export type Say = {
  who: string[]
  text: string
  lang: string
  voice: string
  carry: boolean
  cutoff: boolean
  offscreen: boolean
}

export type Shot = {
  id: string
  /** The sentence, with mentions as literal `@handle` text — see `mentions`. */
  line: string
  /** Duration weight. Seconds fall out of the share, so two rows cannot be out
   *  of order and a cut cannot land outside the clip. */
  beats: number
  pills: ShotPill[]
  say: Say
}

export type Scene = {
  cast: CastMember[]
  shots: Shot[]
  style: string
  grade: string
}

let seq = 0
const uid = (p: string) => `${p}${++seq}`

export const newShot = (line = ''): Shot => ({
  id: uid('s'), line, beats: 1, pills: [],
  say: { who: [], text: '', lang: 'English', voice: '',
         carry: false, cutoff: false, offscreen: false },
})

export const newMember = (kind: CastKind): CastMember =>
  ({ id: uid('c'), kind, name: '', note: '', refs: [] })

export const emptyScene = (): Scene =>
  ({ cast: [], shots: [newShot()], style: '', grade: '' })

// ── handles ─────────────────────────────────────────────────────────────────

/** A name reduced to the characters a mention can be written in. */
export const handleOf = (name: string) =>
  name.toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '')

const MENTION = /@([a-z0-9_]+)/gi

/** The handles a line mentions, in order, once each. */
export function mentions(line: string): string[] {
  const out: string[] = []
  for (const m of line.matchAll(MENTION)) {
    const h = m[1]?.toLowerCase()
    if (h && !out.includes(h)) out.push(h)
  }
  return out
}

/**
 * Renaming rewrites the handle across every row, as a visible find-and-replace.
 *
 * The alternative is storing `@{id}` and showing the person a string they
 * cannot safely edit. This way the mention is the literal text, which has one
 * consequence worth stating rather than discovering: **edit the handle and it
 * stops being a mention.** That is `remap`'s rule one layer up — the user's
 * text is the record.
 */
export const rename = (line: string, from: string, to: string) =>
  from && to && from !== to
    ? line.replace(new RegExp(`@${from}\\b`, 'gi'), `@${to}`)
    : line

// ── derivations ─────────────────────────────────────────────────────────────

export const named = (cast: CastMember[]) => cast.filter((c) => handleOf(c.name))

/**
 * The pool ids that travel, per category, in the order the server will number
 * them.
 *
 * **This is the `<Picture N>` contract.** The label is the file's position in
 * `references[]`, not its order of first appearance in the cast, so whatever
 * this returns is exactly what has to be uploaded, in this order. Number one
 * way here and upload another and every label is well-formed and points at
 * somebody else's face.
 */
export function assets(cast: CastMember[], kind: Media, pool: Record<string, PoolFile>) {
  const out: PoolFile[] = []
  for (const c of cast) {
    for (const r of c.refs) {
      const f = pool[r.fileId]
      if (f && f.kind === kind && !out.some((x) => x.id === f.id)) out.push(f)
    }
  }
  return out
}

/** Seconds each shot occupies, as [start, end], from the beat shares. */
export function times(shots: Shot[], seconds: number): [number, number][] {
  const total = shots.reduce((n, s) => n + s.beats, 0) || 1
  let t = 0
  return shots.map((s) => {
    const at = t
    t += (s.beats / total) * seconds
    return [at, t]
  })
}

export const clock = (t: number) => {
  const m = Math.floor(t / 60)
  return `${String(m).padStart(2, '0')}:${(t - m * 60).toFixed(3).padStart(6, '0')}`
}

/**
 * Is there anything here the compiler would do something with?
 *
 * The degrade, and it has to be exact: one shot, no cast, no pills and the run
 * is the typed prompt byte-for-byte. A scene that is merely *open* changes
 * nothing — the same contract `_compile_h3_prompt` keeps.
 */
export const live = (sc: Scene) =>
  sc.cast.length > 0 || sc.shots.length > 1
  || sc.shots.some((s) => s.pills.length > 0 || s.say.text.trim() !== '')

/**
 * The payload, with pool ids resolved to the positional indices the routes take.
 *
 * Returns null when the scene is not live, which is what keeps this additive:
 * a client that never opens the composer sends no `scene` key at all and the
 * backend takes its existing path.
 */
export function readScene(
  sc: Scene,
  pool: Record<string, PoolFile>,
): { scene: unknown; references: string[]; ref_videos: string[]; ref_audios: string[] } | null {
  if (!live(sc)) return null
  const order: Record<Media, PoolFile[]> = {
    image: assets(sc.cast, 'image', pool),
    video: assets(sc.cast, 'video', pool),
    audio: assets(sc.cast, 'audio', pool),
  }
  const cast = named(sc.cast).map((c) => ({
    id: c.id,
    kind: c.kind,
    name: c.name,
    note: c.note,
    refs: c.refs.flatMap((r) => {
      const f = pool[r.fileId]
      if (!f) return []
      const index = order[f.kind].findIndex((x) => x.id === f.id)
      return index < 0 ? [] : [{ kind: f.kind, index, slots: r.slots }]
    }),
  }))
  return {
    scene: {
      style: sc.style,
      grade: sc.grade,
      cast,
      shots: sc.shots.map((s) => ({
        line: s.line,
        beats: s.beats,
        pills: s.pills.map((p) => ({ key: p.key, ...(p.value !== undefined && { value: p.value }), ...(p.lang && { lang: p.lang }) })),
        say: {
          who: s.say.who, text: s.say.text, lang: s.say.lang,
          voice: s.say.voice, carry: s.say.carry,
          cutoff: s.say.cutoff, offscreen: s.say.offscreen,
        },
      })),
    },
    references: order.image.map((f) => f.b64),
    ref_videos: order.video.map((f) => f.b64),
    ref_audios: order.audio.map((f) => f.b64),
  }
}

