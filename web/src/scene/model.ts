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

/**
 * What a reference *video* is doing, and what an `<Audio N>` is doing.
 *
 * Mirrors `H3_VIDEO_ROLES` and `H3_AUDIO_ROLES`, and it is the one thing that
 * can promote a scene's task type past `reference generation` — the guide is
 * explicit that the mere presence of a video does not create a type. An image
 * has no role here: a picture attached to a slot is already told what it is by
 * the slot it landed on.
 */
export const VIDEO_ROLES = ['reference', 'edit', 'continue'] as const
export const AUDIO_ROLES = ['reference', 'reuse'] as const

/** A pointer plus the roles it plays. One entry per file per bucket: a second
 *  slot on the same file adds a role rather than a second entry, which is what
 *  makes "this photo is both the outfit and the body" one upload. */
export type CastRef = {
  fileId: string
  slots: string[]
  /** Only read for video and audio — see `VIDEO_ROLES`. Left at `reference`,
   *  which is what both tables default to and what the server assumes. */
  role?: string
}

/**
 * The guide's relationship markers for visual content — `H3_RETENTION`.
 *
 * Not the audio list: `fully_copy` is meaningless about a photograph and
 * `fully_preserved` is meaningless about a signal, which is why the server
 * keeps two tables rather than one.
 */
export const RETENTION = ['fully_preserved', 'partially_preserved',
                          'attribute_transfer', 'weak_reference'] as const

/**
 * The same four in words, for the control that sets them.
 *
 * The tokens are a *fixed English value in the output format* — ref-en §4.1 says
 * so in those words — and they go into the document verbatim. They are also
 * jargon off a model card, and the whole thesis here is that nobody should have
 * to learn a text encoder: the prompt is a compilation target, so the control
 * reads in words and the compiler emits the token.
 */
export const RETENTION_LABEL: Record<string, string> = {
  fully_preserved: 'exactly as shown',
  partially_preserved: 'closely',
  attribute_transfer: 'attributes only',
  weak_reference: 'loosely',
}

export type CastMember = {
  id: string
  kind: CastKind
  /** The @handle, and the only text anyone types about them. */
  name: string
  note: string
  /** How much of the reference has to survive into the render. Defaults to the
   *  strictest, because the common case for a face is "this exact person" and a
   *  looser default is a likeness quietly allowed to drift. */
  retention: string
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
  /**
   * Duration weight, and **null is the normal state**: the share is derived from
   * how much you wrote about the shot — see `shares`. Seconds fall out of the
   * share, so two rows cannot be out of order and a cut cannot land outside the
   * clip.
   *
   * A number here is the precision escape hatch, the way a region's four
   * coordinates are: dragging teaches the proportion and the proportion never
   * taught the dragging.
   */
  beats: number | null
  pills: ShotPill[]
  say: Say
}

/**
 * The three clip-level sources — `H3_SOURCES`.
 *
 * Not cast references, which is why they need their own home: a video you are
 * continuing from is not a subject's likeness, it is a property of the clip.
 * A keyframe points into the images, `continue` and `edit` into the videos, and
 * the last two are mutually exclusive because they are different task types and
 * different documents.
 */
export type SourceKind = 'keyframe' | 'continue' | 'edit'
export type Sources = Partial<Record<SourceKind, string[]>>

export const SOURCE_TAKES: Record<SourceKind, Media> =
  { keyframe: 'image', continue: 'video', edit: 'video' }

export type Scene = {
  cast: CastMember[]
  shots: Shot[]
  sources: Sources
  style: string
  grade: string
}

let seq = 0
const uid = (p: string) => `${p}${++seq}`

export const newShot = (line = ''): Shot => ({
  id: uid('s'), line, beats: null, pills: [],
  say: { who: [], text: '', lang: 'English', voice: '',
         carry: false, cutoff: false, offscreen: false },
})

export const newMember = (kind: CastKind): CastMember =>
  ({ id: uid('c'), kind, name: '', note: '', retention: RETENTION[0], refs: [] })

export const emptyScene = (): Scene =>
  ({ cast: [], shots: [newShot()], sources: {}, style: '', grade: '' })

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
 *
 * The cast first and the clip-level sources after it, and only files something
 * still points at. A member you have started and not yet named is not sent —
 * `named` drops it — so uploading its photograph would be megabytes on the wire
 * for a picture no label mentions.
 */
export function assets(sc: Scene, kind: Media, pool: Record<string, PoolFile>) {
  const out: PoolFile[] = []
  const take = (id: string) => {
    const f = pool[id]
    if (f && f.kind === kind && !out.some((x) => x.id === f.id)) out.push(f)
  }
  for (const c of named(sc.cast)) for (const r of c.refs) take(r.fileId)
  for (const k of Object.keys(sc.sources) as SourceKind[]) {
    for (const id of sc.sources[k] ?? []) take(id)
  }
  return out
}

/**
 * The shortest thing that still reads as a shot, in characters.
 *
 * A floor rather than a proportion, because the failure it prevents is at zero:
 * a row you have just added has nothing written in it, and without this its
 * share is zero — a tick with no width on the strip, and a cut time identical to
 * its neighbour's, which is the one thing `_h3_shot_text` refuses to compile.
 */
const MIN_EXTENT = 20

/**
 * How the clip divides, by shot.
 *
 * **The share is a readout, not an input.** A shot's slice of the clip is the
 * length of what you wrote about it — write more about shot 1 and it gets more
 * of the clip — which removes a 9px drag target rather than enlarging it to a
 * thumb-sized one. `beats` overrides it where somebody has dragged, and that is
 * the escape hatch a region's four coordinates are.
 *
 * Dialogue counts toward the extent because a line of it *is* the shot: a row
 * whose whole content is somebody speaking would otherwise sit at the floor
 * while the shot beside it, describing the same beat in prose, took the clip.
 */
export function shares(shots: Shot[]): number[] {
  return shots.map((s) => s.beats ?? Math.max(
    MIN_EXTENT, s.line.trim().length + s.say.text.trim().length))
}

/**
 * What the person actually typed, as one string.
 *
 * The sidecar's `prompt_typed`, and the guard on whether there is anything to
 * render. The compiled document is what runs and this is what you meant — the
 * durable half, which is why Reuse, Copy and the metadata sheet all prefer it.
 *
 * Newline-joined rather than space-joined: the rows *are* separate sentences,
 * and a gallery card showing them run together would be showing a paragraph
 * nobody wrote. With one shot it is that shot's line and nothing else, which is
 * the string the prompt box used to hold.
 */
export const typedProse = (sc: Scene) =>
  sc.shots.map((s) => s.line.trim()).filter(Boolean).join('\n')

/** Seconds each shot occupies, as [start, end], from the shares. */
export function times(shots: Shot[], seconds: number): [number, number][] {
  const w = shares(shots)
  const total = w.reduce((n, x) => n + x, 0) || 1
  let t = 0
  return shots.map((_, i) => {
    const at = t
    t += (w[i]! / total) * seconds
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
 *
 * Style and grade are in it because they are not decoration: with either one set
 * the document opens on a sentence declaring it, which is a different input to
 * the encoder than the bare prose. Empty means "the compiler's own default",
 * which is what `_compile_h3_scene` falls back to, so leaving them alone is
 * still the degrade.
 */
export const live = (sc: Scene) =>
  named(sc.cast).length > 0 || sc.shots.length > 1
  || sc.style.trim() !== '' || sc.grade.trim() !== ''
  || Object.values(sc.sources).some((v) => v.length > 0)
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
    image: assets(sc, 'image', pool),
    video: assets(sc, 'video', pool),
    audio: assets(sc, 'audio', pool),
  }
  const at = (id: string, kind: Media) =>
    order[kind].findIndex((x) => x.id === id)
  const cast = named(sc.cast).map((c) => ({
    id: c.id,
    kind: c.kind,
    name: c.name,
    note: c.note,
    retention: c.retention,
    refs: c.refs.flatMap((r) => {
      const f = pool[r.fileId]
      if (!f) return []
      const index = at(f.id, f.kind)
      return index < 0 ? [] : [{ kind: f.kind, index, slots: r.slots,
                                 role: r.role ?? 'reference' }]
    }),
  }))
  const w = shares(sc.shots)
  const sources: Partial<Record<SourceKind, number[]>> = {}
  for (const k of Object.keys(sc.sources) as SourceKind[]) {
    const idx = (sc.sources[k] ?? [])
      .map((id) => at(id, SOURCE_TAKES[k]))
      .filter((i) => i >= 0)
    if (idx.length) sources[k] = idx
  }
  return {
    scene: {
      style: sc.style,
      grade: sc.grade,
      sources,
      cast,
      shots: sc.shots.map((s, i) => ({
        line: s.line,
        beats: w[i],
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
