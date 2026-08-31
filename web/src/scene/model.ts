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
  /**
   * The Arsenal file this came off, when it came off one.
   *
   * **A recalled character is applied, not imported** — the roadmap's veto in
   * its own words, and the half that was missing. Recall copied the bytes and
   * stopped, so re-shooting `@maya`'s reference left every scene that already
   * had her holding the old picture forever. Harmless while a scene died with
   * the tab; `keep.ts` made a scene outlive the library it was copied from,
   * which is the first moment the two could disagree.
   *
   * Absent for a dropped file, which has no library behind it and is nobody's
   * to update. See `refreshArsenal`.
   */
  from?: { handle: string; file: string }
}

// ── the cast ────────────────────────────────────────────────────────────────

/**
 * There is one kind, because the guide has one label.
 *
 * `<Subject N>` covers "People, animals, or objects / Scenes, backgrounds, or
 * environments / Clothing, props, interfaces, or visual effects / Styles,
 * actions, expressions, or poses" (ref-en §2.1) — four *examples* of what can be
 * a subject, not four types of one. character/place/thing was ours, and it was
 * the closed-vocabulary failure again: three kinds means three sayable things,
 * and a style, a pose or an expression is a legal subject with no box to go in.
 *
 * The type survives as a single member so the payload keeps its shape for a
 * server that still reads the field.
 */
export type CastKind = 'subject'

/**
 * What a subject can be given, by channel.
 *
 * A slot is the media and nothing else. `face`, `wardrobe` and `body` were three
 * ways of saying "a picture of them", and what a given picture is *for* is a
 * sentence now — see `CastRef.note` and the guide's own construction:
 *
 *     <Subject 1> is the woman whose appearance comes from <Picture 1> and
 *     whose walking motion comes from <Video 1>.
 */
export const TAKES: Media[] = ['image', 'audio', 'video']

export const slotFor = (_kind: CastKind, media: Media): string | null =>
  (TAKES.includes(media) ? media : null)

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
  /**
   * What *this* asset provides, in the person's own words.
   *
   * The guide's construction, and the reason a role name is not needed:
   *
   *     <Subject 1> is the woman whose appearance comes from <Picture 1> and
   *     whose walking motion comes from <Video 1>.
   *
   * Empty is the normal state — one photograph of somebody needs no gloss, and
   * `member.note` already says what the subject *is*. This says what a second
   * or third picture is *for*, which is the thing five labelled slots were
   * trying to express with a fixed vocabulary of five.
   */
  note?: string
  /**
   * This picture is a labeled character reference sheet.
   *
   * A *typed* reference rather than a note, because the division of labour
   * turns on it: templated instruction text — the sheet citation, the audio
   * lines, the alignment sentences, the retention grammar — is the compiler's
   * to write, and everything carrying intent is the person's, verbatim. The
   * mark is the one bit that says which sentence this picture gets. Cast sets
   * it automatically (the app made the sheet, so provenance is certain); a
   * sheet arriving as a file is marked with one click on its row.
   */
  sheet?: boolean
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

/** What a new shot runs for until somebody drags it. Four seconds is a beat you
 *  can see rather than a placeholder, and it is well inside one generation. */
export const SHOT_SECONDS = 4

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

/**
 * Push the counter past ids that came from somewhere other than this counter.
 *
 * The restore path is the only caller and the reason is exact: the sequence is
 * module-level, so it restarts at zero on a reload while the shots and cast
 * coming back out of storage still carry `s1`, `c1`. The next row added was a
 * second row with a live row's id — `patchShot` writes to both of them and React
 * keys them as one element. See `keep.ts`.
 */
export const seedShotIds = (ids: string[]) => {
  for (const id of ids) {
    const n = Number(id.slice(1))
    if (Number.isFinite(n) && n > seq) seq = n
  }
}

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
 * The photograph a member is drawn as, if they have one.
 *
 * The rail used to carry a 5px dot, filled or hollow, as the one fact it could
 * hold about somebody — a person reduced to a monogram on a token, in a product
 * whose strongest lever is that it can be handed a face. The dot survives for the
 * case it was always right about: a member with a description and no picture,
 * whom `_h3_label` compiles to prose rather than to a `<Subject N>`.
 */
export const faceOf = (c: CastMember, pool: Record<string, PoolFile>) => {
  for (const r of c.refs) {
    const f = pool[r.fileId]
    if (f && f.kind === 'image') return f
  }
  return null
}

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

/**
 * How long each shot runs, in seconds.
 *
 * **Authored, not derived — and this reverses a rule that shipped.** A shot's
 * slice of the clip used to be the length of what you wrote about it, on the
 * argument that extent drives prominence and it removes a 9px drag target. The
 * owner's objection retires it, and it is not a matter of taste:
 *
 * > Time is not derived from the description because doing so is impossible. If
 * > I'm a director, maybe I want a 5 minute scene where the protagonist sits in
 * > a chair. The director decides, not the model.
 *
 * Nothing about "he sits in the chair" implies five seconds or five minutes, so
 * a readout there invented the one number a director is least likely to be
 * delegating. The 9px drag target the old rule was avoiding is answered by
 * giving time its own axis instead — see `Timeline`, where a shot is a bar you
 * pull rather than a hairline between two rows.
 *
 * `beats` keeps its name and its place in the payload because the server
 * arithmetic already lands: `_compile_h3_scene` spans each shot at
 * `beats / total * seconds`, so with `seconds` sent as the sum of the bars,
 * every span comes out as exactly the number that was dragged.
 */
export function shares(shots: Shot[]): number[] {
  return shots.map((s) => s.beats ?? SHOT_SECONDS)
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

/**
 * How long the whole scene runs — the sum of the bars.
 *
 * **The timeline is the clip's length, not the duration menu.** Once a shot's
 * seconds are dragged rather than derived, a separate control saying how long
 * the clip is is a second authority over the same number, and the two disagree
 * the moment anybody touches either. So the menu keeps the one job it can still
 * hold honestly — still or motion — and the total comes from the track.
 *
 * Null when the scene is not live, which is what preserves the degrade: one
 * shot, nothing else chosen, and the run is the menu's length and the typed
 * sentence, exactly as before there was a timeline.
 */
export const sceneSeconds = (sc: Scene): number | null =>
  sc.shots.length > 1 || sc.shots.some((x) => x.beats != null)
    ? shares(sc.shots).reduce((n, x) => n + x, 0)
    : null

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
                                 ...(r.note ? { note: r.note } : {}),
                                 ...(r.sheet ? { sheet: true } : {}),
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
