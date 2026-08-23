/**
 * The app state, in one store.
 *
 * The vanilla page kept 27 module-level `let`s *and* used the DOM as the rest of
 * its state — `#g-aspect`'s value was the aspect, `#g-regional.on` was the mode,
 * `dataset.touched` was "you chose this". Both are rehomed here. That is the
 * only structural change the port makes, because every one of those values
 * exists for a reason recorded elsewhere.
 *
 * **What deliberately did not come across.** Values that change at pointer rate
 * or are handles on a DOM node stay out — `growing` (a reentrancy latch inside
 * `autoGrow`), `dragEnd`, `menuEl`, `promptCaret`. The rule the split follows:
 * **if a value changes at pointer rate, or is a handle on a DOM node, it is a
 * ref.**
 *
 * A region drag is the one place that rule had to be argued rather than applied.
 * The boxes *are* in the store, and a drag writes them, because the inspector's
 * four numbers are a readout that moves while you drag — that is the whole reason
 * they survived the row they used to live in, so dropping the writes until
 * `pointerup` would delete the feature to save the frames. What the drag does
 * instead is coalesce: one store write per animation frame, however many
 * `pointermove` events arrive inside it. See `regions/RegionLayer.tsx`.
 *
 * **An empty string means "the model decides", and that replaces
 * `data-touched`.** The vanilla page stamped an attribute on a field the moment
 * you typed in it, so that rebuilding the video composer for a new checkpoint
 * would not hand your sampler back to a default. Here a control holds `''` until
 * you set it, and `console/resolve.ts` reads the model's own default through it.
 * Same rule, flag deleted: a value you chose is yours and survives a model
 * change, a value that was only the previous model's default is not — and it
 * survives *only if the new model offers it*, which is what stopped
 * res_multistep quietly overriding the euler that Wan's own templates use.
 */
import { create } from 'zustand'

import type { AppState, ShotGroup, ShotItem, ShotPill, VideoModel } from './api/types'
import type { LoraChip } from './lora/tokens'

export type Kind = 'image' | 'video'
export type Mode = 'generate' | 'train'

/** What the region layer draws. See `Store.edit` for what each one means and why
 *  editing a box's contents and redrawing the box are two states rather than one. */
export type EditMode = 'off' | 'content' | 'geometry'

/**
 * A region is a rectangle in 0..1 of the frame, plus what it is told to be. Its
 * prompt takes the same `<lora:…>` syntax as the main field, which is the whole
 * reason a region has no LoRA select of its own.
 *
 * `w`/`h` rather than the wire's `width`/`height`: the rename happens once, in
 * `readRegions`. What is *not* negotiable is the order — `_pair_boxes` matches
 * box i to row i by original index, so the boxes and the rows have to come out
 * of one list in one order or every face lands in the wrong rectangle with no
 * error anywhere. `id` exists only so React can key a list that reorders.
 */
export type Region = {
  id: string
  x: number
  y: number
  w: number
  h: number
  prompt: string
  /** One per box — the node's shape. A dropdown on the card rather than a token
   *  in the field: the field is for words this performer's description needs,
   *  and which box a LoRA belongs to is said by which card you are looking at. */
  lora: LoraChip | null
  /** Carried as a bool in the job record, never the bytes — it is polled. */
  attachments: Attachment[]
}

let regionSeq = 0
export const newRegion = (r: Partial<Region> = {}): Region => ({
  id: `r${++regionSeq}`,
  x: 0, y: 0, w: 0.5, h: 1, prompt: '', lora: null, attachments: [],
  ...r,
})

/**
 * What a picture given to a place is *for*.
 *
 * Three roles, because three are wired. A photograph in a box is that character's
 * likeness — V9's `regions_json.ref_image`, a latent mold that pulls the rectangle
 * toward that face during sampling. The two frame-scope plates are the scene the
 * picture is generated inside and an outfit transferred onto the subjects. They reach
 * the backend under three different names and they are one thing: a picture, and what
 * it is for. That is the whole reason this type exists rather than a `ref` on a box
 * and a `{scene, outfit}` record somewhere else — two spellings of one idea are two
 * drop handlers, two inspectors, and two places to add the next one.
 *
 * It is also the axis the next capability arrives on. A ControlNet is a picture with a
 * structural role: `depth`, `pose`, `edges`. The V12 node already returns
 * (MODEL, +COND, −COND), so a `ControlNetApplyAdvanced` slots between it and the
 * sampler without touching anything above it — which means adding one should be a
 * catalogue entry, a preprocessor on a CPU container, a branch in the graph builder,
 * and a role here. Dropped on the frame it is frame-wide; dropped on a box it is
 * masked; and neither needs a control that does not already exist. If it ever wants a
 * panel, this type was the wrong shape and the panel is the tell.
 */
export type Role = 'identity' | 'scene' | 'outfit'

/** One picture, and what it is for. The bytes are base64 with no `data:` prefix,
 *  because that is what the wire takes and converting at the edges twice is how a
 *  prefix ends up inside a payload. */
export type Attachment = { role: Role; image: string }

/** Everything a picture can be given to. The frame is one of these, which is what stops
 *  "a photo on a box" and "a photo on the canvas" from being two systems. */
export type Placed = { attachments: Attachment[] }

/** At most one per role per place: a box takes one likeness, the frame takes one scene
 *  and one outfit. That is the node's shape rather than a simplification of it. */
export const attached = (p: Placed | undefined, role: Role): string | null =>
  p?.attachments.find((a) => a.role === role)?.image ?? null

/** Replaces by role rather than appending, and removes the entry on null rather than
 *  storing an empty one — `attached` returning `''` and returning `null` would be two
 *  spellings of "no picture" and the wire only has the one. */
export const setAttached = (
  list: Attachment[],
  role: Role,
  image: string | null,
): Attachment[] => {
  const rest = list.filter((a) => a.role !== role)
  return image ? [...rest, { role, image }] : rest
}

/** Every value an image run is priced by. `''` is "the checkpoint decides", and
 *  is not the same as a zero: the route's `num()` falls back to its default on
 *  `''` and would take a 0 literally. */
export type ImageComposer = {
  model: string
  /** A trained bucket (`"1152x896"`) or `"custom"` — see console/size.ts for why
   *  a bucket is multiplied rather than recomputed from its ratio. */
  aspect: string
  scale: string
  width: number
  height: number
  sampler: string
  scheduler: string
  steps: string
  cfg: string
  shift: string
  seed: string
  n: number
}

export type VideoComposer = {
  model: string
  aspect: string
  tier: string
  seconds: string
  sampler: string
  scheduler: string
  steps: string
  cfg: string
  shift: string
  switchAt: string
  seed: string
  /** `match` scales every reference to the clip's own pixel area; `max` hands
   *  over as much as the run allows. The one control in the sources row that
   *  changes what a take costs rather than what it contains — reference tokens
   *  ride every sampling step, so this is a per-step price. */
  refSize: 'match' | 'max'
}

export type { LoraChip }

export type Keyframes = { first: string | null; last: string | null }

/** Everything that is per-kind and lives at the root while its kind is showing.
 *  `shot` is in here because pills compile *into* the prompt — a rail left
 *  standing after a switch is pills belonging to a sentence no longer on screen. */
export type Composed = {
  prompt: string
  negative: string
  negOn: boolean
  shot: ShotPill[]
  loras: LoraChip[]
}

const IMAGE: ImageComposer = {
  model: 'turbo',
  // 4:3, the ratio the page opens on. Its transpose is not on the menu, so the
  // swap arrow lands this one on Custom — see `swapSize`.
  aspect: '1152x896',
  scale: '1',
  width: 1152,
  height: 896,
  sampler: '', scheduler: '', steps: '', cfg: '', shift: '', seed: '',
  n: 1,
}

const VIDEO: VideoComposer = {
  model: '',
  aspect: '16:9',
  tier: '', seconds: '', sampler: '', scheduler: '',
  steps: '', cfg: '', shift: '', switchAt: '', seed: '',
  refSize: 'match',
}


export type Store = {
  /* ---- served vocabulary. The server is the authority; never restate it. -- */
  state: AppState | null
  stateError: string | null
  setState: (s: AppState) => void
  setStateError: (e: string | null) => void

  /* ---- where you are ---------------------------------------------------- */
  mode: Mode
  kind: Kind
  setMode: (m: Mode) => void
  /**
   * **The composer is per-kind. Only the canvas and the gallery are shared.**
   *
   * This used to say the prompt survives the switch, on the grounds that a shot
   * described as a still is the same sentence you would describe as a clip. True
   * of the sentence and false of everything around it: a Krea 2 LoRA is not a
   * Wan LoRA, and carrying one across loaded it into a run that could not use it
   * — silently, because `loraNote` warns about names that resolve to nothing and
   * about models that read no LoRAs at all, and a LoRA for the wrong
   * architecture is neither.
   *
   * Both buffers are kept. Switch away and back and the sentence, the pills and
   * the chips are exactly as they were, which is the promise `img`/`vid` already
   * make about model, size, steps and seed.
   */
  setKind: (k: Kind) => void
  /**
   * The kind you are not looking at.
   *
   * **One live set, one dormant, and `setKind` is the only place both exist.**
   * Moving these five onto `ImageComposer`/`VideoComposer` is the more literal
   * reading and touches every `s.prompt` in the codebase; this touches one
   * function, and keeps the invalid state unreachable rather than merely
   * unlikely — there is exactly one live set, so no component is able to read
   * the wrong one.
   */
  stash: { image: Composed | null; video: Composed | null }

  /* ---- the composer. Live for `kind`; the other one is in `stash`. ------- */
  prompt: string
  negative: string
  /** Which of the two textareas is showing. Per-kind with the rest: H3 has no
   *  negative branch at all, so a buffer shared with the image side was already
   *  half-fictional. */
  negOn: boolean
  /** The LoRAs plugged into this canvas. Regions carry their own. */
  loras: LoraChip[]
  setPrompt: (p: string) => void
  setNegative: (n: string) => void
  setNegOn: (on: boolean) => void
  /** Picking one already picked takes it out — the tick in the menu is a state,
   *  so a second click has to mean something. */
  toggleLora: (chip: LoraChip) => void
  dropLora: (path: string) => void
  patchLora: (path: string, patch: Partial<LoraChip>) => void

  /** The rail, in the order you built it. */
  shot: ShotPill[]
  /** Which valued pill is expanded, if any. */
  shotOpen: string | null
  peekOpen: boolean
  /** The LoRA box's disclosure. Beside `peekOpen` because it is the same
   *  gesture on the same kind of thing. */
  loraOpen: boolean
  setLoraOpen: (on: boolean) => void
  setShot: (pills: ShotPill[]) => void
  toggleShot: (key: string) => void
  setPill: (key: string, patch: Partial<ShotPill>) => void
  setShotOpen: (key: string | null) => void
  setPeekOpen: (on: boolean) => void

  img: ImageComposer
  vid: VideoComposer
  setImg: (patch: Partial<ImageComposer>) => void
  setVid: (patch: Partial<VideoComposer>) => void
  /** Chosen once a session, and it already confirms a cold start when it
   *  changes — so it lives under the gear rather than in the strip. */
  gpu: { image: string; video: string }
  setGpu: (patch: Partial<{ image: string; video: string }>) => void

  /* ---- regions ---------------------------------------------------------- */
  // There is no `regional` flag. Regions are on when there is a rectangle on the
  // canvas and off when the last one is deleted — a mode you arm is a mode you
  // can forget you are in, and the boxes already say which it is. `regionsLive`
  // derives it where the payload and the notes need a boolean.
  regions: Region[]
  /** Index, not id: `_pair_boxes` is positional and so is every readout. */
  rsel: number
  regionWeight: string
  /** The frame is a place too, and the scene and outfit plates are its attachments.
   *  A record with a slot per plate was the same shape as a region's `ref` written
   *  twice more, and it is what made frame scope and box scope two systems. */
  frame: Placed
  /**
   * What the region layer is *drawing*, which is not the same question as whether
   * regions are on. They are on whenever a box exists; this is only about what is
   * on screen, and it exists because two different acts were being served by one
   * control and one of them was paying for the other.
   *
   * - `off` — the default the moment a render lands. Nothing is drawn. The boxes
   *   are still there, still masking their LoRAs, still sent with the next run;
   *   they are simply *addressable rather than drawn*, which is the Phase 6 rule
   *   arriving early rather than a relaxation of "nothing sits on top of a render".
   *   Hover names what you would be touching; a plain click opens it.
   * - `content` — one region's card, for the frequent act: its sentence and its
   *   photograph. No rectangles and no coordinates: what is drawn is the open box
   *   alone — its hairline, because a card with no scope is a card about nothing in
   *   particular, and its handles, because an open box is adjustable. That last part
   *   is not geometry leaking back in. Arming geometry was gated because it put
   *   *every* box over a render you were judging, and the box you have already
   *   clicked is not one of those; asking for a modifier to move the rectangle in
   *   front of you is friction with nothing behind it.
   * - `geometry` — the rare act: every box, its handles, the snapping, the four
   *   coordinates and the frame's own card. Reached by ⌘-click, or a long press
   *   where there is no modifier to hold. Clearing the canvas first was the earlier answer and it was
   *   invented friction: ⌘ already means "geometry" on the frame, where ⌘-drag is
   *   "a new box, here", so it means the same thing over a render.
   *
   * The mode is per-surface in one respect: the frame, before any render exists,
   * has nothing to protect and is always `geometry`. See `regions/RegionLayer.tsx`.
   */
  edit: EditMode
  /** The pointer is down on a box. Two jobs, and they are the same fact: hold
   *  the boxes up through a drag that started before a render was cleared, and
   *  put the inspector away while you are dragging the thing it describes. */
  boxDrag: boolean
  /** A file is somewhere over the window. The one moment "you can drop a photo
   *  on a box" needs saying — and the only way anyone finds that gesture — so
   *  it brings the boxes back even over a finished render. `body.dragging`
   *  carries the same fact to the stylesheet; this is the half React reads, and
   *  its absence is why the port's boxes stayed hidden through a drag and the
   *  drop could not land on one. */
  fileOver: boolean
  setRegions: (r: Region[]) => void
  patchRegion: (i: number, patch: Partial<Region>) => void
  select: (i: number) => void
  setRegionWeight: (v: string) => void
  /** One picture onto one place. `where` is a region's index or the frame, and that
   *  argument is the entire difference between "this character" and "this scene" —
   *  same gesture, same record, different target. */
  attach: (where: number | 'frame', role: Role, image: string | null) => void
  setEdit: (m: EditMode) => void
  setBoxDrag: (on: boolean) => void
  setFileOver: (on: boolean) => void

  /* ---- every picture the model can be given ----------------------------- */
  keyframe: Keyframes
  refs: string[]
  refVids: string[]
  /** One role per image reference, positional — index i is `<Picture i+1>`, so
   *  removing the second chip has to remove the second role with it. */
  refRoles: string[]
  setKeyframe: (slot: keyof Keyframes, b64: string | null) => void
  setRefs: (refs: string[], roles?: string[]) => void
  setRefVids: (v: string[]) => void
  setRefRoles: (r: string[]) => void
}

export const useStore = create<Store>((set, get) => ({
  state: null,
  stateError: null,
  setState: (s) => set({ state: s, stateError: null }),
  setStateError: (e) => set({ stateError: e }),

  mode: 'generate',
  kind: 'image',
  setMode: (mode) => set({ mode }),
  stash: { image: null, video: null },
  // The swap, and the only statement in the file where both sets exist at once.
  setKind: (kind) => set((s) => {
    if (kind === s.kind) return {}
    const mine = s.stash[kind]
    return {
      kind,
      stash: { ...s.stash, [s.kind]: {
        prompt: s.prompt, negative: s.negative, negOn: s.negOn,
        shot: s.shot, loras: s.loras,
      } },
      prompt: mine?.prompt ?? '',
      negative: mine?.negative ?? '',
      negOn: mine?.negOn ?? false,
      shot: mine?.shot ?? [],
      loras: mine?.loras ?? [],
    }
  }),

  prompt: '',
  negative: '',
  negOn: false,
  loras: [],
  setPrompt: (prompt) => set({ prompt }),
  setNegative: (negative) => set({ negative }),
  setNegOn: (negOn) => set({ negOn }),
  toggleLora: (chip) => set((s) => ({
    loras: s.loras.some((l) => l.path === chip.path)
      ? s.loras.filter((l) => l.path !== chip.path)
      : [...s.loras, chip],
  })),
  dropLora: (path) => set((s) => ({ loras: s.loras.filter((l) => l.path !== path) })),
  patchLora: (path, patch) => set((s) => ({
    loras: s.loras.map((l) => (l.path === path ? { ...l, ...patch } : l)),
  })),

  shot: [],
  shotOpen: null,
  peekOpen: false,
  loraOpen: false,
  setShot: (shot) => set({ shot }),
  setShotOpen: (shotOpen) => set({ shotOpen }),
  setPeekOpen: (peekOpen) => set({ peekOpen }),
  setLoraOpen: (loraOpen) => set({ loraOpen }),
  setPill: (key, patch) =>
    set((s) => ({ shot: s.shot.map((p) => (p.key === key ? { ...p, ...patch } : p)) })),

  toggleShot: (key) => {
    const s = get()
    const vocab = s.state?.shot_vocab ?? []
    const g = shotGroup(vocab, key)
    const it = shotItem(vocab, key)
    if (!g || !it) return
    if (s.shot.some((p) => p.key === key)) {
      set({
        shot: s.shot.filter((p) => p.key !== key),
        shotOpen: s.shotOpen === key ? null : s.shotOpen,
      })
      return
    }
    // The same exclusions `_validate_shot` applies, for a different reason:
    // there they make the rule true, here they make it legible. The guide allows
    // one camera move per clip, and a palette that let you stack three would be
    // teaching the opposite of what it compiles.
    const same = (p: ShotPill) => shotGroup(vocab, p.key) === g
    const kept = g.pick === 'one' || it.solo
      ? s.shot.filter((p) => !same(p))
      : s.shot.filter((p) => !(same(p) && shotItem(vocab, p.key)?.solo))
    const pill: ShotPill = { key }
    if (it.valued) {
      pill.value = ''
      // A language tag is a thing you cannot know to write, which is the whole
      // reason dialogue is a pill instead of something you type in the prompt.
      if (it.valued === 'dialogue') pill.lang = s.state?.shot_langs?.[0] ?? 'English'
    }
    set({ shot: [...kept, pill], shotOpen: it.valued ? key : s.shotOpen })
  },

  img: IMAGE,
  vid: VIDEO,
  setImg: (patch) => set((s) => ({ img: { ...s.img, ...patch } })),
  setVid: (patch) => set((s) => ({ vid: { ...s.vid, ...patch } })),
  gpu: { image: '', video: '' },
  setGpu: (patch) => set((s) => ({ gpu: { ...s.gpu, ...patch } })),

  regions: [],
  rsel: -1,
  regionWeight: '1',
  frame: { attachments: [] },
  edit: 'geometry',
  boxDrag: false,
  fileOver: false,
  setRegions: (regions) => set({ regions }),
  patchRegion: (i, patch) =>
    set((s) => ({ regions: s.regions.map((r, n) => (n === i ? { ...r, ...patch } : r)) })),
  select: (i) => set((s) => ({ rsel: i >= 0 && i < s.regions.length ? i : -1 })),
  setRegionWeight: (regionWeight) => set({ regionWeight }),
  attach: (where, role, image) => set((s) => (
    where === 'frame'
      ? { frame: { attachments: setAttached(s.frame.attachments, role, image) } }
      : {
          regions: s.regions.map((r, n) => (n === where
            ? { ...r, attachments: setAttached(r.attachments, role, image) }
            : r)),
        }
  )),
  setEdit: (edit) => set({ edit }),
  setBoxDrag: (boxDrag) => set({ boxDrag }),
  setFileOver: (fileOver) => set({ fileOver }),

  keyframe: { first: null, last: null },
  refs: [],
  refVids: [],
  refRoles: [],
  setKeyframe: (slot, b64) => set((s) => ({ keyframe: { ...s.keyframe, [slot]: b64 } })),
  setRefs: (refs, roles) => set((s) => ({ refs, refRoles: roles ?? s.refRoles })),
  setRefVids: (refVids) => set({ refVids }),
  setRefRoles: (refRoles) => set({ refRoles }),
}))

/* ---- reading the served vocabulary ------------------------------------- */
/* A pill key is `"{group}.{item}"`. Split here and nowhere else, so a malformed
 * key is one thing to reject rather than a shape three call sites guess at. */

export function shotGroup(vocab: ShotGroup[], key: string): ShotGroup | undefined {
  return vocab.find((g) => g.key === key.split('.')[0])
}

export function shotItem(vocab: ShotGroup[], key: string): ShotItem | undefined {
  const rest = key.split('.').slice(1).join('.')
  return shotGroup(vocab, key)?.items.find((it) => it.key === rest)
}

/** The chosen video family, or null when nothing is selected. The composer shows
 *  only the controls this says the model reads — a control that is present but
 *  ignored is worse than one that is absent. */
/** Regions are on when there is a box, and off when the last is gone — there is no
 *  separate flag. Kind is part of it: the boxes are an image-side thing, so a stack
 *  of rectangles left behind by switching to video is not "regions on". */
export const regionsLive = (s: Pick<Store, 'kind' | 'regions'>): boolean =>
  s.kind === 'image' && s.regions.length > 0

export function videoModel(s: Pick<Store, 'state' | 'vid'>): VideoModel | null {
  return s.state?.video_models.find((m) => m.key === s.vid.model) ?? null
}

/** What the chosen family says it reads. An empty object rather than null, so
 *  callers can ask `supports(s).loras` without a guard at each of the nine
 *  sites that ask. */
export function supports(s: Pick<Store, 'state' | 'vid'>): Record<string, boolean> {
  return videoModel(s)?.supports ?? {}
}

/**
 * Whether this model reads a negative prompt at all.
 *
 * On the video side the model says so. On the image side nothing did, and that
 * is the gap this closes: Krea 2 Turbo is distilled to CFG 1.0, where a negative
 * prompt is not weak but *unread*, and the box sat at the top of Advanced
 * regardless. Read off the effective CFG rather than the checkpoint name, so a
 * Turbo run with CFG typed up to 5 gets the control back — the rule is about the
 * number the sampler uses, not about which file is loaded.
 */
export function negAllowed(s: Pick<Store, 'state' | 'kind' | 'img' | 'vid'>): boolean {
  if (s.kind === 'video') return !!supports(s).negative
  const typed = parseFloat(s.img.cfg)
  const def = s.state?.krea2_defaults?.[s.img.model]?.cfg
  const cfg = Number.isFinite(typed) ? typed : def
  return typeof cfg === 'number' && Number.isFinite(cfg) && cfg > 1
}



export function readShot(shot: ShotPill[]): ShotPill[] {
  return shot.map((p) => {
    const o: ShotPill = { key: p.key }
    if (p.value !== undefined) o.value = p.value
    if (p.lang) o.lang = p.lang
    return o
  })
}
