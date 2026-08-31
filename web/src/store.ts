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
 * res_multistep quietly overriding a sampler the model's own templates use.
 */
import { create } from 'zustand'

import type { AppState, ShotGroup, ShotItem, ShotPill, VideoModel } from './api/types'
import type { LoraChip } from './lora/tokens'
import {
  emptyScene, handleOf, newMember, newShot, rename,
  type CastKind, type CastMember, type PoolFile, type Scene, type Shot,
  type SourceKind,
} from './scene/model'

export type Kind = 'image' | 'video'
// 'sheet' is the character-sheet composer — its own surface, entered by a
// door like Train, because composing a reference sheet is neither generating
// nor training and squatting in either room made it a popover.
// 'playground' is the node room — the backend's own graph on screen, rewired
// by hand. A door on the same argument as Sheet: experimenting on the engine
// is neither generating nor training, and the veto list governs the product
// canvas, not the lab.
export type Mode = 'generate' | 'train' | 'sheet' | 'playground'

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
  /**
   * Who this box is, as an Arsenal handle — and empty is the normal state.
   *
   * A box has never needed a name to render: `prompt` says who is standing
   * there and `_pair_boxes` matches by position, so nothing downstream reads
   * this. It exists because **a character you place is a character you should
   * be able to keep**, and the Arsenal is keyed by handle. Naming a box is what
   * makes it saveable and what makes it recallable — the same rule the cast
   * runs on, that the name you are already typing is the recall.
   *
   * It is deliberately not sent. A region's payload is a rectangle, a sentence,
   * a photograph and a LoRA; a handle is how *you* address the character, and
   * putting it on the wire would be the app asking the encoder to read a
   * filing label.
   */
  name: string
  prompt: string
  /** One per box — the node's shape. A dropdown on the card rather than a token
   *  in the field: the field is for words this performer's description needs,
   *  and which box a LoRA belongs to is said by which card you are looking at. */
  lora: LoraChip | null
  /** Carried as a bool in the job record, never the bytes — it is polled. */
  attachments: Attachment[]
}

let regionSeq = 0

/** The same seeding `seedShotIds` does, for the same reason and the same caller.
 *  See `keep.ts`. */
export const seedRegionIds = (ids: string[]) => {
  for (const id of ids) {
    const n = Number(id.slice(1))
    if (Number.isFinite(n) && n > regionSeq) regionSeq = n
  }
}

export const newRegion = (r: Partial<Region> = {}): Region => ({
  id: `r${++regionSeq}`,
  x: 0, y: 0, w: 0.5, h: 1, name: '', prompt: '', lora: null, attachments: [],
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
export type Role =
  'identity' | 'scene' | 'outfit' | 'object1' | 'object2' | 'style1'

/** One picture, and what it is for. The bytes are base64 with no `data:` prefix,
 *  because that is what the wire takes and converting at the edges twice is how a
 *  prefix ends up inside a payload.
 *
 *  `note` exists for the two object plates and nothing else: V12's sockets 3
 *  and 4 take a photograph plus the user's own sentence about what it is —
 *  "a motorcycle she leans against" — and an object with no sentence does
 *  close to nothing, which is why the backend refuses one. It is the cast
 *  row's `ref.note` wearing the image side's record. */
export type Attachment = {
  role: Role
  image: string
  note?: string
  /** The Arsenal file this came off — see `PoolFile.from`, which carries the
   *  same fact for the video side's pool. The image side has no id to re-point,
   *  so a refresh replaces `image` where it stands. */
  from?: { handle: string; file: string }
}

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
  from?: { handle: string; file: string },
): Attachment[] => {
  const rest = list.filter((a) => a.role !== role)
  // `from` is only ever set by a recall. A hand-dropped photograph replacing a
  // recalled one passes nothing and the provenance goes with the picture it
  // described — which is right: what is on the box now came off the Finder, and
  // a later library edit has no claim on it.
  return image ? [...rest, { role, image, ...(from ? { from } : {}) }] : rest
}

/** The object plates' sentence. Only an attachment that exists takes one —
 *  a note with no photograph would be a row the payload cannot send. */
export const setNote = (
  list: Attachment[],
  role: Role,
  note: string,
): Attachment[] =>
  list.map((a) => (a.role === role ? { ...a, note } : a))

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
  /** The Playground toggle: '' runs the built-in graph, a name runs that saved
   *  workflow with the console's compiled values substituted in. See
   *  _apply_workflow in app.py for what "substituted" means. */
  workflow: string
  /** Values for the workflow's exposed inputs — {nodeKey: {input: value}}.
   *  Only sent while `workflow` is set; kept per-composer so toggling back to
   *  Default does not forget what you dialled in. */
  workflowExtras: Record<string, Record<string, string | number | boolean>>
}

export type VideoComposer = {
  model: string
  aspect: string
  tier: string
  seconds: string
  sampler: string
  scheduler: string
  steps: string
  seed: string
  /** `match` scales every reference to the clip's own pixel area; `max` hands
   *  over as much as the run allows. The one control in the sources row that
   *  changes what a take costs rather than what it contains — reference tokens
   *  ride every sampling step, so this is a per-step price. */
  refSize: 'match' | 'max'
  /** Same pair as the image side — see ImageComposer.workflow. */
  workflow: string
  workflowExtras: Record<string, Record<string, string | number | boolean>>
}

export type { LoraChip }

export type Keyframes = { first: string | null; last: string | null }

/** One generation that has landed, as a link in a scene. `line` is the prose it
 *  was made from, kept so the strip can say what a take was without refetching a
 *  sidecar for something the page watched happen. */
export type SceneTake = { jobId: string; file: string; line: string }

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
  workflow: '', workflowExtras: {},
}

const VIDEO: VideoComposer = {
  model: '',
  aspect: '16:9',
  tier: '', seconds: '', sampler: '', scheduler: '',
  steps: '', seed: '',
  refSize: 'match',
  workflow: '', workflowExtras: {},
}

/** One ComfyUI node as the wire spells it. `_meta.title` is ComfyUI's own
 *  optional label field, carried so a renamed node round-trips. */
export type PgNode = {
  class_type: string
  inputs?: Record<string, unknown>
  _meta?: { title?: string }
}
export type PgGraph = Record<string, PgNode>

export type Playground = {
  /** null until first seeded — the room seeds itself from the console. */
  graph: PgGraph | null
  /** What the graph was seeded from, kept for the save-time diff that finds
   *  the inputs a workflow exposes to the console. */
  seed: PgGraph | null
  /** Which engine runs it — the same choice the console makes by duration. */
  host: Kind
  /** Saved-as name; '' while unsaved. */
  name: string
  dirty: boolean
  /** Selected node key, or null. */
  sel: string | null
  /** {filename: base64} for the files LoadImage-style nodes name. */
  attachments: Record<string, string>
}

const PLAYGROUND: Playground = {
  graph: null, seed: null, host: 'image', name: '', dirty: false,
  sel: null, attachments: {},
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
   * of the sentence and false of everything around it: a Krea 2 LoRA is not an
   * H3 LoRA, and carrying one across loaded it into a run that could not use it
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

  /* ---- the playground --------------------------------------------------- */
  /** The node room's whole state. The graph is the wire format itself — the
   *  API graph /prompt takes — because a second in-editor shape would be a
   *  translation layer with its own bugs, and the PNG round-trip only has to
   *  be honest if what is on screen *is* what runs. Node positions are not
   *  here: layout is computed (the frame is computed, never authored), and a
   *  drag is session-local. */
  pg: Playground
  setPg: (p: Partial<Playground>) => void
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
  /** How hard the style LoRA is applied when style references are attached.
   *  A string for the same reason `regionWeight` is — the field owns the
   *  keystrokes, the route's num() owns the parse. 1 is the trained strength;
   *  the live A/B at 1.0 pulled composition and even the subject's gender
   *  toward the reference, which is what earned this its own number. */
  styleStrength: string
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
  /**
   * Is the selected thing's card on screen?
   *
   * Selection used to *be* the open state, which meant the card arrived on every
   * gesture that touched a box — including the ones that were not about what is
   * inside it. Framing is a run of those: draw, move, reshape, draw again, and a
   * 296px panel rooted in the last box you touched sat over the picture for all of
   * it. Worse than sitting there, it *ate presses*: the layer refuses anything
   * inside `.rins`, so a handle or a box lying under the card was not adjustable at
   * all, and the frame's card parks in the bottom-left corner whenever nothing is
   * selected.
   *
   * So the card is a thing you open. A click or a tap inside a box opens it; a
   * press anywhere that is not the card closes it. A *drag* is not a click and
   * leaves it shut, which is the distinction the whole flag exists for: the same
   * press on the same rectangle means "show me this one" when it lands and stays,
   * and "move this one" when it travels.
   */
  cardOpen: boolean
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
  setStyleStrength: (v: string) => void
  /** One picture onto one place. `where` is a region's index or the frame, and that
   *  argument is the entire difference between "this character" and "this scene" —
   *  same gesture, same record, different target. */
  attach: (where: number | 'frame', role: Role, image: string | null,
           from?: { handle: string; file: string }) => void
  /** The sentence on an object plate — frame-scope only, because only the
   *  frame's attachments carry notes. */
  notePlate: (role: Role, note: string) => void
  setEdit: (m: EditMode) => void
  setBoxDrag: (on: boolean) => void
  setCardOpen: (on: boolean) => void
  setFileOver: (on: boolean) => void

  /* ---- the scene (video only) ------------------------------------------ */
  /**
   * The cast and the timeline — **the record for the video side's prose**.
   *
   * `prompt` above is the image side's. The two used to be one field swapped by
   * `setKind`, and the stash comment there calls that the halfway point: the
   * workflows have diverged past what one string can carry, because H3 reads a
   * document with named fields and the composer is what collects them. A shot's
   * line is where the sentence lives now, and with one shot and nothing else
   * chosen `readScene` returns null and the run is that line byte-for-byte.
   *
   * Not stashed by `setKind`, for the same reason `regions` is not: it belongs
   * to one kind, and a scene left standing while you look at Krea 2 is not a
   * state anything reads.
   */
  scene: Scene
  /** Every dropped file, held once and pointed at by id. See `PoolFile`. */
  pool: Record<string, PoolFile>
  /** Which row is selected — an id rather than an index, because rows are added
   *  in the middle and an index would move a selection somebody did not touch. */
  shotSel: string
  /** Which rail chip has its card open, if any. Cast id, pill key or LoRA path:
   *  one rail, three kinds of chip, one open slot between them. */
  railOpen: string | null
  /** The compiled document is on screen, taking the canvas. See `SourcePane` —
   *  no control opens it, because devtools is a chord and a context menu. */
  docOpen: boolean
  /**
   * The document, **taken over by hand**.
   *
   * `null` is the normal state and means the scene is driving it. A string means
   * somebody edited it and that string is what runs — one bit for the whole
   * document rather than a pin per field, because six independent states is an
   * attempt to make a derived surface partly authoritative. Either the scene is
   * driving it or you have taken it over, and you can always see which.
   */
  doc: string | null
  /**
   * The generations this scene is made of, in order.
   *
   * **A scene is longer than a generation, and that is the only reason this
   * exists.** H3 tops out at `H3_MAX_FRAMES` — 345 frames at 24fps, about 14.4
   * seconds — so anything with more than one beat in it is several runs, and the
   * thing that turns several runs into a scene is not a model capability. It is
   * that the cast, the look and the last frame survive from one to the next, so
   * you are never rebuilding context you already gave.
   *
   * Deliberately **not** inside `scene`: that type mirrors `_validate_scene` and
   * is the request body. A record of what has already rendered is not part of any
   * request, and folding it in would put results in the payload.
   */
  takes: SceneTake[]
  /**
   * The take the next generation continues from — by motion, not by a still.
   *
   * Set by Continue when the finished take saved its sampler latent; the run
   * then opens with the previous clip's actual motion and audio pinned as
   * context rather than restarting from a frame. `null` is the normal state.
   * One anchor at a time: while this is set, `keyframe.first` is not sent —
   * two answers to "where does this take open" is a stale-tab bug the route
   * refuses, not a state the page can reach.
   */
  continueFrom: string | null
  setDocOpen: (on: boolean) => void
  setDoc: (text: string | null) => void
  /** Append what just landed. Called from the run rather than by re-asking the
   *  volume about work the page watched finish — see `useVideo.finish`. */
  addTake: (t: SceneTake) => void
  setContinueFrom: (jobId: string | null) => void
  /** Start over. The cast stays: it belongs to the scene, not to a take. */
  clearTakes: () => void
  setScene: (patch: Partial<Scene>) => void
  /** The prose, one row per line. What `typedProse` joined, taken back apart —
   *  which is what makes Reuse a round trip rather than a re-parse. The cast is
   *  left alone: it is not in the sentence. */
  setProse: (text: string) => void
  selectShot: (id: string) => void
  setRailOpen: (id: string | null) => void
  patchShot: (id: string, patch: Partial<Shot>) => void
  /** A new row after `after`, selected. Returns its id so the caller can focus it. */
  addShot: (after?: string) => string
  /** Cut one shot in two at the caret: the tail becomes the next row, the head
   *  stays put. A caret at the end is `addShot` exactly, which is what lets ⏎
   *  be one gesture instead of two. */
  splitShot: (id: string, caret: number) => string
  dropShot: (id: string) => void
  /** A cast member, named on creation because the gesture that makes one is
   *  typing its name — see the mention menu. Returns it so the caller can insert
   *  the handle it settled on, which is not always the string typed. */
  addCast: (kind: CastKind, name: string) => CastMember
  patchCast: (id: string, patch: Partial<CastMember>) => void
  dropCast: (id: string) => void
  addFile: (f: PoolFile) => void
  /**
   * Move every reference from one pool id to another, and forget the old file.
   *
   * The pool is keyed by content, so a library file whose bytes changed is a
   * *new* entry rather than a mutated one — writing new bytes under an old id
   * would break the hash `<Picture N>` is derived from. Re-pointing is therefore
   * the whole of an update, and it has to be one action: a rewrite followed by a
   * separate delete leaves a window where a ref names a file the pool no longer
   * has, and `assets()` would number around the gap. See `refreshArsenal`.
   *
   * The old entry's object URL is revoked here because this is the only place a
   * file has ever left the pool. Nothing else removes one — `detachRef` drops
   * the pointer and leaves the bytes, which is right while another member may
   * still point at them.
   */
  repointRefs: (from: string, to: string) => void
  dropFile: (id: string) => void
  /** One file onto one slot of one cast member. A second slot on the same file
   *  adds a role rather than a second entry — that is what makes "this photo is
   *  both the wardrobe and the body" one upload. */
  attachSlot: (castId: string, fileId: string, slot: string) => void
  detachSlot: (castId: string, slot: string) => void
  /** A note on one attached file — what that picture or recording provides.
   *  Keyed by file rather than by slot, because a slot is only the channel now
   *  and two pictures of one subject share it. */
  patchRef: (castId: string, fileId: string,
             patch: { note?: string; sheet?: boolean; role?: string }) => void
  /**
   * The clip-level sources' one writer — `scene.sources` had none, anywhere,
   * ever: `setScene` shipped with zero callers, so `H3_SOURCES` sat complete
   * in the compiler while no gesture on the page could say a clip continues
   * from a video, edits one, or opens on a picture the cast run will actually
   * read. Replace semantics per kind, one entry, because `continue[0]` /
   * `edit[0]` is what the compiler reads; null clears. Setting `continue`
   * clears `edit` and the reverse, because the validator refuses the pair as
   * two answers to what this clip is — better unrepresentable than refused at
   * Generate.
   */
  setSource: (kind: SourceKind, fileId: string | null) => void
  /** Remove one attached file. By file, not by slot: every picture shares the
   *  `image` channel now, so removing by slot would take all of them. */
  detachRef: (castId: string, fileId: string) => void

  /* ---- every picture the model can be given ----------------------------- */
  keyframe: Keyframes
  refs: string[]
  refVids: string[]
  /** One role per image reference, positional — index i is `<Picture i+1>`, so
   *  removing the second chip has to remove the second role with it. */
  refRoles: string[]
  setKeyframe: (slot: keyof Keyframes, b64: string | null) => void
  /** True while `keyframe.first` is the still the kind-switch carried over on
   *  its own — see the auto-attach effect in App. The note line reads it to
   *  name the actor: a warning about a frame the app attached is a warning
   *  about a decision the person never made, and has to say so. Cleared by any
   *  hand-write of the first slot, which is what `setKeyframe` does. */
  autoFirst: boolean
  setAutoFirst: (on: boolean) => void
  setRefs: (refs: string[], roles?: string[]) => void
  setRefVids: (v: string[]) => void
  setRefRoles: (r: string[]) => void
}

const FIRST = emptyScene()

/**
 * **Every `set` callback is annotated `: Partial<Store>`, and that is a guard
 * rather than a decoration.**
 *
 * Zustand's `set` accepts a partial, so a callback returning an object with a
 * key that is not a store field is *accepted and ignored* — a wrong write that
 * produces a perfectly valid object. `repointRefs` shipped that way for an
 * afternoon: `sources` returned at the root instead of inside `scene`, so the
 * rewrite went nowhere and a keyframe would have left the payload with nothing
 * on screen saying so. The rest of the file happened not to have one.
 *
 * TypeScript will catch it, but only if it is asked. Excess-property checking
 * fires on an object literal assigned to a known type and does *not* fire
 * through the union an early `return {}` creates, which is the shape most of
 * these have. The explicit return type restores it on every branch: a typo'd
 * `poool` now reads "Did you mean to write 'pool'?" instead of silently doing
 * nothing.
 *
 * So do not strip these as noise. It is the class `check_loras` turned out to be
 * guarding on the prompt buffers, caught by the compiler instead of by an
 * afternoon — raised by visionary-platform-b0, whose reading it is.
 */
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
  setKind: (kind) => set((s): Partial<Store> => {
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
  toggleLora: (chip) => set((s): Partial<Store> => ({
    loras: s.loras.some((l) => l.path === chip.path)
      ? s.loras.filter((l) => l.path !== chip.path)
      : [...s.loras, chip],
  })),
  dropLora: (path) => set((s): Partial<Store> => ({ loras: s.loras.filter((l) => l.path !== path) })),
  patchLora: (path, patch) => set((s): Partial<Store> => ({
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
    set((s): Partial<Store> => ({ shot: s.shot.map((p) => (p.key === key ? { ...p, ...patch } : p)) })),

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
  setImg: (patch) => set((s): Partial<Store> => ({ img: { ...s.img, ...patch } })),
  setVid: (patch) => set((s): Partial<Store> => ({ vid: { ...s.vid, ...patch } })),

  pg: PLAYGROUND,
  setPg: (patch) => set((s): Partial<Store> => ({ pg: { ...s.pg, ...patch } })),
  gpu: { image: '', video: '' },
  setGpu: (patch) => set((s): Partial<Store> => ({ gpu: { ...s.gpu, ...patch } })),

  regions: [],
  rsel: -1,
  regionWeight: '1',
  styleStrength: '1',
  frame: { attachments: [] },
  edit: 'geometry',
  boxDrag: false,
  cardOpen: false,
  fileOver: false,
  setRegions: (regions) => set({ regions }),
  patchRegion: (i, patch) =>
    set((s): Partial<Store> => ({ regions: s.regions.map((r, n) => (n === i ? { ...r, ...patch } : r)) })),
  select: (i) => set((s): Partial<Store> => ({ rsel: i >= 0 && i < s.regions.length ? i : -1 })),
  setRegionWeight: (regionWeight) => set({ regionWeight }),
  setStyleStrength: (styleStrength) => set({ styleStrength }),
  attach: (where, role, image, from) => set((s): Partial<Store> => (
    where === 'frame'
      ? { frame: { attachments: setAttached(s.frame.attachments, role, image, from) } }
      : {
          regions: s.regions.map((r, n) => (n === where
            ? { ...r, attachments: setAttached(r.attachments, role, image, from) }
            : r)),
        }
  )),
  notePlate: (role, note) => set((s): Partial<Store> => (
    { frame: { attachments: setNote(s.frame.attachments, role, note) } }
  )),
  setEdit: (edit) => set({ edit }),
  setBoxDrag: (boxDrag) => set({ boxDrag }),
  setCardOpen: (cardOpen) => set({ cardOpen }),
  setFileOver: (fileOver) => set({ fileOver }),

  scene: FIRST,
  pool: {},
  // The one row a fresh scene has, selected. The strip is a readout of which
  // shot you are in, and an unselected one on the first frame is a bar that
  // means nothing until you click something.
  shotSel: FIRST.shots[0]?.id ?? '',
  railOpen: null,
  docOpen: false,
  doc: null,
  takes: [],
  continueFrom: null,
  setContinueFrom: (continueFrom) => set({ continueFrom }),
  addTake: (t) => set((s): Partial<Store> => ({ takes: [...s.takes, t] })),
  clearTakes: () => set({ takes: [], continueFrom: null }),
  setDocOpen: (docOpen) => set({ docOpen }),
  setDoc: (doc) => set({ doc }),
  setScene: (patch) => set((s): Partial<Store> => ({ scene: { ...s.scene, ...patch } })),
  setProse: (text) => set((s): Partial<Store> => {
    const lines = text.split('\n')
    const shots = (lines.length ? lines : ['']).map((line) => newShot(line))
    return { scene: { ...s.scene, shots }, shotSel: shots[0]?.id ?? '' }
  }),
  selectShot: (shotSel) => set({ shotSel }),
  setRailOpen: (railOpen) => set({ railOpen }),
  patchShot: (id, patch) => set((s): Partial<Store> => ({
    scene: { ...s.scene,
             shots: s.scene.shots.map((x) => (x.id === id ? { ...x, ...patch } : x)) },
  })),
  addShot: (after) => {
    const row = newShot()
    set((s): Partial<Store> => {
      const at = s.scene.shots.findIndex((x) => x.id === after)
      const shots = [...s.scene.shots]
      shots.splice(at < 0 ? shots.length : at + 1, 0, row)
      return { scene: { ...s.scene, shots }, shotSel: row.id }
    })
    return row.id
  },
  splitShot: (id, caret) => {
    const cur = get().scene.shots.find((x) => x.id === id)
    // Only the line moves. The pills, the dialogue and `beats` stay with the
    // head — they were chosen about the shot you were writing, and a cut does
    // not reassign them — and the new row takes the default length rather than a
    // share of its parent's. Splitting the parent's seconds by where the caret
    // sits would derive a duration from the description, which is the derivation
    // `Timeline` records the retirement of: the director decides, so the clip
    // grows and the bar is dragged. That is also what `+` already does, which
    // keeps the two ways of starting a shot one behaviour.
    const row = newShot(cur ? cur.line.slice(caret).trimStart() : '')
    const head = cur ? cur.line.slice(0, caret).trimEnd() : ''
    set((s): Partial<Store> => {
      const at = s.scene.shots.findIndex((x) => x.id === id)
      const shots = s.scene.shots.map((x) => (x.id === id ? { ...x, line: head } : x))
      shots.splice(at < 0 ? shots.length : at + 1, 0, row)
      return { scene: { ...s.scene, shots }, shotSel: row.id }
    })
    return row.id
  },
  // Never to zero rows. A scene with no shots is not an empty scene, it is a
  // scene with nowhere to type — and `_validate_scene` reads no shots as "no
  // scene", so the composer would silently become the old prompt box with the
  // sentence deleted.
  dropShot: (id) => set((s): Partial<Store> => {
    if (s.scene.shots.length < 2) return {}
    const i = s.scene.shots.findIndex((x) => x.id === id)
    const shots = s.scene.shots.filter((x) => x.id !== id)
    return {
      scene: { ...s.scene, shots },
      shotSel: s.shotSel === id ? (shots[Math.max(0, i - 1)]?.id ?? '') : s.shotSel,
    }
  }),
  addCast: (kind, name) => {
    // Uniqueness is settled here rather than refused at the door: two people
    // called Ava is a thing that happens, and the handle is how a shot names
    // somebody, so it has to pick one of them. `_validate_scene` refuses a
    // collision, which is the right answer to a stale tab and the wrong answer
    // to somebody halfway through typing a second name.
    const taken = new Set(get().scene.cast.map((c) => handleOf(c.name)))
    const base = handleOf(name)
    let handle = base
    for (let n = 2; handle && taken.has(handle); n++) handle = `${base}_${String(n)}`
    const member = { ...newMember(kind), name: handle }
    set((s): Partial<Store> => ({ scene: { ...s.scene, cast: [...s.scene.cast, member] } }))
    return member
  },
  // A rename rewrites the handle across every row, as a visible find-and-replace
  // — the alternative is storing `@{id}` and showing a string nobody can safely
  // edit. See `rename` for the consequence that buys.
  patchCast: (id, patch) => set((s): Partial<Store> => {
    const was = s.scene.cast.find((c) => c.id === id)
    const from = was ? handleOf(was.name) : ''
    const to = patch.name === undefined ? from : handleOf(patch.name)
    return {
      scene: {
        ...s.scene,
        cast: s.scene.cast.map((c) => (c.id === id ? { ...c, ...patch } : c)),
        shots: from === to ? s.scene.shots
          : s.scene.shots.map((x) => ({ ...x, line: rename(x.line, from, to) })),
      },
    }
  }),
  // The mentions are left alone. They are the person's own text and deleting a
  // bucket is not a licence to edit their sentences — `_validate_scene` says so
  // out loud instead, naming the handle nobody defines any more, which is a
  // sentence under the timeline rather than words vanishing from a line.
  dropCast: (id) => set((s): Partial<Store> => ({
    scene: { ...s.scene, cast: s.scene.cast.filter((c) => c.id !== id) },
    railOpen: s.railOpen === id ? null : s.railOpen,
  })),
  addFile: (f) => set((s): Partial<Store> => (s.pool[f.id] ? {} : { pool: { ...s.pool, [f.id]: f } })),
  // **New bytes get a new id; they are never written under the old one.** The
  // reason is not visible from here, which is why it is repeated here: `pool` is
  // keyed by a hash of what travels, and `<Picture N>` is that key's position in
  // `references[]` — derived rather than kept in step by hand, which is the whole
  // of what `pool.ts` buys. Mutating a file in place would leave an id that no
  // longer describes its contents, so a photograph and its own re-export would
  // stop deduping and every label after the first would name somebody else's
  // face. It reads like an implementation quirk until somebody simplifies it.
  //
  // `: Partial<Store>` is load-bearing. Without it this shipped with `sources`
  // returned at the store root instead of inside `scene` — a wrong write that
  // produces a perfectly valid object, so the rewrite went nowhere and a
  // keyframe would have quietly left the payload. TypeScript let it past because
  // excess-property checking does not fire through the union an early `return {}`
  // creates; an explicit return type makes it fire on every branch.
  repointRefs: (from, to) => set((s): Partial<Store> => {
    if (from === to) return {}
    const cast = s.scene.cast.map((c) => {
      if (!c.refs.some((r) => r.fileId === from)) return c
      // A member already holding the new id keeps one entry rather than two:
      // the same rule `attachSlot` follows, and the reason the pool is keyed by
      // content at all — two rows pointing at one picture is two `<Picture N>`
      // labels for one upload.
      return { ...c, refs: c.refs.some((r) => r.fileId === to)
        ? c.refs.filter((r) => r.fileId !== from)
        : c.refs.map((r) => (r.fileId === from ? { ...r, fileId: to } : r)) }
    })
    // **`sources` too, and inside `scene` where it lives.** A clip-level source
    // is unlikely to be a recalled file, but `dropFile` removes the old entry
    // either way and `readScene` filters an id the pool has lost — so missing
    // this would take a keyframe out of the payload with nothing on screen
    // saying so, which is the one failure worse than an error.
    const sources = Object.fromEntries(
      Object.entries(s.scene.sources).map(([k, ids]) => [
        k, [...new Set((ids as string[]).map((i) => (i === from ? to : i)))],
      ]),
    ) as Scene['sources']
    return { scene: { ...s.scene, cast, sources } }
  }),
  dropFile: (id) => set((s): Partial<Store> => {
    const f = s.pool[id]
    if (!f) return {}
    // The one place a file leaves the pool, so the one place its object URL can
    // be released without stranding a thumbnail somebody is still looking at.
    if (f.url) URL.revokeObjectURL(f.url)
    const { [id]: _gone, ...rest } = s.pool
    return { pool: rest }
  }),
  attachSlot: (castId, fileId, slot) => set((s): Partial<Store> => ({
    scene: { ...s.scene, cast: s.scene.cast.map((c) => {
      if (c.id !== castId) return c
      // Already here — a second drop of the same file is not a second entry.
      if (c.refs.some((r) => r.fileId === fileId)) return c
      // **Pictures stack; a voice replaces.** This kept one file per slot on the
      // rule that the slot *was* the role, so dropping a second photograph
      // silently replaced the first. With a slot collapsed to the channel, every
      // picture shares `image`, and that rule became "a subject may have exactly
      // one photograph" — which is the opposite of what the guide says: *"One
      // subject may be defined by multiple reference assets."*
      //
      // Audio and video still replace, because a subject has one voice and one
      // motion, and a second of either is a correction rather than an addition.
      const many = slot === 'image'
      const rest = many
        ? c.refs
        : c.refs.filter((r) => !r.slots.includes(slot))
      return { ...c, refs: [...rest, { fileId, slots: [slot] }] }
    }) },
  })),
  detachRef: (castId, fileId) => set((s): Partial<Store> => ({
    scene: { ...s.scene, cast: s.scene.cast.map((c) => (c.id !== castId ? c : {
      ...c, refs: c.refs.filter((r) => r.fileId !== fileId),
    })) },
  })),
  patchRef: (castId, fileId, patch) => set((s): Partial<Store> => ({
    scene: { ...s.scene, cast: s.scene.cast.map((c) => (c.id !== castId ? c : {
      ...c, refs: c.refs.map((r) => (r.fileId === fileId ? { ...r, ...patch } : r)),
    })) },
  })),
  setSource: (kind, fileId) => set((s): Partial<Store> => ({
    scene: { ...s.scene,
             sources: { ...s.scene.sources,
                        [kind]: fileId ? [fileId] : [],
                        ...(fileId && kind === 'continue' ? { edit: [] } : null),
                        ...(fileId && kind === 'edit' ? { continue: [] } : null) } },
  })),
  detachSlot: (castId, slot) => set((s): Partial<Store> => ({
    scene: { ...s.scene, cast: s.scene.cast.map((c) => (c.id === castId ? {
      ...c,
      refs: c.refs
        .map((r) => ({ ...r, slots: r.slots.filter((x) => x !== slot) }))
        .filter((r) => r.slots.length),
    } : c)) },
  })),

  keyframe: { first: null, last: null },
  refs: [],
  refVids: [],
  refRoles: [],
  autoFirst: false,
  setAutoFirst: (autoFirst) => set({ autoFirst }),
  // Any write of the first slot is a hand on the control, so the auto flag
  // drops; the auto-attach effect re-raises it after its own write.
  setKeyframe: (slot, b64) => set((s): Partial<Store> => ({
    keyframe: { ...s.keyframe, [slot]: b64 },
    ...(slot === 'first' ? { autoFirst: false } : null),
  })),
  setRefs: (refs, roles) => set((s): Partial<Store> => ({ refs, refRoles: roles ?? s.refRoles })),
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
 * **Never on video.** H3 is guidance-distilled, so a negative prompt there is
 * not weak but *unread* — and a sidecar that records one is a sidecar that lies
 * about how the clip was made. This asked the model while there were two video
 * families and one of them took CFG; there is one now, and the answer is a
 * constant rather than a lookup.
 *
 * On the image side nothing said so either, and that is the gap this closes:
 * Krea 2 Turbo is distilled to CFG 1.0, where the same thing is true, and the
 * box sat at the top of Advanced regardless. Read off the effective CFG rather
 * than the checkpoint name, so a Turbo run with CFG typed up to 5 gets the
 * control back — the rule is about the number the sampler uses, not about which
 * file is loaded.
 */
export function negAllowed(s: Pick<Store, 'state' | 'kind' | 'img' | 'vid'>): boolean {
  if (s.kind === 'video') return false
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
