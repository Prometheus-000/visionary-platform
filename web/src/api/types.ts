/**
 * The shapes `/api/state` and the job routes actually return.
 *
 * Written against the live response rather than inferred from the route
 * signatures, because several of these are assembled dict-by-dict in Python and
 * the annotation there is `dict[str, Any]`. Where the server is the authority
 * on a vocabulary — shot pills, caption presets, video models — the type says
 * so and the page never restates the contents. That is the same rule
 * `preview_ui.py` follows when it pulls `SHOT_VOCAB` out of `app.py` by AST: a
 * second copy of a vocabulary is a copy that can disagree with the compiler.
 */

/**
 * A shot pill, in the one shape it has anywhere.
 *
 * The key is `"{group}.{item}"` — a bare item key is rejected by name rather
 * than ignored, because a pill silently dropped is indistinguishable from the
 * model ignoring the word.
 *
 * `value` and not `text`, which is what this said first: this exact object is
 * what `/api/generate`, `/api/video` and `/api/compile` take, what
 * `_validate_shot` reads, and what the sidecar records for Reuse to read back.
 * A second spelling on the client would be a translation layer between four
 * places, and three of them are on the far side of the network.
 */
export type ShotPill = { key: string; value?: string; lang?: string }

export type ShotItem = {
  key: string
  label: string
  glyph?: string
  phrase?: string
  /** `"dialogue"` or `"text"` — takes a typed value, preserved verbatim,
   *  punctuation included. Dialogue is the one that also carries a language,
   *  because the guide names the eleven and forbids inventing one. */
  valued?: 'dialogue' | 'text'
  /** The placeholder for that value: what to type, not what the field is. */
  hint?: string
  solo?: boolean
  needs?: 'audio' | null
}

export type ShotGroup = {
  key: string
  label: string
  pick: 'one' | 'many'
  join: string
  slot: number
  field: 'visual' | 'sound' | 'score'
  /** False means the image side cannot read it — the palette dims rather than
   *  hides, and the compiler is what actually drops it. */
  image: boolean
  needs: 'audio' | null
  items: ShotItem[]
}

export type ShotRole = { key: string; label: string; noun: string; retain: string }

export type ModelEntry = {
  key: string
  label: string
  note: string
  family: string
  repo_id: string
  present: boolean
  size_gb?: number
  approx_gb: number
  gated: boolean
}

export type LoraFile = { name: string; bytes: number; path?: string }
export type LoraEntry = {
  name: string
  trigger_word: string
  root: string
  bytes: number
  catalogue: string
  files: LoraFile[]
}

/** The model's own defaults. Every one is optional because the two families
 *  genuinely differ: H3 is guidance-distilled, so it has no `cfg` at all, and a
 *  `0` here would be a CFG nobody chose rather than a control that is absent. */
export type VideoDefaults = {
  steps?: number
  cfg?: number
  shift?: number
  sampler?: string
  scheduler?: string
  tier?: string
  seconds?: number
}

export type VideoModel = {
  key: string
  label: string
  note: string
  /** Tier key → its own label, which already reads "768p" or "544p draft". The
   *  second word is a fact about the run and belongs on the button. */
  tiers: Record<string, string>
  lengths: number[]
  samplers: string[]
  schedulers: string[]
  defaults: VideoDefaults
  /** Which controls this family actually reads. A control that is present but
   *  ignored is worse than one that is absent, so the composer builds from it. */
  supports: Record<string, boolean>
  /** Per task, because a t2v run must never be told to download the 28.6 GB i2v
   *  pair it will not load. */
  tasks: Record<string, { ready: boolean; missing?: string[] }>
  ready: boolean
}

export type GpuChoice = { options: string[]; default: string }

export type AppState = {
  hf_token_set: boolean
  models: ModelEntry[]
  loras: LoraEntry[]
  video_models: VideoModel[]
  wan_experts: string[]
  max_loras: number
  max_refs: number
  max_ref_videos: number
  max_regions: number
  samplers: string[]
  schedulers: string[]
  image_defaults: { sampler: string; scheduler: string }
  krea2_defaults: Record<string, { steps: number; cfg: number }>
  edit_lora: boolean
  gpus: { image: GpuChoice; video: GpuChoice }
  shot_vocab: ShotGroup[]
  shot_langs: string[]
  shot_roles: ShotRole[]
  caption_presets: { key: string; label: string; note: string }[]
  caption_models: { key: string; label: string; note: string }[]
  caption_defaults: { preset: string; model: string }
}

/** What a poll of `/api/status/{job}` can say. `beat` is why a status is
 *  believed: a job record outlives its container, so a bare `running` is not
 *  evidence that anything is running. */
export type JobStatus = {
  status?: 'running' | 'completed' | 'failed' | 'stopped'
  phase?: string
  step?: number
  steps?: number
  pct?: number
  error?: string
  beat?: number
  files?: string[]
  job_id?: string
  [k: string]: unknown
}

export type CompileResult = { prompt: string }

/**
 * What `/api/datasets/{name}/insight` answers: the prose answer to "what is this dataset
 * teaching the model?".
 *
 * Trigger coverage first, because a caption without the trigger trains a LoRA you cannot
 * summon. `duplicates`, `tag_style` and `thin` are defects; `phrases` is not — it is what
 * the set is teaching, and reading it is the point.
 */
export type Insight = {
  images: number
  captioned: number
  uncaptioned: number
  trigger_word: string
  with_trigger: number
  missing_trigger: string[]
  median_words: number
  /** Captions short enough that the image is mostly teaching the trigger word. */
  thin: string[]
  duplicates: { caption: string; images: string[]; count: number }[]
  tag_style: string[]
  phrases: { phrase: string; count: number; share: number; words: number }[]
}

/**
 * One candidate inside a group.
 *
 * Every field here is a column of the same comparison, which is why they
 * arrive together rather than being fetched per tile: the question is never
 * "how big is this one", it is "which of these four is the one to keep", and
 * that is answered by reading across.
 */
export type DupeImage = {
  name: string
  caption: string
  bytes: number
  width: number
  height: number
  megapixels: number
  format: string
  /** Mean neighbour difference at a fixed 64x64. A tie-breaker between copies
   *  of one picture, never a comparison between different ones. */
  sharpness: number
  mtime: number
  /** Distance from the group's keeper, per hash. Both must be under their
   *  threshold for a link — see DupeReport.thresholds. */
  dhash_distance: number
  phash_distance: number
  /** Byte-identical to the keeper — the one fact that makes the call for you. */
  same_file: boolean
  /** What would explain the difference: resized, reformatted, recompressed,
   *  cropped. Named rather than scored, because "0.93 similar" is not a reason. */
  transforms: string[]
  /** Set only on a crop match, where the *direct* distances above sit outside
   *  the threshold that accepted the pair — these are what accepted it. */
  crop_dhash: number | null
  crop_phash: number | null
}

/**
 * A group, and its kind is the whole safety model.
 *
 * `duplicate` is one picture stored more than once — deleting all but one loses
 * nothing, so it arrives with `suggest` set and everything else marked.
 * `similar` is two photographs that look alike, which on a training set is
 * usually a burst and is usually all worth keeping. A similar group carries an
 * empty `suggest` and nothing in it is ever preselected.
 */
export type DupeGroup = {
  key: string
  kind: 'duplicate' | 'similar'
  /** The server's pick, or '' for a similar group. Derived, so it is shown as
   *  derived — see `why`. */
  suggest: string
  why: string
  images: DupeImage[]
}

export type DupeReport = {
  /** The scan ran out of its per-request budget. `groups` is empty — half a
   *  folder groups into half the truth — and the page polls until it clears. */
  scanning?: boolean
  measured?: number
  total?: number
  images: number
  groups: DupeGroup[]
  thresholds: Record<string, { dhash: number; phash: number }>
  summary: {
    duplicate_groups: number
    duplicate_images: number
    similar_groups: number
    similar_images: number
  }
  /** Bytes accepting every suggestion would return. Duplicates only — nothing
   *  in a similar group is marked, so nothing in one is counted. */
  reclaim: number
}
