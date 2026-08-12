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
