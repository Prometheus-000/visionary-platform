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

/** A shot pill as it goes *out*. The key is `"{group}.{item}"` — a bare item
 *  key is rejected by name rather than ignored, because a pill silently
 *  dropped is indistinguishable from the model ignoring the word. */
export type ShotPill = { key: string; text?: string; lang?: string }

export type ShotItem = {
  key: string
  label: string
  glyph?: string
  phrase?: string
  /** Takes a typed value — dialogue. Preserved verbatim, punctuation included. */
  valued?: boolean
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

export type VideoModel = {
  key: string
  label: string
  note: string
  tiers: Record<string, unknown>
  lengths: number[]
  samplers: string[]
  schedulers: string[]
  defaults: Record<string, unknown>
  /** Which controls this family actually reads. A control that is present but
   *  ignored is worse than one that is absent, so the composer builds from it. */
  supports: Record<string, boolean>
  tasks: Record<string, unknown>
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
