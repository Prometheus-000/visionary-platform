import { fileUrl } from '../api/routes'

/**
 * A generation, as `/api/gallery` returns it — newest first, no job id needed.
 *
 * The sidecar carries three prompt-shaped fields and they are not
 * interchangeable. `prompt` is the compiled receipt: what this encoder was
 * told, this once. `prompt_typed` and `shot` are the durable half — what you
 * actually meant. The gallery, Reuse and the metadata sheet all prefer the
 * typed one, which reads as a legibility choice (a card showing a six-field H3
 * document is a card you cannot read) and is really the deeper one: intent
 * recompiles for whatever model comes next, a stored prompt is worth nothing to
 * a checkpoint that wants a different grammar.
 */
export type GalleryItem = {
  job_id: string
  kind: 'image' | 'video'
  files: string[]
  created?: number
  modified?: number
  prompt?: string
  /** Present only when the compiler did something. Absent on every prompt
   *  written before the shot palette, which is why `promptOf` falls back. */
  prompt_typed?: string
  shot?: { key: string; text?: string }[]
  model?: string
  seed?: number
  seeds?: number[]
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  sampler?: string
  scheduler?: string
  shift?: number
  seconds?: number
  fps?: number
  frames?: number
  negative_prompt?: string
  loras?: unknown[]
  regions?: unknown[]
  references?: number
  ref_videos?: number
  region_weight?: number
}

export type Filter = 'all' | 'image' | 'video'

/** What a card and the metadata sheet show. Typed first — see above. */
export function promptOf(it: GalleryItem): string {
  return (it.prompt_typed || it.prompt || '').trim()
}

export function coverUrl(it: GalleryItem): string {
  return fileUrl(it.job_id, it.files[0] ?? '')
}

/**
 * Relative time, in the shape the page already used.
 *
 * Deliberately not `Intl.RelativeTimeFormat`: that pluralises and prefixes
 * ("3 minutes ago"), and these sit in a 12px foot under a picture where the
 * short form is what fits.
 */
export function ago(ts?: number): string {
  if (!ts) return ''
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return 'just now'
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  return d < 7 ? `${d}d ago` : new Date(ts * 1000).toLocaleDateString()
}
