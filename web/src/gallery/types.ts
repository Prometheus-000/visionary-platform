import { coverUrl, fileUrl } from '../api/routes'
import type { ShotPill } from '../api/types'

/**
 * A generation, as `/api/gallery` returns it — newest first, no job id needed.
 *
 * The sidecar carries three prompt-shaped fields and they are not
 * interchangeable. `prompt` is the compiled receipt: what this encoder was
 * told, this once. `prompt_typed` and `shot` are the durable half — what you
 * actually meant.
 *
 * **A card never shows any of them.** The grid carries the picture, its kind
 * and its age; the prompt lives in the metadata sheet, and Reuse and Copy read
 * it from there. Worth stating because "put the prompt on the card" is an
 * obvious-looking addition, and it is the wrong one twice over: a six-field H3
 * document is not a thing you can read at thumbnail size, and a prompt is an
 * implementation detail of whichever encoder was being fed that day.
 *
 * Where the prompt *is* shown — the sheet, Reuse, Copy — the typed one wins.
 * That reads as legibility and is really the deeper choice: intent recompiles
 * for whatever model comes next, while a stored prompt is worth nothing to a
 * checkpoint that wants a different grammar.
 */
export type GalleryItem = {
  job_id: string
  kind: 'image' | 'video'
  files: string[]
  /** Set only by the two callers that open the viewer on something that is not a gallery
   *  row — a dataset image, and the canvas's own full-screen. Never present on anything
   *  `/api/gallery` returned. */
  src?: string
  created?: number
  modified?: number
  prompt?: string
  /** Present only when the compiler did something. Absent on every prompt
   *  written before the shot palette, which is why `promptOf` falls back. */
  prompt_typed?: string
  shot?: ShotPill[]
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
  switch_at?: number
  seconds?: number
  fps?: number
  frames?: number
  negative_prompt?: string
  /** `expert` only exists on the video stack and `applied` only on the image one, which
   *  is why both are optional rather than this being two types. `applied === false` is
   *  the test everywhere — see the `=== false` note in useGenerate. */
  loras?: { name?: string; unet?: number; expert?: string; applied?: boolean
            text_encoder?: number | null }[]
  /** As `_validate_regions` writes them *back*: `lora`/`strength` rather than the
   *  `loras` stack the page sends, and the box as a four-tuple. Reuse reads this shape,
   *  not the one it posted. */
  regions?: { prompt?: string; lora?: string; strength?: number; box?: number[] }[]
  references?: number
  ref_videos?: number
  ref_roles?: string[]
  region_weight?: number
}

export type Filter = 'all' | 'image' | 'video'

/** What a card and the metadata sheet show. Typed first — see above. */
export function promptOf(it: GalleryItem): string {
  return (it.prompt_typed || it.prompt || '').trim()
}

export function fullUrl(it: GalleryItem): string {
  // `src` first, because two things reach the viewer with a URL and no job behind it: a
  // dataset image, which is addressed by set and filename, and the canvas's own
  // full-screen, which already has the src it is displaying. Making those synthesise a
  // job id to get a URL back out would be a lie in a field other code reads.
  return it.src ?? fileUrl(it.job_id, it.files[0] ?? '')
}

/**
 * The same item small: what a card shows, never what the viewer shows.
 *
 * This was one function called `coverUrl` doing both jobs, which is exactly why one
 * route served both — a 232px grid cell was handed the full-resolution PNG, and so was
 * the 36px last-generation button. Two names is the fix, because the distinction is not
 * a detail of the URL, it is the difference between a picture you are judging and a
 * picture you are picking out of a grid.
 *
 * Same `src` precedence as `fullUrl`, so a caller that already holds bytes is never sent
 * to a route that would have to invent a job id to answer.
 */
export function coverOf(it: GalleryItem): string {
  return it.src ?? coverUrl(it.job_id, it.files[0] ?? '')
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
