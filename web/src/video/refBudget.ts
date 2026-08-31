import { assets, live } from '../scene/model'
import type { Store } from '../store'

/**
 * What this run is actually carrying, against what H3 will take.
 *
 * **The caps are real, published and completely invisible.** Nine images, three
 * videos, three audio — and **twelve across all types**, which is the binding one
 * and the one nothing has ever said out loud. Nine photographs plus three voices
 * is exactly twelve, so a fully cast scene has no room left for a reference video
 * at all, and three audio caps how many voices one generation can clone: two
 * people in a diner is fine and the waitress is the ceiling.
 *
 * `/api/video` refuses past twelve with a named error before it rents anything,
 * so nothing here is protecting the run. What it protects is the *discovery* —
 * finding out at Generate, after casting nine people, is finding out too late to
 * be told cheaply. Same argument as the keyframe note beside it.
 *
 * **It counts what will travel, not what is in the trays.** `VideoNote` counted
 * `refs`/`refVids` alone, which was right when those were the only way to attach
 * a picture and blind from the moment the composer arrived: twelve photographs
 * on twelve cast members registered as zero. The rule is `videoBody`'s — the
 * cast's files when there is a cast, the flat trays otherwise, never both.
 */
export type RefBudget = {
  images: number
  videos: number
  audios: number
  total: number
  /** Named limits, from `/api/state` rather than transcribed. */
  max: { images: number; videos: number; audios: number; total: number }
  over: boolean
}

/** H3's own total, which `/api/state` does not serve as its own field — the three
 *  per-type caps are published and the sum is not. Named here rather than left as
 *  `9 + 3` arithmetic, because it is not the sum: images and audio alone reach it
 *  without a single video. See `MAX_H3_REF_TOTAL`. */
const TOTAL = 12

export function refBudget(s: Store): RefBudget {
  const composed = live(s.scene)
  // The first frame rides `references[]` when the scene is live — `readScene`'s
  // third argument — so it spends this budget like any photograph: nine cast
  // photos plus a keyframe is ten, and the refusal would otherwise arrive from
  // the route instead of this note.
  const keyed = composed && !s.continueFrom && s.keyframe.first ? 1 : 0
  const images = (composed ? assets(s.scene, 'image', s.pool).length : s.refs.length) + keyed
  const videos = composed ? assets(s.scene, 'video', s.pool).length : s.refVids.length
  const audios = composed ? assets(s.scene, 'audio', s.pool).length : 0
  const max = {
    images: s.state?.max_refs ?? 9,
    videos: s.state?.max_ref_videos ?? 3,
    audios: s.state?.max_ref_audios ?? 3,
    total: TOTAL,
  }
  const total = images + videos + audios
  return {
    images, videos, audios, total, max,
    over: images > max.images || videos > max.videos
      || audios > max.audios || total > max.total,
  }
}

/**
 * The sentence, or nothing.
 *
 * Only ever says what is wrong — the same rule `loraNote` keeps. A count of what
 * you attached is a line telling you what you can already see on the rail.
 *
 * The per-type message names the *silent* half specifically: `/api/video` slices
 * `references` to nine **before** it sums the twelve, so twelve photographs and
 * three voices passes the total check with three pictures quietly gone, and the
 * refusal then arrives as a cast member pointing past what was uploaded. That is
 * a true error about the wrong thing, which is worse than none.
 */
export function budgetNote(s: Store): string {
  const b = refBudget(s)
  if (!b.over) return ''
  if (b.images > b.max.images) {
    return `${b.images} photographs — H3 takes ${b.max.images}. `
      + 'The rest are dropped before the run sees them.'
  }
  if (b.audios > b.max.audios) {
    return `${b.audios} voices — H3 clones ${b.max.audios} in one generation. `
      + 'Continue the scene to carry the others into the next.'
  }
  if (b.videos > b.max.videos) return `${b.videos} reference clips — H3 takes ${b.max.videos}.`
  return `${b.total} references in all — H3 takes ${b.max.total} `
    + `(${b.images} images + ${b.videos} videos + ${b.audios} audio). `
    + 'Continue the scene to spread them across takes.'
}
