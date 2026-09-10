/**
 * What the run will actually use, resolved through the blanks.
 *
 * This is `data-touched` deleted and replaced by one rule: **a control holds `''`
 * until you set it, and `''` means "the model decides".** The vanilla page stamped
 * an attribute on a field the moment you typed in it, so that rebuilding the video
 * composer for a new checkpoint would not hand your sampler back to a default —
 * `fillSelect` read the flag, kept a touched value and re-derived the rest.
 *
 * The rule it was protecting is worth keeping exactly: *a value you chose is yours
 * and survives a model change; a value that was only the previous model's default
 * is not.* Without it, switching video families kept res_multistep — a sampler
 * nobody picked, quietly overriding the euler the other family's own templates use
 * — because it happened to be in both lists.
 *
 * The other half is validation, and it is what the flag alone never gave: a chosen
 * value is kept **only if the new model offers it**. A sampler you picked on one
 * model that the next does not have falls back to that model's default rather than
 * being sent to a checkpoint that will reject it. There is one video family today,
 * so neither half fires there — this is the image side's rule as much as the video
 * side's, and it is the machinery a second family arrives onto.
 */
import type { Store, VideoComposer } from '../store'
import { videoModel } from '../store'

export type ResolvedVid = {
  tier: string
  seconds: string
  sampler: string
  scheduler: string
  /** Placeholders, not values: an empty box means "the model's default", which
   *  stays right when the default changes under it. */
  stepsPlaceholder: string
}

const keep = (value: string, options: string[], fallback: string | undefined): string =>
  options.includes(value) ? value : (fallback && options.includes(fallback) ? fallback : options[0] ?? '')

export function resolveVid(s: Pick<Store, 'state' | 'vid'>): ResolvedVid {
  const m = videoModel(s)
  const d = m?.defaults ?? {}
  const tiers = Object.keys(m?.tiers ?? {})
  const lengths = (m?.lengths ?? []).map(String)
  return {
    tier: keep(s.vid.tier, tiers, d.tier),
    seconds: keep(s.vid.seconds, lengths, d.seconds != null ? String(d.seconds) : undefined),
    sampler: keep(s.vid.sampler, m?.samplers ?? [], d.sampler),
    scheduler: keep(s.vid.scheduler, m?.schedulers ?? [], d.scheduler),
    stepsPlaceholder: d.steps != null ? String(d.steps) : 'auto',
  }
}

/**
 * Which task this run is, for the *download check only*.
 *
 * Deliberately the coarse read, matching what `/api/video` itself decides: which
 * checkpoint loads. `_h3_task` makes a finer one for the alignment instruction,
 * where first-only, last-only and both are three different sentences about where a
 * picture sits in time — that distinction belongs to the compiler and is not this
 * function's business.
 */
export function videoTask(s: Pick<Store, 'state' | 'vid' | 'refs' | 'refVids' | 'keyframe'>): string {
  const m = videoModel(s)
  if (!m) return 't2v'
  if (m.supports.references && (s.refs.length || s.refVids.length)) return 'ref2va'
  if (m.key === 'h3') return 'fl2va'
  return s.keyframe.first || s.keyframe.last ? 'i2v' : 't2v'
}

/** What this model needs that is not on the volume, per task — so a t2v run is
 *  never told to download the 28.6 GB i2v pair it will not load. */
export function videoReady(s: Pick<Store, 'state' | 'vid' | 'refs' | 'refVids' | 'keyframe'>) {
  const m = videoModel(s)
  const t = m?.tasks?.[videoTask(s)] ?? { ready: true, missing: [] }
  return {
    ready: !!t.ready,
    note: t.ready
      ? (m?.note ?? '')
      : `Not downloaded: ${(t.missing ?? []).join(', ')} — get them under Settings.`,
  }
}

/**
 * A model that cannot take what is already attached would fail at submit.
 *
 * Returns the patch that drops it, or null. Called where the section it came from
 * is visibly gone, which is the only version of this that does not look like the
 * request lost it.
 */
export function dropUnsupported(
  s: Pick<Store, 'state' | 'vid' | 'refs' | 'refVids' | 'keyframe'>,
): { refs?: string[]; refVids?: string[]; refRoles?: string[]; last?: null } | null {
  const m = videoModel(s)
  if (!m) return null
  const out: { refs?: string[]; refVids?: string[]; refRoles?: string[]; last?: null } = {}
  if (!m.supports.references && (s.refs.length || s.refVids.length)) {
    out.refs = []
    out.refVids = []
    out.refRoles = []
  }
  if (!m.supports.last_frame && s.keyframe.last) out.last = null
  return Object.keys(out).length ? out : null
}

/** `steps`, `seed`, `sampler`, `scheduler` — whether any of them is an override.
 *  The resolved numbers alone cannot say whether you chose them or the
 *  checkpoint did, and "why is this 12 steps" is a question you ask days later
 *  off a gallery card. CFG, shift and the expert switch were in this list while
 *  a family read them. */
export function vidEdited(v: VideoComposer): boolean {
  return !!(v.steps || v.sampler || v.scheduler)
}
