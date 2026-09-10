import type { GalleryItem } from './types'

/**
 * What this window made, kept where the volume listing cannot lose it.
 *
 * The gallery reads the volume rather than replaying job ids, and that is still the
 * record — a reload, a redeploy or an expired job all leave the work reachable, which a
 * list of ids in a browser does not. But there is a window where the volume's answer is
 * behind what this page already knows for certain: it watched the run finish and holds
 * the job id and the filenames. Asking the volume to tell it that, and believing the
 * answer when it comes back without it, is how a fresh render fell out of the drawer.
 *
 * So this is a merge, not a replacement. `mine` is only ever *added* to what the volume
 * says, and the volume wins every field on conflict because it has the sidecar.
 *
 * `sessionStorage` rather than `localStorage`, for the reason the drafts session gives:
 * it survives a reload and dies with the tab, which is the right lifetime for something
 * that exists only to cover a lag. A tab reopened tomorrow gets the volume's answer,
 * which by then is the complete one.
 */
const KEY = 'vis-mine'
/** Enough to cover a session's lag, bounded so a long session cannot grow without end.
 *  The volume is what holds the history; this only ever holds the recent tail. */
const CAP = 200

function read(): GalleryItem[] {
  try {
    const raw = sessionStorage.getItem(KEY)
    const rows: unknown = raw ? JSON.parse(raw) : []
    // Shape-checked rather than trusted: this is parsed from storage that survives a
    // reload, so a build that changed the shape would otherwise put undefined into
    // `files[0]` and render a card pointing at nothing.
    return Array.isArray(rows)
      ? (rows as GalleryItem[]).filter((r) => r?.job_id && r.files?.length)
      : []
  } catch {
    return []
  }
}

function write(rows: GalleryItem[]): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(rows.slice(0, CAP)))
  } catch { /* private mode, or the quota — the in-memory merge still works */ }
}

/** Record a finished run. Newest first, and idempotent on job id. */
export function remember(it: GalleryItem): GalleryItem[] {
  const rows = [it, ...read().filter((r) => r.job_id !== it.job_id)]
  write(rows)
  return rows
}

/**
 * Drop what was deleted.
 *
 * Without this, deletion is undone by the next merge: the volume stops listing the
 * folder and this puts it straight back, so the card returns with a picture that now
 * 404s. Delete has to reach both, which is why both delete paths call it with exactly
 * the ids they sent the server — the list is the agreement, on this side too.
 */
export function forget(jobIds: string[]): GalleryItem[] {
  const gone = new Set(jobIds)
  const rows = read().filter((r) => !gone.has(r.job_id))
  write(rows)
  return rows
}

export const mine = read

/**
 * The volume's listing, plus anything of ours it has not caught up to yet.
 *
 * Volume-first on conflict: it carries the sidecar, so it is the richer record of the
 * same job — prompt, seed, model, everything Reuse needs. Ours only fills gaps.
 *
 * Sorted on the same key the server sorts on, with the job id breaking ties, so a
 * session item and a volume item never swap places between reloads.
 */
export function merged(items: GalleryItem[], own: GalleryItem[]): GalleryItem[] {
  const seen = new Set(items.map((i) => i.job_id))
  const extra = own.filter((r) => !seen.has(r.job_id))
  if (!extra.length) return items
  const at = (r: GalleryItem) => r.created ?? r.modified ?? 0
  return [...items, ...extra].sort((a, b) =>
    at(b) - at(a) || (a.job_id < b.job_id ? 1 : a.job_id > b.job_id ? -1 : 0))
}
