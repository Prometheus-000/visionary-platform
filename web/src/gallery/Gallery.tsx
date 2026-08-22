import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { failed, type ApiError } from '../api/client'
import { deleteOutput, fileUrl, gallery, purgeOutputs } from '../api/routes'
import { IconBack, IconExpand, IconRefresh } from '../icons'
import { ErrorNote } from '../ui/ErrorNote'
import { Masonry } from '../ui/Masonry'
import { Menu, type MenuItem } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { useBusy } from '../ui/useBusy'
import { useMedia } from '../ui/useMedia'
import { Card } from './Card'
import { forget, merged, mine, remember } from './mine'
import { reuse } from './reuse'
import { Viewer } from './Viewer'
import { aspectOf, ratioOf, type Filter, type GalleryItem } from './types'

/** How many times a listing that admits it is behind is asked again, and how long
 *  each wait is. Sized against a media transfer finishing rather than a value changing
 *  on a server — `RELOAD_INSIST_BACKOFF`'s own note says a clip on a slow connection is
 *  seconds. About four seconds of patience, at zero cost when the reply is fresh. */
const STALE_TRIES = 2
const STALE_BACKOFF = [1200, 3000]
/** One page. The server clamps its own limit; this is what the grid asks for. */
const PAGE = 200
/** A card's height that is not the picture — the foot's 40px and the two border
 *  pixels. Only the packer needs it, and only so that a column of wide cards is
 *  not packed as though their footers were free. */
const GAL_CHROME = 42
/** The full grid is packed above this and cropped to squares below it, which is
 *  `@media(max-width:1024px)` in the stylesheet read from the other side. */
const GAL_PACKED = '(min-width:1025px)'

/**
 * Three layers, and the third is the one that is easy to miss: the drawer beside the canvas,
 * the full grid, and the picture with its chrome off. That last is entered by *tapping*
 * rather than swiping, and a tap used to close the viewer — so leaving was doing double duty
 * for looking closely, and every attempt to see a render properly dismissed it.
 *
 * Moving between the layers is navigation and lives in the top corners from the moment you
 * are inside — not promoted there after a bottom-centre pill hands you over, which is what a
 * "View all generations" button did and why it read as inconsistent.
 *
 * What this hook holds is one listing assembled from two sources: the volume, which is the
 * record, and what this window watched itself make, which the volume can lag behind. See
 * `mine.ts` for why that is a merge and not a replacement.
 */
export function useGallery() {
  const [items, setItems] = useState<GalleryItem[]>(() => mine())
  const [error, setError] = useState<string | null>(null)
  const [stale, setStale] = useState(false)
  const [total, setTotal] = useState(0)
  // Results this window made. Held in a ref as well as in `items` because every merge
  // needs it and a re-render must not be the thing that keeps it alive.
  const own = useRef<GalleryItem[]>(mine())
  // Which request is current. A ref, not state: state would re-render, and would itself
  // be subject to batching.
  const gen = useRef(0)
  // The retry budget *is* state, unlike `gen`. It was a ref, and that made the one thing
  // it is meant to surface unreachable: `behind` is read during render, a ref does not
  // re-render, and the reply that exhausts the budget sets `stale` to the value it
  // already had — so React bailed out and the button never learned it had given up.
  const [tries, setTries] = useState(0)

  /**
   * One fetch, applied only if it is still the newest.
   *
   * `everyMs` is the first thing anyone reaches for here and it is the wrong tool: it
   * skips a *tick* while one is out, and this is event-driven — Refresh pressed while a
   * land is still fetching is the common case, and there is no tick to skip. So the
   * reply is discarded on arrival instead. Same lesson, other end of the wire.
   *
   * The whole reply is dropped, `stale` with it. A superseded stale answer arming the
   * retry below after a fresh one has landed is the same out-of-order write, one field
   * over.
   */
  const refetch = useCallback(async () => {
    const mine_ = ++gen.current
    const r = await gallery(0, PAGE)
    if (mine_ !== gen.current) return
    if (failed(r)) return setError(r.error)
    const body = r as { items?: GalleryItem[]; total?: number; stale?: boolean }
    setError(null)
    setStale(!!body.stale)
    setTotal(body.total ?? 0)
    setItems(merged((body.items ?? []).filter((i) => i.files?.length), own.current))
  }, [])

  /** A fresh question deserves fresh patience, so every deliberate reload — a land, the
   *  Refresh button, opening the gallery — resets the retry budget. */
  const reload = useCallback(async () => {
    setTries(0)
    await refetch()
  }, [refetch])

  /** A finished run, in the grid before the volume has been asked about it. */
  const record = useCallback((it: GalleryItem) => {
    own.current = remember(it)
    setItems((prev) => merged(prev.filter((r) => r.job_id !== it.job_id), own.current))
  }, [])

  /** Deleted here as well as there, or the next merge puts it straight back. */
  const drop = useCallback((jobIds: string[]) => {
    own.current = forget(jobIds)
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  /**
   * Come back once it has stopped loading pictures — which is what `stale` is for.
   *
   * Three things stop this being a poll, and all three are load-bearing. It is armed by
   * the *reply*, not a clock, so a fresh gallery runs no timers at all. The budget is
   * consecutive and hard, because hammering a container that is telling you it cannot
   * reload is the failure a budget exists to prevent. And the cleanup cancels, so a
   * deliberate reload replaces a pending retry rather than stacking on it.
   */
  useEffect(() => {
    if (!stale || tries >= STALE_TRIES) return
    const t = window.setTimeout(() => {
      setTries((n) => n + 1)
      void refetch()
    }, STALE_BACKOFF[tries] ?? STALE_BACKOFF[STALE_BACKOFF.length - 1])
    return () => window.clearTimeout(t)
  }, [stale, tries, refetch])

  // Only once it has given up. While it is still retrying there is nothing to say —
  // the route's own docstring is right that a listing catching up is not an error and
  // not something to apologise for on screen.
  const behind = stale && tries >= STALE_TRIES
  return { items, error, reload, record, drop, total, behind }
}

export function Gallery({
  items,
  total,
  behind,
  open,
  onClose,
  onReload,
  onDropped,
  onMeta,
  onHandoff,
}: {
  items: GalleryItem[]
  /** How many results exist, not how many are listed. The two differ past the cap, and
   *  the difference used to be invisible — including to the purge dialog. */
  total: number
  /** The listing stopped catching up. Said on the Refresh button and nowhere else. */
  behind: boolean
  open: boolean
  onClose: () => void
  /** May hand back the reload it starts, and should: `remove` and `purge` stay busy until
   *  the listing has actually come back, and a caller that returns `void` ends that wait at
   *  the reply instead — which on a slow volume is the shorter half of the two. */
  onReload: () => void | Promise<void>
  onDropped: (jobIds: string[]) => void
  onMeta: (it: GalleryItem) => void
  onHandoff: (it: GalleryItem, as: 'first' | 'reference' | 'refvideo') => void
}) {
  const [filter, setFilter] = useState<Filter>('all')
  const [viewing, setViewing] = useState<{ rows: GalleryItem[]; i: number } | null>(null)
  const [menuFor, setMenuFor] = useState<GalleryItem | null>(null)
  const pop = usePopover()
  const { busy, run } = useBusy()
  const wide = useMedia(GAL_PACKED)
  // Where `alert(r.error)` used to go. Kept here rather than per card because a delete
  // outlives the menu it was started from — the menu closes on the click — so there is no
  // card-shaped thing left to hang the failure on by the time it arrives.
  const [err, setErr] = useState<ApiError | string | null>(null)

  const shown = useMemo(
    () => (filter === 'all' ? items : items.filter((i) => i.kind === filter)),
    [items, filter],
  )

  const download = (it: GalleryItem) => {
    // One file per click rather than a zip: the browser already knows how to save a file,
    // and a zip route would be a second thing to build and a second thing to get wrong
    // about names.
    for (const f of it.files) {
      const a = document.createElement('a')
      a.href = fileUrl(it.job_id, f)
      a.download = f
      a.click()
    }
  }

  // Keyed by job so the card being deleted is the one that dims, and inside `run` from the
  // confirm onward rather than from the reply: a second Delete while the first is out used
  // to raise a second confirm dialog for a file already on its way to being unlinked.
  const remove = (it: GalleryItem) => run(`del:${it.job_id}`, async () => {
    // Deletion unlinks. The confirm dialog is the safety net, so it has to say what is going
    // and how much of it — a dialog that undersells the blast radius is the failure this
    // replaced.
    const n = it.files.length
    const what = n > 1 ? `${n} files` : it.files[0]
    if (!confirm(`Delete ${what}? This cannot be undone.`)) return
    setErr(null)
    const r = await deleteOutput(it.job_id)
    // Inline, not `alert()`. The sentence got better when `client.ts` split it from the
    // server's report, but a modal can only ever carry the sentence — it has nowhere to
    // fold the report that says *why* the unlink was refused, and it stops the panel to
    // say one line you then have to dismiss before you can look at what it is about.
    if (failed(r)) return setErr(r)
    // Before the reload, not after: the session record is merged into whatever the
    // listing returns, so a delete that only reached the volume comes straight back.
    onDropped([it.job_id])
    // Awaited, so the card stays dim through the refetch too. The reply is not the end of
    // the wait — the grid is wrong until the new listing lands, and that was the half of
    // it with nothing on screen.
    await onReload()
  })

  const purge = () => run('purge', async () => {
    // Spells out the count and the scope, because a filtered gallery and an unfiltered one
    // put the same button in the same place over very different amounts of work.
    const n = shown.length
    if (!n) return
    // "all" only when it is all. The listing is capped, so on a volume holding more than
    // one page this said "all 200", deleted 200 and left the rest — the one control in
    // the app that unlinks, undercounting its own blast radius in the opposite direction
    // from the failure the confirm dialog exists to prevent.
    const rest = filter === 'all' ? Math.max(0, total - n) : 0
    const what = filter === 'all'
      ? (rest ? `${n} of ${total} results` : `all ${n} results`)
      : `${n} ${filter}${n === 1 ? '' : 's'}`
    if (!confirm(`Permanently delete ${what}?\n\n`
      + (rest ? `The ${rest} older than these are not included.\n\n` : '')
      + 'The files are unlinked from the volume. This cannot be undone.')) return
    // The ids, not the filter: what goes is exactly what was counted in the dialog, even if
    // a run lands while it is open.
    const ids = shown.map((i) => i.job_id)
    setErr(null)
    const r = await purgeOutputs({ confirm: 'delete', job_ids: ids })
    // Same reasoning as `remove`, and more so: a purge that half-failed has a server report
    // worth reading, and an alert threw it away.
    if (failed(r)) return setErr(r)
    onDropped(ids)
    await onReload()
  })

  /** What you can do to one result. Reuse first, because it is the reason the sidecar is
   *  kept at all; Delete last and red, because it unlinks. */
  const menuItems = (it: GalleryItem): MenuItem[] => [
    { label: 'Reuse prompt & settings', run: () => reuse(it) },
    ...(it.kind === 'image'
      ? [{ label: 'Animate from this frame', run: () => onHandoff(it, 'first' as const) },
         { label: 'Use as reference', run: () => onHandoff(it, 'reference' as const) }]
      : [{ label: 'Use as video reference', run: () => onHandoff(it, 'refvideo' as const) }]),
    { sep: true },
    { label: 'View metadata', run: () => onMeta(it) },
    { label: it.files.length > 1 ? `Download ${it.files.length} files` : 'Download',
      run: () => download(it) },
    { label: 'Delete', danger: true, run: () => void remove(it) },
  ]

  /**
   * `rows` is what gets drawn; `all` is what the viewer can page through.
   *
   * They were the same array, and the drawer draws `items.slice(0, 24)` — so opening a
   * picture from the drawer handed the viewer 24 rows and its counter read "3 / 24" on
   * a volume holding hundreds. A render cap had quietly become a navigation cap: the
   * chevrons and the count are a claim about how much there is, not about how many
   * cards happened to be rendered beside the canvas.
   */
  const card = (it: GalleryItem, all: GalleryItem[], packed: boolean) => (
    <Card key={`${it.job_id}:${it.files[0]}`} item={it}
          busy={busy === `del:${it.job_id}`}
          aspect={packed ? aspectOf(it) : null}
          onOpen={() => setViewing({ rows: all, i: Math.max(0, all.indexOf(it)) })}
          onMenu={(anchor) => {
            setMenuFor(it)
            pop.at(anchor)
          }} />
  )

  /** The drawer is one column, so it needs no packing — but it is the same picture
   *  as the one in the grid, and letterboxing it in the column beside the canvas
   *  while the full gallery shows it whole is the same render disagreeing with
   *  itself on one screen. Below 1024px it is a horizontal strip whose cards are
   *  sized by flex, and an aspect-ratio there fights the height it is given. */
  const cards = (rows: GalleryItem[], all: GalleryItem[] = rows) =>
    rows.map((it) => card(it, all, wide))

  return (
    <>
      {/* The gallery lives beside the canvas, because the thing you made an hour ago is raw
          material for the thing you are making now — not a destination you leave the studio
          to visit. */}
      <aside className="drawer" id="drawer">
        <div className="drawer-in">
          <div className="drawer-head">
            <span className="grow" />
            <button className="ico" title="Open gallery" type="button" onClick={onReload}>
              <IconExpand />
            </button>
          </div>
          <div id="drawer-grid" className="grid">{cards(items.slice(0, 24), items)}</div>
          {/* A card's Delete is reachable from the drawer with the full gallery shut, and a
              failure rendered only inside `#gal-full` would then be painted behind a panel
              nobody is looking at — silence, which is what replacing the alert must not
              become. Only while it is shut, because the overlay covers this and one failure
              shown twice reads as two. */}
          {!open && <ErrorNote err={err} style={{ marginTop: 6 }} />}
          {!items.length && (
            <p className="muted" style={{ marginTop: 6 }}>Nothing generated yet.</p>
          )}
        </div>
      </aside>

      {open && (
        <div id="gal-full" style={{
          position: 'absolute', inset: 0, background: 'var(--bg)',
          zIndex: 20, overflow: 'auto', padding: '18px 28px 72px',
        }}>
          <div className="row" style={{ gap: 10, marginBottom: 18, flexWrap: 'wrap' }}>
            <button className="ico" title="Back to canvas" type="button" onClick={onClose}>
              <IconBack />
            </button>
            <span className="grow" />
            {(['all', 'image', 'video'] as const).map((f) => (
              <button key={f} type="button"
                      className={`pill${filter === f ? ' on' : ''}`}
                      onClick={() => setFilter(f)}>
                {f === 'all' ? 'All' : f === 'image' ? 'Images' : 'Video'}
              </button>
            ))}
            {/* The one place the listing admits it is behind, and only once it has
                stopped trying. A tooltip naming a state on a control whose home you are
                already in is what the icon rule licenses; a banner over the grid is
                not. */}
            <button className="ico" type="button" onClick={() => void onReload()}
                    disabled={!!busy}
                    title={behind ? 'Refresh — the listing may be behind' : 'Refresh'}>
              <IconRefresh />
            </button>
            {/* Says what it is doing, and shuts while anything else is. `disabled` was
                `!shown.length` alone, so through the purge itself — the longest request in
                the app, one round trip per result — this pill sat there fully live reading
                "Delete 200 results", which is an invitation to press it again. The label
                is the signal; the guard against the second press is in `run`, because
                `disabled` is a paint and a queued click was already past it. */}
            <button className="pill danger" type="button"
                    disabled={!shown.length || !!busy}
                    onClick={() => void purge()}>
              {busy === 'purge' ? 'Deleting…'
                : !shown.length ? 'Delete all'
                : `Delete ${shown.length} ${filter === 'all' ? 'result' : filter}`
                  + (shown.length === 1 ? '' : 's')}
            </button>
          </div>

          {/* Above the grid it is about, which is the thing the alert could not do: a modal
              covers the panel to say one sentence, and you have to agree to it before you
              can look at what failed. This stays until the next attempt clears it. */}
          <ErrorNote err={err} style={{ marginBottom: 14 }} />

          {/* Two layouts, and they are different DOM rather than two stylesheets:
              below 1024px the grid crops to squares — the one place this app trades
              information for density, because there the screen is the constraint —
              and a masonry's columns would hand that grid its cards in chronological
              order down each column instead of across each row. */}
          {wide
            ? <Masonry id="gal-grid" items={shown} ratio={ratioOf} min={232} gap={14}
                       chrome={GAL_CHROME} render={(it) => card(it, shown, true)} />
            : <div id="gal-grid" className="grid">{cards(shown)}</div>}
          {/* Stated rather than silent. The cap was invisible, so a volume holding more
              than a page looked like a volume that had lost the rest. */}
          {filter === 'all' && total > shown.length && (
            <p className="muted" id="gal-more" style={{ marginTop: 14 }}>
              Showing the newest {shown.length} of {total}.
            </p>
          )}
          {!shown.length && (
            <p className="muted">
              {items.length ? 'Nothing of that kind yet.' : 'Nothing generated yet.'}
            </p>
          )}
        </div>
      )}

      {pop.open && menuFor && (
        <Menu anchor={pop.anchor} items={menuItems(menuFor)}
              onClose={() => { pop.close(); setMenuFor(null) }} />
      )}

      {viewing && (
        <Viewer items={viewing.rows} index={viewing.i}
                onIndex={(i) => setViewing({ ...viewing, i })}
                onClose={() => setViewing(null)}
                onAll={() => { setViewing(null); onReload() }} />
      )}
    </>
  )
}
