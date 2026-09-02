import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { failed, type ApiError } from '../api/client'
import { deleteOutput, fileUrl, gallery, purgeOutputs, zipOutputsUrl } from '../api/routes'
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
  /** Job folders from before the record lived inside the file, still being
   *  moved into place by a one-time job. The listing says how many; the
   *  page says so and comes back until it stops saying it. */
  const [migrating, setMigrating] = useState(0)
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
  /** What the volume listing has returned so far, in its own order — the pages, without
   *  the session merge. Paging appends to this and `items` is always re-derived from it,
   *  because appending to the merged array would page from a cursor the session items
   *  moved. */
  const listed = useRef<GalleryItem[]>([])
  /** No older page left. State for the sentinel's render, mirrored in a ref for the
   *  callback that must read it without a stale closure. */
  const [done, setDone] = useState(false)
  const doneRef = useRef(false)
  const finish = (v: boolean) => {
    doneRef.current = v
    setDone(v)
  }
  const fetching = useRef(false)

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
    const body = r as { items?: GalleryItem[]; total?: number; stale?: boolean
                        migrating?: { pending?: number } }
    setError(null)
    setStale(!!body.stale)
    setMigrating(body.migrating?.pending ?? 0)
    setTotal(body.total ?? 0)
    const page = (body.items ?? []).filter((i) => i.files?.length)
    // A refetch is page one again: a land or a delete changed what "newest"
    // means, and older pages stitched onto the old world would double or skip
    // across the seam. The scroll re-earns them.
    listed.current = page
    finish(page.length >= (body.total ?? 0) || page.length < PAGE)
    setItems(merged(page, own.current))
  }, [])

  /**
   * One older page, appended. The cursor is the last listed row's sort key —
   * the same key the server pages on, so the window is stable under runs
   * landing above it. Deduped by job id anyway: a cursor tie or a misbehaving
   * server must cost a skipped row, never the same card twice, and a page
   * that adds nothing marks the listing done rather than asking again forever
   * — the scan loop's own forward-progress rule, at this end of the wire.
   */
  const more = useCallback(async () => {
    if (fetching.current || doneRef.current) return
    const last = listed.current[listed.current.length - 1]
    if (!last?.modified) return
    fetching.current = true
    const mine_ = gen.current
    try {
      const r = await gallery(last.modified, PAGE)
      // A reload replaced the world while this page was in flight; stitching
      // it onto the new page one would be the out-of-order write again.
      if (mine_ !== gen.current || failed(r)) return
      const body = r as { items?: GalleryItem[]; total?: number }
      const seen = new Set(listed.current.map((i) => i.job_id))
      const page = (body.items ?? [])
        .filter((i) => i.files?.length && !seen.has(i.job_id))
      listed.current = [...listed.current, ...page]
      setTotal(body.total ?? 0)
      if (!page.length || listed.current.length >= (body.total ?? 0)) finish(true)
      setItems(merged(listed.current, own.current))
    } finally {
      fetching.current = false
    }
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
  // While older results are being moved into place, come back for them. A
  // clock here rather than a reply-armed retry, because the reply is the
  // same until the job lands and the job is minutes on a deep volume.
  useEffect(() => {
    if (!migrating) return
    const t = window.setTimeout(() => void refetch(), 4000)
    return () => window.clearTimeout(t)
  }, [migrating, items, refetch])

  const behind = stale && tries >= STALE_TRIES
  return { items, error, reload, record, drop, total, behind, more, done, migrating }
}

/**
 * The line that asks for the next page by being scrolled toward.
 *
 * Not `useNearViewport`: that latches once, which is right for a picture that
 * should never unload and wrong for a trigger that must fire once per page.
 * `epoch` is in the effect's deps so each appended page re-observes — the
 * initial observation always delivers an entry, so a page that was too short
 * to push the sentinel out of the margin still asks for the next one instead
 * of stalling until a scroll happens to nudge it.
 */
function LoadMore({ done, epoch, onNear }: {
  done: boolean
  epoch: number
  onNear: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (done || !el) return
    if (typeof IntersectionObserver === 'undefined') return onNear()
    let t: number | undefined
    const io = new IntersectionObserver(
      (es) => {
        if (!es.some((e) => e.isIntersecting)) return
        onNear()
        // Re-armed, because the observer reports crossings and a sentinel
        // that stays on screen never crosses again. A page that was dropped
        // (a reload superseded it mid-flight) leaves the line visible with
        // nothing coming — re-observing delivers a fresh initial entry, so
        // it asks again rather than waiting for a scroll to nudge it. The
        // hook's own guards make the repeat free while a fetch is out.
        io.unobserve(el)
        t = window.setTimeout(() => {
          if (ref.current) io.observe(ref.current)
        }, 1200)
      },
      // A screen of margin: an older page is a listing round trip away, so
      // asking early means the scroll rarely catches the edge.
      { rootMargin: '600px' },
    )
    io.observe(el)
    return () => {
      io.disconnect()
      window.clearTimeout(t)
    }
  }, [done, epoch, onNear])
  if (done) return null
  return <p ref={ref} className="muted" style={{ margin: '14px 0' }}>Loading older results…</p>
}

export function Gallery({
  items,
  total,
  behind,
  migrating,
  done,
  open,
  drawerOpen,
  onClose,
  onReload,
  onMore,
  onDropped,
  onMeta,
  onHandoff,
  onBoard,
}: {
  items: GalleryItem[]
  /** How many results exist, not how many are listed. The two differ past the cap, and
   *  the difference used to be invisible — including to the purge dialog. */
  total: number
  /** The listing stopped catching up. Said on the Refresh button and nowhere else. */
  behind: boolean
  /** How many older results are still being moved into place, or 0. */
  migrating: number
  /** Every page is loaded; the sentinels render nothing. */
  done: boolean
  open: boolean
  /** Whether the drawer beside the canvas is showing. The drawer stays mounted
   *  through the collapse so the reflow animates — but its cards must not be
   *  rendered behind a closed panel: a hidden card never intersects the
   *  viewport, so its cover would never load, and a drawer that opens onto a
   *  column of blanks reads as broken for the seconds the queue takes. */
  drawerOpen: boolean
  onClose: () => void
  /** May hand back the reload it starts, and should: `remove` and `purge` stay busy until
   *  the listing has actually come back, and a caller that returns `void` ends that wait at
   *  the reply instead — which on a slow volume is the shorter half of the two. */
  onReload: () => void | Promise<void>
  /** Fetch one older page and append it. Idempotent under repeat calls — the
   *  hook guards in-flight and exhausted — so two sentinels can share it. */
  onMore: () => void
  onDropped: (jobIds: string[]) => void
  onMeta: (it: GalleryItem) => void
  onHandoff: (it: GalleryItem, as: 'first' | 'reference' | 'refvideo' | 'edit') => void
  /** Pin this render to the storyboard — a pointer, never a copy. Images
   *  only: a panel is a frame, and a clip is a receipt of several. */
  onBoard: (it: GalleryItem) => void
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

  /**
   * The selection, the osx app's model: ⌘-click starts it, shift-click
   * ranges from the anchor, and while anything is selected a plain click
   * toggles. Toggle rather than the app's replace-on-click, deliberately:
   * this grid has no lasso, so a batch is built click by click, and a plain
   * click that replaced the selection would cost the whole batch to one
   * stray press. A plain click at rest keeps opening the viewer — selection
   * never taxes the common case.
   */
  const [sel, setSel] = useState<Set<string>>(() => new Set())
  const anchor = useRef<string | null>(null)
  const clearSel = useCallback(() => {
    setSel(new Set())
    anchor.current = null
  }, [])
  /** A one-line answer to a silent wait: a navigation download reports
   *  nothing while the server builds the zip, so the page says what the
   *  quiet is. Cleared on a timer because nothing else can clear it — the
   *  page is never told when the download starts. */
  const [zipNote, setZipNote] = useState(false)

  // Pruned against the listing, so a card deleted elsewhere cannot stay
  // counted in a bar it no longer has a picture under.
  useEffect(() => {
    setSel((s) => {
      if (!s.size) return s
      const live = new Set(items.map((i) => i.job_id))
      const kept = new Set([...s].filter((id) => live.has(id)))
      return kept.size === s.size ? s : kept
    })
  }, [items])

  // Esc clears — unless the viewer is up, whose own Esc it must not eat.
  useEffect(() => {
    if (!sel.size || viewing) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') clearSel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [sel.size, viewing, clearSel])

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
      a.href = fileUrl(f)
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

  const pick = (it: GalleryItem, e: React.MouseEvent) => {
    // The ⋯ menu stays reachable whatever is selected.
    if ((e.target as HTMLElement).closest('.more')) return
    if (!sel.size && !e.metaKey && !e.ctrlKey && !e.shiftKey) return
    e.preventDefault()
    e.stopPropagation()
    setSel((s) => {
      const next = new Set(s)
      if (e.shiftKey && anchor.current) {
        // The range runs over the visible order, from the anchor — the osx
        // model's own arithmetic, one layer up from Python.
        const order = shown.map((i) => i.job_id)
        const a = order.indexOf(anchor.current)
        const b = order.indexOf(it.job_id)
        if (a >= 0 && b >= 0) {
          for (let k = Math.min(a, b); k <= Math.max(a, b); k++) {
            const id = order[k]
            if (id) next.add(id)
          }
          return next
        }
      }
      if (next.has(it.job_id)) next.delete(it.job_id)
      else next.add(it.job_id)
      anchor.current = it.job_id
      return next
    })
  }

  const picked = useMemo(() => items.filter((i) => sel.has(i.job_id)), [items, sel])

  const removeSel = () => run('delsel', async () => {
    const ids = picked.map((i) => i.job_id)
    const files = picked.reduce((n, i) => n + i.files.length, 0)
    if (!ids.length) return
    // The dialog states the blast radius in both units, because the card is
    // the thing selected and the file is the thing unlinked.
    if (!confirm(`Permanently delete ${ids.length} result${ids.length === 1 ? '' : 's'}`
      + ` — ${files} file${files === 1 ? '' : 's'}?\n\n`
      + 'The files are unlinked from the volume. This cannot be undone.')) return
    setErr(null)
    const r = await purgeOutputs({ confirm: 'delete', job_ids: ids })
    if (failed(r)) return setErr(r)
    onDropped(ids)
    clearSel()
    await onReload()
  })

  const downloadSel = () => {
    const a = document.createElement('a')
    a.href = zipOutputsUrl(picked.map((i) => i.job_id))
    a.download = ''
    a.click()
    setZipNote(true)
    window.setTimeout(() => setZipNote(false), 8000)
  }

  /** What you can do to one result. Reuse first, because it is the reason the sidecar is
   *  kept at all; Delete last and red, because it unlinks. */
  const menuItems = (it: GalleryItem): MenuItem[] => [
    { label: 'Reuse prompt & settings', run: () => reuse(it) },
    ...(it.kind === 'image'
      ? [{ label: 'Edit this image', run: () => onHandoff(it, 'edit' as const) },
         { label: 'Animate from this frame', run: () => onHandoff(it, 'first' as const) },
         { label: 'Use as reference', run: () => onHandoff(it, 'reference' as const) },
         { label: 'Add to storyboard', run: () => onBoard(it) }]
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
   * They were the same array, and the drawer drew a 24-item slice — so opening a
   * picture from the drawer handed the viewer 24 rows and its counter read "3 / 24" on
   * a volume holding hundreds. The slice is gone (see the drawer grid), but the split
   * survives it: the full grid draws a *filtered* view, and the chevrons and the count
   * are a claim about how much there is, not about how many cards happened to match.
   */
  const card = (it: GalleryItem, all: GalleryItem[], packed: boolean) => (
    <Card key={`${it.job_id}:${it.files[0]}`} item={it}
          busy={busy === `del:${it.job_id}`}
          selected={open && sel.has(it.job_id)}
          onPick={open ? (e) => pick(it, e) : undefined}
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
          <div id="drawer-grid" className="grid">
            {/* The whole listing, not a slice. `slice(0, 24)` dated from when
                every mounted card fetched its cover on first paint, so two
                dozen was already the tolerable burst — a cap protecting the
                network. Covers are viewport-gated and queued now, so a card
                below the fold costs nothing until it is scrolled near, and
                the cap had quietly become a wall: the drawer looked like the
                volume stopped at two dozen results. */}
            {(drawerOpen || open) ? cards(items) : null}
          </div>
          {(drawerOpen || open) && (
            <LoadMore done={done} epoch={items.length} onNear={onMore} />
          )}
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
            {sel.size > 0 && (
              /* The batch bar replaces the filter row rather than joining
                 it: two delete buttons with different scopes in one strip is
                 the confusion the purge dialog exists to prevent. */
              <>
                <span className="muted">{sel.size} selected</span>
                <button className="pill" type="button" onClick={downloadSel}>
                  Download zip
                </button>
                <button className="pill danger" type="button" disabled={!!busy}
                        onClick={() => void removeSel()}>
                  {busy === 'delsel' ? 'Deleting…' : `Delete ${sel.size}`}
                </button>
                <button className="pill" type="button" title="Clear selection"
                        onClick={clearSel}>
                  ✕
                </button>
              </>
            )}
            {sel.size === 0 && (['all', 'image', 'video'] as const).map((f) => (
              <button key={f} type="button"
                      className={`pill${filter === f ? ' on' : ''}`}
                      onClick={() => setFilter(f)}>
                {f === 'all' ? 'All' : f === 'image' ? 'Images' : 'Video'}
              </button>
            ))}
            {sel.size === 0 && <>{migrating > 0 && (
              // Says what it is doing while it does it — the wait costs what
              // it shows. Once, in words, where the listing's own controls are.
              <span className="muted" id="gal-migrating">
                Moving {migrating} older result{migrating === 1 ? '' : 's'} into place…
              </span>
            )}{/* The one place the listing admits it is behind, and only once it has
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
            </>}
          </div>

          {zipNote && (
            <p className="muted" style={{ marginBottom: 10 }}>
              Zipping — the download starts when the file is built.
            </p>
          )}

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
          {/* Where "Showing the newest 200 of N" stood. That line existed
              because the cap was invisible and looked like lost work; the
              sentinel replaces the cap itself — scrolling toward the end
              fetches the next page — so what remains to say is only that the
              end on screen is not the end of the record, for the moment it
              takes the page to land. */}
          <LoadMore done={done} epoch={items.length} onNear={onMore} />
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
