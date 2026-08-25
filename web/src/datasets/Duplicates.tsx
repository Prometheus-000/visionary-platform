import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { failed } from '../api/client'
import { datasetDuplicates, imageUrl, removeImage, thumbUrl } from '../api/routes'
import type { DupeGroup, DupeImage, DupeReport } from '../api/types'
import { fmtFileSize } from '../format'
import { IconExpand } from '../icons'
import { Thumb } from '../media/thumb'
import type { useDatasets } from './useDatasets'

type Ds = ReturnType<typeof useDatasets>

/**
 * Duplicate review: every group on one scroll, and you pick what survives.
 *
 * It was one group at a time behind ‹ › paging, and that shape was retired on
 * the owner's own report: "cycling between groups makes no sense to me". The
 * paging hid the size of the job (a count in the bar is not eight groups you
 * can see), hid what was marked everywhere but the group on screen, and made
 * the filter dropdown silently renumber the position you were at. A scroll
 * shows the whole blast radius at once, which is what the confirm dialog was
 * having to describe in words. The rail survives as a map — one segment per
 * group, filled where something is marked — because on a long scroll it is
 * also the way to jump.
 *
 * **The selection is inverted, and that is the whole feature.** Every other
 * delete surface here asks you to name what goes, which is the wrong half of
 * the question for a group of near-identical frames: there are six of them and
 * exactly one you want, so naming the five is five decisions to express one.
 * A group therefore arrives with a keeper already chosen and everything else
 * marked, and the gesture is promotion — touch any marked image and it becomes
 * a keeper too. Demotion is the same touch back, with one refusal: **the last
 * keeper cannot be demoted.** That invariant is what makes the whole screen
 * safe to move quickly through, because there is no sequence of clicks that
 * deletes a group entirely.
 *
 * **The suggestion is derived, so it says which number decided it.** `why`
 * comes off the server beside the pick — "most pixels · 12.2 MP", "same size,
 * least compressed" — because a suggestion you have to re-derive by hand before
 * you can trust it costs more than making the choice yourself.
 *
 * **The facts are laid out as one grid with the pictures in its top row.** The
 * question a duplicate group asks is never "how big is this one", it is "which
 * of these four", and that is read *across*. So resolution, megapixels, weight
 * and encoding are rows of a table whose columns are the images themselves —
 * the label appears once in the gutter, the best cell in each row is marked,
 * and every other cell carries its distance from that best. A per-card stack of
 * the same numbers was the first version and it made you hold four values in
 * your head to compare them.
 */
export function Duplicates({ ds, onLightbox, onExit }: {
  ds: Ds
  onLightbox: (src: string) => void
  onExit: () => void
}) {
  const [mode, setMode] = useState<'all' | 'duplicate' | 'similar'>('all')
  const [report, setReport] = useState<DupeReport | null>(null)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  /** Per group, who survives. Never empty for a group that exists. */
  const [keep, setKeep] = useState<Record<string, string[]>>({})
  /** Groups that have actually been on screen, which the confirm dialog
   *  reports: a suggestion you never looked at is still a suggestion, and a
   *  dialog that does not distinguish the two is a dialog underselling its
   *  blast radius. On a scroll, "seen" means the group entered the viewport. */
  const [seen, setSeen] = useState<Set<string>>(new Set())
  /** One element per group key, for the rail's jump. */
  const groupEls = useRef(new Map<string, HTMLDivElement>())

  const name = ds.open

  const scan = useCallback(async () => {
    if (!name) return
    setScanning(true)
    setError(null)
    /* A scan that ran out of its per-request budget wrote what it measured and
       asked to be called again, so this loops rather than polling on a timer:
       the next request *is* the work, there is nothing to wait for, and a timer
       would only leave the volume idle between measurements.

       `last` is the client half of the server's forward-progress guarantee.
       The server measures at least one image per request whatever the budget,
       so a round that measures nothing cannot happen — and a page that trusts
       that unconditionally is a page that spins forever the day it stops being
       true. Checked from both sides, and it costs one comparison. */
    let last = -1
    for (;;) {
      const r = await datasetDuplicates(name)
      if (failed(r)) {
        setScanning(false)
        setReport(null)
        return setError(r.error)
      }
      if (r.scanning) {
        const done = r.measured ?? 0
        if (done <= last) {
          setScanning(false)
          return setError(
            `The scan stopped making progress at ${done} of ${r.total ?? '?'} images.`
            + ' Rescan to pick it up again.')
        }
        last = done
        setReport(r)
        continue
      }
      setScanning(false)
      setReport(r)
      // A rescan is a different set of groups, so the marks start again from
      // the suggestions rather than being carried onto keys that may not exist.
      // Only a duplicate group preselects: a similar group is two photographs
      // that look alike, which on a training set is usually a burst and usually
      // all worth keeping, so it arrives whole with nothing to undo.
      setKeep(Object.fromEntries(r.groups.map((g) => [
        g.key, g.kind === 'duplicate' ? [g.suggest] : g.images.map((i) => i.name)])))
      setSeen(new Set())
      return
    }
  }, [name])

  useEffect(() => { void scan() }, [scan])

  const all = useMemo(() => report?.groups ?? [], [report])
  const groups = useMemo(
    () => all.filter((g) => mode === 'all' || g.kind === mode), [all, mode])

  const markSeen = useCallback((key: string) => {
    setSeen((s) => (s.has(key) ? s : new Set(s).add(key)))
  }, [])

  const jump = useCallback((key: string) => {
    // Instant, not smooth: a jump from the map is about being there, and a
    // smooth scroll is an animation the browser silently skips in a hidden
    // tab — which is how "the rail does nothing" was reproduced.
    groupEls.current.get(key)?.scrollIntoView({ block: 'start' })
  }, [])

  const kept = useCallback((g: DupeGroup, k: Record<string, string[]>) =>
    k[g.key] ?? (g.kind === 'duplicate' ? [g.suggest] : g.images.map((i) => i.name)), [])

  const toggle = useCallback((g: DupeGroup, img: string) => {
    setKeep((k) => {
      const cur = k[g.key] ?? (g.kind === 'duplicate' ? [g.suggest]
        : g.images.map((i) => i.name))
      if (!cur.includes(img)) return { ...k, [g.key]: [...cur, img] }
      // The one refusal on this screen. Everything else is reversible by
      // clicking again; a group with nothing left in it is not.
      if (cur.length === 1) return k
      return { ...k, [g.key]: cur.filter((n) => n !== img) }
    })
  }, [])

  /* Over every group, never the filtered view. The delete button states a
     blast radius, and a number that changes when you touch a dropdown you did
     not think was a decision is the specific way a confirm dialog stops being
     believed. Every image is in at most one group, so no name can appear
     twice here — that invariant is the server's, and it is what this count
     rests on. */
  const marked = useMemo(() => {
    const out: { name: string; bytes: number; key: string }[] = []
    for (const g of all) {
      const kept = keep[g.key] ?? (g.kind === 'duplicate' ? [g.suggest]
        : g.images.map((i) => i.name))
      for (const img of g.images) {
        if (!kept.includes(img.name)) out.push({ name: img.name, bytes: img.bytes, key: g.key })
      }
    }
    return out
  }, [all, keep])

  const commit = async () => {
    if (!name || !marked.length || deleting) return
    const unseen = new Set(marked.map((m) => m.key)).size
      - new Set(marked.map((m) => m.key).filter((k) => seen.has(k))).size
    const bytes = marked.reduce((n, m) => n + m.bytes, 0)
    // The dialog is the whole safety net — deletion unlinks and there is no
    // trash behind it — so it states the count, the weight, and the part of the
    // blast radius that is still only a suggestion.
    if (!confirm(
      `Permanently delete ${marked.length} image${marked.length === 1 ? '' : 's'}`
      + ` (${fmtFileSize(bytes)})?\n\n`
      + `Every group keeps at least one image.`
      + (unseen
        ? `\n${unseen} of the groups involved ${unseen === 1 ? 'has' : 'have'} not been`
          + ` opened, so ${unseen === 1 ? 'its' : 'their'} keeper is still the suggested one.`
        : '')
      + `\n\nImages and their captions are unlinked from the volume.`
      + ` This cannot be undone.`,
    )) return
    setDeleting(true)
    const r = await removeImage(name, { images: marked.map((m) => m.name) })
    setDeleting(false)
    if (failed(r)) return setError(r.error)
    await ds.loadTiles(name)
    await ds.load()
    await scan()
  }

  /* ---- keys ------------------------------------------------------------ */
  /* The scroll is the navigation now, so the keyboard keeps only what the
     scroll cannot do: Escape leaves. The arrow/digit chords went with the
     paging — a digit meant "column N of the group on screen", and a scroll
     has no single group on screen for it to mean. */
  const live = useRef({ onExit })
  live.current = { onExit }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
      if (e.key === 'Escape') { e.preventDefault(); live.current.onExit() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  /* ---- states before there is anything to review ----------------------- */
  if (error) {
    return (
      <div className="dupes">
        <div className="err-box">{error}</div>
        <button className="s" type="button" onClick={() => void scan()}>Try again</button>
      </div>
    )
  }
  if (scanning && !report?.groups.length) {
    // Named as a decode rather than a search, and *counted*, because the first
    // scan of a folder of phone photos is tens of seconds and a bare spinner
    // for that long is indistinguishable from a hang. The count is real: the
    // server reports what it has actually measured, and the figure survives a
    // container dying mid-scan because the cache behind it does.
    const done = report?.measured ?? 0
    const total = report?.total ?? ds.images.length
    return (
      <div className="dupes">
        <div className="dupes-empty">
          <b>Measuring {total} image{total === 1 ? '' : 's'}…</b>
          <p className="muted">
            {done ? `${done} of ${total}` : 'Reading the folder'} · measured once, then cached
          </p>
          <div className="bar" style={{ width: 'min(340px,60vw)' }}>
            <i style={{ width: `${total ? Math.round((done / total) * 100) : 0}%` }} />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="dupes">
      <div className="opts dupes-bar">
        <span className="muted" id="dupes-count">
          {report
            ? `${groups.length} group${groups.length === 1 ? '' : 's'} of ${report.images} images`
            : ''}
        </span>
        {/* Similar is not a softer duplicate. Filtering only changes what is on
            screen — never what is marked, which is why the delete count is
            computed over every group rather than over this list. */}
        <div className="opt">
          <select id="dupes-mode" value={mode} title="Which groups to show"
                  onChange={(e) => setMode(e.target.value as typeof mode)}>
            <option value="all">All groups</option>
            <option value="duplicate">Duplicates</option>
            <option value="similar">Similar</option>
          </select>
        </div>
        <button className="s" id="dupes-rescan" type="button" disabled={scanning}
                title="Measure the set again" onClick={() => void scan()}>
          {scanning ? 'Scanning…' : 'Rescan'}
        </button>
        <span className="actions">
          {/* What deleting would cost, before you press it. A count with no
              weight beside it is the half of the blast radius that is easy to
              state and the one nobody is deciding on. */}
          <button className="b danger" id="dupes-delete" type="button"
                  disabled={!marked.length || deleting}
                  onClick={() => void commit()}>
            {deleting ? 'Deleting…'
              : marked.length
                ? `Delete ${marked.length} · ${fmtFileSize(marked.reduce((n, m) => n + m.bytes, 0))}`
                : 'Nothing marked'}
          </button>
          <button className="s" id="dupes-done" type="button" onClick={onExit}>Done</button>
        </span>
      </div>

      {report && (
        <div className="dupes-summary" aria-label="What the scan found">
          <span>
            {report.summary.duplicate_images} copies in {report.summary.duplicate_groups}
            {' '}group{report.summary.duplicate_groups === 1 ? '' : 's'}
          </span>
          {/* Amber, and never green: a similar group is something to look at,
              and colouring it like a decided one is the page implying a
              verdict the classifier deliberately did not reach. */}
          <span className="review">
            {report.summary.similar_images} similar in {report.summary.similar_groups}
            {' '}group{report.summary.similar_groups === 1 ? '' : 's'}
          </span>
          {/* Not amber. Amber on this row means "look at this yourself", and
              the reclaim figure is a consequence of the duplicate groups rather
              than a third thing to judge. */}
          <span>
            {report.reclaim ? `${fmtFileSize(report.reclaim)} recoverable` : 'nothing to reclaim'}
          </span>
          {/* Should never render: everything the upload accepts, the scan
              decodes. When it does, it is the difference between "the scan
              missed obvious duplicates" and a named decode fault. */}
          {!!report.unreadable?.length && (
            <span className="review" title={report.unreadable.join(', ')}>
              {report.unreadable.length} image{report.unreadable.length === 1 ? '' : 's'} could
              not be decoded and {report.unreadable.length === 1 ? 'is' : 'are'} missing from
              every group
            </span>
          )}
        </div>
      )}

      {!groups.length ? (
        <div className="dupes-empty">
          <b>{mode === 'all' ? 'Nothing repeats in this set.'
            : mode === 'duplicate' ? 'No duplicates.' : 'Nothing merely similar.'}</b>
          <p className="muted">
            {report?.images ?? 0} images measured, every pair compared.
          </p>
        </div>
      ) : (
        <>
          {/* The rail is the map: one segment per group, filled where something
              is marked, hollow where you have kept everything. It earns its row
              by being content rather than a control — and on a scroll it is
              also how you jump to a group you remember. */}
          <div className="dupes-rail" id="dupes-rail">
            {groups.map((g, i) => {
              const cuts = g.images.length - kept(g, keep).length
              return (
                <button key={g.key} type="button"
                        className={['seg', cuts ? 'cuts' : 'clean',
                                    seen.has(g.key) ? 'seen' : ''].filter(Boolean).join(' ')}
                        title={`Group ${i + 1} — ${g.kind}, ${g.images.length} images,`
                               + ` ${cuts} marked`}
                        onClick={() => jump(g.key)} />
              )
            })}
          </div>

          {/* Every group, one scroll. What is marked anywhere is visible by
              scrolling, which is the same fact the delete button's count
              states — the two can be checked against each other. */}
          {groups.map((g) => (
            <SeenGate key={g.key} onSeen={() => markSeen(g.key)}
                      refFor={(el) => {
                        if (el) groupEls.current.set(g.key, el)
                        else groupEls.current.delete(g.key)
                      }}>
              <Group group={g} kept={kept(g, keep)}
                     dataset={name!} onToggle={(img) => toggle(g, img)}
                     onReset={() => setKeep((k) => ({
                       ...k,
                       [g.key]: g.kind === 'duplicate' ? [g.suggest]
                         : g.images.map((i) => i.name),
                     }))}
                     onKeepAll={() => setKeep((k) => (
                       { ...k, [g.key]: g.images.map((i) => i.name) }))}
                     onLightbox={onLightbox} />
            </SeenGate>
          ))}
        </>
      )}
    </div>
  )
}

/**
 * Marks a group as seen once it has actually been in the viewport, for the
 * confirm dialog's "N groups were never opened" caveat. Margin zero, unlike
 * the thumbnail prefetch: "about to scroll in" is close enough to fetch a
 * picture and not close enough to count as having looked at it.
 */
function SeenGate({ onSeen, refFor, children }: {
  onSeen: () => void
  refFor: (el: HTMLDivElement | null) => void
  children: React.ReactNode
}) {
  const box = useRef<HTMLDivElement>(null)
  const seenRef = useRef(onSeen)
  seenRef.current = onSeen
  useEffect(() => {
    const el = box.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') {
      seenRef.current()
      return
    }
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) {
        seenRef.current()
        io.disconnect()
      }
    })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return (
    <div ref={(el) => { box.current = el; refFor(el) }}>
      {children}
    </div>
  )
}

/** The best value in each row, so a cell can mark itself and every other one can
 *  say how far off it is. Computed per group rather than per set: "the largest"
 *  is a claim about these four files and nothing else. */
function bestOf(images: DupeImage[]) {
  return {
    pixels: Math.max(...images.map((i) => i.width * i.height)),
    bytes: Math.max(...images.map((i) => i.bytes)),
    sharp: Math.max(...images.map((i) => i.sharpness)),
    // The encoding the keep rank would prefer, so the marked cell is the one
    // that actually counted rather than whichever sorts first.
    format: [...images].sort((a, b) =>
      FORMAT_RANK.indexOf(a.format) - FORMAT_RANK.indexOf(b.format))[0]?.format,
  }
}

/** Mirrors `_FORMAT_RANK` in app.py. Only used to mark a cell — the ranking
 *  that decides anything is the server's. */
const FORMAT_RANK = ['PNG', 'WEBP', 'AVIF', 'BMP', 'JPEG']

const pct = (v: number, best: number) =>
  (v === best ? '' : `−${Math.round((1 - v / best) * 100)}%`)

function Group({ group, kept, dataset, onToggle, onReset, onKeepAll, onLightbox }: {
  group: DupeGroup
  kept: string[]
  dataset: string
  onToggle: (name: string) => void
  onReset: () => void
  onKeepAll: () => void
  onLightbox: (src: string) => void
}) {
  const best = bestOf(group.images)
  const cuts = group.images.length - kept.length
  // Bounded, not `1fr`: a column wide enough to be worth looking at, and never
  // so wide that the square above it eats the whole canvas. Past what fits, the
  // wrapper scrolls sideways — a photograph you cannot judge is worse than one
  // you have to scroll to.
  const cols = `92px repeat(${group.images.length},minmax(150px,300px))`

  const dup = group.kind === 'duplicate'
  /** The first row of the group is the anchor every distance is measured from,
   *  and it says so: a distance is meaningless without naming what it is from.
   *  Below it, the transforms — what would explain the difference — because
   *  "resized, recompressed" is a reason and "0.94 similar" is not. */
  const anchor = group.images[0]?.name
  const match = (i: DupeImage) =>
    i.name === anchor ? 'the one below is measured from this'
      : i.same_file ? 'the same file, byte for byte'
        // A crop is matched on its crop distance, not its direct one, and the
        // direct one is outside the threshold by definition. Showing only the
        // number that did not accept the pair reads as the page contradicting
        // itself, so a cropped row quotes the number that did — labelled, since
        // it is measured between two different rectangles than every other
        // distance on this screen.
        //
        // `typeof`, not `!== null`: a server that has not been restarted sends
        // the field as *absent*, `undefined !== null` is true, and the row then
        // reads "d undefined p undefined" — which is how this was found.
        : typeof i.crop_dhash === 'number'
          ? `${i.transforms.join(', ')} · crop d${i.crop_dhash} p${i.crop_phash}`
          : [i.transforms.join(', ') || 'same framing',
             `d${i.dhash_distance} p${i.phash_distance}`].join(' · ')

  return (
    <div className="dupes-group" id="dupes-group">
      <div className="dupes-head">
        <b>
          {group.images.length} {dup ? 'copies of one picture' : 'images that look alike'}
        </b>
        {/* Derived, and visible as derived — including the case where nothing
            was derived, which a similar group has to say out loud or its empty
            marks read as the scan having failed to decide. */}
        <span className="muted">
          {dup ? `Keeping ${group.suggest} — ${group.why}`
            : 'Nothing preselected. Alike is not the same picture, and a burst is '
              + 'usually worth keeping whole.'}
        </span>
        <span className="actions">
          <span className={`muted dupes-tally${cuts ? ' cuts' : ''}`}>
            {cuts ? `${cuts} marked to delete` : 'keeping all'}
          </span>
          <button className="s" type="button" title="Keep every image in this group"
                  onClick={onKeepAll}>Keep all</button>
          <button className="s" type="button" disabled={!cuts}
                  title={dup ? 'Back to the suggested keeper' : 'Back to keeping everything'}
                  onClick={onReset}>Reset</button>
        </span>
      </div>

      {/* One grid: pictures in the first row, one fact per row under them, the
          label once in the gutter. Scrolls sideways rather than shrinking,
          because a column narrow enough to fit eight of them is a column you
          cannot judge a photograph in. */}
      <div className="dupes-tableWrap">
        <div className="dupes-table" style={{ gridTemplateColumns: cols }}>
          <div className="gut" />
          {group.images.map((i, n) => {
            const on = kept.includes(i.name)
            const last = on && kept.length === 1
            return (
              <div key={i.name} className={`cand${on ? ' keep' : ' cut'}`}>
                <button className="shot" type="button"
                        title={on
                          ? (last ? 'The last keeper in this group cannot be dropped'
                            : 'Mark this one for deletion')
                          : 'Keep this one too'}
                        onClick={() => onToggle(i.name)}>
                  <Thumb url={thumbUrl(dataset, i.name)} />
                  <span className="mark">{on ? 'Keep' : 'Delete'}</span>
                  <span className="num">{n + 1}</span>
                </button>
                {/* The lightbox is its own target rather than the picture's
                    click, because the picture's click is the decision — one
                    gesture, one meaning. */}
                <button className="peek" type="button" title="Full size"
                        onClick={() => onLightbox(imageUrl(dataset, i.name))}>
                  <IconExpand />
                </button>
                <span className="fname" title={i.name}>{i.name}</span>
              </div>
            )
          })}

          <div className="gut">Resolution</div>
          {group.images.map((i) => (
            <div key={i.name} className={`cell${i.width * i.height === best.pixels ? ' best' : ''}`}>
              {i.width}×{i.height}
            </div>
          ))}

          <div className="gut">Megapixels</div>
          {group.images.map((i) => (
            <div key={i.name} className={`cell${i.width * i.height === best.pixels ? ' best' : ''}`}>
              {i.megapixels} MP
              <i className="off">{pct(i.width * i.height, best.pixels)}</i>
            </div>
          ))}

          <div className="gut">File size</div>
          {group.images.map((i) => (
            <div key={i.name} className={`cell${i.bytes === best.bytes ? ' best' : ''}`}>
              {fmtFileSize(i.bytes)}
              <i className="off">{pct(i.bytes, best.bytes)}</i>
            </div>
          ))}

          <div className="gut">Type</div>
          {group.images.map((i) => (
            <div key={i.name} className={`cell${i.format === best.format ? ' best' : ''}`}>
              {i.format}
            </div>
          ))}

          {/* In the table because it breaks ties in `_keep_rank`, and a
              suggestion decided by a number the page does not show is a
              suggestion you cannot check. Never read across different
              pictures — it only means anything between copies of one. */}
          <div className="gut">Detail</div>
          {group.images.map((i) => (
            <div key={i.name} className={`cell${i.sharpness === best.sharp ? ' best' : ''}`}>
              {i.sharpness}
              <i className="off">{pct(i.sharpness, best.sharp)}</i>
            </div>
          ))}

          <div className="gut">Caption</div>
          {group.images.map((i) => (
            <div key={i.name} className={`cell${i.caption ? ' best' : ''}`}
                 title={i.caption || 'No caption'}>
              {i.caption ? i.caption : <span className="muted">none</span>}
            </div>
          ))}

          <div className="gut">Match</div>
          {group.images.map((i) => (
            // `match()` already opens with the transforms. Appending them again
            // here was printing every one twice — the title carries the whole
            // string because this cell ellipsises like every other one.
            <div key={i.name} className="cell muted" title={match(i)}>
              {match(i)}
            </div>
          ))}
        </div>
      </div>

      <p className="muted dupes-keys">
        Click a picture to move it between Keep and Delete · Esc to leave.
        {' '}The last Keep in a group cannot be dropped.
      </p>
    </div>
  )
}
