import { useCallback, useEffect, useMemo, useState } from 'react'

import { failed } from '../api/client'
import { deleteOutput, fileUrl, gallery, purgeOutputs } from '../api/routes'
import { IconBack, IconExpand, IconRefresh } from '../icons'
import { Menu, type MenuItem } from '../ui/Menu'
import { usePopover } from '../ui/Popover'
import { Card } from './Card'
import { reuse } from './reuse'
import { Viewer } from './Viewer'
import type { Filter, GalleryItem } from './types'

/**
 * Three layers, and the third is the one that is easy to miss: the drawer beside the canvas,
 * the full grid, and the picture with its chrome off. That last is entered by *tapping*
 * rather than swiping, and a tap used to close the viewer — so leaving was doing double duty
 * for looking closely, and every attempt to see a render properly dismissed it.
 *
 * Moving between the layers is navigation and lives in the top corners from the moment you
 * are inside — not promoted there after a bottom-centre pill hands you over, which is what a
 * "View all generations" button did and why it read as inconsistent.
 */
export function useGallery() {
  const [items, setItems] = useState<GalleryItem[]>([])
  const [error, setError] = useState<string | null>(null)

  const reload = useCallback(async () => {
    const r = await gallery()
    if (failed(r)) return setError(r.error)
    setError(null)
    setItems(((r as { items?: GalleryItem[] }).items ?? []).filter((i) => i.files?.length))
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  return { items, error, reload }
}

export function Gallery({
  items,
  open,
  onClose,
  onReload,
  onMeta,
  onHandoff,
}: {
  items: GalleryItem[]
  open: boolean
  onClose: () => void
  onReload: () => void
  onMeta: (it: GalleryItem) => void
  onHandoff: (it: GalleryItem, as: 'first' | 'reference' | 'refvideo') => void
}) {
  const [filter, setFilter] = useState<Filter>('all')
  const [viewing, setViewing] = useState<{ rows: GalleryItem[]; i: number } | null>(null)
  const [menuFor, setMenuFor] = useState<GalleryItem | null>(null)
  const pop = usePopover()

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

  const remove = async (it: GalleryItem) => {
    // Deletion unlinks. The confirm dialog is the safety net, so it has to say what is going
    // and how much of it — a dialog that undersells the blast radius is the failure this
    // replaced.
    const n = it.files.length
    const what = n > 1 ? `${n} files` : it.files[0]
    if (!confirm(`Delete ${what}? This cannot be undone.`)) return
    const r = await deleteOutput(it.job_id)
    if (failed(r)) return alert(r.error)
    onReload()
  }

  const purge = async () => {
    // Spells out the count and the scope, because a filtered gallery and an unfiltered one
    // put the same button in the same place over very different amounts of work.
    const n = shown.length
    if (!n) return
    const what = filter === 'all' ? `all ${n} results` : `${n} ${filter}${n === 1 ? '' : 's'}`
    if (!confirm(`Permanently delete ${what}?\n\n`
      + 'The files are unlinked from the volume. This cannot be undone.')) return
    // The ids, not the filter: what goes is exactly what was counted in the dialog, even if
    // a run lands while it is open.
    const r = await purgeOutputs({ confirm: 'delete', job_ids: shown.map((i) => i.job_id) })
    if (failed(r)) return alert(r.error)
    onReload()
  }

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

  const cards = (rows: GalleryItem[]) => rows.map((it, i) => (
    <Card key={`${it.job_id}:${it.files[0]}`} item={it}
          onOpen={() => setViewing({ rows, i })}
          onDownload={() => download(it)}
          onDelete={() => void remove(it)}
          onMenu={(anchor) => {
            setMenuFor(it)
            pop.at(anchor)
          }} />
  ))

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
          <div id="drawer-grid" className="grid">{cards(items.slice(0, 24))}</div>
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
            <button className="ico" title="Refresh" type="button" onClick={onReload}>
              <IconRefresh />
            </button>
            <button className="pill danger" type="button" disabled={!shown.length}
                    onClick={() => void purge()}>
              {!shown.length ? 'Delete all'
                : `Delete ${shown.length} ${filter === 'all' ? 'result' : filter}`
                  + (shown.length === 1 ? '' : 's')}
            </button>
          </div>

          <div id="gal-grid" className="grid">{cards(shown)}</div>
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
