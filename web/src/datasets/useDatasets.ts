import { useCallback, useEffect, useRef, useState } from 'react'

import type { ApiError } from '../api/client'
import { failed } from '../api/client'
import {
  createDataset, datasetInsight, deleteDataset, getDataset, listDatasets, saveDataset,
  setDatasetMeta,
} from '../api/routes'
import type { Insight } from '../api/types'
import { useBusy } from '../ui/useBusy'
import { SESSION, beat, keepAlive } from './session'

/**
 * The dataset list, the open set, and its insight.
 *
 * **Saving a set is a choice, and it is the only thing `drafts/` means.** Dropping images
 * makes a draft: it captions, filters and trains exactly like a saved set — same folder,
 * same sidecars, same code path — and the only difference is which parent it sits under.
 * The page never asks for a name before the images are in front of you, because "is this
 * worth keeping" is not a question you can answer at drop time.
 */
export type DatasetRow = {
  name: string
  /** Images. Deliberately not "media": every reader of this means images by it
   *  — the trainer's gate, the rail's line, the button that refuses an empty
   *  set — and only one of the two numbers can be trained on today. */
  count: number
  videos?: number
  uncaptioned?: number
  cover?: string
  saved?: boolean
  trigger_word?: string
}

export type DatasetImage = {
  name: string
  /** A clip has no dimensions here and no thumbnail route: web_image has no
   *  ffmpeg, so the tile paints its own first frame the way a gallery card
   *  does. Absent means image, so a set listed by an older deploy still reads. */
  kind?: 'image' | 'video'
  caption: string
  bytes: number
  width?: number
  height?: number
}

/**
 * The last listing, module-level so it survives the hook unmounting.
 *
 * The hook lives inside Train, so leaving the screen used to throw the rows
 * away — every visit to Sets started from nothing and showed a blank page for
 * the length of the fetch. Now a revisit paints the previous rows instantly
 * and the refresh lands behind them, the same stale-while-revalidate the
 * gallery gets from keeping its listing mounted. Never cleared: a stale row
 * is corrected by the refresh already in flight, and an emptied one is a
 * blank screen again.
 */
let cachedRows: DatasetRow[] | null = null

/** Fill the cache before anyone opens Sets — called once at app mount, the
 *  same warmth the sessions door already gets, so the first click usually
 *  finds the rows waiting. Fire and forget: nothing on screen depends on it,
 *  and a slow first listing is the only cost of it failing. */
export function warmDatasets(): void {
  void listDatasets().then((r) => {
    if (!failed(r)) cachedRows = (r as { datasets?: DatasetRow[] }).datasets ?? []
  })
}

export function useDatasets() {
  const [rows, setRows] = useState<DatasetRow[]>(() => cachedRows ?? [])
  const [loading, setLoading] = useState(cachedRows == null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [images, setImages] = useState<DatasetImage[]>([])
  const [insight, setInsight] = useState<Insight | null>(null)
  const [trigger, setTrigger] = useState('')
  /** An `ApiError` as well as a string, so a failure that came off a route can hand the
   *  server's own report to `ErrorNote` instead of dropping it. The raisers that never
   *  had a server behind them ("Give the set a name to save it.") still pass a bare
   *  string. */
  const [editError, setEditError] = useState<string | ApiError | null>(null)
  const drafts = useRef(false)
  const { busy, run } = useBusy()

  const openRow = rows.find((r) => r.name === open)
  const saved = !!openRow?.saved

  const load = useCallback(async () => {
    const r = await listDatasets()
    if (failed(r)) {
      setLoading(false)
      return setError(r.error)
    }
    setError(null)
    const next = (r as { datasets?: DatasetRow[] }).datasets ?? []
    cachedRows = next
    setRows(next)
    setLoading(false)
  }, [])

  const refreshInsight = useCallback(async (name: string, trig: string) => {
    const r = await datasetInsight(name, trig)
    setInsight(failed(r) ? null : r)
  }, [])

  const loadTiles = useCallback(async (name: string) => {
    const d = await getDataset(name)
    if (failed(d)) return setEditError(d.error)
    setEditError(null)
    const rec = d as { images?: DatasetImage[]; trigger_word?: string }
    setImages(rec.images ?? [])
    setTrigger((cur) => cur || rec.trigger_word || '')
    await refreshInsight(name, rec.trigger_word ?? '')
  }, [refreshInsight])

  const choose = useCallback(async (name: string | null) => {
    setOpen(name)
    setEditError(null)
    setImages([])
    setInsight(null)
    if (name) await loadTiles(name)
  }, [loadTiles])

  // One beat on load whatever the state: it registers this window and sweeps out the
  // drafts of the one you closed.
  useEffect(() => {
    void beat()
  }, [])

  // Only while there is something to keep alive.
  const hasDrafts = rows.some((r) => !r.saved)
  useEffect(() => {
    drafts.current = hasDrafts
    return keepAlive(hasDrafts)
  }, [hasDrafts])

  /** Dropping onto the empty screen is how most sets begin, so the set is created here
   *  rather than being a thing you had to make first. */
  const create = useCallback(async (name: string) => {
    const r = await createDataset({ name, session: SESSION })
    if (failed(r)) {
      setError(r.error)
      return null
    }
    await load()
    await choose(name)
    return name
  }, [choose, load])

  /** A name that is free to use, so dropping never stops to ask for one. A zip carries a
   *  name worth keeping; loose images do not, so they get a numbered one. Either way it is
   *  provisional — this is the draft's handle, and the name that lasts is the one typed
   *  into Save. */
  const suggestName = useCallback((files: File[]) => {
    const zip = files.find((f) => /\.zip$/i.test(f.name))
    const clean = (zip ? zip.name.replace(/\.zip$/i, '') : '')
      .replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 64)
    if (clean && !rows.some((d) => d.name === clean)) return clean
    let n = 1
    while (rows.some((d) => d.name === `set_${n}`)) n++
    return `set_${n}`
  }, [rows])

  const save = useCallback(async (name: string) => {
    if (!open) return
    if (!name.trim()) {
      setEditError('Give the set a name to save it.')
      return
    }
    const r = await saveDataset(open, { name: name.trim() })
    if (failed(r)) return setEditError(r.error)
    setEditError(null)
    await load()
    await choose(String((r as { name?: string }).name ?? name.trim()))
  }, [choose, load, open])

  /** Confirm on anything with images in it, saved or not. A draft is disposable by design,
   *  but forty files you just waited on an upload for are not, and the dashed border does
   *  not make a stray click on a small round × any less easy. The dialog is the whole
   *  safety net — there is no `.trash` behind it — so it says how much is going and that
   *  it is not coming back. */
  const remove = useCallback(async (name: string) => {
    // Checked before the dialog rather than only inside `run`, because the dialog is the
    // expensive half: a delete already in flight has already been confirmed, and asking
    // again puts a modal question about a set that will not exist by the time it is
    // answered. The rail's ✕ painted nothing while the unlink was out, so it was pressed
    // twice as a matter of course.
    if (busy) return
    const d = rows.find((x) => x.name === name)
    const held = (d?.count ?? 0) + (d?.videos ?? 0)
    if (held && !confirm(
      `Permanently delete “${name}”?\n\n${d!.count} image${d!.count === 1 ? '' : 's'}`
      + (d!.videos ? ` and ${d!.videos} clip${d!.videos === 1 ? '' : 's'}` : '')
      + ' and their captions are unlinked from the volume. This cannot be undone.',
    )) return
    // Keyed by name: the rail is a column of ✕ buttons, and the one that is working has
    // to be the one that says so.
    await run(`remove:${name}`, async () => {
      const r = await deleteDataset(name)
      if (failed(r)) return setError(r.error)
      if (name === open) await choose(null)
      await load()
    })
  }, [busy, choose, load, open, rows, run])

  const commitTrigger = useCallback(async (v: string) => {
    if (!open) return
    await setDatasetMeta(open, { trigger_word: v.trim() })
    await refreshInsight(open, v.trim())
  }, [open, refreshInsight])

  return {
    rows, error, open, openRow, saved, images, insight, trigger, editError,
    /** The set whose delete is out, or null. The name rather than a boolean because the
     *  rail draws one ✕ per row and only one of them is doing anything — the others just
     *  go quiet while it does. */
    removing: busy?.startsWith('remove:') ? busy.slice('remove:'.length) : null,
    setTrigger, setImages, setEditError,
    load, loading, choose, loadTiles, create, suggestName, save, remove, commitTrigger,
    refreshInsight,
  }
}
